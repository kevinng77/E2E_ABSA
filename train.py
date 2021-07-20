from torch import optim
from transformers import BertModel
import torch
from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup
import os
from models.pretrain_model import PretrainModel
from utils.data_utils import E2EABSA_dataset, Tokenizer
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from utils.metrics import F1, FocalLoss
from utils.result_helper import init_logger
from config import config
import time
from allennlp.modules.elmo import Elmo


logger = init_logger(logging_folder=config.working_path + 'checkout',
                     logging_file=config.working_path + "checkout/training_log.txt")


class Trainer(object):
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        if args.load_model:
            self.model.load_state_dict(torch.load(args.state_dict_path))
        self.model = self.model.to(args.device)
        self.model_name = f"{args.model_name}-{args.downstream}"

        self.max_val_step = 0
        self.max_val_acc = 0
        self.train_metric = 0
        self.train_loss = 0
        self.step = 0
        self.time = 0
        self.min_metrics = 0.40  # min F1 metrics to save model

        if args.loss == "CE":
            self.weight = [0.1, 0.8, 1., 1., 1.2, 1.2, 1.2, 1., 1., 1.]
            criterion_weight = torch.tensor(self.weight).to(self.args.device)
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.target_pad_token_id,
                                                 weight=criterion_weight)
        elif args.loss == "focal":
            self.weight = [args.alpha for _ in range(self.args.num_classes)]
            self.criterion = FocalLoss(class_num=args.num_classes,
                                       alpha=self.weight,
                                       gamma=args.gamma,
                                       ignore_index=self.tokenizer.target_pad_token_id,
                                       device=args.device)
        else:
            assert f"loss function {args.loss} only implement 'CE' , 'focal"
        self.optimizer = self.args.optimizer(self.model.parameters(),
                                             **self.args.optimizer_kwargs)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=args.warmup_steps,
                                                         num_training_steps=args.max_steps)

        if args.metrics == "f1":  # future work, change metrics
            self.metrics = F1(args.num_classes, downstream=args.downstream)
        else:
            assert "--metrics only implement f1"

        self.dev_dataloader = DataLoader(E2EABSA_dataset(file_path=self.args.file_path['dev'],
                                                         tokenizer=self.tokenizer),
                                         batch_size=self.args.batch_size,
                                         shuffle=self.args.shuffle,
                                         drop_last=True)
        self.train_dataloader = DataLoader(E2EABSA_dataset(file_path=self.args.file_path['train'],
                                                           tokenizer=self.tokenizer),
                                           batch_size=self.args.batch_size,
                                           shuffle=self.args.shuffle,
                                           drop_last=True)

    def _gen_inputs(self, data):
        inputs = data["text_ids"].to(self.args.device)
        target = data["pred_ids"].to(self.args.device)
        attention_mask = data["att_mask"].to(self.args.device)
        return inputs, target, attention_mask

    def _train_epoch(self, epoch):
        self.model.train()
        TP, FP, FN = 0, 0, 0
        for data in self.train_dataloader:
            self.optimizer.zero_grad()
            inputs, target, attention_mask = self._gen_inputs(data)
            # print(inputs.shape)
            # print(inputs[0])
            # print(target[0])
            # print(attention_mask[0])
            # print(attention_mask.shape)
            if self.model_name.endswith('crf'):
                loss, logits = self.model(inputs, attention_mask=attention_mask, labels=target)
                loss = loss / self.args.batch_size
                output = self.model.downstream.viterbi_tags(logits=logits, mask=attention_mask)
                output = [x + [self.tokenizer.target_pad_token_id] * (self.args.max_seq_len - len(x))
                          for x in output]
                output = torch.tensor(output, dtype=torch.long, device=self.args.device)
            else:
                output = self.model(inputs, attention_mask=attention_mask)
                loss = self.criterion(output.view(-1, self.args.num_classes), target.view(-1))
            # print(output[0])
            # print(output.shape)
            loss.backward()
            dTP, dFP, dFN = self.metrics(output, target, attention_mask)
            TP += dTP
            FP += dFP
            FN += dFN

            self.optimizer.step()
            self.scheduler.step()
            self.train_loss += loss
            self.step += 1

            if self.step % self.args.step == 0:
                self.train_metric = self.metrics.get_f1(TP, FP, FN)
                self._checkout(epoch)
                self.time = time.time()
                self.train_loss, self.train_metric = 0, 0
                TP, FP, FN = 0, 0, 0
                self.model.train()

    def _dev_epoch(self):
        self.model.eval()
        dev_losses = 0
        TP, FP, FN = 0, 0, 0
        count = 0
        with torch.no_grad():
            for data in self.dev_dataloader:
                count += 1
                inputs, target, attention_mask = self._gen_inputs(data)
                if self.model_name.endswith('crf'):
                    loss, logits = self.model(inputs, attention_mask=attention_mask, labels=target)
                    loss = loss / self.args.batch_size
                    output = self.model.downstream.viterbi_tags(logits=logits, mask=attention_mask)
                    # padding outputs
                    output = [x + [self.tokenizer.target_pad_token_id] * (self.args.max_seq_len - len(x))
                              for x in output]
                    output = torch.tensor(output, dtype=torch.long, device=self.args.device)
                else:
                    output = self.model(inputs, attention_mask=attention_mask)
                    loss = self.criterion(output.view(-1, self.args.num_classes), target.view(-1))

                dTP, dFP, dFN = self.metrics(output, target, attention_mask)
                TP += dTP
                FP += dFP
                FN += dFN
                dev_losses += loss
        return dev_losses / count, self.metrics.get_f1(TP, FP, FN, verbose=self.args.verbose)

    def run(self):
        if not os.path.exists('checkout/state_dict'):
            os.mkdir('checkout/state_dict')

        logger.info(f"\n>>>>>>>>>>>>>>>>>>>>>{datetime.now()}>>>>>>>>>>>>>>>>>>>>>>>>")
        for arg in vars(self.args):
            logger.info(f'>>> {arg}: {getattr(self.args, arg)}')
        logger.info(f">>> class weight(or alpha) {self.weight}")

        self.time = time.time()
        for epoch in range(self.args.epochs + 1):
            self._train_epoch(epoch)
            if self.step > self.args.max_steps:
                break

        path = f'checkout/state_dict/{self.model_name}_{self.args.mode}_final.pth'
        torch.save(self.model.state_dict(), path)
        print(f'>> saved: {path}')

    def _checkout(self, epoch):
        train_loss = self.train_loss / self.args.step
        dev_loss, dev_metrics = self._dev_epoch()

        logger.info(f"> Epoch: {epoch} Step: {self.step}, "
                    f"train loss: {train_loss:.4f} "
                    f"{self.metrics.name}: {self.train_metric * 100:.2f}% "
                    f"dev loss: {dev_loss:.4f} "
                    f"{self.metrics.name}: {dev_metrics * 100:.2f}% "
                    f"{(time.time() - self.time) / 60:.2f} min")

        if dev_metrics > self.max_val_acc:
            self.max_val_acc = dev_metrics
            if dev_metrics > self.min_metrics:
                path = f'checkout/state_dict/{self.model_name}_' \
                       f'{self.args.mode}_seed{self.args.seed}.pth'
                torch.save(self.model.state_dict(), path)
                print(f'>> saved: {path}')


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
        'adamw': AdamW,
        'Adafactor': Adafactor
    }
    default_optim_kwargs = {'lr': args.lr, 'weight_decay': args.weight_decay}
    optimizers_kwargs = {
        'adadelta': {},
        'adagrad': {},
        'adam': {"betas": (args.adam_beta1, args.adam_beta2),
                 "eps": args.adam_epsilon,
                 "amsgrad": args.adam_amsgrad},
        'adamax': {},
        'asgd': {},
        'rmsprop': {},
        'sgd': {},
        'adamw': {"betas": (args.adam_beta1, args.adam_beta2),
                  "eps": args.adam_epsilon},
        'Adafactor': {"scale_parameter": False, "relative_step": False}
    }
    assert args.optimizer in list(optimizers.keys()), \
        f"Optimizer only support {list(optimizers.keys())}"
    args.optimizer_kwargs = optimizers_kwargs[args.optimizer]
    args.optimizer_kwargs.update(default_optim_kwargs)
    args.optimizer = optimizers[args.optimizer]
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = None
    tokenizer = Tokenizer(args=args)
    if args.model_name.startswith("bert"):
        print(f"> Loading bert model {args.pretrained_bert_name}")
        bert = BertModel.from_pretrained(args.pretrained_bert_name)
        model = PretrainModel(pretrain_model=bert, args=args)
    elif args.model_name.startswith("elmo"):
        print(f"> Loading elmo model")
        elmo = Elmo(options_file=config.options_file,
                    weight_file=config.weight_file,
                    num_output_representations=1,
                    dropout=0,
                    requires_grad=args.finetune_elmo)
        model = PretrainModel(pretrain_model=elmo, args=args)
    else:
        assert f"model {args.model_name} not implement "
    trainer = Trainer(model=model, tokenizer=tokenizer, args=args)
    trainer.run()


if __name__ == "__main__":
    main(config.args)
