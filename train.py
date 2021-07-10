import argparse
from transformers import BertModel
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import os
from models.BERT_BASE import BERT
from utils.data_utils import E2EABSA_dataset, Tokenizer
import torch.nn as nn
import logging
import sys
from torch.utils.data import DataLoader
from datetime import datetime
from utils.metrics import Accuracy, F1, FocalLoss
from utils.result_helper import init_logger
from config import config
import time


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
        self.model_name = args.model_name

        self.max_val_step = 0
        self.max_val_acc = 0
        self.train_metric = 0
        self.train_loss = 0
        self.step = 0
        self.min_metrics = 0.58  # min metrics to save model

        if args.loss == "CE":
            self.weight = [1.0 for _ in range(self.args.num_classes)]
            self.weight[0] = 7e-2  # Adam ok
            self.weight[-4] = 1.1
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
            assert f"loss function {args.loss} not in 'CE' , 'focal"
        self.optimizer = self.args.optimizer(self.model.parameters(),
                                             lr=self.args.lr,
                                             weight_decay=self.args.l2reg)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=args.warmup_steps,
                                                         num_training_steps=3000)

        if args.metrics == "f1":  # future work, change metrics
            self.metrics = F1(args.num_classes)
        else:
            assert "--metrics only support f1"

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
        time1 = time.time()
        for data in self.train_dataloader:
            self.optimizer.zero_grad()
            inputs, target, attention_mask = self._gen_inputs(data)
            # print(inputs[0])
            # print(target[0])
            output = self.model(inputs, attention_mask=attention_mask)
            loss = self.criterion(output.view(-1, self.args.num_classes), target.view(-1))
            loss.backward()
            self.optimizer.step()
            dTP, dFP, dFN = self.metrics(output, target, attention_mask)
            TP += dTP
            FP += dFP
            FN += dFN
            self.train_loss += loss
            self.step += 1

            if self.step % self.args.step == 0:
                self.train_metric = self.metrics.get_f1(TP, FP, FN)
                self._checkout(epoch, time1)
                time1 = time.time()
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

        for epoch in range(self.args.epochs + 1):
            self._train_epoch(epoch)

        path = f'checkout/state_dict/{self.model_name}_{self.args.mode}_final.pth'
        torch.save(self.model.state_dict(), path)
        logger.info(f'>> saved: {path}')

    def _checkout(self, epoch, times):
        train_loss = self.train_loss / self.args.step
        dev_loss, dev_metrics = self._dev_epoch()

        logger.info(f"> Epoch: {epoch} Step: {self.step}, "
                    f"train loss: {train_loss:.4f} "
                    f"{self.metrics.name}: {self.train_metric * 100:.2f}% "
                    f"dev loss: {dev_loss:.4f} "
                    f"{self.metrics.name}: {dev_metrics * 100:.2f}% "
                    f"{(time.time() - times) / 60:.2f} min")

        if dev_metrics > self.max_val_acc:
            self.max_val_acc = dev_metrics
            if dev_metrics > self.min_metrics:
                path = f'checkout/state_dict/{self.model_name}_' \
                       f'{self.args.mode}_{self.metrics.name}_{dev_metrics * 100:.2f}.pth'
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
        'adamw':AdamW,
    }

    assert args.optimizer in list(optimizers.keys()), \
        f"Optimizer only support {list(optimizers.keys())}"
    args.optimizer = optimizers[args.optimizer]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = None
    model = None
    if args.model_name == "bert":
        print(f"> Loading bert model {args.pretrained_bert_name}")
        tokenizer = Tokenizer(args.max_seq_len, args.pretrained_bert_name)
        bert = BertModel.from_pretrained(args.pretrained_bert_name)
        model = BERT(bert, args)
    elif args.model_name == "elmo":
        # TODO add elmo model
        pass
    else:
        assert f"model {args.model} not in bert, elmo "
    trainer = Trainer(model=model, tokenizer=tokenizer, args=args)
    trainer.run()


if __name__ == "__main__":
    main(config.args)
