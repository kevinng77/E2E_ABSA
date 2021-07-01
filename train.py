import argparse
from transformers import BertModel, BertConfig
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
import os
from models.BERT_BASE import BERT
from utils.data_utils import E2EABSA_dataset, Tokenizer
import torch.nn as nn
import logging
import sys
from torch.utils.data import DataLoader
from datetime import datetime
from utils import metrics
from config import config


logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not os.path.exists(config.working_path + 'checkout'):
    os.mkdir(config.working_path + 'checkout')
handler = logging.FileHandler(config.working_path + "checkout/data_processing_log.txt")
handler.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.addHandler(handler)


class Trainer(object):
        def __init__(self, model, tokenizer, args):
            self.model = model
            self.tokenizer = tokenizer
            self.args = args
            if args.load_model:
                self.model.load_state_dict(torch.load(args.state_dict_path))
            self.model = self.model.to(args.device)
            self.step = 0
            self.name = args.model_name

            weight = [1.2 for _ in range(self.args.num_classes)]
            weight[0] = 0.8
            criterion_weight = torch.tensor(weight).to(self.args.device)
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.tokenizer.pad_token_id,
                                            # weight=criterion_weight
            )

        def _gen_inputs(self, data):
            inputs = data["text_ids"].to(self.args.device)
            target = data["pred_ids"].to(self.args.device)
            attention_mask = data["att_mask"].to(self.args.device)
            return inputs, target, attention_mask

        def _train_epoch(self,epoch):
            self.model.train()
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                inputs, target, attention_mask = self._gen_inputs(data)
                output = self.model(inputs, attention_mask=attention_mask)
                loss = self.criterion(output.view(-1, self.args.num_classes), target.view(-1))
                loss.backward()
                self.optimizer.step()
                acc = metrics.compute_acc(output, target, attention_mask)
                self.train_acc += acc
                self.train_loss += loss
                self.step += 1
                if self.step % self.args.step == 0:
                    train_loss = self.train_loss/(self.args.step * self.args.batch_size)
                    train_acc = self.train_acc/self.args.step
                    self._checkout(train_loss,train_acc,epoch)
                    self.train_loss, self.train_acc = 0, 0

        def _dev_epoch(self):
            self.model.eval()
            losses = 0
            acces = 0
            num_samples = len(self.dev_dataloader.dataset)
            with torch.no_grad():
                for data in self.dev_dataloader:
                    inputs, target, attention_mask = self._gen_inputs(data)
                    output = self.model(inputs, attention_mask=attention_mask)
                    loss = self.criterion(output.view(-1, self.args.num_classes), target.view(-1))
                    acc = metrics.compute_acc(output, target, attention_mask)
                    losses += loss
                    acces += acc
            return losses / num_samples, self.args.batch_size * acces / num_samples

        def _checkout(self,loss,acc,epoch):
            dev_loss, dev_acc = self._dev_epoch()
            logger.info(f"> Epoch: {epoch} Step: {self.step}, train loss: {loss * 100:.4f} acc: {acc * 100:.3f}% "
                        f"dev loss: {dev_loss * 100:.4f} acc: {dev_acc * 100:.3f}%")
            if dev_acc > self.max_val_acc:
                self.max_val_acc = dev_acc
                if not os.path.exists('checkout/state_dict'):
                    os.mkdir('checkout/state_dict')
                path = f'checkout/state_dict/{self.name}_val_acc_{dev_acc * 100:.2f}.pth'
                torch.save(self.model.state_dict(), path)
                logger.info(f'>> saved: {path}')

        def run(self):
            self.train_loss = 0
            self.train_acc = 0
            self.max_val_acc = 0
            self.max_val_step = 0
            self.optimizer = self.args.optimizer(self.model.parameters()
                                            , lr=self.args.lr, weight_decay=self.args.l2reg)

            for arg in vars(self.args):
                logger.info(f'>>> {arg}: {getattr(self.args, arg)}')

            self.train_dataloader = DataLoader(E2EABSA_dataset(file_path=self.args.file_path['train'],
                                                          tokenizer=self.tokenizer),
                                          batch_size=self.args.batch_size,
                                          shuffle=self.args.shuffle,
                                          drop_last=True)
            self.dev_dataloader = DataLoader(E2EABSA_dataset(file_path=self.args.file_path['dev'],
                                                        tokenizer=self.tokenizer),
                                        batch_size=self.args.batch_size,
                                        shuffle=self.args.shuffle,
                                        drop_last=True)
            for epoch in range(self.args.epochs+1):
                self._train_epoch(epoch)
                # if epoch % self.args.step == 0:
                    # dev_loss, dev_acc = self._dev_epoch()

                    # logger.info(f"> epoch: {epoch}, train loss: {loss*100:.4f} acc: {acc * 100:.3f}% "
                    #             f"dev loss: {dev_loss*100:.4f} acc: {dev_acc * 100:.3f}%")

                    # if dev_acc > max_val_acc:
                    #     max_val_acc = dev_acc
                    #     max_val_epoch = epoch
                    #     if not os.path.exists('checkout/state_dict'):
                    #         os.mkdir('checkout/state_dict')
                    #     path = f'checkout/state_dict/{self.name}_val_acc_{dev_acc * 100:.2f}.pth'
                    #     torch.save(self.model.state_dict(), path)
                    #     logger.info(f'>> saved: {path}')
                    # if epoch - max_val_epoch >= (self.args.patience * self.args.step):
                    #     print('>> early stop.')
                    #     break
            path = f'checkout/state_dict/{self.name}_val_acc_final.pth'
            torch.save(self.model.state_dict(), path)
            logger.info(f'>> saved: {path}')


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
    }

    args.state_dict_path = "checkout/state_dict" + args.state_dict_path
    args.file_path = config.processed_data_path[args.mode]
    args.optimizer = optimizers[args.optimizer]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"> Loading bert model {args.pretrained_bert_name}")
    tokenizer = Tokenizer(args.max_seq_len, args.pretrained_bert_name)
    bert = BertModel.from_pretrained(args.pretrained_bert_name)
    model = BERT(bert, args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=args)
    trainer.run()


if __name__ == "__main__":
    main(config.args)
