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

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not os.path.exists('checkout'):
    os.mkdir('checkout')
handler = logging.FileHandler("checkout/log.txt")
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

    def _gen_inputs(self, data):
        inputs = data["text_ids"].to(self.args.device)
        target = data["pred_ids"].to(self.args.device)
        attention_mask = data["att_mask"].to(self.args.device)
        return inputs, target, attention_mask

    def _train_epoch(self, dataloader, criterion, optimizer):
        self.model.train()
        losses = 0
        acces = 0
        num_samples = len(dataloader.dataset)
        for data in dataloader:
            optimizer.zero_grad()
            inputs, target, attention_mask = self._gen_inputs(data)
            output = self.model(inputs, attention_mask=attention_mask)
            loss = criterion(output.view(-1, self.args.num_classes), target.view(-1))
            loss.backward()
            optimizer.step()
            acc = metrics.compute_acc(output, target, attention_mask)
            acces += acc
            losses += loss
        return losses / num_samples, self.args.batch_size * acces / num_samples

    def _dev_epoch(self, dataloader, criterion):
        self.model.eval()
        losses = 0
        acces = 0
        num_samples = len(dataloader.dataset)
        with torch.no_grad():
            for data in dataloader:
                inputs, target, attention_mask = self._gen_inputs(data)
                output = self.model(inputs, attention_mask=attention_mask)
                loss = criterion(output.view(-1, self.args.num_classes), target.view(-1))
                acc = metrics.compute_acc(output, target, attention_mask)
                losses += loss
                acces += acc
        return losses / num_samples, self.args.batch_size * acces / num_samples

    def run(self):
        max_val_acc = 0
        max_val_epoch = 0
        for arg in vars(self.args):
            logger.info(f'>>> {arg}: {getattr(self.args, arg)}')
        weight = [1.2 for _ in range(self.args.num_classes)]
        weight[0] = 0.8
        criterion_weight = torch.tensor(weight).to(self.args.device)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id,
                                        weight=criterion_weight)
        optimizer = self.args.optimizer(self.model.parameters()
                                        , lr=self.args.lr, weight_decay=self.args.l2reg)

        train_dataloader = DataLoader(E2EABSA_dataset(file_path=self.args.file_path['train'],
                                                      tokenizer=self.tokenizer),
                                      batch_size=self.args.batch_size,
                                      shuffle=self.args.shuffle,
                                      drop_last=True)
        dev_dataloader = DataLoader(E2EABSA_dataset(file_path=self.args.file_path['dev'],
                                                    tokenizer=self.tokenizer),
                                    batch_size=self.args.batch_size,
                                    shuffle=self.args.shuffle,
                                    drop_last=True)
        for epoch in range(self.args.epochs+1):
            loss, acc = self._train_epoch(dataloader=train_dataloader,
                                          criterion=criterion,
                                          optimizer=optimizer)
            if epoch % self.args.fre_verbose == 0:
                dev_loss, dev_acc = self._dev_epoch(dataloader=dev_dataloader,
                                                    criterion=criterion)

                logger.info(f"> epoch: {epoch}, train loss: {loss*100:.4f} acc: {acc * 100:.3f}% "
                            f"dev loss: {dev_loss*100:.4f} acc: {dev_acc * 100:.3f}%")

                if dev_acc > max_val_acc:
                    max_val_acc = dev_acc
                    max_val_epoch = epoch
                    if not os.path.exists('checkout/state_dict'):
                        os.mkdir('checkout/state_dict')
                    now = str(datetime.now())[11:19]
                    path = f'checkout/state_dict/{now}_val_acc_{dev_acc * 100:.2f}.pth'
                    torch.save(self.model.state_dict(), path)
                    logger.info(f'>> saved: {path}')
                if epoch - max_val_epoch >= (self.args.patience * self.args.fre_verbose):
                    print('>> early stop.')
                    break
        now = str(datetime.now())[11:19]
        path = f'checkout/state_dict/{now}_val_acc_final.pth'
        torch.save(self.model.state_dict(), path)
        logger.info(f'>> saved: {path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--l2reg", type=float, default=1e-5)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--fre_verbose", type=int, default=5,
                        help="checkout for each _ number of training epoch")

    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--file", type=str, default="restaurant2014",
                        help="'restaurant2014', 'laptop2014' or 'restaurant2016'")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--shuffle",action='store_true', default=False)
    parser.add_argument("--load_model",action='store_true',default=False)
    parser.add_argument("--pretrained_bert_name", type=str, default='bert-base-uncased')
    parser.add_argument("--state_dict_path",type=str,default="2021-06-16 08:40:51_val_acc_0.34.pth")
    parser.add_argument("--seed",type=int,default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_files = {
        'restaurant2014': {
            'train': './data/Semeval2014/processed/Restaurants_Train.csv',
            'dev': './data/Semeval2014/processed/Restaurants_dev.csv',
            'test': './data/Semeval2014/processed/Restaurants_test.csv'
        },
        'debug':{
            'train': './data/Semeval2014/processed/debug.csv',
            'dev': './data/Semeval2014/processed/debug.csv',
            'test': './data/Semeval2014/processed/Restaurants_test.csv'
        },

        'laptop2014': {
            # todo adjust path
            'train': './data/semeval14/Laptops_Train.xml.seg',
            'test': './data/datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }

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
    args.file_path = dataset_files[args.file]
    args.optimizer = optimizers[args.optimizer]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # config = BertConfig(args.pretrained_bert_name,
    #                     num_labels=args.num_classes,
    #                     finetuning_task=args.task_name,
    #                     cache_dir="./cache")

    print(f"> Loading bert model {args.pretrained_bert_name}")
    tokenizer = Tokenizer(args.max_seq_len, args.pretrained_bert_name)
    bert = BertModel.from_pretrained(args.pretrained_bert_name)
    model = BERT(bert, args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=args)
    trainer.run()


if __name__ == "__main__":
    main()
