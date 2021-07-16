import argparse
from transformers import BertModel
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
from models.BERT_BASE import BERT
from utils.data_utils import E2EABSA_dataset, Tokenizer
import torch.nn as nn
import sys
from config import config
from utils.metrics import F1
from utils.result_helper import init_logger
from torch.utils.data import DataLoader
import logging
import os
from utils import result_helper


def load_model():
    args = config.args
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer(args.max_seq_len, args.pretrained_bert_name)
    bert = BertModel.from_pretrained(args.pretrained_bert_name)
    model = BERT(bert, args).to(args.device)
    model.load_state_dict(torch.load(args.state_dict_path))
    return model, tokenizer, args


def test():
    model, tokenizer, args = load_model()
    model.eval()
    metrics = F1(num_classes=args.num_classes,downstream=args.downstream)
    torch.autograd.set_grad_enabled(False)
    test_dataloader = DataLoader(E2EABSA_dataset(file_path=args.file_path['test'],
                                                 tokenizer=tokenizer),
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 drop_last=False)

    confusion = torch.zeros([4, 4], device=args.device)
    aspect = torch.zeros([2, 2], device=args.device)
    broken = 0
    with torch.no_grad():
        for data in test_dataloader:
            inputs = data["text_ids"].to(args.device)
            target = data["pred_ids"].to(args.device)
            attention_mask = data["att_mask"].to(args.device)

            if args.downstream != "crf":
                output = model(inputs, attention_mask=attention_mask)
                pred = torch.argmax(output,dim=-1).view(-1)
            else:
                _, logits = model(inputs, attention_mask=attention_mask, labels=target)
                output = model.downstream.viterbi_tags(logits=logits, mask=attention_mask)
                # padding outputs
                output = [x + [tokenizer.target_pad_token_id] * (args.max_seq_len - len(x))
                          for x in output]
                pred = torch.tensor(output, dtype=torch.long, device=args.device).view(-1)

            d_aspect,d_confusion,d_broken = result_helper.gen_confusion_matrix(outputs=pred,
                                                                      targets=target.view(-1))
            aspect = aspect + d_aspect.to(args.device)
            confusion = confusion + d_confusion.to(args.device)
            broken += d_broken

    f1_aspect,f1_polarity,f1_total = result_helper.gen_metrics(confusion)
    logger.info(f'{model.model_name}\t'
                f'{args.mode}\t{metrics.name}\t'
                f'aspect {f1_aspect*100:.2f}%\t'
                f'polarity {f1_polarity*100:.2f}%\t'
                f'total {f1_total*100:.2f}%\t'
                f'number of broken prediction {broken}')


def demo():
    model, tokenizer, args = load_model()
    model.eval()
    torch.autograd.set_grad_enabled(False)
    print("input your sentence: ")
    with torch.no_grad():
        while 1:
            a = sys.stdin.readline().strip()
            if a == 'exit':
                break
            token_list = tokenizer.text_to_ids(a)
            attention_mask = torch.tensor(
                [1 if x != 0 else 0 for x in token_list]).view(1, -1).to(args.device)
            inputs = torch.tensor(token_list).view(1, -1).to(args.device)
            print(tokenizer.ids_to_tokens(token_list))
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            # print(torch.argmax(outputs, dim=-1).cpu().numpy())
            outputs = torch.masked_select(torch.argmax(outputs, dim=-1),
                                          attention_mask == 1).cpu().numpy()
            pred = tokenizer.ids_to_tokens(outputs, is_target=True)
            print(pred)


if __name__ == "__main__":
    if config.args.demo:
        demo()
    else:
        logger = init_logger(logging_folder=config.working_path + 'checkout',
                             logging_file=config.working_path + "checkout/test_log.txt")

        test()
