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


def main():
    args = config.args
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer(args.max_seq_len, args.pretrained_bert_name)
    bert = BertModel.from_pretrained(args.pretrained_bert_name)
    model = BERT(bert, args).to(args.device)
    model.load_state_dict(torch.load(args.state_dict_path))
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
            outputs = torch.masked_select(torch.argmax(outputs, dim=-1),
                                          attention_mask == 1).cpu().numpy()
            pred = tokenizer.ids_to_tokens(outputs, is_pred=True)
            print(pred)


if __name__ == "__main__":
    main()
