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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=0.1)
    # parser.add_argument("--l2reg", type=float, default=1e-5)
    parser.add_argument("--num_classes", type=int, default=10)
    # parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=256)
    # parser.add_argument("--lr", type=float, default=2e-5)
    # parser.add_argument("--epochs", type=int, default=1000)
    # parser.add_argument("--fre_verbose", type=int, default=5,
    #                     help="checkout for each _ number of training epoch")
    #
    # parser.add_argument("--patience", type=int, default=10)
    # parser.add_argument("--file", type=str, default="restaurant2014",
    #                     help="'restaurant2014', 'laptop2014' or 'restaurant2016'")
    # parser.add_argument("--optimizer", type=str, default="adam")
    # parser.add_argument("--shuffle",action='store_true', default=False)
    # parser.add_argument("--load_model",action='store_true',default=False)
    parser.add_argument("--pretrained_bert_name", type=str, default='bert-base-uncased')
    parser.add_argument("--state_dict_path",type=str,default="12:55:31_val_acc_90.22.pth")
    args = parser.parse_args()
    args.state_dict_path = "checkout/state_dict/" + args.state_dict_path
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
            # a = "the design and atmosphere is just as good."
            a = sys.stdin.readline().strip()
            if a == 'exit':
                break
            token_list = tokenizer.text_to_ids(a)
            print(token_list)
            attention_mask = torch.tensor(
                [1 if x != 0 else 0 for x in token_list]).view(1,-1).to(args.device)
            inputs = torch.tensor(token_list).view(1,-1).to(args.device)
            print(tokenizer.ids_to_tokens(token_list))
            outputs = model(input_ids=inputs,attention_mask=attention_mask)
            outputs = torch.argmax(outputs, dim=-1).cpu().numpy()
            print(outputs)
            # outputs = outputs.item()
            pred = tokenizer.ids_to_tokens(outputs[0],is_pred=True)
            print(pred)


if __name__ == "__main__":
    main()