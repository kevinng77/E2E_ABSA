import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
import os


class Tokenizer:
    def __init__(self, max_seq_len, pretrained_bert_name, pos_token=None, senti_token=None):
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len
        if senti_token is None:
            senti_token = ["pos", "neg", "neu"]
        if pos_token is None:
            pos_token = ["B", "I", "E"]
        self.target_pad_token_id = 99
        vocab = {"O": 0, self.bert_tokenizer.pad_token: self.target_pad_token_id}
        token = 1
        for senti in senti_token:
            for pos in pos_token:
                vocab[pos + "-" + senti] = token
                token += 1
        self.vocab = vocab
        self.id2vocab = {vocab[x]:x for x in vocab.keys()}

    def edit_len(self, text, is_target):
        if is_target:
            padding = self.target_pad_token_id
        else:
            padding = self.bert_tokenizer.pad_token_id
        if len(text) > self.max_seq_len:
            return text[:self.max_seq_len]
        else:
            return text + [padding] * (self.max_seq_len - len(text))

    def tokens_to_ids(self, text, is_target=False):
        if is_target:
            sequence = [self.vocab[x] for x in text]
        else:
            sequence = self.bert_tokenizer.convert_tokens_to_ids(text)
        if len(sequence) == 0:
            sequence = [0]
        return self.edit_len(sequence, is_target)

    def text_to_ids(self, text,is_target=False):
        sequence = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        return self.edit_len(sequence,is_target)

    def text_to_tokens(self, text, train=True, aspect=None):
        tokens = self.bert_tokenizer.tokenize(text)
        return tokens

    def ids_to_tokens(self, ids, is_target = False):
        if is_target:
            sequence = [self.id2vocab[x] for x in ids if x != self.target_pad_token_id]
        else:
            sequence = [self.bert_tokenizer._convert_id_to_token(x)
                        for x in ids if x != self.bert_tokenizer.pad_token_id]
        return sequence


class E2EABSA_dataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, "r")as fp:
            lines = fp.readlines()
        dataset = []
        for i in range(0, len(lines), 2):
            text = lines[i]
            gold = lines[i + 1]
            text_ids = tokenizer.tokens_to_ids(text.split())
            pred_ids = tokenizer.tokens_to_ids(gold.split(), is_target=True)
            att_mask = [1 if x != tokenizer.bert_tokenizer.pad_token_id else 0 for x in text_ids]
            data = {
                "text_ids": torch.tensor(text_ids,dtype=torch.long),
                "att_mask": torch.tensor(att_mask,dtype=torch.long),
                "pred_ids": torch.tensor(pred_ids,dtype=torch.long)
            }
            dataset.append(data)
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
