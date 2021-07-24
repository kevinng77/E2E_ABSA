import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
import os
from transformers.file_utils import ModelOutput
from allennlp.modules.elmo import batch_to_ids


class Tokenizer:
    def __init__(self, 
                 args,
                 pos_token=None, 
                 senti_token=None,
                 ):
        self.model_name = args.model_name
        assert self.model_name in ["bert","elmo","glove"],\
            f"{self.model_name} not implemented"
        if self.model_name == "bert":
            self.bert_tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_name)
            self.pad_token = self.bert_tokenizer.pad_token
        elif self.model_name == "elmo":
            self.max_char_len = 50
            self.pad_token = "[PAD]"
        elif self.model_name == "glove":
            self.idx_path = 'data/glove/glove_idx.npy'
            self.pad_token = 5101

        self.max_seq_len = args.max_seq_len

        # Vocab for target tokenizer
        if senti_token is None:
            senti_token = ["pos", "neg", "neu"]
        if pos_token is None:
            pos_token = ["B", "I", "E"]
        self.target_pad_token_id = 99
        vocab = {"O": 0, self.pad_token: self.target_pad_token_id}
        token = 1
        for senti in senti_token:
            for pos in pos_token:
                vocab[pos + "-" + senti] = token
                token += 1
        self.vocab = vocab
        self.id2vocab = {vocab[x]:x for x in vocab.keys()}

    def edit_len(self, text_ids, is_target):
        """
        padding ids for data batch
        is_target bool: is padding for target token ids?
        """
        if is_target:
            padding = self.target_pad_token_id
            if len(text_ids) > self.max_seq_len:
                return text_ids[:self.max_seq_len]
            else:
                return text_ids + [padding] * (self.max_seq_len - len(text_ids))
        elif self.model_name == "elmo":
            # text_ids [batch_size, len_seq, max_char_len] for elmo
            return torch.cat([text_ids,
                                   torch.zeros([self.max_seq_len - text_ids.shape[0],
                                                self.max_char_len])]).numpy().tolist()

        elif self.model_name == "bert":
            padding = self.bert_tokenizer.pad_token_id
            if len(text_ids) > self.max_seq_len:
                return text_ids[:self.max_seq_len]
            else:
                return text_ids + [padding] * (self.max_seq_len - len(text_ids))

        elif self.model_name == "glove":
            padding = self.pad_token
            if len(text_ids) > self.max_seq_len:
                return text_ids[:self.max_seq_len]
            else:
                return text_ids + [padding] * (self.max_seq_len - len(text_ids))

    def tokens_to_ids(self, text, is_target=False):
        """
        text: list[str] list of words in a sentence.
        """
        if is_target:
            sequence = [self.vocab[x] for x in text]
        elif self.model_name == "bert":
            sequence = self.bert_tokenizer.convert_tokens_to_ids(text)
        elif self.model_name == "elmo":
            sequence = batch_to_ids([text])[0]
        else:
            path = self.idx_path
            word2idx = np.load(path,allow_pickle=True)
            word2idx  = word2idx.tolist()
            sequence = list(map(lambda w: word2idx.get(w, 0), text))
        if len(sequence) == 0:
            sequence = [0]
        return self.edit_len(sequence, is_target)

    def text_to_ids(self, text, is_target=False):
        if self.model_name == "bert":
            sequence = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(text))
        elif self.model_name == "elmo":
            sequence = batch_to_ids([text.split()])[0]
        else:
            path = self.idx_path
            word2idx = np.load(path,allow_pickle=True)
            word2idx  = word2idx.tolist()
            sequence = list(map(lambda w: word2idx.get(w, 0), text.split()))
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
            att_mask = [1 if x != tokenizer.target_pad_token_id else 0 for x in pred_ids]
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
