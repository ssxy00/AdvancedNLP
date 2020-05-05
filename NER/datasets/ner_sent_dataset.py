# -*- coding: utf-8 -*-
# @Time        : 2020/5/3 00:06
# @Author      : ssxy00
# @File        : ner_sent_dataset.py
# @Description :

import logging
from typing import Dict
import torch
from torch.utils.data import Dataset

class NerSentDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, label2idx, max_seq_len=-1): # 默认不限制长度
        super(NerSentDataset, self).__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        logging.info("reading and indexing data")
        tokens, labels = self.read_data(dataset_path)
        logging.info("padding data")
        self.pad_tokens = self.pad_data(tokens, pad_id=self.tokenizer.pad_token_id, padding_label=False)
        self.pad_labels = self.pad_data(labels, pad_id=self.label2idx["PAD"])

    def __len__(self):
        return self.pad_tokens.shape[0]

    def __getitem__(self, item):
        return self.pad_tokens[item, :], self.pad_labels[item, :]


    def read_data(self, dataset_path):
        tokens_of_all = []
        labels_of_all = []
        with open(dataset_path) as fin:
            for line in fin:
                if not line.strip():
                    continue
                token_with_labels = line.split()
                tokens_of_line = []
                labels_of_line = []
                for token_with_label in token_with_labels:
                    token, label = token_with_label.split("<###>")
                    sub_tokens = self.tokenizer.tokenize(token)
                    tokens_of_line.extend(self.tokenizer.convert_tokens_to_ids(sub_tokens))
                    labels_of_line.extend([self.label2idx[label]] + [self.label2idx["X"]] * (len(sub_tokens) - 1))
                if self.max_seq_len and not len(tokens_of_line) > self.max_seq_len:
                    # 丢弃掉长度太长的序列
                    tokens_of_all.append(tokens_of_line)
                    labels_of_all.append(labels_of_line)
        return tokens_of_all, labels_of_all

    def pad_data(self, tokens, pad_id, padding_label=True):
        # padding, adding [CLS] and [SEP], converting to pytorch tensor
        max_seq_len = max(len(line) for line in tokens)
        pad_data = []
        for line in tokens:
            if padding_label:
                # label 的 [CLS] [SEP] 位置不算入 loss
                pad_data.append([pad_id] + line + [pad_id] + [pad_id] * (max_seq_len - len(line)))
            else:
                pad_data.append([self.tokenizer.cls_token_id] + line + [self.tokenizer.sep_token_id] +
                                  [pad_id] * (max_seq_len - len(line)))
        pad_data = torch.tensor(pad_data, dtype=torch.long)
        return pad_data





