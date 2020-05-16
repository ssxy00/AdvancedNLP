# -*- coding: utf-8 -*-
# @Time        : 2020/5/6 16:07
# @Author      : ssxy00
# @File        : bert_crf_ner_model.py
# @Description :

import torch.nn as nn
from transformers import BertModel
from models.crf import CRF

class BertCRFNerModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path, n_labels):
        super(BertCRFNerModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.n_labels = n_labels
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_labels)
        self.crf = CRF(num_tags=n_labels, batch_first=True)
        self.init_weights(self.classifier)

    def init_weights(self, module):
        """ Initialize the weights of Linear layer"""
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            return loss, logits

        return logits