# -*- coding: utf-8 -*-
# @Time        : 2020/5/3 17:57
# @Author      : ssxy00
# @File        : bert_ner_model.py
# @Description :

import torch
import torch.nn as nn
from transformers import BertModel
from trainer.loss import LabelSmoothingCrossEntropy, FocalLoss

class BertNerModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path, n_labels, loss_type="ce", label_pad_id=0, ce_weight=None,
                 gamma=2):
        super(BertNerModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.n_labels = n_labels
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_labels)
        self.init_weights(self.classifier)
        assert loss_type in ["ce", "focal", "smoothing"], "unknown loss type"
        if loss_type == "ce":
            if ce_weight is not None:
                ce_weight = torch.tensor(ce_weight, dtype=torch.float)
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=label_pad_id, weight=ce_weight)
        elif loss_type == "smoothing":
            self.loss_fn = LabelSmoothingCrossEntropy(ignore_index=label_pad_id)
        else:
            self.loss_fn = FocalLoss(ignore_index=label_pad_id, gamma=gamma)
            print(f"loss type: {loss_type}, gamma: {gamma}")


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
            loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return loss, logits
        return logits


