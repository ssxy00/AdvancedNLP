from transformers import BertTokenizer, BertForSequenceClassification,BertPreTrainedModel,BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_mean_model():
    return BertForSequenceClassification.from_pretrained('bert-base-uncased')

class BertForSC1(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSC1, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[0]
        pooled_output = pooled_output[:,0,:]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForSC2(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSC2, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = FeedForward(config.hidden_size,400,config.num_labels,config.hidden_dropout_prob)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[0]
        pooled_output = pooled_output[:,0,:]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class BertForSC3(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSC3, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = torch.max(outputs[0].masked_fill((-1*attention_mask+1).bool().unsqueeze(-1),value=0),1)[0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class FeedForward(nn.Module):
    def __init__(self,in_features,middle_features,num_labels,dropout):
        super(FeedForward,self).__init__()
        self.layer1=nn.Linear(in_features,middle_features)
        self.layer2=nn.Linear(middle_features,num_labels)
        self.dropout=nn.Dropout(dropout)
        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.layer1.weight,std=0.02)
        nn.init.normal_(self.layer2.weight,std=0.02)

    def forward(self, x):
        x=F.relu(self.layer1(x))
        x=self.dropout(x)
        x=self.layer2(x)
        return x