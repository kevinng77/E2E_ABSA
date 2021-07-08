import torch.nn as nn
import torch
from transformers import BertModel


# embedding:


class BERT(nn.Module):
    def __init__(self, bert, args):
        super(BERT, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(bert.config.hidden_size, args.num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs, _ = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,  # 0 if padding
                               token_type_ids=token_type_ids)  # segment id
        # ignore pooling
        outputs = self.dropout(outputs)
        outputs = self.classifier(outputs)
        return outputs

# model
