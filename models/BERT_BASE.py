import torch.nn as nn
import torch
from transformers import BertModel
import sys
sys.path.append("..")
from models.downstream import SelfAttention

# embedding:


class BERT(nn.Module):
    def __init__(self, bert, args):
        super(BERT, self).__init__()
        self.bert = bert
        self.d_model = bert.config.hidden_size
        self.dropout = nn.Dropout(args.dropout)
        self.downstream = args.downstream
        self.classifier = nn.Linear(self.d_model, args.num_classes)
        if args.downstream == "linear":
            pass
        elif args.downstream == "lstm":
            pass
            # todo ADD LSTM
        elif args.downstream == "self_attention":
            self.downstream = SelfAttention(d_model=self.d_model,
                                            num_heads=args.num_heads,
                                            dropout=args.dropout)
        elif args.downstream == "crf":
            pass
            # TODO
        else:
            assert f"downstream model {args.downstream} not in linear, lstm, self_attention or crf"

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        attention_mask:
            mask on padding position to avoid attention.
            Different from attention mask in transformers
        """
        outputs, _ = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,  # 0 if padding
                               token_type_ids=token_type_ids)  # segment id
        # ignore pooling
        outputs = self.dropout(outputs)
        if self.downstream == "self_attention":
            outputs = outputs.transpose(0,1)  # to: (len_seq, batch_size, d_model)
            outputs = self.downstream(outputs, key_padding_mask=attention_mask)
            outputs = outputs.transpose(0, 1)
        elif self.downstream == "lstm":
            pass
            # TODO
        elif self.downstream == "crf":
            pass
            # TODO
        final_outputs = self.classifier(outputs)
        return final_outputs

