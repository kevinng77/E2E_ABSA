import torch.nn as nn
import torch
from transformers import BertModel
from models.downstream import SelfAttention, LSTM, CRF
import sys

sys.path.append("..")


class BERT(nn.Module):
    def __init__(self, bert, args):
        super(BERT, self).__init__()
        self.bert = bert
        self.d_model = bert.config.hidden_size
        self.dropout = nn.Dropout(args.dropout)
        self.ds_name = args.downstream
        self.classifier = nn.Linear(self.d_model, args.num_classes)
        self.model_name = f"{args.model_name}-{args.downstream}"
        assert args.downstream in ['linear', 'lstm', "self_attention", "crf"], \
            f"downstream model {args.downstream} not in linear, lstm, self_attention or crf"

        if args.downstream == "linear":
            pass
        elif args.downstream == "lstm":
            self.downstream = LSTM(
                d_model=self.d_model,
                hidden_dim=self.d_model,
                num_layers=args.num_layers,
                args=args
            )
        elif args.downstream == "self_attention":
            self.downstream = SelfAttention(d_model=self.d_model,
                                            num_heads=args.num_heads,
                                            dropout=args.dropout)
        elif args.downstream == "crf":
            self.downstream = CRF(num_tags=args.num_classes,
                                  include_start_end_transitions=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
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
        if self.ds_name == "self_attention":
            # to: (len_seq, batch_size, d_model)
            outputs = self.downstream(outputs.transpose(0, 1), key_padding_mask=(attention_mask == 1))
            outputs = outputs.transpose(0, 1)
        elif self.ds_name == "lstm":
            outputs = self.downstream(outputs)

        outputs = self.classifier(outputs)

        if self.ds_name == 'crf':
            loss = -self.downstream(inputs=outputs,
                                     tags=labels,
                                     mask=attention_mask)
            return loss, outputs
        else:
            return outputs
