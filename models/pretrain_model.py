import torch.nn as nn
import sys
import numpy as np
import torch

sys.path.append("..")
from models.downstream import SelfAttention, LSTM, CRF


class PretrainModel(nn.Module):
    def __init__(self, pretrain_model, args):
        super(PretrainModel, self).__init__()
        if args.model_name == "bert":
            self.bert = pretrain_model
            self.d_model = pretrain_model.config.hidden_size
        elif args.model_name == "elmo":
            self.elmo = pretrain_model
            self.d_model = 1024
        elif args.model_name == "glove":
            def create_emb_layer(non_trainable=False):
                path = args.glove_path
                weights_matrix = torch.tensor(np.load(path,allow_pickle=True))
                num_embeddings, embedding_dim = weights_matrix.size()
                emb_layer = nn.Embedding(num_embeddings, embedding_dim,padding_idx=5101)
                emb_layer.load_state_dict({'weight': weights_matrix})
                if non_trainable:
                    emb_layer.weight.requires_grad = False
                return emb_layer,  embedding_dim
            self.glove,self.d_model = create_emb_layer()
        self.pretrain_type = args.model_name
        self.dropout = nn.Dropout(args.dropout)
        self.ds_name = args.downstream
        self.classifier = nn.Linear(self.d_model, args.num_classes)
        self.model_name = f"{args.model_name}-{args.downstream}"
        assert args.downstream in ['linear', 'lstm', "san", "crf","lstm-crf"], \
            f"downstream model {args.downstream} not in linear, lstm, san or crf"

        if args.downstream == "linear":
            pass
        elif args.downstream == "lstm":
            self.lstm = LSTM(
                d_model=self.d_model,
                hidden_dim=self.d_model,
                num_layers=args.num_layers,
                args=args)
        elif args.downstream == "san":
            self.san = SelfAttention(d_model=self.d_model,
                                     num_heads=args.num_heads,
                                     dropout=args.dropout)
        elif args.downstream == "crf":
            self.crf = CRF(num_tags=args.num_classes,
                           include_start_end_transitions=True)

        elif args.downstream == "lstm-crf":
            self.lstm = LSTM(
                d_model=self.d_model,
                hidden_dim=self.d_model,
                num_layers=args.num_layers,
                args=args)
            self.crf = CRF(num_tags=args.num_classes,
                           include_start_end_transitions=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        attention_mask:  mask on padding position.
        """
        if self.pretrain_type == "bert":
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,  # 0 if padding
                                token_type_ids=token_type_ids)  # segment id
            outputs = self.dropout(outputs.last_hidden_state)

        elif self.pretrain_type == 'elmo':
            outputs = self.elmo(inputs=input_ids)
            outputs = self.dropout(outputs['elmo_representations'][0])
        
        else:
            outputs = self.glove(input_ids)
            outputs = self.dropout(outputs)

        if self.ds_name == "san":
            outputs = self.san(outputs.transpose(0, 1),
                               key_padding_mask=(attention_mask == 1))
            outputs = outputs.transpose(0, 1)
        elif self.ds_name.startswith("lstm"):
            outputs = self.lstm(outputs)

        # linear projector after lstm, self-attention and before crf
        outputs = self.classifier(outputs)

        if self.ds_name.endswith('crf'):
            loss = -self.crf(inputs=outputs,
                             tags=labels,
                             mask=attention_mask)
            return loss, outputs
        else:
            return outputs
