import torch.nn as nn
import torch


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(SelfAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model,
                                               num_heads=num_heads,
                                               dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs, attention_mask=None, key_padding_mask=None):
        inputs_att, _ = self.self_attn(inputs, inputs, inputs,
                                       attn_mask=attention_mask,
                                       key_padding_mask=key_padding_mask)
        outputs = inputs + self.dropout(inputs_att)
        outputs = self.layer_norm(outputs)
        return outputs


class LSTM(nn.Module):
    pass
    # TODO