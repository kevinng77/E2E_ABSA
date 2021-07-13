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
    def __init__(self, d_model, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(d_model, hidden_dim, layer_dim, batch_first=True)# （batch_size,seq_len,input_size)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # out-->(batch_size,seq_len,hidden_size)
        # out[:, -1, :] --> just want last time step hidden states! 
        # out = self.fc(out[:, -1, :]) 
        return out