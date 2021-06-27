import torch
import torch.nn as nn


def compute_acc(outputs, target, attention_mask):
    """
    outputs: hidden state after softmax
    outputs = torch.tensor([[[0.9,0.5,0.3],[0.1,0.2,0.4],[0.2,0.3,0.1],[0.6,0.3,0.1]],
                       [[0.9,0.5,0.3],[0.1,0.2,0.4],[0.2,0.3,0.1],[0.6,0.3,0.1]]])
    attention_mask = torch.tensor([[1,1,1,0],[1,1,0,0]])
    targets = torch.tensor([[0,1,1,0],[1,2,1,0]])
    """
    with torch.no_grad():
        pred = torch.argmax(outputs, dim=-1)
        print(torch.sum(pred).item())
        acc = torch.sum((pred == target) & (attention_mask != 0)) / torch.sum(attention_mask)
    # print(torch.sum((0 == target) & (attention_mask != 0)) / torch.sum(attention_mask))
    return acc.item()
