import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, target):
        if not (target.size() == inputs.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), inputs.size()))

        max_val = (-inputs).clamp(min=0)
        loss = inputs - inputs * target + max_val + \
               ((-max_val).exp() + (-inputs - max_val).exp()).log()

        invprobs = F.logsigmoid(-inputs * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()


class Accuracy(object):
    def __init__(self):
        self.name = "acc"

    def __call__(self, outputs, target, attention_mask):
        """
        outputs: hidden state after softmax
        outputs = torch.tensor([[[0.9,0.5,0.3],[0.1,0.2,0.4],[0.2,0.3,0.1],[0.6,0.3,0.1]],
                           [[0.9,0.5,0.3],[0.1,0.2,0.4],[0.2,0.3,0.1],[0.6,0.3,0.1]]])
        attention_mask = torch.tensor([[1,1,1,0],[1,1,0,0]])
        targets = torch.tensor([[0,1,1,0],[1,2,1,0]])
        """

        with torch.no_grad():
            pred = torch.argmax(outputs, dim=-1)
            # print(torch.sum(pred).item())
            acc = torch.sum((pred == target) & (attention_mask != 0)) / torch.sum(attention_mask)
        return acc.item()


class F1(object):
    def __init__(self, num_classes):
        self.name = "F1"
        self.num_classes = num_classes

    def get_confusion_matrix(self, target, outputs, num_classes):
        """
        ground_truth [batch,len_seq] torch.long
        prediction [batch,len_seq,num_classes] torch.float

        output [num_classes, num_classes]: the confusion matrix
        """
        outputs = torch.argmax(outputs.view(-1, num_classes), axis=-1)
        indices_1d = outputs * num_classes + target.view(-1, )
        temp = torch.zeros(num_classes ** 2)
        indices_1d = indices_1d.bincount(minlength=num_classes)
        temp[:len(indices_1d)] = indices_1d
        return temp.view(num_classes, num_classes)

    def __call__(self, outputs, target, attention_mask):
        """
        ground_truth [batch,len_seq] torch.long
        prediction [batch,len_seq,num_classes] torch.float

        output [num_classes, num_classes]: the confusion matrix

        outputs: macro F1
        """
        mask = attention_mask == 1
        conf_mat = self.get_confusion_matrix(target[mask], outputs[mask], self.num_classes)  # confusion matrix
        TP = conf_mat.diagonal()
        FP = conf_mat.sum(1) - TP
        FN = conf_mat.sum(0) - TP

        return TP,FP,FN

    def get_f1(self,TP,FP,FN):
        # macro average
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        precision = precision.mean()
        recall = recall.mean()

        # micro average
        # TP = TP.sum()
        # FP = FP.sum()
        # FN = FN.sum()
        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)

        return 2 * precision * recall / (recall + precision + 1e-8)

