import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_kl_loss(p, q, pad_mask=None):
    if pad_mask is not None:
        q = q[pad_mask != 0]
        p = p[pad_mask != 0]

    # 官方的r-drop针对序列模型提供的方法为对两次序列中对应的单词计算kl损失
    # 类似于局部与局部的对比学习与数据增强
    # TODO 考虑整个句子的对比学习增强
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='sum')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='sum')

    loss = (p_loss + q_loss) / 2
    return loss


class FocalLoss(nn.Module):
    def __init__(self,
                 class_num: int,
                 alpha: float = None,
                 gamma: int = 2,
                 ignore_index: int = 99,
                 size_average: bool = True,
                 device: str = 'cpu'):
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            self.alpha = torch.ones(class_num, 1).to(device)
        else:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha.to(device)
            else:
                self.alpha = torch.tensor(alpha).to(device)
        assert self.alpha.shape[0] == class_num, \
            f"alpha shape {alpha.shape[0]} not match class number {class_num}"
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        inputs: [batch_size * len_seq, num_classes]
        outputs: [batch_size * len_seq,]
        """
        padding_mask = targets != self.ignore_index
        targets = targets[padding_mask]
        inputs = inputs[padding_mask]
        alpha = self.alpha[targets.view(-1)]
        prob = F.softmax(inputs, dim=-1)
        mask = torch.zeros_like(prob)
        mask.scatter_(1, targets.view(-1, 1), 1.)
        prob = (prob * mask).sum(1).view(-1)
        losses = -alpha * (torch.pow(1 - prob, self.gamma)) * torch.log(prob)
        loss = losses.mean()
        return loss


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
            acc = torch.sum((pred == target) & (attention_mask != 0)) / torch.sum(attention_mask)
        return acc.item()


def get_confusion_matrix(target, outputs, num_classes, is_logit=True):
    """
    target [batch,len_seq] torch.long
    outputs [batch,len_seq,num_classes] torch.float
    output [num_classes, num_classes]: the confusion matrix
    """
    if is_logit:
        outputs = torch.argmax(outputs.view(-1, num_classes), dim=-1)
    indices_1d = outputs * num_classes + target.view(-1, )
    temp = torch.zeros(num_classes ** 2)
    indices_1d = indices_1d.bincount(minlength=num_classes)
    temp[:len(indices_1d)] = indices_1d
    return temp.view(num_classes, num_classes)


class F1(object):
    def __init__(self,
                 num_classes: int,
                 downstream: str = "linear",
                 avg_type: str = "macro"):
        self.name = "F1"
        self.num_classes = num_classes
        self.type = avg_type
        self.ds_name = downstream

    def __call__(self, outputs, target, attention_mask):
        """
        outputs [batch,len_seq] torch.long
        target [batch,len_seq,num_classes] torch.float

        return : tp, fp, fn for all classes
        """
        mask = attention_mask == 1
        conf_mat = get_confusion_matrix(target[mask], outputs[mask],
                                        num_classes=self.num_classes,
                                        is_logit=self.ds_name != 'crf')  # confusion matrix
        TP = conf_mat.diagonal()
        FP = conf_mat.sum(1) - TP
        FN = conf_mat.sum(0) - TP
        return TP, FP, FN

    def get_f1(self, tp, fp, fn, verbose=False):
        if self.type == "macro":
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            if verbose:
                print("precision: ", precision)
                print("recall: ", recall)

            # [1:] to ignore "O" precision and recall
            precision = precision.mean()
            recall = recall.mean()
        else:
            # micro average
            TP = tp.sum()
            FP = fp.sum()
            FN = fn.sum()
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

        return 2 * precision * recall / (recall + precision + 1e-8)
