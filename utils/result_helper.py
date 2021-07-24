import logging
import os
import sys
import torch


def gen_metrics(confusion_matrix):
    f_polarity = gen_metric(confusion_matrix[1:, 1:])
    aspect = torch.tensor([[confusion_matrix[0, 0], confusion_matrix[0, 1:].sum()],
                           [confusion_matrix[1:, 0].sum(), confusion_matrix[1:, 1:].sum()]])
    f_aspect = gen_metric(aspect)
    f_total = gen_metric(confusion_matrix)
    return f_aspect, f_polarity, f_total


def gen_metric(confusion_matrix):
    TP = confusion_matrix.diagonal()
    FP = confusion_matrix.sum(dim=1) - TP
    FN = confusion_matrix.sum(dim=0) - TP
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    precision = precision.mean()
    recall = recall.mean()
    f1 = 2 * precision * recall / (recall + precision + 1e-8)
    return f1


def gen_confusion_matrix(outputs, targets, ignore_index=99):
    """
    get confusion matrix for one input sentence.
    inputs: torch.Tensor [len_seq,]
    targets: torch.Tensor [len_seq,]
    return:
        confusion matrix of sentiment classify and aspect term classify
        row: prediction, columns: targets, sorted by classes id
        polarity: torch.Tensor [pos[], neg[], neu[]]
        aspect: torch.Tensor [neg[], pos[]]
    """
    aspect = torch.zeros([2, 2])
    confusion = torch.zeros([4, 4])
    i = 0
    broken = 0
    mask = targets != ignore_index
    outputs = outputs[mask]
    targets = targets[mask]
    while i < len(targets):
        if targets[i] == 0:
            aspect[1 - (outputs[i] == 0).int(), 0] += 1
            confusion[(outputs[i] - 1).item() // 3 + 1, 0] += 1
            i += 1
        else:
            # 检测到target的目标词开头
            start = i
            target_senti = targets[start] // 3  # 1-pos, 2-neg, 3-neu
            while targets[i] != 0:
                i += 1
            end = i

            if torch.all(outputs[start:end] == targets[start:end]):
                # 预测与target的情感和提取编码完全一致
                aspect[1, 1] += 1
                confusion[target_senti + 1, target_senti + 1] += 1

            elif torch.all(((-outputs[start:end] + targets[start:end]) % 3) == 0):
                # 抓取词正确，情感预测一致，但是情感预测错了
                aspect[1, 1] += 1
                pred_senti = outputs[start] // 3
                confusion[pred_senti + 1, target_senti + 1] += 1
            else:
                # 情感词不一致，或者抓取不完整
                confusion[0, target_senti + 1] += 1
                aspect[0, 1] += 1
                if (outputs[start]%3==1) and (outputs[end]%3==0) and (outputs[end] != 0):
                    # print(targets)
                    # print(outputs)
                    broken += 1  # 词抓取到，但是情感不一致
    return aspect, confusion, broken


def init_logger(logging_folder, logging_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not os.path.exists(logging_folder):
        os.mkdir(logging_folder)
    handler = logging.FileHandler(logging_file)
    handler.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(handler)
    return logger
