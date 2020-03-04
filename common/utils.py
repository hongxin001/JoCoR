import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F



def plot_result(accuracy_list,pure_ratio_list,name="test.png"):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(accuracy_list, label='test_accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(pure_ratio_list, label='test_pure_ratio')
    plt.savefig(name)


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]
