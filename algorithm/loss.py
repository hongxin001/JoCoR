import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)




def loss_jocor(y_1, y_2, t, forget_rate, ind, noise_or_not, co_lambda=0.1):

    loss_pick_1 = F.cross_entropy(y_1, t, reduce = False) * (1-co_lambda)
    loss_pick_2 = F.cross_entropy(y_2, t, reduce = False) * (1-co_lambda)
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y_1, y_2,reduce=False) + co_lambda * kl_loss_compute(y_2, y_1, reduce=False)).cpu()


    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    pure_ratio = np.sum(noise_or_not[ind[ind_sorted[:num_remember]]])/float(num_remember)

    ind_update=ind_sorted[:num_remember]

    # exchange
    loss = torch.mean(loss_pick[ind_update])

    return loss, loss, pure_ratio, pure_ratio


