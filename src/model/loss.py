import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable

def J_loss(mask, initial_mel, label):
    pred = mask * initial_mel

    loss_mat = np.abs(pred - label)

    loss = np.sum(loss_mat)

    return loss

def J_batch_loss(masks, initial_mels, labels):
    pass