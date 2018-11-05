import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable

def J_batch_loss(masks, initial_mels, labels):
    """
    masks & initial_mels : batch * 1 * 256 * 128
    label : batch * 1 * 256 * 128
    """
    pred = masks * initial_mels
    fn = nn.L1Loss()

    loss = fn(pred, labels)

    return loss

def J_whole_loss(maskss, initial_mels, labels, use_gpu = False):
    """
    maskss : [mask1s, mask2s, mask3s ...]
    initial_mels : batch * 1 * 256 * 128
    label : batch * 1 * 256 * 128
    """
    loss = Variable(torch.zeros(1)).cuda()
    
    if use_gpu:
        loss = loss.cuda()

    fn = nn.L1Loss()

    for masks in maskss:
        pred = masks * initial_mels
    
        loss += fn(pred, labels)

    return loss

def J_loss(maskss, initial_mels, left, right):
    """
    maskss : [mask1s, mask2s, mask3s ...]
    mask1s : batch * 2 * 256 * 128
    initial_mels : batch * 1 * 256 * 128
    label : batch * 1 * 256 * 128
    """
    left_maskss = [ m[:,0:1,:,:] for m in maskss ]
    right_maskss = [ m[:,1:2,:,:] for m in maskss ]

    left_loss = whole_J_loss(left_maskss, initial_mels, left)
    right_loss = whole_J_loss(right_maskss, initial_mels, right)

    loss = left_loss + right_loss

    return loss


