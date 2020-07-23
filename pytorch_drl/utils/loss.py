import numpy as np
import torch


def huber_loss(x, k):
    mask = (abs(x) < k).float()
    return x.pow(2)/2 * mask + k * (x.abs() - k/2) * (1-mask)
    

def quantile_huber_loss(x, tau, k=1):
    return (tau - (x.detach()<0).float()).abs() * huber_loss(x, k)


