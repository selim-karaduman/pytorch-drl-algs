import torch
import torch.nn.functional as F

def tanh_expand(l, r, x):
    return l + (r - l) * (x + 1) / 2

def squish_tanh(l, r, x):
    return -1 + (x - l) / (r - l) * 2 

def index_to_onehot(idx, size):
    """
    idx: tensor of shape (*B, )
    """
    return F.one_hot(idx, size)
    
def onehot_to_index(x, size):
    return one_hot.argmax(-1)
