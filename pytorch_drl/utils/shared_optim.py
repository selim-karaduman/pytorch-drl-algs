import torch

# Lock free optimizers
# https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py
class AdamShared(torch.optim.Adam):
    def __init__(self,  *args, **kwargs):
        super(AdamShared, self).__init__(*args, **kwargs)
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                amsgrad = group['amsgrad']
                state = self.state[p]  
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p).share_memory_()
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(p).share_memory_()
