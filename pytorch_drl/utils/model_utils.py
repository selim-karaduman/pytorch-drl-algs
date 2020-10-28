import torch

def soft_update_model(source_net, target_net, tau):
    for s_param, t_param in zip(source_net.parameters(),
                                target_net.parameters()):
        t_param.data.copy_((1 - tau) * t_param + tau * s_param)

def hard_update_model(source_net, target_net):
    target_net.load_state_dict(source_net.state_dict())


def transfer_gradients(source_net, target_net):
    for s_param, t_param in zip(source_net.parameters(),
                                target_net.parameters()):
        if (t_param.grad is not None) or (s_param.grad is None):
            return
        t_param._grad = s_param.grad

def get_flat_parameters(model, grad=False):
    with torch.set_grad_enabled(grad):
        return torch.cat([torch.flatten(p) for p in model.parameters()])

def set_parameters(model, parameters):
    # Parameters: flat tensor
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            shape = p.shape
            n = torch.prod(torch.tensor(shape))
            p.data.copy_((parameters[i:i+n]).view(shape))
            i += n

def add_delta(model, delta):
    new_params = get_flat_parameters(model) + delta
    set_parameters(model, new_params)