

def soft_update_model(source_net, target_net, tau):
    for s_param, t_param in zip(source_net.parameters(),
                                target_net.parameters()):
        t_param.data.copy_((1 - tau) * t_param + tau * s_param)


def transfer_gradients(source_net, target_net):
    for s_param, t_param in zip(source_net.parameters(),
                                target_net.parameters()):
        if (t_param.grad is not None) or (s_param.grad is None):
            return
        t_param._grad = s_param.grad

