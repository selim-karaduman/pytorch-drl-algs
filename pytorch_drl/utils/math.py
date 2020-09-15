import torch

def conjugate_gradient(Af, b, max_steps=10, eps=1e-8):
    """
     Af: Method like dot product wrt matrix A
     b: shape: (N,)
     Return x for Ax =~ b (for symmetric +definite A)
    """
    x = torch.zeros_like(b).to(b)
    r = b.clone()
    d = r.clone()
    r_dot_r = r.T.dot(r)
    for i in range(max_steps):
        A_dot_d = Af(d)
        alpha =  r_dot_r / d.T.dot(A_dot_d)
        x = x + alpha * d
        r = r - alpha * A_dot_d
        r_dot_r_new = r.T.dot(r)
        if r_dot_r_new.item() < eps:
            break
        beta = r_dot_r_new / r_dot_r
        d = r + beta * d
        r_dot_r = r_dot_r_new

    return x

def hvp(outputf, f_inputs, model, vector, damping=None):
    """
    outputf: function that will generate 1 element tensor
    inputs: generator for tuple of tensors: N elements in total; net.params()
    vector: N element vector
    """
    outputs = outputf(model)# outputf(*f_inputs)
    jacobians = torch.autograd.grad(outputs, model.parameters(), create_graph=True)
    flat_jacobian = torch.cat([torch.flatten(p) for p in jacobians])
    g_dot_v = (vector * flat_jacobian).sum() 
    # grad_x(g(x) dot v) = H(x)v
    Hv = torch.autograd.grad(g_dot_v, model.parameters()) 
    Hv = torch.cat([torch.flatten(p) for p in Hv]).detach()
    if damping:
        Hv + damping * vector
    return Hv
