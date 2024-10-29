import torch
from .core import convert_to_tensor
import xitorch
import xitorch.linalg

def cholesky(a, **kwargs):
    a = convert_to_tensor(a)
    return torch.linalg.cholesky(a, **kwargs)

def eigh(a, s=None, **kwargs):
    a = convert_to_tensor(a)
    a = xitorch.LinearOperator.m(a)
    if s is not None:
        s = convert_to_tensor(s)
        s = xitorch.LinearOperator.m(s)
    return xitorch.linalg.symeig(a, M=s, **kwargs)

def inv(a, **kwargs):
    a = convert_to_tensor(a)
    return torch.linalg.inv(a, **kwargs)

def norm(x, ord=None, axis=None, keepdims=False, **kwargs):
    x = convert_to_tensor(x)
    return torch.linalg.norm(x, ord, axis, keepdims, **kwargs)
