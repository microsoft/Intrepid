import torch
from torch.autograd import Variable


def cuda_tensor(t):
    if torch.cuda.is_available():
        return t.cuda()
    else:
        return t


def cuda_var(t, volatile=False, requires_grad=False):
    if volatile:
        return Variable(cuda_tensor(t), volatile=True, requires_grad=requires_grad)
    else:
        return Variable(cuda_tensor(t), requires_grad=False)
