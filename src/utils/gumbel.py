import torch
import torch.nn.functional as F

from utils.cuda import cuda_var


def _sample_gumbel(input_size):
    noise = torch.rand(input_size)
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return cuda_var(noise)


def gumbel_sample(input, temperature):
    noise = _sample_gumbel(input.size())
    x = (input + noise) / temperature
    prob = F.softmax(x, dim=1)
    log_prob = F.log_softmax(x, dim=1)
    return prob, log_prob
