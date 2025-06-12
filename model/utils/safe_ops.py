import torch


SIGMOID_MAX = 9.21
LOGIT_MAX = 0.99999

def safe_sigmoid(tensor):
    tensor = torch.clamp(tensor, -SIGMOID_MAX, SIGMOID_MAX)
    return torch.sigmoid(tensor)

def safe_inverse_sigmoid(tensor):
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX)
    return torch.log(tensor / (1 - tensor))
