import torch.nn as nn, torch
from typing import NamedTuple
from torch import Tensor
import torch.nn.functional as F

from mmengine import MODELS
from mmengine.model import BaseModule, Sequential
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout

SIGMOID_MAX = 9.21
LOGIT_MAX = 0.99999

class GaussianPrediction(NamedTuple):
    means: Tensor
    scales: Tensor
    rotations: Tensor
    opacities: Tensor
    semantics: Tensor

class SuperQuadricPrediction(NamedTuple):
    means: Tensor
    scales: Tensor
    rotations: Tensor
    opacities: Tensor
    u: Tensor
    v: Tensor
    semantics: Tensor

def safe_sigmoid(tensor):
    tensor = torch.clamp(tensor, -SIGMOID_MAX, SIGMOID_MAX)
    return torch.sigmoid(tensor)

def safe_inverse_sigmoid(tensor):
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX)
    return torch.log(tensor / (1 - tensor))

def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers

def get_rotation_matrix(tensor):
    assert tensor.shape[-1] == 4

    tensor = F.normalize(tensor, dim=-1)
    mat1 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat1[..., 0, 0] = tensor[..., 0]
    mat1[..., 0, 1] = - tensor[..., 1]
    mat1[..., 0, 2] = - tensor[..., 2]
    mat1[..., 0, 3] = - tensor[..., 3]
    
    mat1[..., 1, 0] = tensor[..., 1]
    mat1[..., 1, 1] = tensor[..., 0]
    mat1[..., 1, 2] = - tensor[..., 3]
    mat1[..., 1, 3] = tensor[..., 2]

    mat1[..., 2, 0] = tensor[..., 2]
    mat1[..., 2, 1] = tensor[..., 3]
    mat1[..., 2, 2] = tensor[..., 0]
    mat1[..., 2, 3] = - tensor[..., 1]

    mat1[..., 3, 0] = tensor[..., 3]
    mat1[..., 3, 1] = - tensor[..., 2]
    mat1[..., 3, 2] = tensor[..., 1]
    mat1[..., 3, 3] = tensor[..., 0]

    mat2 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat2[..., 0, 0] = tensor[..., 0]
    mat2[..., 0, 1] = - tensor[..., 1]
    mat2[..., 0, 2] = - tensor[..., 2]
    mat2[..., 0, 3] = - tensor[..., 3]
    
    mat2[..., 1, 0] = tensor[..., 1]
    mat2[..., 1, 1] = tensor[..., 0]
    mat2[..., 1, 2] = tensor[..., 3]
    mat2[..., 1, 3] = - tensor[..., 2]

    mat2[..., 2, 0] = tensor[..., 2]
    mat2[..., 2, 1] = - tensor[..., 3]
    mat2[..., 2, 2] = tensor[..., 0]
    mat2[..., 2, 3] = tensor[..., 1]

    mat2[..., 3, 0] = tensor[..., 3]
    mat2[..., 3, 1] = tensor[..., 2]
    mat2[..., 3, 2] = - tensor[..., 1]
    mat2[..., 3, 3] = tensor[..., 0]

    mat2 = torch.conj(mat2).transpose(-1, -2)
    
    mat = torch.matmul(mat1, mat2)
    return mat[..., 1:, 1:]


def cartesian(anchor, pc_range):
    xyz = safe_sigmoid(anchor[..., :3])
    xxx = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    yyy = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    zzz = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    xyz = torch.stack([xxx, yyy, zzz], dim=-1)
    
    return xyz