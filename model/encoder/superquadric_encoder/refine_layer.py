from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Scale

from ..gaussian_encoder.utils import linear_relu_ln, safe_sigmoid, GaussianPrediction, LOGIT_MAX
import torch, torch.nn as nn
import torch.nn.functional as F


@MODELS.register_module()
class SuperQuadric3DRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        pc_range=None,
        scale_range=None,
        unit_xyz=None,
        semantic_dim=0,
        include_opa=True,
    ):
        super(SuperQuadric3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.semantic_dim = semantic_dim
        self.output_dim = 12 + int(include_opa) + semantic_dim
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float), False)
        self.register_buffer('unit_xyz', torch.tensor(unit_xyz, dtype=torch.float), False)
        self.register_buffer('scale_range', torch.tensor(scale_range, dtype=torch.float), False)
        
        self.output_layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim))
    
    def safe_inverse_sigmoid(self, x, range):
        x = (x - range[:3]) / (range[3:] - range[:3])
        x = torch.clamp(x, 1 - LOGIT_MAX, LOGIT_MAX)
        # x = torch.clamp(x, 1 - LOGIT_MAX, LOGIT_MAX).detach() + x - x.detach()
        return torch.log(x / (1 - x))

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
    ):
        output = self.output_layers(instance_feature + anchor_embed)

        # refine xyz
        delta_xyz = (2 * safe_sigmoid(output[..., :3]) - 1) * self.unit_xyz
        xyz = safe_sigmoid(anchor[..., :3]) * (self.pc_range[3:] - self.pc_range[:3]) + self.pc_range[:3]
        xyz = xyz + delta_xyz
        xyz = self.safe_inverse_sigmoid(xyz, self.pc_range)

        # refine scale
        scale = output[..., 3:6]

        # refine rot
        rot = torch.nn.functional.normalize(output[..., 6:10], p=2, dim=-1)

        # refine feature like opa \ uv \ temporal feat \ semantic
        feat = output[..., 10:]

        anchor_refine = torch.cat([xyz, scale, rot, feat], dim=-1)

        return anchor_refine