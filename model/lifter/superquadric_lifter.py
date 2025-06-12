import numpy as np
import torch
from torch import nn
import numpy as np
from mmengine import MODELS
from mmengine.model import BaseModule
from ..utils.safe_ops import safe_inverse_sigmoid


@MODELS.register_module()
class SuperQuadricLifter(BaseModule):
    def __init__(
        self,
        embed_dims,
        num_anchor=25600,
        anchor_grad=True,
        feat_grad=True,
        semantic_dim=0,
        include_opa=True,
    ):
        super().__init__()
        self.embed_dims = embed_dims

        xyz = torch.rand(num_anchor, 3, dtype=torch.float)
        xyz = safe_inverse_sigmoid(xyz)
        scale = torch.ones(num_anchor, 3, dtype=torch.float) * 0.5
        scale = safe_inverse_sigmoid(scale)
        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1
        opacity = safe_inverse_sigmoid(0.5 * torch.ones((num_anchor, int(include_opa)), dtype=torch.float))
        u = safe_inverse_sigmoid(0.5 * torch.ones(num_anchor, 1, dtype=torch.float))
        v = safe_inverse_sigmoid(0.5 * torch.ones(num_anchor, 1, dtype=torch.float))
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)

        anchor = torch.cat([xyz, scale, rots, opacity, u, v, semantic], dim=-1)

        self.num_anchor = num_anchor
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.instance_feature = nn.Parameter(
            torch.zeros([num_anchor, self.embed_dims]),
            requires_grad=feat_grad,
        )
        
    def init_weights(self):
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def forward(self, mlvl_img_feats):
        bs = mlvl_img_feats[0].shape[0]
        anchor = torch.tile(self.anchor[None], (bs, 1, 1))
        instance_feature = torch.tile(self.instance_feature[None], (bs, 1, 1))
        return anchor, instance_feature