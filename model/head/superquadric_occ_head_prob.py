import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from mmengine import MODELS
from mmengine.model import BaseModule
from ..encoder.gaussian_encoder.utils import \
    cartesian, safe_sigmoid, SuperQuadricPrediction, get_rotation_matrix


@MODELS.register_module()
class SuperQuadricOccHeadProb(BaseModule):
    def __init__(
        self,
        empty_label=17,
        num_classes=18,
        cuda_kwargs=dict(
            scale_multiplier=3,
            H=200, W=200, D=16,
            pc_min=[-40.0, -40.0, -1.0],
            grid_size=0.4),
        use_localaggprob=True,
        pc_range=[],
        scale_range=[],
        u_range=[],
        v_range=[],
        include_opa=True,
        semantics_activation='softmax'
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_localaggprob = use_localaggprob
        import local_aggregate_prob_sq
        self.aggregator = local_aggregate_prob_sq.LocalAggregator(**cuda_kwargs)
        self.empty_label = empty_label
        self.pc_range = pc_range
        self.scale_range = scale_range
        self.u_range = u_range
        self.v_range = v_range
        self.include_opa = include_opa
        self.semantic_start = 12 + int(include_opa)
        self.semantic_dim = self.num_classes
        self.semantics_activation = semantics_activation
        xyz = self.get_meshgrid(pc_range, [cuda_kwargs['H'], cuda_kwargs['W'], cuda_kwargs['D']], cuda_kwargs['grid_size'])
        self.register_buffer('gt_xyz', torch.tensor(xyz)[None])
    
    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz # x, y, z, 3
    
    def anchor2gaussian(self, anchor):
        xyz = cartesian(anchor, self.pc_range)
        gs_scales = safe_sigmoid(anchor[..., 3:6])
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales
        rot = anchor[..., 6: 10]
        opas = safe_sigmoid(anchor[..., 10: (10 + int(self.include_opa))])
        uv = safe_sigmoid(anchor[..., (10 + int(self.include_opa)): (12 + int(self.include_opa))])
        u = self.u_range[0] + (self.u_range[1] - self.u_range[0]) * uv[..., :1]
        v = self.v_range[0] + (self.v_range[1] - self.v_range[0]) * uv[..., 1:]
        semantics = anchor[..., self.semantic_start: (self.semantic_start + self.semantic_dim)]
        if self.semantics_activation == 'softmax':
            semantics = semantics.softmax(dim=-1)
        elif self.semantics_activation == 'softplus':
            semantics = F.softplus(semantics)
        
        gaussian = SuperQuadricPrediction(
            means=xyz,
            scales=gs_scales,
            rotations=rot,
            opacities=opas,
            u=u,
            v=v,
            semantics=semantics
        )
        return gaussian
    
    def prepare_gaussian_args(self, gaussians):
        means = gaussians.means # b, g, 3
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        opacities = gaussians.semantics # b, g, c
        origi_opa = gaussians.opacities # b, g, 1
        u = gaussians.u                 # b, g, 1
        v = gaussians.v                 # b, g, 2
        
        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)
        assert opacities.shape[-1] == self.num_classes - 1
        opacities = opacities.softmax(dim=-1)
        opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)

        rots = get_rotation_matrix(rotations) # b, g, 3, 3
        return means, origi_opa, opacities, scales, rots, u, v
    
    def prepare_gt_xyz(self, tensor):
        B, G, C = tensor.shape
        gt_xyz = self.gt_xyz.repeat([B, 1, 1, 1, 1]).to(tensor.dtype)
        return gt_xyz

    def forward(self, anchors, label, output_dict, return_anchors=False):
        B, F, G, _ = anchors.shape
        assert B==1
        anchors = anchors.flatten(0, 1)
        gaussians = self.anchor2gaussian(anchors)
        means, origi_opa, opacities, scales, rots, u, v = self.prepare_gaussian_args(gaussians)

        gt_xyz = self.prepare_gt_xyz(anchors)        # bf, x, y, z, 3
        sampled_xyz = gt_xyz.flatten(1, 3).float()
        origi_opa = origi_opa.flatten(1, 2)
        u = u.flatten(1, 2)
        v = v.flatten(1, 2)
        
        semantics = []
        bin_logits = []
        density = []
        for i in range(len(sampled_xyz)):
            semantic = self.aggregator(
                sampled_xyz[i:(i+1)], 
                means[i:(i+1)], 
                origi_opa[i:(i+1)],
                u[i:(i+1)],
                v[i:(i+1)],
                opacities[i:(i+1)],
                scales[i:(i+1)],
                rots[i:(i+1)],) # n, c
            if self.use_localaggprob:
                sem = semantic[0][:, :-1] * semantic[1].unsqueeze(-1)
                geo = 1 - semantic[1].unsqueeze(-1)
                geosem = torch.cat([sem, geo], dim=-1)
                semantics.append(geosem)
                bin_logits.append(semantic[1])
                density.append(semantic[2])
            else:
                semantics.append(semantic)
        semantics = torch.stack(semantics, dim=0).transpose(1, 2)
        bin_logits = torch.stack(bin_logits, dim=0)
        density = torch.stack(density, dim=0)
        spatial_shape = label.shape[2:]
        
        output_dict.update({
            'ce_input': semantics.unflatten(-1, spatial_shape), # F, 17, 200, 200, 16
            'ce_label': label.squeeze(0),                       # F, 200, 200, 16
            'bin_logits': bin_logits,
            'density': density,
        })
        if return_anchors:
            output_dict.update({'anchors': {
                'means': means,
                'opa': origi_opa,
                'sem': opacities,
                'scales': scales,
                'u': u,
                'v': v,
                'rot':rots
            }})
        return output_dict

