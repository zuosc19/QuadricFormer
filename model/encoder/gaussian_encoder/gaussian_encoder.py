# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Union
import torch, torch.nn as nn

from mmengine import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class GaussianEncoder(BaseModule):
    def __init__(
        self,
        anchor_encoder: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        num_encoder: int = 6,
        spconv_layer: dict = None,
        operation_order: Optional[List[str]] = None,
        return_layer_idx: Optional[List[int]] = None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.num_encoder = num_encoder
        self.return_layer_idx =return_layer_idx

        if operation_order is None:
            operation_order = [
                "spconv",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_encoder
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg):
            if cfg is None:
                return None
            return MODELS.build(cfg)
        
        self.anchor_encoder = build(anchor_encoder)
        self.op_config_map = {
            "norm": norm_layer,
            "ffn": ffn,
            "deformable": deformable_model,
            "refine": refine_layer,
            "spconv": spconv_layer,
        }
        self.layers = nn.ModuleList(
            [
                build(self.op_config_map.get(op, None))
                for op in self.operation_order
            ]
        )
        
    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
        self,
        anchor,
        instance_feature: torch.Tensor,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        anchor_embed = self.anchor_encoder(anchor)
        # if instance_feature is None:
        #     instance_feature = anchor_embed
        # else:
        #     instance_feature += anchor_embed

        prediction = []
        refine_layer_idx = 0
        # if self.training:
        #     return_idx = torch.randint(low=0, high=len(self.return_layer_idx), size=[]).item()
        # else:
        #     return_idx = -1
        # return_layer_idx = self.return_layer_idx[return_idx]
        for i, op in enumerate(self.operation_order):
            if op == 'spconv':
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor)
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
            elif "refine" in op:
                anchor = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                )
                if refine_layer_idx in self.return_layer_idx:
                    prediction.append(anchor)
                refine_layer_idx += 1
                
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                    # instance_feature += anchor_embed
            else:
                raise NotImplementedError(f"{op} is not supported.")

        return prediction, instance_feature