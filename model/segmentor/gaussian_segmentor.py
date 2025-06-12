import torch
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmseg.registry import MODELS as MODELS_SEG


@MODELS.register_module()
class GaussianSegmentor(BaseModule):

    def __init__(
        self,
        backbone=None,
        neck=None,
        lifter=None,
        encoder=None,
        head=None, 
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        if backbone is not None:
            try:
                self.backbone = MODELS.build(backbone)
            except:
                self.backbone = MODELS_SEG.build(backbone)
        if neck is not None:
            try:
                self.neck = MODELS.build(neck)
            except:
                self.neck = MODELS_SEG.build(neck)
        if lifter is not None:
            self.lifter = MODELS.build(lifter)
        if encoder is not None:
            self.encoder = MODELS.build(encoder)
        if head is not None:
            self.head = MODELS.build(head)

    def extract_img_feat(self, imgs):
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = self.neck(img_feats_backbone)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped
    
    def obtain_anchor(self, imgs, metas):
        B, F, N, C, H, W = imgs.shape
        imgs = imgs.reshape(B*F, N, C, H, W)
        mlvl_img_feats = self.extract_img_feat(imgs)
        anchor, instance_feature = self.lifter(mlvl_img_feats)    # bf, g, c
        anchor, instance_feature = self.encoder(anchor, instance_feature, mlvl_img_feats, metas) # bf, g, c
        return anchor, instance_feature
    
    def forward(
        self,
        imgs=None,
        metas=None,
        label=None,
        return_anchors=False,
        **kwargs,
    ):
        B, F, N, C, H, W = imgs.shape
        assert B==1, 'bs > 1 not supported'

        anchor, instance_feature = self.obtain_anchor(imgs, metas)

        output_dict = dict()
        anchor = torch.stack(anchor, dim=1)     # bf, n, g, c
        label = label.repeat(1, anchor.shape[1], 1, 1, 1)
        output_dict = self.head(
            anchors=anchor, 
            label=label, 
            output_dict=output_dict,
            return_anchors=return_anchors)

        return output_dict