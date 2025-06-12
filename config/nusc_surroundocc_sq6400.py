# =========== misc config ==============
optimizer_wrapper = dict(
    optimizer = dict(
        type='AdamW',
        lr=4e-4,
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),}
    ),
)
grad_max_norm = 35
amp = False

# =========== base config ==============
seed = 1
print_freq = 50
eval_freq = 1
max_epochs = 20
load_from = None
find_unused_parameters = False

# =========== data config ==============
ignore_label = 0
empty_idx = 17   # 0 noise, 1~16 objects, 17 empty
cls_dims = 18
pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
image_size = [864, 1600]
resize_lim = [1.0, 1.0]
flip = True
num_frames = 1
offset = 0

# =========== model config =============
_dim_ = 128
num_cams = 6
num_heads = 4
num_levels = 4
drop_out = 0.1
semantics_activation = 'identity'
semantic_dim = 17
include_opa = True
wempty = False
freeze_perception = False

num_anchor = 6400
scale_range = [0.01, 3.2]
u_range = [0.1, 2]
v_range = [0.1, 2]
num_learnable_pts = 6
learnable_scale = 3
scale_multiplier = 5
num_encoder = 4
return_layer_idx = [2, 3]

anchor_encoder = dict(
    type='SuperQuadric3DEncoder',
    embed_dims=_dim_, 
    include_opa=include_opa,
    semantic_dim=semantic_dim,
)

ffn = dict(
    type="AsymmetricFFN",
    in_channels=_dim_,
    embed_dims=_dim_,
    feedforward_channels=_dim_ * 4,
    ffn_drop=drop_out,
    add_identity=False,
)

deformable_layer = dict(
    type='DeformableFeatureAggregation',
    embed_dims=_dim_,
    num_groups=num_heads,
    num_levels=num_levels,
    num_cams=num_cams,
    attn_drop=0.15,
    use_deformable_func=True,
    use_camera_embed=True,
    residual_mode="none",
    kps_generator=dict(
        type="SparseGaussian3DKeyPointsGenerator",
        embed_dims=_dim_,
        num_learnable_pts=num_learnable_pts,
        learnable_scale=learnable_scale,
        fix_scale=[
            [0, 0, 0],
            [0.45, 0, 0],
            [-0.45, 0, 0],
            [0, 0.45, 0],
            [0, -0.45, 0],
            [0, 0, 0.45],
            [0, 0, -0.45],
        ],
        pc_range=pc_range,
        scale_range=scale_range),
)

refine_layer = dict(
    type='SuperQuadric3DRefinementModule',
    embed_dims=_dim_,
    pc_range=pc_range,
    scale_range=scale_range,
    unit_xyz=[4.0, 4.0, 1.0],
    semantic_dim=semantic_dim,
    include_opa=include_opa,
)

spconv_layer=dict(
    type='SparseConv3DBlock',
    in_channels=_dim_,
    embed_channels=_dim_,
    pc_range=pc_range,
    use_out_proj=True,
    grid_size=[1.0, 1.0, 1.0],
    kernel_size=[5, 5, 5],
    stride=[1, 1, 1],
    padding=[2, 2, 2],
    dilation=[1, 1, 1],
    spatial_shape=[100, 100, 8],
)

model = dict(
    type='GaussianSegmentor',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp=True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True),
        init_cfg=dict(
          type='Pretrained',
          checkpoint='pretrain/r101_dcn_fcos3d_pretrain.pth'),
    ),
    neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=1,
        out_channels=_dim_,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048]),
    lifter=dict(
        type='SuperQuadricLifter',
        embed_dims=_dim_,
        num_anchor=num_anchor,
        anchor_grad=True,
        feat_grad=False,
        include_opa=include_opa,
        semantic_dim=semantic_dim),
    encoder=dict(
        type='GaussianEncoder',
        return_layer_idx=return_layer_idx,
        num_encoder=num_encoder,
        anchor_encoder=anchor_encoder,
        norm_layer=dict(type="LN", normalized_shape=_dim_),
        ffn=ffn,
        deformable_model=deformable_layer,
        refine_layer=refine_layer,
        spconv_layer=spconv_layer,
        operation_order=[
            "identity",
            "deformable",
            "add",
            "norm",
            "identity",
            "ffn",
            "add",
            "norm",
            "identity",
            "spconv",
            "add",
            "norm",
            "identity",
            "ffn",
            "add",
            "norm",
            "refine",
        ] * num_encoder),
    head=dict(
        type='SuperQuadricOccHeadProb',
        empty_label=empty_idx,
        num_classes=cls_dims,
        cuda_kwargs=dict(
            scale_multiplier=scale_multiplier,
            H=200, W=200, D=16,
            pc_min=[-50.0, -50.0, -5.0],
            grid_size=0.5),
        use_localaggprob=True,
        pc_range=pc_range,
        scale_range=scale_range,
        u_range=u_range,
        v_range=v_range,
        include_opa=include_opa,
        semantics_activation=semantics_activation
    )
)


loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='CELoss',
            weight=10.0,
            cls_weight=[
                1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ],
            ignore_label=ignore_label,
            use_softmax=False,
            input_dict={
                'ce_input': 'ce_input',
                'ce_label': 'ce_label'}),
        dict(
            type='LovaszLoss',
            weight=1.0,
            empty_idx=empty_idx,
            ignore_label=ignore_label,
            use_softmax=False,
            input_dict={
                'lovasz_input': 'ce_input',
                'lovasz_label': 'ce_label'}),
    ]
)

data_path = 'data/surroundocc'

train_dataset_config = dict(
    type='NuScenes_Scene_SurroundOcc_Dataset',
    data_path = data_path,
    num_frames = num_frames,
    offset = offset,
    empty_idx=empty_idx,
    imageset = 'data/nuscenes_temporal_infos_train.pkl',
)

val_dataset_config = dict(
    type='NuScenes_Scene_SurroundOcc_Dataset',
    data_path = data_path,
    num_frames = num_frames,
    offset = offset,
    empty_idx=empty_idx,
    imageset = 'data/nuscenes_temporal_infos_val.pkl',
)

train_wrapper_config = dict(
    type='NuScenes_Scene_Occ_DatasetWrapper',
    final_dim = image_size,
    resize_lim = resize_lim,
    flip = flip,
    phase='train', 
)

val_wrapper_config = dict(
    type='NuScenes_Scene_Occ_DatasetWrapper',
    final_dim = image_size,
    resize_lim = resize_lim,
    flip = flip,
    phase='val', 
)

train_loader_config = dict(
    batch_size = 1,
    shuffle = True,
    num_workers = 8,
)
    
val_loader_config = dict(
    batch_size = 1,
    shuffle = False,
    num_workers = 8,
)