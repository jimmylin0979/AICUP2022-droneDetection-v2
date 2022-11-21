pwd = "/home/user/桌面/FredH/AICUP/AICUP2022-droneDetection"
_base_ = f'{pwd}/mmdetection/configs/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco.py'

# 
work_dir = f'{pwd}/results/deformable_detr_v2_pro'
dataset_type = 'CocoDataset'
pwd_dataset = '/home/user/桌面/FredH/AICUP/AICUP2022-droneDetection/data/Training Dataset_v5'
data_root = f'{pwd_dataset}'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[{
            'type':
            'Resize',
            'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                          (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                          (736, 1333), (768, 1333), (800, 1333)],
            'multiscale_mode':
            'value',
            'keep_ratio':
            True
        }],
                  [{
                      'type': 'Resize',
                      'img_scale': [(400, 4200), (500, 4200), (600, 4200)],
                      'multiscale_mode': 'value',
                      'keep_ratio': True
                  }, {
                      'type': 'RandomCrop',
                      'crop_type': 'absolute_range',
                      'crop_size': (384, 600),
                      'allow_negative_crop': True
                  }, {
                      'type':
                      'Resize',
                      'img_scale': [(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                      'multiscale_mode':
                      'value',
                      'override':
                      True,
                      'keep_ratio':
                      True
                  }]]),
    dict(type='Rotate', level=2),
    dict(type='Shear', level=2),
    dict(type='Translate', level=2),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

#
CLASSES = ('car', 'hov', 'person', 'motorcycle')
pwd_data_train = f'{pwd_dataset}/train_coco_train_tiled'
pwd_data_valid = f'{pwd_dataset}/train_coco_valid_tiled'
pwd_data_test = f'/home/user/桌面/FredH/AICUP/AICUP2022-droneDetection/data/Public Testing Dataset_v2/public_tiled/data'
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file=f'{pwd_data_train}/labels.json',
        img_prefix=f'{pwd_data_train}/data/',
        classes=CLASSES,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='AutoAugment',
                policies=[[{
                    'type':
                    'Resize',
                    'img_scale': [(480, 1333), (512, 1333), (544, 1333),
                                  (576, 1333), (608, 1333), (640, 1333),
                                  (672, 1333), (704, 1333), (736, 1333),
                                  (768, 1333), (800, 1333)],
                    'multiscale_mode':
                    'value',
                    'keep_ratio':
                    True
                }],
                          [{
                              'type': 'Resize',
                              'img_scale': [(400, 4200), (500, 4200),
                                            (600, 4200)],
                              'multiscale_mode': 'value',
                              'keep_ratio': True
                          }, {
                              'type': 'RandomCrop',
                              'crop_type': 'absolute_range',
                              'crop_size': (384, 600),
                              'allow_negative_crop': True
                          }, {
                              'type':
                              'Resize',
                              'img_scale': [(480, 1333), (512, 1333),
                                            (544, 1333), (576, 1333),
                                            (608, 1333), (640, 1333),
                                            (672, 1333), (704, 1333),
                                            (736, 1333), (768, 1333),
                                            (800, 1333)],
                              'multiscale_mode':
                              'value',
                              'override':
                              True,
                              'keep_ratio':
                              True
                          }]]),
            dict(type='Rotate', level=2),
            dict(type='Shear', level=2),
            dict(type='Translate', level=2),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=1),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        filter_empty_gt=False),
    val=dict(
        type='CocoDataset',
        ann_file=f'{pwd_data_valid}/labels.json',
        img_prefix=f'{pwd_data_valid}/data/',
        classes=CLASSES,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 1333),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=1),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        classes=CLASSES,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 1333),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=1),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox', save_best='auto')

# 
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(interval=600, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = f"{pwd}/checkpoints/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth"
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=32)

#
custom_hooks = [
    dict(type='NumClassCheckHook'),     
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49
    )
]

#
# TODO : Change backbone to ResNeXt-101
model = dict(
    type='DeformableDETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='DeformableDETRHead',
        num_query=300,
        num_classes=len(CLASSES),
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=True,
        transformer=dict(
            type='DeformableDetrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        with_box_refine=True),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100))

# 
optimizer = dict(
    type='AdamW',
    lr=0.0002 / 8,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[40, 90, 140, 190, 240, 290])

# Runner
seed = 0
gpu_ids = range(1)
device = 'cuda'
runner = dict(type='EpochBasedRunner', max_epochs=300)

# Inference
checkpoint = f"{work_dir}/best_bbox_mAP_epoch_77.pth"