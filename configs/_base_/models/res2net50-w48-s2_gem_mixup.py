model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Res2Net',
        depth=50,
        scales=2,
        base_width=48,
        deep_stem=False,
        avg_down=False,
    ),
    neck=dict(type='GeneralizedMeanPooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=2,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_soft=True)),
    train_cfg=dict(
        augments=dict(type='BatchMixup', alpha=0.2, num_classes=2,
                      prob=1.)))
