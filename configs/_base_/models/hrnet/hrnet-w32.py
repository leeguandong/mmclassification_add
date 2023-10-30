# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='HRNet', arch='w32'),
    neck=[
        dict(type='HRFuseScales', in_channels=(32, 64, 128, 256)),
        dict(type='GlobalAveragePooling'),
    ],
    head=dict(
        type='LinearClsHead',
        in_channels=2048,
        num_classes=16,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
