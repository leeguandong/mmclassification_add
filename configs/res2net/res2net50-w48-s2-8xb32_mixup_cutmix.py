_base_ = [
    '../_base_/models/res2net50-w48-s2_mixup_cutmix.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]
