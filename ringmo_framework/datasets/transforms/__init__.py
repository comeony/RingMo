"""transforms of ring-framework"""
from ringmo_framework.datasets.transforms.aug_policy import ImageNetPolicyV2,\
    ImageNetPolicy, SubPolicy, SVHNPolicy
from ringmo_framework.datasets.transforms.auto_augment import rand_augment_transform
from ringmo_framework.datasets.transforms.mixup import Mixup
from ringmo_framework.datasets.transforms.random_erasing import RandomErasing
