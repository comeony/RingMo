from ringmo.src.datasets.pretrain_dataset import create_pretrain_dataset
from ringmo.src.datasets.finetune_dataset import create_finetune_dataset


def build_dataset(config, is_pretrain=True, is_train=True):
    if is_pretrain:
        return create_pretrain_dataset(config)
    else:
        return create_finetune_dataset(config, is_train=is_train)
