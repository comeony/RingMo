def _check_pretrain_dataset_config(config: dict):
    _check_config_type(config)
    config.pretrain_dataset.arch = config.arch

    if config.arch == "simmim" or config.arch == "ringmo" or config.arch == "ringmo_mm":
        config.pretrain_dataset.mask_patch_size = config.model.mask_patch_size

    if config.train_config.batch_size:
        config.pretrain_dataset.batch_size = config.train_config.batch_size

    if config.train_config.image_size:
        config.pretrain_dataset.image_size = config.train_config.image_size

    if config.model.patch_size:
        config.pretrain_dataset.patch_size = config.model.patch_size

    if config.model.mask_ratio:
        config.pretrain_dataset.mask_ratio = config.model.mask_ratio

    if config.device_num:
        config.pretrain_dataset.device_num = config.device_num

    if config.local_rank is not None:
        config.pretrain_dataset.local_rank = config.local_rank

    if config.model.inside_ratio:
        config.pretrain_dataset.inside_ratio = config.model.inside_ratio

    if config.model.use_lbp:
        config.pretrain_use_lbp = config.model.use_lbp


def _check_finetune_dataset_config(config: dict):
    _check_config_type(config)
    config.finetune_dataset.arch = config.arch

    if config.arch == "simmim" or config.arch == "ringmo" or config.arch == "ringmo_mm":
        config.finetune_dataset.mask_patch_size = config.model.mask_patch_size

    if config.train_config.batch_size:
        config.finetune_dataset.batch_size = config.train_config.batch_size

    if config.train_config.image_size:
        config.finetune_dataset.image_size = config.train_config.image_size

    if config.model.patch_size:
        config.finetune_dataset.patch_size = config.model.patch_size

    if config.device_num:
        config.finetune_dataset.device_num = config.device_num

    if config.local_rank is not None:
        config.finetune_dataset.local_rank = config.local_rank

    if config.train_config.num_classes:
        config.finetune_dataset.num_classes = config.train_config.num_classes


def _check_config_type(config):
    if not isinstance(config, dict):
        raise TypeError("dataset config should be dict type, but get {}".format(type(config)))
