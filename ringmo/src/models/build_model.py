from ringmo.src.models.backbone.vit import build_vit
from ringmo.src.models.backbone.swin_transformer import build_swin


def build_model(config):
    if config.model.backbone == 'vit':
        model = build_vit(config)
    elif config.model.backbone == 'swin':
        model = build_swin(config)
    else:
        raise NotImplementedError
    return model
