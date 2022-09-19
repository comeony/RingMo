from .mae import build_mae
from .simmim import build_simmim
from .ringmo import build_ringmo


def build_model(config):
    if config.arch == 'mae':
        model = build_mae(config)
    elif config.arch == 'simmim':
        model = build_simmim(config)
    elif config.arch == "ringmo":
        model = build_ringmo(config)
    else:
        raise NotImplementedError("This arch {} should not support".format(config.arch))
    return model
