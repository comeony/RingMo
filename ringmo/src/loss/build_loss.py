from ringmo.src.loss.loss import get_loss


def build_loss(config):
    return get_loss(config)
