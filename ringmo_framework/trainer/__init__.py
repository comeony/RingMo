"""trainer of ringmo_framework"""
from ringmo_framework.trainer.trainer import build_wrapper
from ringmo_framework.trainer.ema import EMACell
from ringmo_framework.trainer.clip_grad import clip_by_global_norm
