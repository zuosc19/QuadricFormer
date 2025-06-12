from mmengine import build_from_cfg
from mmengine.registry import MODELS
from .backbone import *
from .neck import *
from .lifter import *
from .encoder import *
from .segmentor import *
from .head import *


def build_model(model_config):
    model = build_from_cfg(model_config, MODELS)
    model.init_weights()
    return model