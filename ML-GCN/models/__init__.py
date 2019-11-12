from __future__ import absolute_import

from .resnet import *
from .gcn_resnet import *

__model_factory = {
    # image classification models
    'resnet50': resnet50,
    'resnet50_fc512': resnet50_fc512,
    'resnet101': resnet101,
    'gcn_resnet50': gcn_resnet50
}

def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __model_factory[name](*args, **kwargs)
