from __future__ import absolute_import

from .antisymmetric_backbone import *
from .symmetric_backbone import *
from .joint_classifier import *

__model_factory = {
    # image classification models
    'resnet50_antisymm': resnet50_antisymm,
    'resnet50_symm': resnet50_symm,
    'joint_classifier': joint_classifier
}

def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __model_factory[name](*args, **kwargs)
