from __future__ import absolute_import

from .resnet50_feature import *
from .symmetric_classifier import *
from .antisymmetric_classifier import *

__model_factory = {
    # image classification models
    'resnet50_feature': resnet50_feature,
    'symmetric_classifier': symmetric_classifier,
    'antisymmetric_classifier': antisymmetric_classifier
}

def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __model_factory[name](*args, **kwargs)
