from __future__ import absolute_import
from __future__ import division

from models.resnet_feature import *
from models.resnet_feature_l1 import *
from models.resnet_feature_l2 import *
from models.resnet_feature_l3 import *

__all__ = ['resnet50_feature']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""


def init_pretrained_weights(model, model_url):
    """
    Initialize model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print("Initialized model with pretrained weights from {}".format(model_url))


def resnet50_feature(pretrained='imagenet',model_type='l4', **kwargs):
    
    if model_type=='l4':
        print("PARTITION AT %s"%(model_type))
        model = ResNet(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            last_stride=2,
            fc_dims=None,
            dropout_p=None,
            **kwargs
        )
    elif model_type=='l3':
        print("PARTITION AT %s"%(model_type))
        model = ResNet_l3(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            last_stride=2,
            fc_dims=None,
            dropout_p=None,
            **kwargs
        )
    elif model_type=='l2':
        print("PARTITION AT %s"%(model_type))
        model = ResNet_l2(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            last_stride=2,
            fc_dims=None,
            dropout_p=None,
            **kwargs
        )
    elif model_type=='l1':
        print("PARTITION AT %s"%(model_type))
        model = ResNet_l1(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            last_stride=2,
            fc_dims=None,
            dropout_p=None,
            **kwargs
        )
    if pretrained == 'imagenet':
        init_pretrained_weights(model, model_urls['resnet50'])
    return model
