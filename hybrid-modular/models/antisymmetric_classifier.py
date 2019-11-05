from __future__ import absolute_import
from __future__ import division

import torch.nn as nn

__all__ = ['antisymmetric_classifier']

class AntiSymm_Classifier(nn.Module):
    
    def __init__(self, num_classes=79, feat_dim=2048, *args):
        super(AntiSymm_Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        
    def forward(self, x, *args):
        x = self.fc(x)
        return x
    
def antisymmetric_classifier(feat_dim=2048, num_classes=79, *args):
    print('Loading Anti-Symmetric Classifier.')
    clf = AntiSymm_Classifier(num_classes, feat_dim)

    return clf
