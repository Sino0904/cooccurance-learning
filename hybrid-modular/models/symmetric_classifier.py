from __future__ import absolute_import
from __future__ import division

import torch.nn as nn

__all__ = ['symmetric_classifier']

class Symm_Classifier(nn.Module):
    
    def __init__(self, num_classes=79, feat_dim=2048, *args):
        super(Symm_Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        
    def forward(self, x, *args):
        x = self.fc(x)
        return x
    
def symmetric_classifier(feat_dim=2048, num_classes=79, test=False, *args):
    print('Loading Symmetric Classifier.')
    clf = Symm_Classifier(num_classes, feat_dim)

    return clf
