from __future__ import absolute_import
from __future__ import division

import torch.nn as nn

__all__ = ['joint_classifier']

class JointClassifier(nn.Module):
    
    def __init__(self, num_classes=80, feat_dim=80+79, *args):
        super(JointClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        
    def forward(self, x, *args):
        x = self.fc(x)
        return x
    
def joint_classifier(feat_dim=80+79, num_classes=80, test=False, *args):
    print('Loading Joint Classifier.')
    clf = JointClassifier(num_classes, feat_dim)

    return clf