from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import sys
import numpy as np


class MultiLabelSoftmaxLoss(nn.Module):
    """Multi label softmax los.
    
    Reference:
    1) https://gombru.github.io/2018/05/23/cross_entropy_loss/
    
    2) Exploring the Limits ofWeakly Supervised Pretraining(https://research.fb.com/wp-content/uploads/2018/05/exploring_the_limits_of_weakly_supervised_pretraining.pdf?)
    
    
    Args:
    - num_classes (int): number of classes
    - use_gpu (bool): whether to use gpu devices
    """
    def __init__(self):
        super(MultiLabelSoftmaxLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (batch_size, num_classes)
        """
        if not targets.is_same_size(inputs):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), inputs.size()))

        log_probs = self.logsoftmax(inputs)
        scale_factor = 1/targets.sum(1)
        cum_score_per_img = (- targets * log_probs).sum(1)
        loss = (cum_score_per_img * scale_factor).mean()
        if(torch.isnan(loss).any()):
            print(inputs)
            print(targets)
            print(log_probs)
            print(scale_factor)
            print(cum_score_per_img)
            print(loss)     
            sys.exit()   
        return loss
    
    
# class MultiLabelSoftmaxLoss(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, inputs, targets):
#         """
#         Args:
#         - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
#         - targets: ground truth labels with shape (batch_size, num_classes)
#         """
#         if not targets.is_same_size(inputs):
#             raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), inputs.size()))
            
#         ctx.save_for_backward(inputs, targets)
        
#         logsoftmax = nn.LogSoftmax(dim=1)
#         log_probs = logsoftmax(inputs)
#         scale_factor = 1/targets.sum(1)
#         cum_score_per_img = (- targets * log_probs).sum(1)
#         loss = (cum_score_per_img * scale_factor).mean()
#         #print(loss)
#         return loss
  
#     @staticmethod
#     def backward(ctx, grad_output):
#         inputs,targets = ctx.saved_tensors
#         grad_inputs = grad_output.clone()
#         softmax = nn.Softmax(dim=1)
#         delta = softmax(inputs) # If the class label is 0, the gradient is equal to probs
#         scale_factor = 1/targets.sum(1)
#         for img_id in range(inputs.size(0)):
#             scale_factor = 1/targets.sum(1)[img_id]
#             for cls_id in range(inputs.size(1)):
#                 if targets[img_id, cls_id].item() != 0:
#                     delta[img_id, cls_id] = scale_factor * (delta[img_id, cls_id] - 1) + (1 - scale_factor) * delta[img_id, cls_id]
#         #print(delta/inputs.size(0))
#         return delta/inputs.size(0), None
