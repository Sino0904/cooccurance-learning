#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import math
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import sys

class HNHEvalMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False):
        super(HNHEvalMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)
            
        

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            rr (FloatTensor): 1xN tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        
        rr = torch.zeros(self.scores.size(0))
       
        for k in range(self.scores.size(0)):
            # sort scores
            scores = self.scores[k, :]
            targets = self.targets[k, :]
            
            #get targets with [correct_candidate_idx, wrong_candidate_idx]
            targets[targets == -1] = 0
            candidate_idxs = targets.nonzero().squeeze()
            targets_new = targets[candidate_idxs]
            targets_new[targets_new == -2] = 0
            scores = F.softmax(scores, dim=0)
            scores_new = scores[candidate_idxs]
            # compute average precision
            rr[k] = HNHEvalMeter.reciprocal_rank(scores_new, targets_new, self.difficult_examples)
        return rr.mean().item()

    @staticmethod
    def reciprocal_rank(output, target, difficult_examples=True):
        sorted_output, rank  = output.sort(descending=True)
        correct_candidate_idx = target.argmax().item()
        rank_numpy = rank.cpu().numpy()
        pred_rank = np.where(rank_numpy == correct_candidate_idx)[0][0] + 1.0
        reciprocal_rank = 1.0/pred_rank
        return reciprocal_rank

    def acc(self):
        if self.scores.numel() == 0:
            return 0

        acc = torch.zeros(self.scores.size(0))
        
        for k in range(self.scores.size(0)):
            scores = self.scores[k, :]
            targets = self.targets[k, :]
            
            #get targets with [correct_candidate_idx, wrong_candidate_idx]
            targets[targets == -1] = 0
            candidate_idxs = targets.nonzero().squeeze()
            targets_new = targets[candidate_idxs]
            targets_new[targets_new == -2] = 0
            scores = F.softmax(scores, dim=0)
            scores_new = scores[candidate_idxs]
            sorted_output, rank  = scores_new.sort(descending=True)
            acc[k] = targets_new[rank[0]].item()
        return acc.mean().item()*100.0
    
    def acc_topk(self,k_int):
        if self.scores.numel() == 0:
            return 0

        acc = torch.zeros(self.scores.size(0))
        
        for k in range(self.scores.size(0)):
            scores = self.scores[k, :]
            targets = self.targets[k, :]
            
            #get targets with [correct_candidate_idx, wrong_candidate_idx]
            targets[targets == -1] = 0
            candidate_idxs = targets.nonzero().squeeze()
            targets_new = targets[candidate_idxs]
            targets_new[targets_new == -2] = 0
            scores = F.softmax(scores, dim=0)
            scores_new = scores[candidate_idxs]
            sorted_output, rank  = scores_new.sort(descending=True)
            acc[k] = targets_new[rank[0:k_int]].sum().item()
                
        return acc.mean().item()*100.0
        

