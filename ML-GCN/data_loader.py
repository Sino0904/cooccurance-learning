#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
import json
import pickle

CLASSES = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']

    
    
class DatasetLoader_semantics(Dataset):
    def __init__(self, data_path, img_set='train', annotation=None, transform=None, inp_name=None):
        self.img_path = os.path.join(data_path, img_set, 'images')
        self.anno = json.load(open(annotation))
        self.transform = transform
        self.num_classes = len(CLASSES)
        self.img_name = [v['name'] for k,v in self.anno.items()]
        self.img_cats = [[CLASSES.index(x) for x in v['category']] for k,v in self.anno.items()]
        
        if inp_name is not None:
            with open(inp_name, 'rb') as f:
                self.inp = pickle.load(f)
        else:
            self.inp = np.identity(80)
        self.inp_name = inp_name
        

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_name[index]))
        img = img.convert('RGB')
        img_name = self.img_name[index]
        if self.transform is not None:
            img = self.transform(img)
            
        target = np.zeros(self.num_classes, np.float32) - 1
        target[self.img_cats[index]] = 1
        label = torch.from_numpy(target)
        return (img, img_name, self.inp), label
    
    def __len__(self):
        return len(self.img_name)

    
class DatasetLoader_semantics_HNH(Dataset):
    def __init__(self, data_path, img_set='val', annotation=None, transform=None, inp_name=None):
        self.img_path = os.path.join(data_path, img_set, 'images')
        self.anno = json.load(open(annotation))
        self.transform = transform
        self.num_classes = len(CLASSES)
        self.img_name = [v['name'] for k,v in self.anno.items()]
        self.correct_cat = [[CLASSES.index(x) for x in v['correct_candidate']] for k,v in self.anno.items()]
        self.wrong_cat = [[CLASSES.index(x) for x in v['wrong_candidate']] for k,v in self.anno.items()]
        
        if inp_name is not None:
            with open(inp_name, 'rb') as f:
                self.inp = pickle.load(f)
        else:
            self.inp = np.identity(80)
        self.inp_name = inp_name
        

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_name[index]))
        img = img.convert('RGB')
        img_name = self.img_name[index]
        if self.transform is not None:
            img = self.transform(img)
            
        target = np.zeros(self.num_classes, np.float32) - 1
        target[self.correct_cat[index]] = 1
        target[self.wrong_cat[index]] = -2
        label = torch.from_numpy(target)
        return (img, img_name, self.inp), label
    
    def __len__(self):
        return len(self.img_name)
    