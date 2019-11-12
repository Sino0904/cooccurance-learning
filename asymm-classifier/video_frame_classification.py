#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk

import torchvision
import torchvision.transforms as transforms



from engine import *
import models
from util import *
from util_hnh import *
from losses import *
from args import argument_parser
from data_loader import DatasetLoader, DatasetLoader_HNH
import json
import argparse
from os import listdir
from os.path import isfile, join
from PIL import Image
import sys

CLASSES = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']


parser = argparse.ArgumentParser(description='HNH I2L Simple per image inference codebase')

parser.add_argument('--arch', default='resnet50', type=str)
parser.add_argument('--model', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--image', default=None, type=str)
parser.add_argument('--video', default='/mnt/nfs/scratch1/ashishsingh/DATASETS/coco_half_label')
parser.add_argument('--output_path', default='/mnt/nfs/scratch1/ashishsingh/DATASETS/coco_half_label')
parser.add_argument('--output_name', default='/mnt/nfs/scratch1/ashishsingh/DATASETS/coco_half_label')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                        help='number of epochs to change learning rate')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
#os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    
def get_model():
    model = models.init_model(name=args.arch, num_classes=len(CLASSES), pretrained = None, use_gpu=True)
    model = model.cuda()
    print("Model size: {:.3f} M".format(count_num_param(model)))
    assert os.path.isfile(args.model), 'Error: no directory found!'
    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model

def test_per_image(image_name,model):
    
    classes = CLASSES
    
    image = Image.open(image_name).convert('RGB')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    preprocess =  transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
    
    tensor = preprocess(image)
    prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad=True)
    
    if use_cuda:
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model, device_ids=args.device_ids).cuda()
        
    with torch.no_grad():
        model.eval()
        prediction = model(prediction_var)
        pred_probabilities = F.softmax(prediction, dim=1).data.squeeze()
        class_idx = [int(((x.cpu()).numpy())) for x in list(topk(pred_probabilities,79)[1])]
        pred_class = [classes[x] for x in class_idx]
        class_prob = [float(((x.cpu()).numpy())) for x in list(topk(pred_probabilities,79)[0])]
        prediction_scores = np.ravel(prediction.cpu().numpy()).sort() 
    
    return pred_class, class_prob, prediction_scores


def main():
    
    if not os.path.exists(os.path.join(args.output_path)):
        os.makedirs(args.output_path)
        
    result = {}
    model = get_model()
    
    video_folderpath = args.video
    video_name = ''
    video_path = os.path.join(video_folderpath, video_name)
    
    cap = cv2.VideoCapture(video_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(0,length+25,25):
        cap.set(1, i-1)
        res, frame = cap.read()
        pred_class, pred_probabilities, prediction = test_per_image(frame, model)
        result[i]['CLASSES'] = pred_class
        result[i]['PROB'] = pred_probabilities
        result[i]['SCORE'] = prediction
        
    with open(os.path.join(args.output_path,"results_" + args.output_name + ".json"), 'w') as result_jsonfile:
        json.dump(result,result_jsonfile)
    
        
if __name__ == '__main__':
    main()
