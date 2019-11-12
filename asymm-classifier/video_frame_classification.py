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
import cv2

from os import listdir
from os.path import isfile, join


CLASSES = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']


parser = argparse.ArgumentParser(description='HNH I2L Simple per image inference codebase')

parser.add_argument('--arch', default='resnet50', type=str)
parser.add_argument('--model', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--image', default=None, type=str)
parser.add_argument('--video', default='/srv/data1/ashishsingh/youtube-random')
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


def cv_to_pil(cv_img):
    cv_img2 = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    cv_img_pil = Image.fromarray(cv_img2)
    return cv_img_pil
    
def get_model():
    model = models.init_model(name=args.arch, num_classes=len(CLASSES), pretrained = None, use_selfatt=False, use_gpu=True)
    model = model.cuda()
    print("Model size: {:.3f} M".format(count_num_param(model)))
    assert os.path.isfile(args.model), 'Error: no directory found!'
    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model

def test_per_image(image,model):
    
    classes = CLASSES
    
    image = image.convert('RGB')
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
        prediction_scores = [float(x) for x in list(np.sort(np.ravel(prediction.cpu().numpy()))[::-1])]
        pred_probabilities = F.softmax(prediction, dim=1).data.squeeze()
        class_idx = [int(((x.cpu()).numpy())) for x in list(topk(pred_probabilities,79)[1])]
        #class_idx_scores = list(np.argsort(np.ravel(prediction.cpu().numpy()))[::-1])
        pred_class = [classes[x] for x in class_idx]
        #pred_class_scores = [classes[x] for x in class_idx_scores]
        class_prob = [float(((x.cpu()).numpy())) for x in list(topk(pred_probabilities,79)[0])]
        #prediction_scores = np.sort(np.ravel(prediction.cpu().numpy()))[::-1]
     
    return pred_class, class_prob, prediction_scores


def main():
    
    if not os.path.exists(os.path.join(args.output_path)):
        os.makedirs(args.output_path)
        
    result = {}
    model = get_model()
    
    video_folderpath = args.video
    video_name_list = [f for f in listdir(video_folderpath) if isfile(join(video_folderpath, f))]
    video_path_list = [os.path.join(video_folderpath, video_name) for video_name in video_name_list]
    
    for l in range(len(video_path_list)):
        video_name = video_name_list[l]
        video_path = video_path_list[l]
        print(video_name)
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(0,length,25):
            cap.set(1, i-1)
            res, frame = cap.read()
            pred_class, pred_probabilities, prediction = test_per_image(cv_to_pil(frame), model)
            result[i] = {}
            result[i]['CLASSES'] = pred_class
            result[i]['PROB'] = pred_probabilities
            result[i]['SCORE'] = prediction
            if(i%10000 == 0):
                print(str(i) + 'frames done!')

        with open(os.path.join(args.output_path,"results_" + video_name.split('.')[0] + ".json"), 'w') as result_jsonfile:
            json.dump(result,result_jsonfile)
    
        
if __name__ == '__main__':
    main()
