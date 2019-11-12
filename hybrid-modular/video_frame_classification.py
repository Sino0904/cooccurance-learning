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
use_gpu = torch.cuda.is_available()

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
    #model = models.init_model(name=args.arch, num_classes=len(CLASSES), pretrained = None, use_gpu=True)
    
    feature_model = models.init_model(name='resnet50_feature', pretrained = None, model_type=args.model_type, use_gpu=use_gpu)
    antisymm_classifier_model = models.init_model(name='antisymmetric_classifier', num_classes=79)
    symm_classifier_model = models.init_model(name='symmetric_classifier', num_classes=79)
    
    
    assert os.path.isfile(args.model), 'Error: no directory found!'
    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
    if isinstance(checkpoint, dict) and 'state_dict_feature' in checkpoint and 'state_dict_antisymm' in checkpoint and 'state_dict_symm' in checkpoint:
        feature_model.load_state_dict(checkpoint['state_dict_feature'])
        antisymm_classifier_model.load_state_dict(checkpoint['state_dict_antisymm'])
        symm_classifier_model.load_state_dict(checkpoint['state_dict_symm'])

    return feature_model, antisymm_classifier_model, symm_classifier_model


def test_per_image(image,feature_model, antisymm_classifier_model, symm_classifier_model):
    
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
        feature_model = torch.nn.DataParallel(feature_model, device_ids=args.device_ids).cuda()
        antisymm_classifier_model = torch.nn.DataParallel(antisymm_classifier_model, device_ids=args.device_ids).cuda()
        symm_classifier_model = torch.nn.DataParallel(symm_classifier_model, device_ids=args.device_ids).cuda()
        
    with torch.no_grad():
        feature_model.eval()
        antisymm_classifier_model.eval()
        symm_classifier_model.eval()
        
        _ , prediction_feature_asymm = feature_model(prediction_var)
        prediction = antisymm_classifier_model(prediction_feature_asymm)
        prediction_scores = [float(x) for x in list(np.sort(np.ravel(prediction.cpu().numpy()))[::-1])]
        pred_probabilities = F.softmax(prediction, dim=1).data.squeeze()
        class_idx = [int(((x.cpu()).numpy())) for x in list(topk(pred_probabilities,79)[1])]
        pred_class = [classes[x] for x in class_idx]
        class_prob = [float(((x.cpu()).numpy())) for x in list(topk(pred_probabilities,79)[0])]
        
        _ , prediction_feature_symm = feature_model(prediction_var)
        prediction_symm = symm_classifier_model(prediction_feature_symm)
        prediction_scores_symm = [float(x) for x in list(np.sort(np.ravel(prediction_symm.cpu().numpy()))[::-1])]
        pred_probabilities_symm = F.softmax(prediction_symm, dim=1).data.squeeze()
        class_idx_symm = [int(((x.cpu()).numpy())) for x in list(topk(pred_probabilities_symm,79)[1])]
        pred_class_symm = [classes[x] for x in class_idx_symm]
        class_prob_symm = [float(((x.cpu()).numpy())) for x in list(topk(pred_probabilities_symm,79)[0])]
       
    return pred_class, class_prob, prediction_scores, pred_class_symm, class_prob_symm, prediction_scores_symm


def main():
    
    if not os.path.exists(os.path.join(args.output_path)):
        os.makedirs(args.output_path)
        
    result_antisymm = {}
    result_symm = {}
    feature_model, antisymm_classifier_model, symm_classifier_model = get_model()
    
    video_folderpath = args.video
    video_name_list = [f for f in listdir(video_folderpath) if isfile(join(video_folderpath, f))]
    video_path_list = [os.path.join(video_folderpath, video_name) for video_name in video_name_list]
    
    for l in range(len(video_path_list)):
        video_name = video_name_list[l]
        video_path = video_path_list[l]
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(0,length,25):
            cap.set(1, i-1)
            res, frame = cap.read()
            pred_class, class_prob, prediction_scores, pred_class_symm, class_prob_symm, prediction_scores_symm = test_per_image(cv_to_pil(frame), model)
            
            result_antisymm[i] = {}
            result_antisymm[i]['CLASSES'] = pred_class
            result_antisymm[i]['PROB'] = pred_probabilities
            result_antisymm[i]['SCORE'] = prediction
            
            result_symm[i] = {}
            result_symm[i]['CLASSES'] = pred_class_symm
            result_symm[i]['PROB'] = pred_probabilities_symm
            result_symm[i]['SCORE'] = prediction_symm
            
            if(i%10000 == 0):
                print(str(i) + 'frames done!')

        with open(os.path.join(args.output_path,"results_" + video_name.split('.')[0] + "_antisymm.json"), 'w') as result_jsonfile:
            json.dump(result_antisymm,result_jsonfile)
            
        with open(os.path.join(args.output_path,"results_" + video_name.split('.')[0] + "_symm.json"), 'w') as result_jsonfile2:
            json.dump(result_symm,result_jsonfile2)
    
        
if __name__ == '__main__':
    main()
