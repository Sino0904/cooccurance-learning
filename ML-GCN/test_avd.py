#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
import sys
#sys.append.path('')

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
#from data_loader import DatasetLoader, DatasetLoader_HNH
import json
import argparse
from os import listdir
from os.path import isfile, join
from PIL import Image
import sys
import pickle

CLASSES = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parking meter', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']

parser = argparse.ArgumentParser(description='HNH I2L Simple per image inference codebase')

parser.add_argument('--arch', default='resnet50', type=str)
parser.add_argument('--model', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--image', default=None, type=str)
parser.add_argument('--dataset_path', default='/mnt/nfs/scratch1/ashishsingh/DATASETS/coco_half_label')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                        help='number of epochs to change learning rate')

args = parser.parse_args()


state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
use_cuda = torch.cuda.is_available()
use_gpu = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    

    
def mean_reciprocal_rank(correct_img_list, ranked_img_names_list):
    reciprocal_rank_complete = []
    for i in range(len(correct_img_list)):
        correct_img = correct_img_list[i]
        ranked_list = ranked_img_names_list[i]
        rank = ranked_list.index(correct_img) + 1
        reciprocal_rank = 1/rank
        reciprocal_rank_complete.append(reciprocal_rank)
    mean_reciprocal_rank = np.mean(np.array(reciprocal_rank_complete))
    return mean_reciprocal_rank

def avg_rank(correct_img_list, ranked_img_names_list):
    rank_complete = []
    for i in range(len(correct_img_list)):
        correct_img = correct_img_list[i]
        ranked_list = ranked_img_names_list[i]
        rank = ranked_list.index(correct_img) + 1
        rank_complete.append(rank)
    mean_rank = np.mean(np.array(rank_complete))
    return mean_rank

def accuracy(correct_img_list, ranked_img_names_list, n=1):
    acc_complete = []
    for i in range(len(correct_img_list)):
        correct_img = correct_img_list[i]
        ranked_list = ranked_img_names_list[i]
        predicted_rank_order_n = ranked_list[n -1]
        if correct_img == predicted_rank_order_n:
            acc_complete.append(1)
        else:
            acc_complete.append(0)
    final_acc = np.mean(np.array(acc_complete))
    return final_acc

def performance_eval(result):
    correct_img_list = []
    ranked_img_names_list = []
    for img_id, candidates_data in result.items():
        correct_img_name = "%0.12d_left.png" %int(img_id)
        correct_img_list.append(correct_img_name)
        scores = candidates_data['scores']
        img_names = candidates_data['img_names']
        ranked_img_names = [img_names[x] for x in list(np.argsort(np.array(scores))[::-1])]
        ranked_img_names_list.append(ranked_img_names)
        
    mrr = mean_reciprocal_rank(correct_img_list, ranked_img_names_list)
    average_rank = avg_rank(correct_img_list, ranked_img_names_list)
    acc = accuracy(correct_img_list, ranked_img_names_list)
    print("MEAN RECIPROCAL RANK: " + str(mrr))
    print("AVERAGE RANK: " + str(average_rank))
    print("ACCURACY: " + str(acc))
    

inp = pickle.load(open('./Image_to_Label/baselines/ML-GCN/data/coco/hnh_glove_word2vec.pkl', 'rb'))

def get_model():
    model = models.init_model(name=args.arch, num_classes=len(CLASSES),t=0.4, adj_file='./Image_to_Label/baselines/ML-GCN/data/coco/hnh_adj.pkl', pretrained = None, use_gpu=use_cuda)
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
    inp_tensor = Variable((torch.from_numpy(inp).unsqueeze(0)).cuda(), requires_grad=True).float().detach()

    if use_cuda:
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model, device_ids=args.device_ids).cuda()
        
    with torch.no_grad():
        model.eval()
        prediction = model(prediction_var, inp_tensor)
        pred_probabilities = F.softmax(prediction, dim=1).data.squeeze()
        class_idx = [int(((x.cpu()).numpy())) for x in list(topk(pred_probabilities,79)[1])]
        pred_class = [classes[x] for x in class_idx]
        class_prob = [float(((x.cpu()).numpy())) for x in list(topk(pred_probabilities,79)[0])]
        prediction_scores = np.ravel(prediction.cpu().numpy()).sort() 
    
    return pred_class, class_prob, prediction_scores

    
def test_per_set(img_paths, target_label,model):
    #img_paths = [os.path.join(args.dataset_path,"test", x) for x in img_set]
    imgs_scores = []
    for i in range(len(img_paths)):
        pred_class, class_prob, _ = test_per_image(img_paths[i],model)
        target_class = target_label
        target_prob_score = class_prob[pred_class.index(target_class)]
        imgs_scores.append(target_prob_score)
    return [float(x) for x in imgs_scores]


def main():
    data_root = args.dataset_path
    gallery_sets = [str(x) for x in range(1,len(listdir(data_root))+1)]
    gallery_sets_paths = [os.path.join(data_root, x) for x in gallery_sets]
    target_all_list = []
    gallery_list = []
    correct_list = []
    for gallery_set_path in gallery_sets_paths:
        files_per_gallery_set = [f for f in listdir(gallery_set_path) if isfile(join(gallery_set_path, f)) and f.endswith('.jpg')]
        target_file = open(os.path.join(gallery_set_path,"target.txt"), "r")
        id_file = open(os.path.join(gallery_set_path,"id.txt"), "r")
        target = target_file.read().strip().split(':')[-1]
        correct_id = id_file.read().strip().split(':')[-1]
        target_all_list.append(target)
        correct_list.append(correct_id)
        gallery_list.append(files_per_gallery_set)
        target_file.close()
        

    #result_jsonfile = os.path.join(args.output_path,"results.json")
    #result = {}
    cor = 0
    model = get_model()
    for i in range(len(gallery_list)):
        target_label = target_all_list[i]
        gallery = gallery_list[i]
        gallery_set_path = gallery_sets_paths[i]
        correct_gt = correct_list[i] 
        img_paths = [os.path.join(gallery_set_path, x) for x in gallery]
        scores_all = test_per_set(img_paths, target_label, model)
        ranked_id_list = [gallery[y].split('.')[0] for y in list(np.argsort(np.array(scores_all))[::-1])]
        if ranked_id_list[0] == correct_gt:
            cor += 1
            verdict = 'CORRECT'
        else:
            verdict = 'WRONG'
        
        print()    
        print("GALLERY SET: %d" %(i+1))
        print("TARGET OBJECT: %s" %(target_label))
        print("SCORES:")
        for j in range(len(scores_all)):
            print("\t IMAGE NAME: %s | SCORE: %f"%(gallery[j], scores_all[j]))
        print("CLASSIFICATION: %s" %(verdict))
       
    print() 
    print("ACCURACY: %f" %(cor*100.0/len(gallery_list)))
    print()

    
if __name__ == '__main__':
    main()    
