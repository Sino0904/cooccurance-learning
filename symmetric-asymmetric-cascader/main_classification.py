#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

from engine import *
import models
from util import *
from util_hnh import *
from losses import *
from args import argument_parser
#from data_loader import DatasetLoader, DatasetLoader_HNH
from coco_dataloader import *
import json

# global variables
parser = argument_parser()
args = parser.parse_args()

 #Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.manualSeed)


def main():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    num_classes = 80
    
    # define state params
    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes, 'load': args.load, 'test': args.test}
    state['difficult_examples'] = True
    state['save_model_path'] = args.checkpoint
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    state['device_ids'] = args.device_ids
    if args.evaluate:
        state['evaluate'] = True
    if args.test:
        state['test'] = True
    
    if not args.test:
        train_dataset = COCO2014(args.data, phase='train')
        val_dataset = COCO2014(args.data, phase='val')
        
        print("Initializing model: {}".format(args.arch))
        feature_model_antisymmetric = models.init_model(name='resnet50_antisymm', num_classes=79, pretrained = 'antisymmetric', model_load=args.load_antisymm, use_gpu=use_gpu)
        print("Model size: {:.3f} M".format(count_num_param(feature_model_antisymmetric)))
        
        feature_model_symmetric = models.init_model(name='resnet50_symm', num_classes=num_classes, pretrained = 'imagenet', use_gpu=use_gpu)
        print("Model size: {:.3f} M".format(count_num_param(feature_model_symmetric)))
        
        joint_classifier_model = models.init_model(name='joint_classifier', num_classes=num_classes)
        
        models_learning = {'feature':{'symm':feature_model_symmetric, 'antisymm': feature_model_antisymmetric}, 'classifier': joint_classifier_model}
            
        # define loss function (criterion)
        #criterion = nn.MultiLabelSoftMarginLoss()
        #criterion = MultiLabelSoftmaxLoss()
        criterion = nn.BCEWithLogitsLoss()
        criterion_val = criterion
        
        # define optimizer
        optim_params = list(feature_model_symmetric.parameters()) + list(joint_classifier_model.parameters())
        optimizer = torch.optim.SGD(optim_params,
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        
        engine = SymmetricMultiLabelMAPEngine(state)
        engine.learning(models_learning, criterion, criterion_val, train_dataset, val_dataset, optimizer)
        
if __name__ == '__main__':
    main()
