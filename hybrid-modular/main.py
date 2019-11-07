#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

from engine import *
import models
from util import *
from util_hnh import *
from losses import *
from args import argument_parser
from data_loader import DatasetLoader, DatasetLoader_HNH

import json
import os

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
    num_classes = 79
    
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
        #TODO: Make annotation paths more general
        train_dataset = DatasetLoader(args.data, img_set='train', annotation=os.path.join('/srv/data1/ashishsingh/Half_and_Half_Data/I2L/annotation', 'hybrid_trainset_annotation.json'))

        if args.val_hnh:
            val_dataset = DatasetLoader_HNH(args.data, img_set='val', annotation=os.path.join('/srv/data1/ashishsingh/Half_and_Half_Data/I2L/annotation', 'i2l_valset_annotation.json'))
        else:
            val_dataset = DatasetLoader(args.data, img_set='val_coco', annotation=os.path.join('/srv/data1/ashishsingh/Half_and_Half_Data', 'valset_complete_metadata.json'))
            
        print("Initializing model: {}".format(args.arch))
        if args.pretrained:
            feature_model = models.init_model(name='resnet50_feature', pretrained='imagenet', model_type=args.model_type, use_gpu=use_gpu)
        else:
            feature_model = models.init_model(name='resnet50_feature', pretrained = None, model_type=args.model_type, use_gpu=use_gpu)
            
        print("Model size: {:.3f} M".format(count_num_param(feature_model)))
        
        symm_classifier_model = models.init_model(name='symmetric_classifier', num_classes=num_classes)
        antisymm_classifier_model = models.init_model(name='antisymmetric_classifier', num_classes=num_classes)
        
        models_learning = {'classifier':{'symm':symm_classifier_model, 'antisymm': antisymm_classifier_model}, 'feature': feature_model}
            
        # define loss function (criterion)
        #criterion_train_symm = nn.MultiLabelSoftMarginLoss()
        #criterion_train_antisymm = MultiLabelSoftmaxLoss()
        criterion_train_symm = nn.MultiLabelSoftMarginLoss()
        criterion_train_antisymm = nn.MultiLabelSoftMarginLoss()
        criterion_val = nn.CrossEntropyLoss()
        criterions_learning = {'train':{'symm':criterion_train_symm, 'antisymm': criterion_train_antisymm}, 'eval': criterion_val}
       
        # define optimizer
        optim_params = list(feature_model.parameters()) + list(symm_classifier_model.parameters())+ list(antisymm_classifier_model.parameters())
        optimizer = torch.optim.SGD(optim_params,
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        #Validate using test-like objective
        engine = SymmetricMultiLabelHNHEngine(state)
      
        engine.learning(models_learning, criterions_learning, train_dataset, val_dataset, optimizer)

        
    else:
        test_dataset = DatasetLoader_HNH(args.data, img_set='test_cleaned', annotation=os.path.join('/srv/data1/ashishsingh/Half_and_Half_Data/I2L/annotation', 'i2l_testset_annotation.json'))
        
        print("Initializing model: {}".format(args.arch))
        feature_model = models.init_model(name='resnet50_feature', pretrained = None, model_type=args.model_type, use_gpu=use_gpu)
        antisymm_classifier_model = models.init_model(name='antisymmetric_classifier', num_classes=num_classes)
        
        models_eval = {'classifier':{'antisymm': antisymm_classifier_model}, 'feature': feature_model}
        
        criterion_test = nn.CrossEntropyLoss()
        criterions_eval = {'eval':criterion_test}
        engine = SymmetricMultiLabelHNHEngine(state)
        engine.test(models_eval, criterions_eval, test_dataset)
        
        
        
if __name__ == '__main__':
    main()
