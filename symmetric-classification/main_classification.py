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
        model = models.init_model(name=args.arch, num_classes=num_classes, pretrained = 'imagenet', use_gpu=use_gpu)
        print("Model size: {:.3f} M".format(count_num_param(model)))
            
        # define loss function (criterion)
        criterion = nn.MultiLabelSoftMarginLoss()
        #criterion = MultiLabelSoftmaxLoss()
        criterion_val = criterion
        
        # define optimizer
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        
        engine = SymmetricMultiLabelMAPEngine(state)
        engine.learning(model, criterion, criterion_val, train_dataset, val_dataset, optimizer)
        
if __name__ == '__main__':
    main()
