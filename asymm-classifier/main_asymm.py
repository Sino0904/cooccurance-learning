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
        train_dataset = DatasetLoader(args.data, img_set='train', annotation=os.path.join('/srv/data1/ashishsingh/Half_and_Half_Data/I2L/annotation', 'antisymm_multilabel_nofreq_trainset_annotation.json'))

        if args.val_hnh:
            val_dataset = DatasetLoader_HNH(args.data, img_set='val', annotation=os.path.join('/srv/data1/ashishsingh/Half_and_Half_Data/I2L/annotation', 'i2l_valset_annotation.json'))
        else:
            val_dataset = DatasetLoader(args.data, img_set='val_coco', annotation=os.path.join('/srv/data1/ashishsingh/Half_and_Half_Data', 'valset_complete_metadata.json'))
            
        print("Initializing model: {}".format(args.arch))
        if args.pretrained:
            model = models.init_model(name=args.arch, num_classes=num_classes, pretrained = 'symmetric', model_load=args.load_symm, use_selfatt=True, use_gpu=use_gpu)
        else:
            model = models.init_model(name=args.arch, num_classes=num_classes, pretrained = None,model_load=args.load_symm, use_selfatt=True, use_gpu=use_gpu)
        print("Model size: {:.3f} M".format(count_num_param(model)))
            
        # define loss function (criterion)
        #criterion = nn.MultiLabelSoftMarginLoss()
        criterion = MultiLabelSoftmaxLoss()
        if args.val_hnh or args.test:
            criterion_val = nn.CrossEntropyLoss()
        else:
            criterion_val = nn.MultiLabelSoftMarginLoss()

        # define optimizer
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        #Validate using test-like objective
        if args.val_hnh:
            engine = SymmetricMultiLabelHNHEngine(state)
        else:
            engine = SymmetricMultiLabelMAPEngine(state)

        engine.learning(model, criterion, criterion_val, train_dataset, val_dataset, optimizer)

        
    else:
        test_dataset = DatasetLoader_HNH(args.data, img_set='test_cleaned', annotation=os.path.join('/srv/data1/ashishsingh/Half_and_Half_Data/I2L/annotation', 'i2l_testset_annotation.json'))
        
        print("Initializing model: {}".format(args.arch))
        model = models.init_model(name=args.arch, num_classes=num_classes, pretrained = None,model_load=args.load_symm,  use_selfatt=True, use_gpu=use_gpu)
        criterion_test = nn.CrossEntropyLoss()
        engine = SymmetricMultiLabelHNHEngine(state)
        engine.test(model, criterion_test, test_dataset)
        
        
        
if __name__ == '__main__':
    main()
