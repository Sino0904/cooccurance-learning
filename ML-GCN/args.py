#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='HNH I2L Symmetric-multilabel Training')
    parser.add_argument('--data', metavar='DIR',
                        help='path to dataset (e.g. data/')
    parser.add_argument('--image-size', '-i', default=448, type=int,
                        metavar='N', help='image size (default: 224)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                        help='number of epochs to change learning rate')
    parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                        help='number of epochs to change learning rate')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                        metavar='LR', help='learning rate for pre-trained layers')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=0, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--load', default='', type=str, metavar='PATH',
                        help='path to checkpoint for evaluation (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-t', '--test', dest='test', action='store_true',
                        help='evaluate model on HNH I2L test set')
    parser.add_argument('--val_hnh', dest='val_hnh', action='store_true',
                        help='use HNH validation set')
    parser.add_argument('-a', '--arch', type=str, default='resnet101')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to save checkpoint (default: none)')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    return parser
    
    
#def optimizer_kwargs(parsed_args):
#    """
#    Build kwargs for optimizer in optimizer.py from
#    the parsed command-line arguments.
#    """
#    return {
#        'optim': parsed_args.optim,
#        'lr': parsed_args.lr,
#        'weight_decay': parsed_args.weight_decay,
#        'momentum': parsed_args.momentum,
#        'sgd_dampening': parsed_args.sgd_dampening,
#        'sgd_nesterov': parsed_args.sgd_nesterov,
#        'rmsprop_alpha': parsed_args.rmsprop_alpha,
#        'adam_beta1': parsed_args.adam_beta1,
#        'adam_beta2': parsed_args.adam_beta2
#    }
