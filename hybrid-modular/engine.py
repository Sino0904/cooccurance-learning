#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import os
import sys
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from util import *
from util_hnh import *

tqdm.monitor_interval = 0
class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('image_size') is None:
            self.state['image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 64

        if self._state('workers') is None:
            self.state['workers'] = 25

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('evaluate') is None:
            self.state['evaluate'] = False
            
        if self._state('validate_hnh') is None:
            self.state['validate_hnh'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self, training, models, criterions, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, models, criterions, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].value()[0]
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}'.format(self.state['epoch'], loss=loss))
            else:
                print('Test: \t Loss {loss:.4f}'.format(loss=loss))
        return loss

    def on_start_batch(self, training, models, criterions, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, models, criterions, data_loader, optimizer=None, display=True):

        # record loss
        self.state['loss_batch'] = self.state['loss'].item()
        self.state['meter_loss'].add(self.state['loss_batch'])

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

    def on_forward(self, training, models, criterions, data_loader, optimizer=None, display=True):

        input_var = torch.autograd.Variable(self.state['input'])
        target_var_symm = torch.autograd.Variable(self.state['target_symm'])
        target_var_antisymm = torch.autograd.Variable(self.state['target_antisymm'])
        
        # compute output
        output_feature_symm, output_feature_antisymm  = models['feature'](input_var)
        
        if training:
            output_symm = models['classifier']['symm'](output_feature_symm)
            output_antisymm = models['classifier']['antisymm'](output_feature_antisymm)
            self.state['output'] = output_antisymm
            #output_cpu = self.state['output'].cpu()
            self.state['loss_symm'] = criterions['train']['symm'](output_symm, target_var_symm)
            self.state['loss_antisymm'] = criterions['train']['antisymm'](output_antisymm, target_var_antisymm)
            self.state['loss'] = self.state['loss_symm'] + self.state['loss_antisymm']
        else:
            output_antisymm = models['classifier']['antisymm'](output_feature_antisymm)
            self.state['output'] = output_antisymm
            self.state['loss'] = criterions['eval'](self.state['output'], target_var.nonzero()[:,1])

        
        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()
      
    def init_learning(self):

        if self._state('train_transform') is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.state['train_transform'] = transforms.Compose([
                #transforms.Resize((512, 512)),
                #MultiScaleCrop(self.state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            print(self.state['train_transform'])

        if self._state('val_transform') is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.state['val_transform'] = transforms.Compose([
                #Warp(self.state['image_size']),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])

        self.state['best_score'] = 0

    def learning(self, models, criterions, train_dataset, val_dataset, optimizer=None):
        
        
        self.init_learning()

        # define train and val transform
        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')

        # data loading code
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True,
                                                   num_workers=self.state['workers'])

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])
        
            
        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['load'], map_location=lambda storage, loc: storage)
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                if isinstance(checkpoint, dict) and 'state_dict_feature' in checkpoint and 'state_dict_antisymm' in checkpoint:
                    models['feature'].load_state_dict(checkpoint['state_dict_feature'])
                    models['classifier']['antisymm'].load_state_dict(checkpoint['state_dict_antisymm'])
                    models['classifier']['symm'].load_state_dict(checkpoint['state_dict_symm'])
                else:
                    #model.load_state_dict(checkpoint)
                    print("ERROR IN LOADING THE MODEL")
                    sys.exit()
                    
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))


        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True


            models['feature'] = torch.nn.DataParallel(models['feature'], device_ids=self.state['device_ids']).cuda()
            models['classifier']['symm'] = torch.nn.DataParallel(models['classifier']['symm'], device_ids=self.state['device_ids']).cuda()
            models['classifier']['antisymm'] = torch.nn.DataParallel(models['classifier']['antisymm'], device_ids=self.state['device_ids']).cuda()
            
            
            criterions['train']['antisymm'] = criterions['train']['antisymm'].cuda()
            criterions['train']['symm'] = criterions['train']['symm'].cuda()
            criterions['eval'] = criterions['eval'].cuda()

        if self.state['evaluate']:
            self.validate(val_loader,models, criterions)
            return

        # TODO define optimizer

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:',lr)

            # train for one epoch
            self.train(train_loader, models, criterions, optimizer, epoch)
            # evaluate on validation set
            prec1 = self.validate(val_loader, models, criterions)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.state['best_score']
            self.state['best_score'] = max(prec1, self.state['best_score'])
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self._state('arch'),
                'state_dict_feature': models['feature'].module.state_dict() if self.state['use_gpu'] else models['feature'].state_dict(),
                'state_dict_symm':  models['classifier']['symm'].module.state_dict() if self.state['use_gpu'] else models['classifier']['symm'].state_dict(),
                'state_dict_antisymm':  models['classifier']['antisymm'].module.state_dict() if self.state['use_gpu'] else models['classifier']['antisymm'].state_dict(),
                'best_score': self.state['best_score'],
            }, is_best)

            print(' *** best={best:.3f}'.format(best=self.state['best_score']))
        return self.state['best_score']

    def train(self, data_loader, models, criterions, optimizer, epoch):

        # switch to train mode
        models['feature'].train()
        models['classifier']['symm'].train()
        models['classifier']['antisymm'].train()

        self.on_start_epoch(True, models, criterions, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target_symm'] = target[0]
            self.state['target_antisymm'] = target[1]

            self.on_start_batch(True, models, criterions, data_loader, optimizer)

            if self.state['use_gpu']:
                self.state['target_symm'] = self.state['target_symm'].cuda(async=True)
                self.state['target_antisymm'] = self.state['target_antisymm'].cuda(async=True)

            self.on_forward(True, models, criterions, data_loader, optimizer)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, models, criterions, data_loader, optimizer)

        self.on_end_epoch(True, models, criterions, data_loader, optimizer)

    def validate(self, data_loader, models, criterions):
        
        with torch.no_grad():

            # switch to evaluate mode
            models['feature'].eval()
            models['classifier']['symm'].eval()
            models['classifier']['antisymm'].eval()

            self.on_start_epoch(False, models, criterions, data_loader)

            if self.state['use_pb']:
                data_loader = tqdm(data_loader, desc='Test')

            end = time.time()
            for i, (input, target) in enumerate(data_loader):
                # measure data loading time
                self.state['iteration'] = i
                self.state['data_time_batch'] = time.time() - end
                self.state['data_time'].add(self.state['data_time_batch'])

                self.state['input'] = input
                self.state['target_antisymm'] = target

                self.on_start_batch(False, models, criterions, data_loader)

                if self.state['use_gpu']:
                    self.state['target_antisymm'] = self.state['target_antisymm'].cuda(async=True)


                self.on_forward(False, models, criterions, data_loader)

                # measure elapsed time
                self.state['batch_time_current'] = time.time() - end
                self.state['batch_time'].add(self.state['batch_time_current'])
                end = time.time()
                # measure accuracy
                self.on_end_batch(False, models, criterions, data_loader)

            score = self.on_end_epoch(False, models, criterions, data_loader)

            return score
        
#------------------------------------------------------------------------------------------------
    def test(self,models, criterions, test_dataset, optimizer=None):

        if self._state('test_transform') is None:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.state['test_transform'] = transforms.Compose([
                #Warp(self.state['image_size']),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])

        self.state['best_score'] = 0
        
        # define test transform
        test_dataset.transform = self.state['test_transform']
        test_dataset.target_transform = self._state('test_target_transform')

        # data loading code

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])
        
            
        # load the checkpoint
        if self._state('load') is not None:
            if os.path.isfile(self.state['load']):
                print("=> loading checkpoint '{}'".format(self.state['load']))
                checkpoint = torch.load(self.state['load'], map_location=lambda storage, loc: storage)
                if isinstance(checkpoint, dict) and 'state_dict_feature' in checkpoint and 'state_dict_antisymm' in checkpoint:
                    models['feature'].load_state_dict(checkpoint['state_dict_feature'])
                    models['classifier']['antisymm'].load_state_dict(checkpoint['state_dict_antisymm'])
                else:
                    #model.load_state_dict(checkpoint)
                    print("ERROR IN LOADING THE MODEL")
                    sys.exit()
                
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['test'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['load']))


        if self.state['use_gpu']:
            test_loader.pin_memory = True
            cudnn.benchmark = True

            models['feature'] = torch.nn.DataParallel(models['feature'], device_ids=self.state['device_ids']).cuda()
            models['classifier']['antisymm'] = torch.nn.DataParallel(models['classifier']['antisymm'], device_ids=self.state['device_ids']).cuda()

            criterions['eval'] = criterions['eval'].cuda()

        with torch.no_grad():

            # switch to evaluate mode
            models['feature'].eval()
            models['classifier']['antisymm'].eval()

            self.on_start_epoch(False, models, criterions, test_loader)

            if self.state['use_pb']:
                test_loader = tqdm(test_loader, desc='Test')

            end = time.time()
            for i, (input, target) in enumerate(test_loader):
                # measure data loading time
                self.state['iteration'] = i
                self.state['data_time_batch'] = time.time() - end
                self.state['data_time'].add(self.state['data_time_batch'])

                self.state['input'] = input
                self.state['target_antisymm'] = target

                self.on_start_batch(False, models, criterions, test_loader)

                if self.state['use_gpu']:
                    self.state['target_antisymm'] = self.state['target_antisymm'].cuda(async=True)

                self.on_forward(False, models, criterions, test_loader)

                # measure elapsed time
                self.state['batch_time_current'] = time.time() - end
                self.state['batch_time'].add(self.state['batch_time_current'])
                end = time.time()
                # measure accuracy
                self.on_end_batch(False, models, criterions, test_loader)

            score = self.on_end_epoch(False, models, criterions, test_loader)

            return score


    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'], 'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_list = []
        decay = 0.1 if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)

    
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
                
class HNHMultiClassEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])
        self.state['hnh_meter'] = HNHEvalMeter(self.state['difficult_examples'])

    def on_start_epoch(self, training, models, criterions, data_loader, optimizer=None, display=True):
        Engine.on_start_epoch(self, training, models, criterions, data_loader, optimizer)
        if training:
            self.state['ap_meter'].reset()
        else:
            self.state['hnh_meter'].reset()

    def on_end_epoch(self, training,models, criterions, data_loader, optimizer=None, display=True):
        if training:
            map = 100 * self.state['ap_meter'].value().mean()
            loss = self.state['meter_loss'].value()[0]
            OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
            score_ = map

        else:
            mrr = self.state['hnh_meter'].value()
            loss = self.state['meter_loss'].value()[0]
            ACC_R1 = self.state['hnh_meter'].acc()
            ACC_R3 = self.state['hnh_meter'].acc_topk(3)
            score_ = mrr
            
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}\t'
                      'mAP {map:.3f}'.format(self.state['epoch'], loss=loss, map=map))
                print('OP: {OP:.4f}\t'
                      'OR: {OR:.4f}\t'
                      'OF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\t'
                      'CR: {CR:.4f}\t'
                      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
            else:
                print('Test: \t Loss {loss:.4f}\t MRR {mrr:.3f}'.format(loss=loss, mrr=mrr))
                print('ACC_R1: {ACC_R1:.4f}\t'
                      'ACC_R3: {ACC_R3:.4f}'.format(ACC_R1=ACC_R1, ACC_R3=ACC_R3))
               
        return score_

    def on_start_batch(self, training, models, criterions, data_loader, optimizer=None, display=True):
        self.state['target_symm_gt'] = self.state['target_symm'].clone()
        self.state['target_antisymm_gt'] = self.state['target_antisymm'].clone()
        self.state['target_symm'][self.state['target_symm'] == 0] = 1
        self.state['target_symm'][self.state['target_symm'] == -1] = 0
        self.state['target_antisymm'][self.state['target_antisymm'] == 0] = 1
        self.state['target_antisymm'][self.state['target_antisymm'] == -1] = 0


        input = self.state['input']
        self.state['input'] = input[0]
        self.state['name'] = input[1]
        

    def on_end_batch(self, training, models, criterions, data_loader, optimizer=None, display=True):
        
        Engine.on_end_batch(self, training, models, criterions, data_loader, optimizer, display=False)
        
        if training:
            # measure mAP
            self.state['ap_meter'].add(self.state['output'].data, self.state['target_antisymm_gt'])
        else:
            self.state['hnh_meter'].add(self.state['output'].data, self.state['target_antisymm_gt'])

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
                
                
                
class SymmetricMultiLabelHNHEngine(HNHMultiClassEngine):
    def on_forward(self, training, models, criterions, data_loader, optimizer=None, display=True):
        feature_var = torch.autograd.Variable(self.state['feature']).float()
        
        # compute output
        output_feature_symm, output_feature_antisymm = models['feature'](feature_var)
        
        if training:
            output_symm = models['classifier']['symm'](output_feature_symm)
            output_antisymm = models['classifier']['antisymm'](output_feature_antisymm)
            self.state['output'] = output_antisymm
            #output_cpu = self.state['output'].cpu()
            target_var_symm = torch.autograd.Variable(self.state['target_symm']).float()
            target_var_antisymm = torch.autograd.Variable(self.state['target_antisymm']).float()
            self.state['loss_symm'] = criterions['train']['symm'](output_symm, target_var_symm)
            self.state['loss_antisymm'] = criterions['train']['antisymm'](output_antisymm, target_var_antisymm)
            self.state['loss'] = 0.1*self.state['loss_symm'] + 0.9*self.state['loss_antisymm']
            
        else:
            output_antisymm = models['classifier']['antisymm'](output_feature_antisymm)
            self.state['output'] = output_antisymm
            target_var_antisymm = torch.autograd.Variable(self.state['target_antisymm']).float()
            self.state['loss'] = criterions['eval'](self.state['output'], target_var_antisymm.nonzero()[:,1])

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            nn.utils.clip_grad_norm_(models['feature'].parameters(), max_norm=10.0)
            #nn.utils.clip_grad_norm_(models['classifier']['symm'].parameters(), max_norm=10.0)
            #nn.utils.clip_grad_norm_(models['classifier']['antisymm'].parameters(), max_norm=10.0)
            optimizer.step()


    def on_start_batch(self, training, models, criterions, data_loader, optimizer=None, display=True):
        if training:   
            self.state['target_symm_gt'] = self.state['target_symm'].clone()
            self.state['target_symm'][self.state['target_symm'] == 0] = 1
            self.state['target_symm'][self.state['target_symm'] == -1] = 0
            self.state['target_antisymm_gt'] = self.state['target_antisymm'].clone()
            self.state['target_antisymm'][self.state['target_antisymm'] == 0] = 1
            self.state['target_antisymm'][self.state['target_antisymm'] == -1] = 0

            input = self.state['input']
            self.state['feature'] = input[0]
            self.state['out'] = input[1]
        else:
            self.state['target_antisymm_gt'] = self.state['target_antisymm'].clone()
            self.state['target_antisymm'][self.state['target_antisymm'] == 0] = 1
            self.state['target_antisymm'][self.state['target_antisymm'] == -1] = 0
            self.state['target_antisymm'][self.state['target_antisymm'] == -2] = 0 #wrong candidates
                
            input = self.state['input']
            self.state['feature'] = input[0]
            self.state['out'] = input[1]
        
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
       

# class MultiLabelMAPEngine(Engine):
#     def __init__(self, state):
#         Engine.__init__(self, state)
#         if self._state('difficult_examples') is None:
#             self.state['difficult_examples'] = False
#         self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])

#     def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
#         Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
#         self.state['ap_meter'].reset()

#     def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
#         map = 100 * self.state['ap_meter'].value().mean()
#         loss = self.state['meter_loss'].value()[0]
#         OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
#         OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state['ap_meter'].overall_topk(3)
#         if display:
#             if training:
#                 print('Epoch: [{0}]\t'
#                       'Loss {loss:.4f}\t'
#                       'mAP {map:.3f}'.format(self.state['epoch'], loss=loss, map=map))
#                 print('OP: {OP:.4f}\t'
#                       'OR: {OR:.4f}\t'
#                       'OF1: {OF1:.4f}\t'
#                       'CP: {CP:.4f}\t'
#                       'CR: {CR:.4f}\t'
#                       'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
#             else:
#                 print('Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=loss, map=map))
#                 print('OP: {OP:.4f}\t'
#                       'OR: {OR:.4f}\t'
#                       'OF1: {OF1:.4f}\t'
#                       'CP: {CP:.4f}\t'
#                       'CR: {CR:.4f}\t'
#                       'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
#                 print('OP_3: {OP:.4f}\t'
#                       'OR_3: {OR:.4f}\t'
#                       'OF1_3: {OF1:.4f}\t'
#                       'CP_3: {CP:.4f}\t'
#                       'CR_3: {CR:.4f}\t'
#                       'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))

#         return map

#     def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

#         self.state['target_gt'] = self.state['target'].clone()
#         self.state['target'][self.state['target'] == 0] = 1
#         self.state['target'][self.state['target'] == -1] = 0

#         input = self.state['input']
#         self.state['input'] = input[0]
#         self.state['name'] = input[1]

#     def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

#         Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)

#         # measure mAP
#         self.state['ap_meter'].add(self.state['output'].data, self.state['target_gt'])

#         if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
#             loss = self.state['meter_loss'].value()[0]
#             batch_time = self.state['batch_time'].value()[0]
#             data_time = self.state['data_time'].value()[0]
#             if training:
#                 print('Epoch: [{0}][{1}/{2}]\t'
#                       'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
#                       'Data {data_time_current:.3f} ({data_time:.3f})\t'
#                       'Loss {loss_current:.4f} ({loss:.4f})'.format(
#                     self.state['epoch'], self.state['iteration'], len(data_loader),
#                     batch_time_current=self.state['batch_time_current'],
#                     batch_time=batch_time, data_time_current=self.state['data_time_batch'],
#                     data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
#             else:
#                 print('Test: [{0}/{1}]\t'
#                       'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
#                       'Data {data_time_current:.3f} ({data_time:.3f})\t'
#                       'Loss {loss_current:.4f} ({loss:.4f})'.format(
#                     self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
#                     batch_time=batch_time, data_time_current=self.state['data_time_batch'],
#                     data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
                
                
# class SymmetricMultiLabelMAPEngine(MultiLabelMAPEngine):
#     def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
#         feature_var = torch.autograd.Variable(self.state['feature']).float()
#         target_var = torch.autograd.Variable(self.state['target']).float()


#         # compute output
#         self.state['output'] = model(feature_var)
#         self.state['loss'] = criterion(self.state['output'], target_var)

#         if training:
#             optimizer.zero_grad()
#             self.state['loss'].backward()
#             nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)
#             optimizer.step()


#     def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

#         self.state['target_gt'] = self.state['target'].clone()
#         self.state['target'][self.state['target'] == 0] = 1
#         self.state['target'][self.state['target'] == -1] = 0

#         input = self.state['input']
#         self.state['feature'] = input[0]
#         self.state['out'] = input[1]
                
#-----------------------------------------------------------------------------------------------------
