# by CEN Jun
# Revised and Reconstruct by Liguang Zhou

import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models

import pdb
import matplotlib.pyplot as plt
import json
import datetime
import numpy as np

from model import Object_CDOPM_ResNet18, Object_CDOPM_ResNet50, Object_IOM, Fusion_CDOPM_ResNet50, Fusion_CIOM, Fusion_CDOPM_ResNet18
from arguments import arguments_parse
from dataset import ImageFolderWithPaths, DatasetSelection


best_prec1 = 0

def main():
    global args, best_prec1,discriminative_matrix
    args = arguments_parse.argsParser()
    print(args)
    
    # dataset selection for training
    dataset_selection = DatasetSelection(args.dataset)
    one_hot, data_dir, model = dataset_selection.datasetSelection()
    discriminative_matrix = dataset_selection.discriminative_matrix_estimation()


    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')

    # create the model for training
    if args.om_type == 'ciom_resnet50':
        object_idt = Object_IOM()
        classifier = Fusion_CIOM(args.num_classes)

    elif args.om_type == 'cdopm_resnet18':
        model_dir = './weights'
        model_arch = 'resnet18'
        object_idt = Object_CDOPM_ResNet18()
        classifier = Fusion_CDOPM_ResNet18(args.num_classes)
        # load the resnet18 checkpoint pretrained over the place365
        print("=> creating model '{}'".format(model_arch))
        model_file = os.path.join(model_dir, 'resnet18_places365.pth.tar')

        # pretrained on the place365
        model = models.__dict__['resnet18'](num_classes=365)
        checkpoint = torch.load(model_file)

        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.cuda()

        for param in model.parameters():
            param.requires_grad = True
        print(model)

    elif args.om_type == 'cdopm_resnet50':
        object_idt = Object_CDOPM_ResNet50()
        classifier = Fusion_CDOPM_ResNet50(args.num_classes)

        model_arch = 'resnet50'
        print("=> creating model '{}'".format(model_arch))
        # model_file='./weights/resnet50_best_res50.pth.tar'
        model_file = './weights/resnet50_best_res50.pth.tar'
        model = models.__dict__[model_arch](num_classes=14)
        checkpoint = torch.load(model_file)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.cuda()
        for param in model.parameters():
            param.requires_grad = True
        print(model)

    object_idt.cuda()
    classifier.cuda()
    # object_idt = torch.nn.DataParallel(object_idt).cuda()
    # classifier = torch.nn.DataParallel(classifier).cuda()

    latest_model_name =  './weights/' + args.om_type + '_latest' + '.pth.tar'
    best_model_name = './weights/' + args.om_type + '_best' + '.pth.tar'

    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['model_state_dict'])
            object_idt.load_state_dict(checkpoint['obj_state_dict'])
            classifier.load_state_dict(checkpoint['classifier_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        for param in model.parameters():
            param.requires_grad = False

    cudnn.benchmark = True


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    
    train_dataset = ImageFolderWithPaths(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = ImageFolderWithPaths(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    params = list(object_idt.parameters())+list(classifier.parameters())
    # params = list(model.parameters())

    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, one_hot, object_idt, classifier, criterion)
        return
    
    accuracies_list = []
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        if epoch != 0 and epoch % 10 == 0:
            print("=> loading checkpoint '{}'".format(best_model_name))
            checkpoint = torch.load(best_model_name)

            model.load_state_dict(checkpoint['model_state_dict'])
            object_idt.load_state_dict(checkpoint['obj_state_dict'])
            classifier.load_state_dict(checkpoint['classifier_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # train for one epoch
        train(train_loader, model, one_hot, object_idt, classifier, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, one_hot, object_idt, classifier, criterion)
        accuracies_list.append("%.2f"%prec1.tolist())

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print(best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'model_state_dict': model.state_dict(),
            'obj_state_dict': object_idt.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, best_model_name, latest_model_name)
        print("The best accuracy obtained during training is = {}".format(best_prec1))

def train(train_loader, model, one_hot, object_idt, classifier, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    object_idt.train()
    classifier.train()

    end = time.time()
    for i, (input, target, path) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()

        # get the one-hot object vector through dictionary        
        obj_id_batch = []
        for j in range(len(path)):
            row = one_hot[path[j]]

            if args.om_type == 'ciom_resnet50':
                # print('ciom_model')
                obj_hot_vector = row

            elif args.om_type == 'copm_resnet50':
                row = np.array(row)
                row = row.reshape(1,row.shape[0])
                column = row.T
                object_pair_matrix = np.dot(column,row)
                obj_hot_vector = object_pair_matrix.reshape(22500).tolist()

            elif args.om_type == 'cdopm_resnet50' or args.om_type == 'cdopm_resnet18':
                row = np.array(row)
                row = row.reshape(1,row.shape[0])
                column = row.T
                object_pair_matrix = np.dot(column,row)
                object_discriminative_matrix = object_pair_matrix*discriminative_matrix*args.DIS_SCALE
                obj_hot_vector = object_discriminative_matrix.reshape(22500).tolist()
            
            obj_id_batch.append(obj_hot_vector)

        t = torch.autograd.Variable(torch.FloatTensor(obj_id_batch)).cuda()
        # print('t.shape:', t.shape)

        # compute output
        output_conv = my_forward(model,input_var)
        # print('output_conv.shape:', output_conv.shape)
        output_idt = object_idt(t)
        # print('output_idt:', output_idt.shape)
        output = classifier(output_conv,output_idt) 
        # print('output:', output.shape) 
        # print('target:', target.shape)
        loss = criterion(output, target)
        # print('loss:', loss.shape, loss)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def validate(val_loader, model, one_hot, object_idt, classifier, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    object_idt.eval()
    classifier.eval()

    end = time.time()
    for i, (input, target, path) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        with torch.no_grad():

            obj_id_batch = []
            for j in range(len(path)):
                row = one_hot[path[j]]

                if args.om_type == 'ciom_resnet50':
                    # print('ciom_model')
                    obj_hot_vector = row

                elif args.om_type == 'copm_resnet50':
                    row = np.array(row)
                    row = row.reshape(1,row.shape[0])
                    column = row.T
                    object_pair_matrix = np.dot(column,row)
                    obj_hot_vector = object_pair_matrix.reshape(22500).tolist()

                elif args.om_type == 'cdopm_resnet50' or args.om_type == 'cdopm_resnet18':
                    row = np.array(row)
                    row = row.reshape(1,row.shape[0])
                    column = row.T
                    object_pair_matrix = np.dot(column,row)
                    object_discriminative_matrix = object_pair_matrix*discriminative_matrix*args.DIS_SCALE
                    obj_hot_vector = object_discriminative_matrix.reshape(22500).tolist()
                    
                obj_id_batch.append(obj_hot_vector)

            # compute output
            output_conv = my_forward(model,input_var)
            t = torch.autograd.Variable(torch.FloatTensor(obj_id_batch)).cuda()
            output_idt = object_idt(t)
            output = classifier(output_conv,output_idt)  
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss, input.size(0))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, best_model_name, latest_model_name):
    torch.save(state, latest_model_name)
    if is_best:
        shutil.copyfile(latest_model_name, best_model_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def my_forward(model, x):
    mo = nn.Sequential(*list(model.children())[:-1])
    feature = mo(x)
#    print(feature.size())
    feature = feature.view(x.size(0), -1)
    output= model.fc(feature)
    return feature

if __name__ == '__main__':
    main()
