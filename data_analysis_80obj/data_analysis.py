# by CEN Jun

import argparse
import os
import shutil
import time
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import pdb
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import json
import datetime


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='Training of Combined model')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--num_classes',default=7, type=int,
                    help='num of class in the model')



best_prec1 = 0
proba={}

def load_dict(filename):
    with open(filename,"r") as json_file:
        dic = json.load(json_file)
    return dic
one_hot=load_dict('80obj_7classes_train.json')
one_hot_val=load_dict('80obj_7classes_val.json')
#result will be used to store the occurance count of every objct in every scene, number will be used to store the total image count of every scene
result=np.zeros(shape=(7,80))
number=np.zeros(shape=(7,1))


def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__['resnet50'](num_classes=14)
    model.cuda()
    
    class Object_Linear(nn.Module):
        def __init__(self):
            super(Object_Linear, self).__init__()
            self.fc = nn.Linear(80, 7)

        def forward(self, x):
            out = self.fc(x)
            return out
    object_idt = Object_Linear()
    object_idt = torch.nn.DataParallel(object_idt).cuda()
    
    class ImageFolderWithPaths(datasets.ImageFolder):
        """Custom dataset that includes image file paths. Extends
        torchvision.datasets.ImageFolder
        """

        # override the __getitem__ method. this is the method dataloader calls
        def __getitem__(self, index):
            # this is what ImageFolder normally returns 
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            # the image file path
            path = self.imgs[index][0]
            # make a new tuple that includes original and the path
            tuple_with_path = (original_tuple + (path,))
            return tuple_with_path
    
    data_dir = '/data/cenj/places365_train'
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
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
    k = train_dataset.classes
    print(k)

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

    params = list(model.parameters())

    optimizer = torch.optim.Adam(params, args.lr,weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, object_idt, criterion)
        return
    
    accuracies_list = []
    for epoch in range(args.start_epoch, args.epochs):

        # get the distribution for training dataset
        train(train_loader, model,object_idt, criterion, optimizer, epoch)

        # evaluate on validation set
#        prec1 = validate(val_loader, model, object_idt, criterion)
    np.save("number_80.npy", number)
    np.save("result_80.npy", result)

def train(train_loader, model,object_idt, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    object_idt.train()


    end = time.time()
    for i, (input, target, path) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()

        obj_id_batch = []
        for j in range(len(path)):
            number[target[j]][0] += 1
            obj_hot_vector = one_hot[path[j]]
            # Calculate occurance count of every objct in every scene
            for k in range(80):
                if obj_hot_vector[k] == 1:
                    result[target[j]][k] += 1
            obj_id_batch.append(obj_hot_vector)
        t = torch.autograd.Variable(torch.FloatTensor(obj_id_batch)).cuda()
        output = object_idt(t)
        loss = criterion(output, target)

        # measure accuracy and record loss
#        print(output.data)
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

def validate(val_loader, model, object_idt, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    object_idt.eval()


    end = time.time()
    for i, (input, target, path) in enumerate(val_loader):
        target = target.cuda()
#        input_var = torch.autograd.Variable(input).cuda()
        with torch.no_grad():

            # compute output

            obj_id_batch = []
            for j in range(len(path)):
                img = Image.open(path[j])
                sized = letterbox_image(img, m.width, m.height)
                boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
                img, objects = plot_boxes(img, boxes, 'predictions.jpg', class_names)
                number[target[j]][0] += 1
                indices = [class_names.index(x) for x in objects]
                for k in indices:
                    result[target[j]][k] += 1
                obj_hot_vector = get_hot_vector(objects, class_names)
                proba[path[j]]=obj_hot_vector
                obj_id_batch.append(obj_hot_vector)
            t = torch.autograd.Variable(torch.FloatTensor(obj_id_batch)).cuda()
            output = object_idt(t)
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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename + '_latest_word_vector.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest_word_vector.pth.tar', filename + '_best_word_vector.pth.tar')


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


if __name__ == '__main__':
    main()
