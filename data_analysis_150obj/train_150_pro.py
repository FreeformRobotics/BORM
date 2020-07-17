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
import csv



os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Training of Discriminative 150objects model')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--num_classes',default=7, type=int,
                    help='num of class in the model')
parser.add_argument('--count',default=0, type=int,
                    help='theshold num of noise')
parser.add_argument('--threshold',default=0, type=float,
                    help='discriminative threshold')

global args, best_prec1
args = parser.parse_args()
print(args)

def load_dict(filename):
    with open(filename,"r") as json_file:
        dic = json.load(json_file)
    return dic

one_hot=load_dict('150_7classes.json')

best_prec1 = 0

fileName='result_150.npy'
num_sp = np.load(fileName)
#print(num_sp)
fileName='number_150.npy'
num_total=np.load(fileName)
#print(num_total)
file_name='categories_places365_home.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
        
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])-1] = row[5].split(";")[0]

p_o_c=np.zeros(shape=(7,150))

for i in range(7):
    for j in range(150):
        # Get rid of objects which rarely appear in certain scene which is noise
        if num_sp[i][j]<args.count:
            num_sp[i][j]=0
            
# Calculate the objects distribution for each scene
for i in range(7):
    p_o_c[i]=num_sp[i]/num_total[i]

p_c_o=np.zeros(shape=(150,7))
max_dis=np.zeros(shape=(150))

# Calculate the scene distribution for each object using Bayes theorem
for i in range(150):
    sum=0
    for j in range(7):
        sum += 1/7*p_o_c[j][i]
    if sum==0:
        p_c_o[i]=0
        continue
    for j in range(7):
        p_c_o[i][j]=1/7*p_o_c[j][i]/sum
    # Calculate the discriminative value based on scene distribution
    max_dis[i]=p_c_o[i].std()

y = np.argsort(max_dis,axis=0)
# Print the most discriminative objects
#for i in y:
#    print(names[i])
#    print(max_dis[i])
discriminative=[]
# Store the index of discriminative objects in a list
for j in range(149,-1,-1):
    if max_dis[y[j]]>args.threshold:
        discriminative.append(y[j])
#for i in discriminative:
#    print(names[i])

for i in range(150):
    print(names[i])
    print(max_dis[i])

print(max_dis.shape)

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    class Object_Linear(nn.Module):
        def __init__(self):
            super(Object_Linear, self).__init__()
            self.fc1 = nn.Linear(150, 32)
            self.fc2 = nn.Linear(32, args.num_classes)

        def forward(self, x):
            out1 = self.fc1(x)
            out2 = self.fc2(out1)
            return out2
    object_idt = Object_Linear()
    object_idt=object_idt.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            object_idt.load_state_dict(checkpoint['obj_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    data_dir = '/data/cenj/places365_train'
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

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

    params = list(object_idt.parameters())

    optimizer = torch.optim.SGD(params, args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay)


    if args.evaluate:
        validate(val_loader, model, object_idt, criterion)
        return
    
    accuracies_list = []
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        if epoch != 0 and epoch % 10 == 0:
            print("=> loading checkpoint '{}'".format('resnet50_best_150obj_dis'+str(args.count)+str(args.threshold)+'.pth.tar'))
            checkpoint = torch.load('resnet50_best_150obj_dis'+str(args.count)+str(args.threshold)+'.pth.tar')
            object_idt.load_state_dict(checkpoint['obj_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
              
        # train for one epoch
        train(train_loader,object_idt, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, object_idt, criterion)
        accuracies_list.append("%.2f"%prec1.tolist())
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'obj_state_dict': object_idt.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
        print("The best accuracy obtained during training is = {}".format(best_prec1))


def train(train_loader, object_idt, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    object_idt.train()


    end = time.time()
    for i, (input, target, path) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()

        obj_id_batch = []
        for j in range(len(path)):
            obj_hot_vector = one_hot[path[j]]
            # Ignore objects which are not discriminative
            obj_hot_vector = obj_hot_vector*max_dis*5;
            obj_id_batch.append(obj_hot_vector)
        t = torch.autograd.Variable(torch.FloatTensor(obj_id_batch)).cuda()
        
        output = object_idt(t)
        loss = criterion(output, target)

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

def validate(val_loader, object_idt, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    object_idt.eval()


    end = time.time()
    for i, (input, target, path) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        with torch.no_grad():

            # compute output

            obj_id_batch = []
            for j in range(len(path)):
                obj_hot_vector = one_hot[path[j]]
                obj_hot_vector = obj_hot_vector*max_dis*5
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


def save_checkpoint(state, is_best, filename='resnet50'):
    torch.save(state, filename + '_latest_150obj_dis'+str(args.count)+str(args.threshold)+'.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest_150obj_dis'+str(args.count)+str(args.threshold)+'.pth.tar', filename + '_best_150obj_dis'+str(args.count)+str(args.threshold)+'.pth.tar')


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
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
#    lr = args.lr * (args.lr ** (epoch // 3))
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


if __name__ == '__main__':
    main()
