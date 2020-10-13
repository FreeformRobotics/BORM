# Developed by Liguang Zhou, 2020.9.30

import argparse
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
# print('model_names:', model_names)
BATCH_SIZE = 128
Epochs = 40
Learning_rate = 0.01
Momentum = 0.9
Weight_decay = 1e-4
Num_classes = 14
DIS_SCALE = 1.0

class arguments_parse(object):
    def argsParser():
        parser = argparse.ArgumentParser(description='Training of CIOM model')
        parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                            help='model architecture: ' +
                                ' | '.join(model_names) +
                                ' (default: resnet50)')
        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('--epochs', default=Epochs, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('-b', '--batch-size', default=BATCH_SIZE, type=int,
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--lr', '--learning-rate', default=Learning_rate, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)')
        parser.add_argument('--print-freq', '-p', default=500, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')
        parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                            help='use pre-trained model')
        parser.add_argument('--num_classes',default=Num_classes, type=int,
                            help='num of class in the model')
        parser.add_argument('--om_cls_num',default=150,type=int,
                            help='num of class of the model model')    
        parser.add_argument('--dataset',default='Places365-14',type=str,
                            help='Choose the dataset used for training')
        parser.add_argument('--om_type', default='copm_resnet50', type=str,
                            help='choose the type of object model')
        parser.add_argument('--DIS_SCALE',default=DIS_SCALE,type=float,
                            help='choose the scale of discriminative matrix')
        args = parser.parse_args()

        return args
    
    def test_argsParser():
        parser = argparse.ArgumentParser(description='Testing of CIOM model')
        parser.add_argument('--dataset',default='Places365-14',type=str,
                            help='Choose the dataset used for training')
        parser.add_argument('--num_classes',default=14, type=int,
                            help='num of class in the model')
        parser.add_argument('--om_type', default='copm_resnet50', type=str,
                            help='choose the type of object model')
        parser.add_argument('--DIS_SCALE',default=DIS_SCALE,type=float,
                            help='choose the scale of discriminative matrix')

        args = parser.parse_args()
        return args

