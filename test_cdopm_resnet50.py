# by CEN Jun
# Revised and Reconstruct by Liguang Zhou

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import json
import datetime
import numpy as np
import csv

from model import Object_CDOPM_ResNet18, Object_CDOPM_ResNet50, Object_IOM, Fusion_CDOPM_ResNet50, Fusion_CIOM, Fusion_CDOPM_ResNet18
from arguments import arguments_parse
from dataset import ImageFolderWithPaths, DatasetSelection



global args
best_prec1 = 0


def my_forward(model, x):
    mo = nn.Sequential(*list(model.children())[:-1])
    feature = mo(x)
#    print(feature.size())
    feature = feature.view(x.size(0), -1)
    output = model.fc(feature)
    return feature

if __name__ == '__main__':

    args = arguments_parse.test_argsParser()
    # dataset selection for training
    dataset_selection = DatasetSelection(args.dataset)
    one_hot, data_dir, classes = dataset_selection.datasetSelection()
    discriminative_matrix = dataset_selection.discriminative_matrix_estimation()

    if args.dataset == 'sun':
        valdir = os.path.join(data_dir, 'test')
    else:
        valdir = os.path.join(data_dir, 'val')
    print('valdir:', valdir)

    # create the model for training
    if args.om_type == 'ciom_resnet50':
        # print('model is ciom')
        object_idt = Object_IOM()
        classifier = LinClassifier_CIOM(args.num_classes)

    elif args.om_type == 'cdopm_resnet18':
        object_idt = Object_CDOPM_ResNet18()
        classifier = Fusion_CDOPM_ResNet18(args.num_classes)

    elif args.om_type == 'cdopm_resnet50':
        object_idt = Object_CDOPM_ResNet50()
        classifier = Fusion_CDOPM_ResNet50(args.num_classes)

    # object_idt = torch.nn.DataParallel(object_idt).cuda()
    # classifier = torch.nn.DataParallel(classifier).cuda()

    object_idt.cuda()
    classifier.cuda()

    # th architecture to use
    if args.om_type == 'cdopm_resnet50':
        arch = 'resnet50'
        model = models.__dict__[arch](num_classes=14)
        # load the pre-trained weights
        best_model_name = './weights/' + args.om_type + '_latest' + '.pth.tar'
        print('best_model_name:', best_model_name)
        checkpoint = torch.load(best_model_name)
        model_state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        obj_state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['obj_state_dict'].items()}
        classifier_state_dict = {str.replace(k, 'module.', ''): v for k, v in
                                 checkpoint['classifier_state_dict'].items()}

        model.load_state_dict(model_state_dict)
        object_idt.load_state_dict(obj_state_dict)
        classifier.load_state_dict(classifier_state_dict)

        model.eval()
        object_idt.eval()
        classifier.eval()

        model.cuda()
        object_idt.cuda()
        classifier.cuda()

    elif args.om_type == 'cdopm_resnet18':
        # the architecture to use
        arch = 'resnet18'
        # model = models.__dict__[arch](num_classes=cls_num)

        # load the pre-trained weights
        model_dir = 'weights'
        model_file = os.path.join(model_dir, 'cdopm_resnet18_best.pth.tar')
        checkpoint = torch.load(model_file)

        # pretrained on the place365
        model = models.__dict__[arch](num_classes=365)

        model_state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        obj_state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['obj_state_dict'].items()}
        classifier_state_dict = {str.replace(k, 'module.', ''): v for k, v in
                                 checkpoint['classifier_state_dict'].items()}
        model.load_state_dict(model_state_dict)
        object_idt.load_state_dict(obj_state_dict)
        classifier.load_state_dict(classifier_state_dict)

        # evaluation mode
        model.eval()
        object_idt.eval()
        classifier.eval()
        model.cuda()
        object_idt.cuda()
        classifier.cuda()
        print(checkpoint['best_prec1'])

    # load the image transformer
    centre_crop = trn.Compose([
        trn.Resize((256, 256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    correct_list = []
    totalnumber_list = []
    for class_name in os.listdir(valdir):
        # print('class_name:', class_name)
        # print('valdir:', valdir)
        correct, count = 0, 0
        for img_name in os.listdir(os.path.join(valdir, class_name)):
            img_dir = os.path.join(valdir, class_name, img_name)
            img = Image.open(img_dir)
            input_img = V(centre_crop(img).unsqueeze(0)).cuda()

            # forward pass
            output_conv = my_forward(model, input_img)
            row = one_hot[os.path.join(valdir, class_name,img_name)]

            if args.om_type == 'ciom_resnet50':
                obj_hot_vector = row

            elif args.om_type == 'copm_resnet50':
                row = np.array(row)
                row = row.reshape(1, row.shape[0])
                column = row.T
                object_pair_matrix = np.dot(column, row)
                obj_hot_vector = object_pair_matrix.reshape(22500).tolist()

            elif args.om_type == 'cdopm_resnet50' or args.om_type == 'cdopm_resnet18':
                row = np.array(row)
                row = row.reshape(1, row.shape[0])
                column = row.T
                object_pair_matrix = np.dot(column, row)
                object_discriminative_matrix = object_pair_matrix * discriminative_matrix * args.DIS_SCALE
                obj_hot_vector = object_discriminative_matrix.reshape(22500).tolist()

            t = torch.autograd.Variable(torch.FloatTensor(obj_hot_vector)).cuda()

            output_idt = object_idt(t)
            output_idt = output_idt.unsqueeze(0)
            # print('output_idt:', output_idt.shape)
            logit = classifier(output_conv, output_idt)
            # print('logit:', logit.shape)
            h_x = F.softmax(logit, 1).data.squeeze()
            # print('h_x:', h_x.shape, h_x)

            probs, idx = h_x.sort(0, True)
            # print('probs:', probs, 'idx[0]:', idx[0], 'idx:', idx)

            result = classes[idx[0]]

            if (result == class_name):
                correct += 1
            count += 1
        accuracy = 100 * correct / float(count)
        print('Accuracy of {} class is {:2.2f}%'.format(class_name, accuracy))
        correct_list.append(correct)
        totalnumber_list.append(count)

    print('Average test accuracy is = {:2.2f}%'.format(100 * sum(correct_list) / float(sum(totalnumber_list))))