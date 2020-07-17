# by CEN Jun
import argparse
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image, ImageDraw

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import json
import datetime
import numpy as np
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='80objects Evaluation')
parser.add_argument('--dataset',default='sun',help='dataset to test')
parser.add_argument('--home',default='data_home1',help='specific home dataset of vpc to test')
parser.add_argument('--envtype',default='home',help='home or office type environment')

centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

global args
args = parser.parse_args()


file_name='categories_places365_home.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

# load the pre-trained weights
model_file = 'resnet50_best_80obj.pth.tar'

def load_dict(filename):
    with open(filename,"r") as json_file:
        dic = json.load(json_file)
    return dic
# load the dictionary which contains objects for every image in dataset
one_hot_val=load_dict('80obj_7classes_SUN.json')

class Object_Linear(nn.Module):
    def __init__(self):
        super(Object_Linear, self).__init__()
        self.fc1 = nn.Linear(80, 32)
        self.fc2 = nn.Linear(32, 7)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(out1)
        return out2
object_idt = Object_Linear()
object_idt=object_idt.cuda()

checkpoint = torch.load(model_file)
obj_state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['obj_state_dict'].items()}
object_idt.load_state_dict(obj_state_dict)
object_idt.eval()

if(args.dataset == 'places'):
    data_dir = '/data/cenj/places365_train'
    valdir = os.path.join(data_dir, 'val')
elif(args.dataset == 'sun'):
    valdir = os.path.join('/data/cenj/SUNRGBD_val')
elif(args.dataset == 'vpc'):
    data_dir = '/data/cenj/Home/'+args.home
    valdir = os.path.join(data_dir, 'val')


correct_list = []
totalnumber_list = []
for class_name in os.listdir(valdir):
    correct,count=0,0
    for img_name in os.listdir(os.path.join(valdir,class_name)):
        img_dir = os.path.join(valdir,class_name,img_name)
        img = Image.open(img_dir)
        input_img = V(centre_crop(img).unsqueeze(0))

        # forward pass

        obj_hot_vector = one_hot_val[valdir+'/'+class_name+'/'+img_name]
        obj_hot_vector=np.array(obj_hot_vector)
        t = torch.autograd.Variable(torch.FloatTensor(obj_hot_vector)).cuda()
        logit = object_idt(t)
        h_x = F.softmax(logit).data.squeeze()
        probs, idx = h_x.sort(0, True)
        result=classes[idx[0]]

        if(result == class_name):
            correct+=1
        count+=1
    accuracy = 100*correct/float(count)
    print('Accuracy of {} class is {:2.2f}%'.format(class_name,accuracy))
    correct_list.append(correct)
    totalnumber_list.append(count)
print('Average test accuracy is = {:2.2f}%'.format(100*sum(correct_list)/float(sum(totalnumber_list))))
