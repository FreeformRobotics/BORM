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

parser = argparse.ArgumentParser(description='150objects joint Evaluation')
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
model_file = 'resnet50_best_150obj_joint_dis00.pth.tar'

def load_dict(filename):
    with open(filename,"r") as json_file:
        dic = json.load(json_file)
    return dic
# load the dictionary which contains objects for every image in dataset
one_hot_val=load_dict('150_7classes.json')

fileName='result_150.npy'
num_sp = np.load(fileName)

fileName='number_150.npy'
num_total=np.load(fileName)

matrix_p_o_c=np.zeros(shape=(7,150,150))
for i in range(7):
    for j in range(150):
        if num_sp[i][j]<0:
            num_sp[i][j]=0
for i in range(7):
    X=[]
    Y=[]
    Z=[]
    p_o_c=num_sp[i]/num_total[i]
    p_o_c=p_o_c.reshape(1,p_o_c.shape[0])
#    print(p_o_c)
    p_o_c_tran=p_o_c.T
#    print(p_o_c_tran)
    matrix_p_o_c[i]=np.dot(p_o_c_tran,p_o_c)
    
    
matrix_p_c_o=np.zeros(shape=(7,150,150))
matrix_max=np.zeros(shape=(150,150))
temp=np.zeros(shape=7)
for i in range(150):
    for j in range(150):
        sum=0
        for k in range(7):
            sum += matrix_p_o_c[k][i][j]*1/7
        if sum == 0:
            matrix_p_c_o[k][i][j]=0
            continue
        for k in range(7):
            matrix_p_c_o[k][i][j]=matrix_p_o_c[k][i][j]*1/7/sum
            temp[k]=matrix_p_c_o[k][i][j]
        matrix_max[i][j]=temp.std()
print('matrix_max')
print(matrix_max)
matrix_dis=np.zeros(shape=(150,150))
for i in range(150):
    for j in range(150):
        if matrix_max[i][j]>0:
            matrix_dis[i][j]=1

class Object_Linear(nn.Module):
    def __init__(self):
        super(Object_Linear, self).__init__()
        self.fc1 = nn.Linear(22500, 8192)
        self.fc2 = nn.Linear(8192, 2048)
        self.fc3 = nn.Linear(2048, 7)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(8192)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
object_idt = Object_Linear()
object_idt=object_idt.cuda()

checkpoint = torch.load(model_file)
obj_state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['obj_state_dict'].items()}
object_idt.load_state_dict(obj_state_dict)
object_idt.eval()
print(checkpoint['best_prec1'])

if(args.dataset == 'places'):
    data_dir = '/data/cenj/places365_train'
    valdir = os.path.join(data_dir, 'val')
elif(args.dataset == 'sun'):
    valdir = '/data/cenj/SUNRGBD/test'
elif(args.dataset == 'vpc'):
    data_dir = '/data/cenj/Home/'+args.home
    valdir = os.path.join(data_dir, 'val')


correct_total = 0
totalnumber_list = 0
for class_name in os.listdir(valdir):
    correct,count=0,0
    for img_name in os.listdir(os.path.join(valdir,class_name)):
        img_dir = os.path.join(valdir,class_name,img_name)
        img = Image.open(img_dir)
        input_img = V(centre_crop(img).unsqueeze(0))

        # forward pass

        row = one_hot_val[valdir+'/'+class_name+'/'+img_name]
        row = np.array(row)
        row = row.reshape(1,row.shape[0])
        column = row.T
        matrix = np.dot(column,row)
        matrix = matrix*matrix_max
        obj_hot_vector=matrix.reshape(22500).tolist()
#        obj_hot_vector=np.array(obj_hot_vector)
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
    correct_total += correct
    totalnumber_list += count
    print('Average test accuracy is = {:2.2f}%'.format(100*correct_total/totalnumber_list))
