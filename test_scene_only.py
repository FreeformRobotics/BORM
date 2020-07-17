# python test_scene_only.py --dataset=places --envtype=home
# python test_scene_only.py --dataset=vpc --hometype=home1 --floortype=data_1

# Prediction for Scene_Only model
#
# by Anwesan Pal

import argparse
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
from config import places365_dir, vpc_dir
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='DEDUCE Scene_Only Evaluation')
parser.add_argument('--dataset',default='places',help='dataset to test')
parser.add_argument('--hometype',default='home1',help='home type to test')
parser.add_argument('--floortype',default='data_0',help='data type to test')
parser.add_argument('--envtype',default='home',help='home or office type environment')

global args
args = parser.parse_args()

# th architecture to use
arch = 'resnet18'

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []

    for idx in class_idx:
        print(idx)
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
#        print(cam_img)
        output_cam.append(imresize(cam_img, size_upsample))
    print(output_cam)
    return output_cam

# load the pre-trained weights
model_file = 'resnet18_best_home.pth.tar'
#model_file = 'resnet18_places365.pth.tar'
model = models.__dict__[arch](num_classes=7)
checkpoint = torch.load(model_file)
print(checkpoint['best_prec1'])
print(checkpoint['epoch'])
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
#for k,v in checkpoint['state_dict'].items():
#    print(k)
#    print(v)
model.load_state_dict(state_dict)
#print(model)
model.cuda()
model.eval()
#features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
#for name in features_names:
#    model._modules.get(name).register_forward_hook(hook_feature)

#device = torch.device("cuda:0,3" if torch.cuda.is_available() else "cpu")#第一行代码
#if torch.cuda.device_count() > 1:
#   model = torch.nn.DataParallel(model)
#model.to(device)#第二行代码


#params = list(model.parameters())
#weight_softmax = params[-2].data.cpu().numpy()
#weight_softmax[weight_softmax<0] = 0

# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365_{}.txt'.format(args.envtype)
#file_name='IO_places365.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)
print(classes)

if(args.dataset == 'places'):
    data_dir = '/data/cenj/places365_train'
    valdir = os.path.join(data_dir, 'val')
elif(args.dataset == 'sun'):
    valdir='/data/cenj/SUNRGBD_val'
elif(args.dataset == 'vpc'):
    data_dir = vpc_dir
    home_dir = os.path.join(data_dir, 'data_'+args.hometype)
    valdir = os.path.join(home_dir,args.floortype)

accuracies_list = []
for class_name in os.listdir(valdir):
#    print(os.listdir(valdir))
#    print(class_name)
    correct,count=0,0
    for img_name in os.listdir(os.path.join(valdir,class_name)):
        img_dir = os.path.join(valdir,class_name,img_name)
#        print(img_dir)
        img = Image.open(img_dir)
        input_img = V(centre_crop(img).unsqueeze(0)).cuda()

        # forward pass
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        
        wrong_dir = os.path.join('/data/cenj/wrong_dir',img_name+'_'+classes[idx[0]])
        

        if(classes[idx[0]] == class_name):
                correct+=1
        
#        print(features_blobs[0])
#        CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
#        print(CAMs[0])
#        # render the CAM and output
#        img = cv2.imread(img_dir)
#        height, width, _ = img.shape
#        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
#        result = heatmap * 0.4 + img * 0.6
#        cv2.imwrite(wrong_dir, result)
        count+=1
    print(count)
    accuracy = 100*correct/float(count)
    print('Accuracy of {} class is {:2.2f}%'.format(class_name,accuracy))
    accuracies_list.append(accuracy)
print('Average test accuracy is = {:2.2f}%'.format(sum(accuracies_list)/len(accuracies_list)))
