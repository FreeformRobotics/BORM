# by Liguang Zhou, 2020.9.30

import torch
import torch.nn as nn


class Object_IOM(nn.Module):
        def __init__(self):
            super(Object_IOM, self).__init__()
            self.fc1 = nn.Linear(150, 512)

        def forward(self, x):
            out = self.fc1(x)
            return out

class Object_CDOPM_ResNet18(nn.Module):
    def __init__(self):
        super(Object_CDOPM_ResNet18, self).__init__()
        self.fc1 = nn.Linear(22500, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.fc1(x)
#        out = self.dropout(out)
#        out = self.relu(out)
        out = self.fc2(out)
        return out

class Object_CDOPM_ResNet50(nn.Module):
        def __init__(self):
            super(Object_CDOPM_ResNet50, self).__init__()
            self.fc1 = nn.Linear(22500, 8192)
            self.fc2 = nn.Linear(8192, 2048)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            return out

class Fusion_CDOPM_ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(Fusion_CDOPM_ResNet18, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, conv, idt):
        out = torch.cat((conv,idt),1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Fusion_CDOPM_ResNet50(nn.Module):
        def __init__(self, num_classes):
            super(Fusion_CDOPM_ResNet50, self).__init__()
            self.num_classes = num_classes
            self.fc1 = nn.Linear(4096, 512)
            self.fc2 = nn.Linear(512, self.num_classes)
            self.dropout = nn.Dropout(0.5)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, conv, idt):
            out = torch.cat((conv,idt),1)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            return out

class Fusion_CIOM(nn.Module):
    def __init__(self,num_classes):
        super(LinClassifier_CIOM, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(2560, num_classes)

    def forward(self, conv, idt):
        out = torch.cat((conv,idt),1)
        out = self.fc(out)
        return out