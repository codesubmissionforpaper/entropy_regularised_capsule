import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from constants import *

class Conv2(nn.Module):
      def __init__(self):
          super(Conv2,self).__init__()
          self.conv1 = nn.Sequential(nn.Conv2d(3,32,3),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU())
          self.conv2 = nn.Sequential(nn.Conv2d(32,32,3,2),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU())
          self.pooling1 = nn.MaxPool2d(3,2)
          self.conv3 = nn.Sequential(nn.Conv2d(32,32,5),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU())
          self.conv4 = nn.Sequential(nn.Conv2d(32,64,3),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU())
          self.pooling2 = nn.MaxPool2d(3,2)
          self.conv5 = nn.Sequential(nn.Conv2d(64,64,3),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU())
          self.pooling3 = nn.MaxPool2d(3,1)
          self.conv6 = nn.Sequential(nn.Conv2d(64,64,5),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU())
          self.pooling4 = nn.MaxPool2d(3,1)
          self.layer = nn.Linear(64,2)
          #self.softmax = nn.Softmax(dim=1)

      def forward(self,x):
          x = self.conv1(x)
          x = self.conv2(x)
          x = self.pooling1(x)
          x = self.conv3(x)
          x = self.conv4(x)
          x = self.pooling2(x)
          x = self.conv5(x)
          x = self.pooling3(x)
          x = self.conv6(x)
          x = self.pooling4(x)
          batch_size = x.size(0)
          x = x.view(batch_size,64)
          x = self.layer(x)
          #x = self.softmax(x)
          return x
          
          

