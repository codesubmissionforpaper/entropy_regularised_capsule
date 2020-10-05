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
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4

class Conv4(nn.Module):
      def __init__(self):
          super(Conv4,self).__init__()
          self.conv1 = nn.Sequential(P4ConvZ2(3,32,3),
                                      nn.BatchNorm3d(32),
                                      nn.ReLU())
          self.conv2 = nn.Sequential(P4ConvP4(32,32,3,2),
                                     nn.BatchNorm3d(32),
                                     nn.ReLU())
          self.conv3 = nn.Sequential(P4ConvP4(32,32,5),
                                     nn.BatchNorm3d(32),
                                     nn.ReLU())
          self.conv4 = nn.Sequential(P4ConvP4(32,64,3),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU())
          self.conv5 = nn.Sequential(P4ConvP4(64,64,3),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU())
          self.conv6 = nn.Sequential(P4ConvP4(64,64,5),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU())
          self.layer = nn.Linear(256,2)
          self.softmax = nn.Softmax(dim=1)

      def forward(self,x):
          x = self.conv1(x)
          x = self.conv2(x)
          x = plane_group_spatial_max_pooling(x,3,2)
          x = self.conv3(x)
          x = self.conv4(x)
          x = plane_group_spatial_max_pooling(x,3,2)
          x = self.conv5(x)
          x = plane_group_spatial_max_pooling(x,3,1)
          x = self.conv6(x)
          x = plane_group_spatial_max_pooling(x,3,1)
          batch_size = x.size(0)
          x = x.view(batch_size,64*4)
          x = self.layer(x)
          x = self.softmax(x)
          return x
          
          

