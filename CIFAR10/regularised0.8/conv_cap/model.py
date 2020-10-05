import torch
from utils import *
import torch.nn as nn
import torch.nn.functional as F

#This file contains an architecture for a convolutional SOVNET.

#device is for a single gpu
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

class ConvSov(nn.Module):
    '''Create a convolutional SOV layer that transfer capsule layer L to capsule layer L+1 by dynamic routing.
    Args:
        num_in_capsules: input number of types of capsules
        num_out_capsules: output number of types of capsules
        kernel_size: kernel size of convolution
        in_dim: dimension of input capsule 
        out_dim: dimension of output capsule
        stride: stride of convolution
    Shape:
        input: (batch_size, num_in_capsules, H, W, in_dim)
        output: (batch_size, num_out_capsules, H', W', out_dim)
        H', W' is computed the same way as for a convolution layer
        NOTE: this layer does not support separate activation
    '''
    def __init__(self, num_in_capsules, num_out_capsules, kernel_size, in_dim, out_dim, stride=1):
        super(ConvSov, self).__init__()
        self.num_in_capsules = num_in_capsules
        self.num_out_capsules = num_out_capsules
        self.kernel_size = kernel_size
        self.out_dim = out_dim
        self.stride = stride
        self.projector = nn.Sequential(
                                       nn.Linear(in_dim,out_dim*num_out_capsules),
                                       nn.ReLU(),
                                       nn.Linear(out_dim,128),
                                       nn.ReLU(),
                                       nn.Linear(128,out_dim*num_out_capsules)  
                                       )
    
    def forward(self, input_capsules):
        batch_size, num_in_capsules, H, W, in_dim = input_capsules.size()
        #(batch_size, num_in_capsules, H', W', kernel_size, kernel_size, in_dim)
        transformed_input_capsules = self.create_patches(input_capsules)
        #(batch_size, num_in_capsules, H', W', kernel_size, kernel_size, out_dim*num_out_capsules)
        output_predictions = self.projector(transformed_input_capsules)
        H, W = output_predictions.size(2), transformed_input_capsules.size(3)        
        #(batch_size, num_in_capsules, H', W', kernel_size, kernel_size, num_out_capsules, out_dim)
        output_predictions = output_predictions.view(batch_size,num_in_capsules,H,W,self.kernel_size,self.kernel_size,self.num_out_capsules,self.out_dim)
        #(batch_size, num_in_capsules, kernel_size, kernel_size, num_out_capsules, H', W', out_dim)
        output_predictions = output_predictions.permute(0,1,4,5,6,2,3,7).contiguous()
        #(batch_size, num_out_capsules, H', W', out_dim)       
        output_capsules, c_ij = self.dynamic_routing(output_predictions)  
        return output_capsules, c_ij
    
    def squash(self, x, dim):
        norm_squared = (x ** 2).sum(dim, keepdim=True)
        part1 = norm_squared / (1 +  norm_squared)
        part2 = x / torch.sqrt(norm_squared+ 1e-16)
        output = part1 * part2 
        return output
    
    def create_patches(self, input_capsules):
        """
            Input: (batch_size, num_capsules, H, W, in_dim)
            Output: (batch_size, num_capsules, H', W', kernel_size, kernel_size, in_dim)
        """
        batch_size, num_capsules, H, W, in_dim = input_capsules.size()
        #print(self.kernel_size)
        oH = int((H - self.kernel_size)/(self.stride)) + 1 
        oW = int((W - self.kernel_size)/(self.stride)) + 1
        idxs = [ [(h_idx + k_idx) for k_idx in range(0, self.kernel_size)] for h_idx in range(0, H - self.kernel_size + 1, self.stride) ]
        if len(idxs) != 0:
           input_capsules = input_capsules[:, :, idxs, :, :]
           input_capsules = input_capsules[:, :, :, :, idxs, :]
        else:
             input_capsules = input_capsules.unsqueeze(2).unsqueeze(4)
        input_capsules = input_capsules.contiguous()
        input_capsules = input_capsules.permute(0,1,2,4,3,5,6)
        #print(input_capsules.size())
        return input_capsules

    def dynamic_routing(self, predictions,ITER=3):
        """
        Input: (batch_size, num_in_capsules, kernel_size, kernel_size, num_out_capsules, H', W', out_dim)
        Output: (batch_size, num_out_capsules, H', W', out_dim)
        """
        batch_size,num_in_capsules,kernel_size,kernel_size,num_out_capsules,H,W,out_dim= predictions.size()
        #(batch_size,num_in_capsules)    
        predictions = predictions.view(batch_size,num_in_capsules*kernel_size*kernel_size,
                      num_out_capsules,H,W,out_dim)
        num_capsules = predictions.size(1)
        #dynamic routing (batch_size,num_capsules,num_out_capsules,H,W,1)
        b_ij = predictions.new_zeros(batch_size,num_capsules,num_out_capsules,H,W,1)
        for i in range(ITER):
            c_ij = F.softmax(b_ij, dim=2)
            #(batch_size,1,num_out_capsules,H,W,out_dim)
            s_j  = (c_ij * predictions).sum(dim=1, keepdim=True)
            #(batch_size,1,num_out_capsules,H,W,out_dim) 
            v_j  = self.squash(s_j, dim=-1)
            if i < ITER -1:
                #(batch_size,num_in_capsules,num_out_capsules,H,W,1)
                a_ij = (predictions * v_j).sum(dim=-1, keepdim=True)               
                b_ij = b_ij + a_ij
        c_ij = c_ij.squeeze(5).unsqueeze(3)
        c_ij = unif_act_wt_entropy(c_ij)
        v_j = v_j.squeeze(1)
                
        return v_j, c_ij
  
class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.
       Code from https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
       Args:
            in_planes: the number of in channels
            planes: the number of out channels
            stride: the stride of the convolution
       Used for convolution layers
       Input: (batch_size,C1,H1,W1)
       Output: (batch_size,C2,H3,W2)
    '''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    """
       ResNet block before capsule code from https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
       Args:
            block: the residual block used
            num_blocks: the number of blocks in a layer
       Input:
             (batch_size,C,H,W)
             (batch_size,C'',H',W')     
    """
    def __init__(self, block, num_blocks):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

        
class PrimaryCapsules(nn.Module):
    """
    Primary Capsule layer is formed from convolution
    Args:
        in_channel: input channel
        num_capsules: number of types of capsules.
        capsule_dim: dimensionality of the capsule type   
    Input:(batch_size, in_channel, H, W)
    Output: (batch_size, num_capsules, H', W', capsule_dim)              
    """
    def __init__(self,in_channel, num_capsules, capsule_dim):
        super(PrimaryCapsules, self).__init__()
        self.in_channel = in_channel
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.projector = nn.Conv2d(in_channels=in_channel,out_channels=capsule_dim*num_capsules,
                                                kernel_size=1,stride=1) 
                                                 
    def squash(self, x, dim):
        norm_squared = (x ** 2).sum(dim, keepdim=True)
        part1 = norm_squared / (1 +  norm_squared)
        part2 = x / torch.sqrt(norm_squared+ 1e-16)
        output = part1 * part2 
        return output   
    
    def forward(self, x):
        x = self.projector(x)
        batch_size,_,H,W = x.size()
        output = x.view(batch_size,self.num_capsules,self.capsule_dim,H,W)
        output = output.permute(0,1,3,4,2).contiguous()
        output = self.squash(output,4)
        return output

class ResidualBlock(nn.Module):
      def __init__(self,num_in_capsules,num_out_capsules,in_dim,out_dim,stride=2):
          super(ResidualBlock,self).__init__()
          self.num_in_capsules = num_in_capsules
          self.num_out_capsules = num_out_capsules
          self.in_dim = in_dim
          self.out_dim = out_dim
          self.conv_capsule1 = ConvSov(num_in_capsules,num_out_capsules,3,in_dim,out_dim,stride)
          self.conv_capsule2 = ConvSov(num_out_capsules,num_out_capsules,1,out_dim,out_dim,1)
          self.conv_capsule3 = ConvSov(num_out_capsules,num_out_capsules,1,out_dim,out_dim,1)

      def squash(self, x, dim):
        norm_squared = (x ** 2).sum(dim, keepdim=True)
        part1 = norm_squared / (1 +  norm_squared)
        part2 = x / torch.sqrt(norm_squared+ 1e-16)
        output = part1 * part2 
        return output

      def forward(self,x):
          x, c_ij_one = self.conv_capsule1(x)
          capsule_two, c_ij_two = self.conv_capsule2(x)
          capsule_three, c_ij_three = self.conv_capsule3(capsule_two)
          output_capsule = capsule_three + x
          output_capsule = self.squash(output_capsule,-1)
          output_c_ij = c_ij_one + c_ij_two + c_ij_three
          return output_capsule, output_c_ij 

'''class Model(nn.Module):
      """
         Capsule network
         Args:
              in_channel: the number of input_channel
              im_size: the size of the image 
      """
      def __init__(self):
          super(Model,self).__init__()
          #(batch_size,1,32,32) -> (batch_size,256,30,30) 
          self.conv = nn.Sequential(nn.Conv2d(3,64,3),
                                    nn.ReLU()
                       )
          #(batch_size,32,30,30,16) 
          self.primary_capsule =  PrimaryCapsules(64,16,8)
          #(batch_size,32,14,14,16)
          self.capsule_layer_one = ResidualBlock(16,16,8,8)#ConvSov(16,16,3,8,8,2)
          #(batch_size,16,7,7,16)
          self.capsule_layer_two = ResidualBlock(16,16,8,8)#ConvSov(16,16,2,8,16,2)
          #(batch_size,16,5,5,16)
          self.capsule_layer_three = ConvSov(16,16,3,16,16,1)
          #(batch_size,16,3,3,16)
          self.capsule_layer_four = ConvSov(16,16,3,16,16,1)
          #(batch_size,10,1,1,16)
          self.class_capsule = ConvSov(16,10,3,16,16,1)
          #(batch_size,in_channel,im_size,im_size)
          self.linear = nn.Linear(16,1)            
       
      def forward(self,input):
          conv = self.conv(input)
          primary_capsule = self.primary_capsule(conv)
          capsule_one, capsule_one_c_ij = self.capsule_layer_one(primary_capsule)
          capsule_two, capsule_two_c_ij = self.capsule_layer_two(capsule_one)
          capsule_three, capsule_three_c_ij = self.capsule_layer_three(capsule_two)
          capsule_four, capsule_four_c_ij = self.capsule_layer_four(capsule_three)
          class_capsule, class_capsule_c_ij = self.class_capsule(capsule_four)
          class_capsule = class_capsule.squeeze(2).squeeze(2)
          class_predictions = self.linear(class_capsule).squeeze(2)
          return class_predictions, capsule_one_c_ij+capsule_two_c_ij+capsule_three_c_ij+capsule_four_c_ij+class_capsule_c_ij'''

class Model(nn.Module):
      """
         Capsule network
         Args:
              in_channel: the number of input_channel
              im_size: the size of the image 
      """
      def __init__(self):
          super(Model,self).__init__()
          #(batch_size,1,32,32) -> (batch_size,64,16,16) 
          self.conv = nn.Sequential(nn.Conv2d(1,64,3,2,1)
                       )
          #(batch_size,32,16,16,16) 
          self.primary_capsule =  PrimaryCapsules(64,32,16)
          #(batch_size,16,7,7,16)
          self.capsule_layer_one = ResidualBlock(32,16,16,16)
          #(batch_size,16,3,3,16)
          self.capsule_layer_two = ResidualBlock(32,32,16,16)
          #(batch_size,5,1,1,16)
          self.class_capsule = ConvSov(16,5,3,16,16,1)
          #(batch_size,in_channel,im_size,im_size)
          self.linear = nn.Linear(16,1)            
       
      def forward(self,input):
          conv = self.conv(input)
          primary_capsule = self.primary_capsule(conv)
          capsule_one, capsule_one_c_ij = self.capsule_layer_one(primary_capsule)
          capsule_two, capsule_two_c_ij = self.capsule_layer_two(capsule_one)
          class_capsule, class_capsule_c_ij = self.class_capsule(capsule_two)
          class_capsule = class_capsule.squeeze(2).squeeze(2)
          class_predictions = self.linear(class_capsule).squeeze(2)
          return class_predictions, capsule_one_c_ij+capsule_two_c_ij+class_capsule_c_ij
