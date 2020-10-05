import torch
from torch import nn
from torch.nn import functional as F
from utils import *
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4
from constants import *

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = P4ConvP4(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) # stride 1: 64 -> 64 | stride 2: 64 -> 32
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = P4ConvP4(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # 64 -> 64 ie same
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                P4ConvP4(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.selu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.selu(out)
        return out

class ResNetPreCapsule(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNetPreCapsule, self).__init__()
        self.in_planes = 16

        self.conv1 = P4ConvZ2(1, 16, kernel_size=3, stride=1, padding=1, bias=False)#(b_size,16,32,32)
        self.bn1 = nn.BatchNorm3d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)#(b_size,16,32,32)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)#(b_size,32,16,16)
        #self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)#(b_size,64,8,8)
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.selu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        return out

def convertToCaps(x):
    return x.unsqueeze(2)

class PrimaryCapsules(nn.Module):
    def __init__(self,in_channels,num_capsules,out_dim,H=16,W=16):
        super(PrimaryCapsules,self).__init__()
        self.in_channels = in_channels
        self.num_capsules = num_capsules
        self.out_dim = out_dim
        self.H = H
        self.W = W
        self.preds = nn.Sequential(P4ConvP4(in_channels,num_capsules*out_dim,kernel_size=1),
                                   nn.SELU(),
                                   nn.LayerNorm((num_capsules*out_dim,4,H,W)))

    def forward(self,x):
        # x : (b,64,4,16,16)
        primary_capsules = self.preds(x) #(b,16*8,4,16,16)
        primary_capsules = primary_capsules.view(-1,self.num_capsules,self.out_dim,4,self.H,self.W)
        return primary_capsules #(b,16,8,4,16,16)

class ConvCapsule(nn.Module):
    def __init__(self,in_caps,in_dim,out_caps,out_dim,kernel_size,stride,padding):
        super(ConvCapsule,self).__init__()
        self.in_caps = in_caps
        self.in_dim = in_dim
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.preds = nn.Sequential(P4ConvP4(in_dim,out_caps*out_dim,kernel_size=kernel_size,stride=stride,padding=padding),
                                   nn.BatchNorm3d(out_caps*out_dim),
                                   nn.SELU())
     
    def forward(self,in_capsules,ITER=3):
        # in_capsules : (b,16,8,4,16,16)
        batch_size, _, _, _, H, W = in_capsules.size()
        in_capsules = in_capsules.view(batch_size*self.in_caps,self.in_dim,4,H,W) #(b*16,8,4,16,16)
        predictions = self.preds(in_capsules) # (b,)
        _,_,_, H, W = predictions.size()
        predictions = predictions.view(batch_size, self.in_caps, self.out_caps*self.out_dim, 4, H, W)
        predictions = predictions.view(batch_size, self.in_caps, self.out_caps, self.out_dim, 4, H, W)
        out_capsules, cij_entr= self.dynamic_routing(predictions,ITER)
        return out_capsules, cij_entr

    def unif_act_wt_entropy(self, c_ij):
        N, I, J, _, _, H, W = c_ij.shape
        return (-1/(N*I*H*W*4)) * torch.sum(torch.sum(c_ij * (torch.log10(c_ij + EPS)/0.69897000433), dim=2))    

    def squash(self, inputs, dim):
        norm = torch.norm(inputs, p=2, dim=dim, keepdim=True)
        scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
        return scale * inputs

    def dynamic_routing(self,predictions,ITER=3):
        batch_size,_,_, _, _, H, W = predictions.size()
        b_ij = torch.zeros(batch_size,self.in_caps,self.out_caps,1,4,H,W).to(DEVICE)
        for it in range(ITER):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * predictions).sum(dim=1, keepdim=True)
            v_j = self.squash(inputs=s_j, dim=3)
            if it < ITER - 1: 
               delta = (predictions * v_j).sum(dim=3, keepdim=True)
               b_ij = b_ij + delta
        return v_j.squeeze(dim=1), self.unif_act_wt_entropy(c_ij)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.resnet_precaps = ResNetPreCapsule(BasicBlock,[2,2]) 
        self.primary_caps = PrimaryCapsules(32,16,8,16,16)#for cifar10, H,W = 16, 16. For MNIST etc. H,W = 14,14.
        self.conv_caps1 = ConvCapsule(in_caps=16,in_dim=8,out_caps=32,out_dim=16,kernel_size=3,stride=2,padding=0) # (7,7)
        self.conv_caps2 = ConvCapsule(in_caps=32,in_dim=16,out_caps=16,out_dim=16,kernel_size=3,stride=1,padding=0) # (5,5)
        self.class_caps = ConvCapsule(in_caps=16,in_dim=16,out_caps=10,out_dim=16,kernel_size=5,stride=2,padding=0) # (1,1)
        #self.conv_caps3 = ConvCapsule(in_caps=32,in_dim=16,out_caps=32,out_dim=16,kernel_size=3,stride=1,padding=0) # (3,3)
        #self.class_caps = ConvCapsule(in_caps=32,in_dim=16,out_caps=5,out_dim=16,kernel_size=3,stride=1,padding=0) # (1,1)
        self.linear = nn.Linear(16,1)
        

    def forward(self,x):
        #print(f"\n\nx.shape : {x.shape}")
        resnet_output = self.resnet_precaps(x)
        #print(f"resnet_output.shape : {resnet_output.shape}")
        primary_caps = self.primary_caps(resnet_output)
        #print(f"primary_caps.shape : {primary_caps.shape}")
        conv_caps1, cij_entr1 = self.conv_caps1(primary_caps)
        #print(f"conv_caps1.shape : {conv_caps1.shape}")
        conv_caps2, cij_entr2 = self.conv_caps2(conv_caps1)
        #print(f"conv_caps2.shape : {conv_caps2.shape}")
        class_caps, cij_entr3 = self.class_caps(conv_caps2)
        #print(f"class_caps.shape : {class_caps.shape}")
        class_caps = class_caps.squeeze().permute(0,1,3,2).contiguous()
        #print(f"class_caps.shape : {class_caps.shape}")
        class_predictions = self.linear(class_caps).squeeze()
        #print(f"class_predictions.shape : {class_predictions.shape}")
        class_predictions, _ = torch.max(class_predictions,2)
        #assert False
        #print(f'cij_entr1 : {cij_entr1} | cij_entr2 : {cij_entr2} | cij_entr3 : {cij_entr3} | cij_entr4 : {cij_entr4}')
        if(torch.isnan(cij_entr1) or torch.isnan(cij_entr2) or torch.isnan(cij_entr3)):
            print(f'cij_entr1 : {cij_entr1} | cij_entr2 : {cij_entr2} | cij_entr3 : {cij_entr3}')
            assert(False)
        torch.cuda.empty_cache()
        return class_predictions, (cij_entr1 + cij_entr2 + cij_entr3)

