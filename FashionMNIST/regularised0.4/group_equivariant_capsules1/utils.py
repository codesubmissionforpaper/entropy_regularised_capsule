import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as func
from constants import *

def one_hot(tensor, num_classes=5):
    return torch.eye(num_classes).cuda().index_select(dim=0, index=tensor.cuda()) # One-hot encode
#     return torch.eye(num_classes).index_select(dim=0, index=tensor).numpy() # One-hot encode

def reconst_loss(recnstrcted, data, loss=torch.nn.MSELoss()):
    #print(f"recnstrcted.shape : {recnstrcted.shape} | data.shape : {data.shape}")
    loss_val = loss(recnstrcted.view(recnstrcted.shape[0], -1), data.view(recnstrcted.shape[0], -1))
    #print(f"loss_val : {loss_val}")
    return loss_val

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


########################

def unif_act_wt_entropy(c_ij):
    #c_ij = c_ij + EPS
    if len(c_ij.shape) == 6:
       N, H, W, _, J, I = c_ij.shape
       '''
       if not torch.all(torch.eq(c_ij,c_ij)):
           print("SAI: ",c_ij)
           assert False
       '''
       entr = (-1/(N*I*H*W)) * torch.sum(torch.sum(c_ij * (torch.log10(c_ij + EPS)/1.505149978319906), dim=-2))
       if entr < 0:
           tmp = c_ij.permute(0,1,2,3,5,4).contiguous().view(-1,16)
           tmp1 = torch.sum(tmp,dim=-1).unsqueeze(-1)
           tmp = torch.cat(tmp,tmp1)
           np.savetxt('c_ij.csv',tmp.detach().cpu().numpy(),delimeter=',')

       '''
       if torch.isnan(entr):
           print("RAM: ", entr)
           assert False
       '''
       return entr
    else:
       N, I, J, _ = c_ij.shape
       '''if not torch.all(torch.eq(c_ij,c_ij)):
           print("SAI: ",c_ij)
           assert False'''
       entr = (-1/(N*I)) * torch.sum(torch.sum(c_ij * (torch.log10(c_ij + EPS)/0.6989700043360189), dim=-2))
       '''if torch.isnan(entr):
           print("RAM: ", entr)
           assert False'''
       return entr
       

def unif_act_wt_gini(c_ij):
    N, I, J, _, H, W = c_ij.shape
    return (1/(N*I*H*W)) * torch.sum(1. - torch.sum(c_ij*c_ij, dim=2))

def unif_act_wt_mcls(c_ij):
    N, I, J, _, H, W = c_ij.shape
    return (1/(N*I*H*W)) * torch.sum(1. - (torch.max(c_ij,dim=2))[0])

def get_entropies(c_ij):
    # c_ij: (I, J, 1)
    #print(f"entropy : c_ij.shape - {c_ij.shape}")
    I, J, _, H, W = c_ij.shape
    '''
    for i in range(H):
      for j in range(W):
        print("SAI :", torch.sum(c_ij[0,:,0,i,j]))
        t = c_ij[0,:,0,i,j]
        print("RAM: ", (-1)*torch.sum(t*(torch.log10(t)/0.6989700043360189)))
    I, J, _, H, W = c_ij.shape
    '''
    #t = -1 * torch.sum(c_ij * (torch.log10(c_ij)/0.6989700043360189), dim=1)
    #print(t[0])
    return (-1/(H*W)) * torch.sum(torch.sum(c_ij * (torch.log10(c_ij)/0.6989700043360189), dim=1),dim=(-1,-2,-3))

def get_ginis(c_ij):
    # c_ij: (I, J, 1)
    #print(f"gini : c_ij.shape - {c_ij.shape}")
    I, J, _, H, W = c_ij.shape
    return (1/(H*W)) * torch.sum(1. - torch.sum(c_ij*c_ij, dim=1),dim=(-1,-2,-3))

def get_mcLosses(c_ij):
    # c_ij: (I, J, 1)
    #print(f"mcls : c_ij.shape - {c_ij.shape}")
    I, J, _, H, W = c_ij.shape
    return (1/(H*W)) * torch.sum(1 - (torch.max(c_ij,dim=1))[0],dim=(-1,-2,-3))

def wted_act_wt_entropy(c_ij, v_i):
    N, I, J, _ = c_ij.shape
    v_i = torch.norm(v_i, dim=2).squeeze()
    return (-1/N)*torch.sum((1./torch.sum(v_i,dim=1))*torch.sum(v_i * torch.sum(c_ij*(torch.log10(c_ij)/torch.log10(5)),dim=2).squeeze(),dim=1))


def squash(s, dim=-1):
    norm = torch.norm(s, dim=dim, keepdim=True)
    return (norm /(1 + norm**2 + eps)) * s

def softmax_3d(x, dim):
  return (torch.exp(x) / torch.sum(torch.sum(torch.sum(torch.exp(x), dim=dim[0], keepdim=True), dim=dim[1], keepdim=True), dim=dim[2], keepdim=True))

def one_hot(tensor, num_classes=5):
    return torch.eye(num_classes).cuda().index_select(dim=0, index=tensor.cuda()) # One-hot encode
#     return torch.eye(num_classes).index_select(dim=0, index=tensor).numpy() # One-hot encode

def accuracy(indices, labels):
    correct = 0.0
    for i in range(indices.shape[0]):
        if float(indices[i]) == labels[i]:
            correct += 1
    return correct

