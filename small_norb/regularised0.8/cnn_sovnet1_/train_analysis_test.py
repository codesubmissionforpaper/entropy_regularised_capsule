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
from utils import *

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from constants import * 
from smallNorb import SmallNORB

def build_dataloaders(batch_size, valid_size, train_dataset, valid_dataset, test_dataset):
  # Compute validation split
  train_size = len(train_dataset)
  indices = list(range(train_size))
  split = int(np.floor(valid_size * train_size))
  np.random.shuffle(indices)
  train_idx = indices[split:]
  train_sampler = SubsetRandomSampler(train_idx)
  #valid_sampler = SubsetRandomSampler(valid_idx)
  
  # Create dataloaders
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             sampler=train_sampler)
  #valid_loader = torch.utils.data.DataLoader(valid_dataset,
  #                                           batch_size=batch_size,
  #                                           sampler=valid_sampler)
  test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
  return train_loader, test_loader


def load_small_norb(batch_size):
    path = SMALL_NORB_PATH
    train_transform = transforms.Compose([
                          transforms.Resize(48),
                          transforms.RandomCrop(32),
                          transforms.ColorJitter(brightness=32./255, contrast=0.5),
                          transforms.ToTensor(),
                          transforms.Normalize((0.0,), (0.3081,))
                      ])
    valid_transform = transforms.Compose([
                          transforms.Resize(48),
                          transforms.CenterCrop(32),
                          transforms.ToTensor(),
                          transforms.Normalize((0.,), (0.3081,))
                      ])
    test_transform = transforms.Compose([
                          transforms.Resize(48),
                          transforms.CenterCrop(32),
                          transforms.ToTensor(),
                          transforms.Normalize((0.,), (0.3081,))
                      ])
    
    train_dataset = SmallNORB(path, train=True, download=True, transform=train_transform)
    valid_dataset = SmallNORB(path, train=True, download=True, transform=valid_transform)
    test_dataset = SmallNORB(path, train=False, transform=test_transform)
    valid_size = 0 #DEFAULT_VALIDATION_SIZE 
    return build_dataloaders(batch_size, valid_size, train_dataset, valid_dataset, test_dataset)

def train(epoch,trainloader,trial):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0.0
    second_term = 0.0
    correct = 0.0
    total = 0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs, entropy = model(inputs)
        loss = 0.2*loss_criterion(outputs, targets) + 0.8*entropy 
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        second_term += entropy.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        #if batch_idx%100 == 0:
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                       % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        #print(train_loss/(batch_idx+1),second_term/(batch_idx+1))
    scheduler.step()
    with torch.no_grad():
         #save checkpoint (not for restarting training. Only for analysis.
         state = {
                  'model': model.state_dict(),
                  'loss': train_loss/(batch_idx+1),
                  'acc': correct/total,
                  'entropy': second_term/(batch_idx+1),
                  'epoch': epoch
                  }
         torch.save(state,'./checkpoints/epoch_'+str(epoch)+'_trial_'+str(trial)+'.pth')


def test(epoch,testloader,trial):
    global best_accuracy
    model.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, entropy = model(inputs)
            loss = 0.2*loss_criterion(outputs, targets) + 0.2*entropy

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*float(correct)/total
    print('test accuracy: ',acc)
    if acc > best_accuracy:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': test_loss/(batch_idx+1),  
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, './checkpoints/trial_'+str(trial)+'_best_accuracy.pth')
        best_accuracy = acc

def get_mean_variance(batch_size,entropies,old_mean_entropies,old_var_entropies,):
    mean_entropies = []
    var_entropies = []
    new_batch_size = entropies[1].size(0)
    for entropy, old_mean_entropy, old_var_entropy in zip(entropies,old_mean):
        new_mean_entropy = torch.mean(entropy,dim=0)
        mean_entropy = (batch_size*old_mean_entropy+new_batch_size*new_mean_entropy)/(batch_size+new_batch_size)
        new_var_entropy = torch.var(entropy,dim=0,unbiased=False)
        var_entropy = (batch_size*old_var_entropy+new_batch_size*new_var_entropy)/(batch_size+new_batch_size)
        var_entropy += (batch_size*new_batch_size)*((old_mean_entropy-new_mean_entropy)/(batch_size_new_batch_size))**2
        mean_entropies.append(mean_entropy)
        var_entropies.append(var_entropy)
    return mean_entropies, var_entropies

def analysis(path,loader,trial):
    model = nn.DataParallel(ResnetCnnsovnetDynamicRouting(analysis=True)).to(DEVICE)
    model.load_state_dict(path)
    total = 0.0
    model.eval()
    with torch.no_grad():
         for batch_idx, (data,label) in enumerate(loader):
             data, label = data.to(DEVICE), label.to(DEVICE)
             _, entropies = model(data)
             if batch_idx == 0:
                mean_entropies = []
                var_entropies = []
                for entropy in entropies:
                    mean_entropies.append(torch.mean(entropy,dim=0))
                    var_entropies.append(torch.var(entropy,dim=0,unbiased=False))
             else:
                  mean_entropies, var_entropies = get_mean_variance(total,entropies,mean_entropies,var_entropies)
             total += label.size(0)
    return mean_entropies, var_entropies 

for trial in range(NUMBER_OF_TRIALS):
    trainloader, testloader = load_small_norb(BATCH_SIZE)
    best_accuracy = 0.0
    num_epochs = 300
    loss_criterion = CrossEntropyLoss()
    model = nn.DataParallel(ResnetCnnsovnetDynamicRouting()).to(DEVICE)
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    #lr_lambda = lambda epoch: 0.95**(epoch)
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 250], gamma=0.1)
    for epoch in range(num_epochs):
        train(epoch,trainloader,trial)
        test(epoch,testloader,trial)
