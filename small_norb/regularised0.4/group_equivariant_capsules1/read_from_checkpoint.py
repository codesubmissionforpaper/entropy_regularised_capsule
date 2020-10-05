import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from constants import *
from smallNorb import *
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *

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

def test(testloader,model):
    global best_accuracy
    model.eval()
    e= 0.0
    total = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, entropy = model(inputs)
            total += outputs.size(0)
            e = e + entropy
        e = e/total
    return e.item()

num_trials = 3
_, test_loader = load_small_norb(BATCH_SIZE)
accuracies = []
entropies = []
for i in range(num_trials):
    checkpoint  = torch.load('checkpoints/trial'+'_'+str(i)+'_best'+'_accuracy.pth')
    accuracies.append(checkpoint['acc'])
    model = Model().to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    entropies.append(test(test_loader,model))
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_entropy = np.mean(entropies)
std_entropy = np.std(entropies)
print(mean_entropy,std_entropy)
print(mean_accuracy,std_accuracy)
