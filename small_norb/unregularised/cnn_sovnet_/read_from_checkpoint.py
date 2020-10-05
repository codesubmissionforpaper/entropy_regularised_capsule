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
from data_loaders import *
from utils import *

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
    model = nn.DataParallel(ResnetCnnsovnetDynamicRouting()).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    entropies.append(test(test_loader,model))
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_entropy = np.mean(entropies)
std_entropy = np.std(entropies)
print(mean_entropy,std_entropy)
print(mean_accuracy,std_accuracy)
