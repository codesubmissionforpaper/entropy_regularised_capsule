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

def get_data_loaders():
    transform_train = transforms.Compose([
                      #transforms.RandomCrop(32, padding=4),
                      #transforms.RandomHorizontalFlip(),
                      transforms.RandomAffine(0,(0.1,0.1)),
                      #transforms.RandomHorizontalFlip(),
                      #transforms.ColorJitter(brightness=32./255, contrast=0.5),
                      transforms.ToTensor(),
                      transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
                      ])

    transform_test = transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
                     ])

    trainset = torchvision.datasets.SVHN(root='../../data', split='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.SVHN(root='../../data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader

_, testloader = get_data_loaders()
test_entropy = 0.0
model = nn.DataParallel(ResnetCnnsovnetDynamicRouting()).to(DEVICE)
checkpoint = torch.load('checkpoints/trial_0_best_accuracy.pth')
model.load_state_dict(checkpoint['model'])
model.eval()
with torch.no_grad():
     for batch_idx, (inputs, targets) in enumerate(testloader):
         inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
         outputs, entropy = model(inputs)
         test_entropy = test_entropy + entropy.item()
         _, predicted = outputs.max(1)
print(test_entropy/(batch_idx+1))

