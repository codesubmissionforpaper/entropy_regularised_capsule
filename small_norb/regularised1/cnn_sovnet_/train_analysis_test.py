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
from data_loaders import *
from utils import *

def train(epoch,trainloader,trial):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0.0
    correct = 0.0
    total = 0.0
    weight = max((-0.8/299)*(epoch+1) + (1+(0.8/299)),0.2)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs, entropy_sum = model(inputs)
        criterion = loss_criterion(outputs, targets)
        loss = weight*criterion + (1-weight)*entropy_sum.sum()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx%100 == 0:
           progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                       % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    scheduler.step()
    with torch.no_grad():
         #save checkpoint (not for restarting training. Only for analysis.
         state = {
                  'model': model.state_dict(),
                  'loss': train_loss/(batch_idx+1),
                  'acc': correct/total,
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
    weight = max((-0.8/299)*(epoch+1) + (1+(0.8/299)),0.2)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs, entropy_sum = model(inputs)
            criterion = loss_criterion(outputs, targets)

            test_loss += weight*criterion.item() + (1-weight)*entropy_sum.sum()
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
    num_epochs = 250
    loss_criterion = CrossEntropyLoss()
    model = nn.DataParallel(ResnetCnnsovnetDynamicRouting()).to(DEVICE)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
    for epoch in range(num_epochs):
        train(epoch,trainloader,trial)
        test(epoch,testloader,trial)
