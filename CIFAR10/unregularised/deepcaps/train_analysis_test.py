import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
from model import *
from constants import *
from data_loaders import *

def train(trainloader, testloader, model, num_epochs, lr, batch_size, lamda, m_plus, m_minus, hard, trial):
    optimizer = optim.Adam(model.parameters(),lr=lr)
    lr_lambda = lambda epoch: 0.5**(epoch // 10)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_lambda)
    global best_accuracy
    for epoch in range(num_epochs):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0.0
        correct = 0.0
        total = 0.0
        first_term = 0.0
        impurity_reg = 0.0
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            targets = one_hot(labels)  
            optimizer.zero_grad()
            outputs, pred, masked, reconstructed, indices, cij_reg = model(inputs,targets)
            loss = loss_(outputs, reconstructed, inputs, targets, lamda, m_plus, m_minus)
            loss.backward()
            optimizer.step()
            train_loss += float(loss)
            first_term += float(loss_(outputs, reconstructed, inputs, targets, lamda, m_plus, m_minus))
            impurity_reg += float(cij_reg.sum())
            total += targets.size(0)
            correct += pred.eq(labels).sum().item()
            torch.cuda.empty_cache()
            if batch_idx%150 == 0:
               progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                       % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            #save checkpoint (not for restarting training. Only for analysis.
        if hard == True:
            state = {
                     'model': model.state_dict(),
                     'loss': train_loss/(batch_idx+1),
                     'loss_caps': first_term/(batch_idx+1),
                     'impurity': impurity_reg/(batch_idx),
                     'acc': correct/total,
                     'epoch': epoch + num_epochs
                    }
            torch.save(state,'./checkpoints/epoch_'+str(epoch+num_epochs)+'_trial_'+str(trial)+'.pth') 
        else:
             state = {
                     'model': model.state_dict(),
                     'loss': train_loss/(batch_idx+1),
                     'loss_caps': first_term/(batch_idx+1),
                     'impurity': impurity_reg/(batch_idx),
                     'acc': correct/total,
                     'epoch': epoch
                    } 
             torch.save(state,'./checkpoints/epoch_'+str(epoch)+'_trial_'+str(trial)+'.pth')
        with torch.no_grad():
             test_accuracy, test_loss = test(testloader, model, lamda, m_plus, m_minus, trial)
             if test_accuracy >= best_accuracy:
                print('saving')
                if hard == True:
                    state = {
                         'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'loss': test_loss,  
                         'acc': test_accuracy,
                         'epoch': epoch + num_epochs,
                        }
                else:
                     state = {
                         'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'loss': test_loss,  
                         'acc': test_accuracy,
                         'epoch': epoch,
                        } 
                torch.save(state, './checkpoints/trial_'+str(trial)+'_best_accuracy.pth')
                best_accuracy = test_accuracy
        scheduler.step()

def test(testloader, model, lamda, m_plus, m_minus, trial):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0.0
    for batch_idx, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        targets = one_hot(labels)  
        outputs, pred, masked, reconstructed, indices, cij_reg = model(inputs)
        loss = loss_(outputs, reconstructed, inputs, targets, lamda, m_plus, m_minus)
        test_loss += float(loss)
        total += targets.size(0)
        correct += pred.eq(labels).sum().item()
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    test_loss= test_loss/(batch_idx+1)   
    # Save checkpoint.
    test_accuracy = float(correct)/total
    return test_accuracy, test_loss
        

def get_mean_variance(batch_size,entropies,old_mean_entropies,old_var_entropies):
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

def margin_loss(x, labels, lamda, m_plus, m_minus):
    v_c = torch.norm(x, dim=2, keepdim=True)
    tmp1 = func.relu(m_plus - v_c).view(x.shape[0], -1) ** 2
    tmp2 = func.relu(v_c - m_minus).view(x.shape[0], -1) ** 2
    loss = labels*tmp1 + lamda*(1-labels)*tmp2
    loss = loss.sum(dim=1)
    return loss
    
def reconst_loss(recnstrcted, data):
    mse_loss = nn.MSELoss(reduction="none")
    loss = mse_loss(recnstrcted.view(recnstrcted.shape[0], -1), data.view(recnstrcted.shape[0], -1))
    return 0.01 * loss.sum(dim=1)
    
def loss_(x, recnstrcted, data, labels, lamda=0.5, m_plus=0.9, m_minus=0.1):
    m = margin_loss(x, labels, lamda, m_plus, m_minus) 
    r = reconst_loss(recnstrcted, data)
    loss = m + r
    return loss.mean()

def one_hot(tensor, num_classes=5):
    return torch.eye(num_classes).cuda().index_select(dim=0, index=tensor.cuda())

for trial in range(NUMBER_OF_TRIALS):
    trainloader, testloader = get_data_loaders()
    num_epochs = 50
    best_accuracy = 0
    #model = nn.DataParallel(Model()).to(DEVICE)
    model = Model().to(DEVICE)
    train(trainloader, testloader, model, num_epochs=num_epochs, lr=LR, batch_size=BATCH_SIZE, lamda=0.5, m_plus=0.9,  m_minus=0.1, hard=False, trial=trial)
    print("\n\n\n\nHard-Training\n")
    train(trainloader, testloader, model, num_epochs=num_epochs, lr=2e-4, batch_size=BATCH_SIZE, lamda=0.8, m_plus=0.95,  m_minus=0.05, hard=True, trial=trial)
