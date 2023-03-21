import math
import torch 
import pickle
import numpy as np
import matplotlib.pyplot as plt
#import wandb # to log the params

import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.autograd import Variable

from torchvision import models
from efficientnet_pytorch import EfficientNet
from Densnet import densenet_cifar

from torchvision.datasets import CIFAR10,CIFAR100
from torch.utils.data.sampler import SubsetRandomSampler

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CyclicLR

import Binaryconnect as bc
from EarlyStopper import EarlyStopper

# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'

## We set a seed manually so as to reproduce the results easily
seed  = 2147483647
torch.manual_seed(seed)

# ------------------------------------------

## Normalization adapted for CIFAR
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transformation to add gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        tensor = self.to_tensor(image)
        tensor += torch.randn_like(tensor) * self.std + self.mean
        return transforms.ToPILImage()(tensor)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1),
    transforms.RandomRotation(45),
    AddGaussianNoise(0., 0.001),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,])


## Hyperparameters
num_classes = 100
num_epochs = 150
batch_size = 32
learning_rate = 0.001
weight_decay = 0.00004
momentum = 0.9
end_sched = int(3*num_epochs/4)

## Base directory from EFDL to EFDL_storage
base_dir = '../EFDL_storage'

# The data from CIFAR100 will be downloaded in the following folder
rootdir = base_dir+'/data/cifar100'

# adapt the set for test
c100train = CIFAR100(rootdir,train=True,download=True,transform=transform_train)
c100test = CIFAR100(rootdir,train=False,download=True,transform=transform_test)

trainloader = DataLoader(c100train,batch_size=batch_size,shuffle=True)
testloader = DataLoader(c100test,batch_size=batch_size) 

## number of target samples for the final dataset
num_train_examples = len(c100train)
num_samples_subset = 15000

# Model definition
model_name = 'densnet121'
training = 'base'
dataset = 'cifar100'
model = densenet_cifar(num_classes).to(device) #densnet121
#model = EfficientNet.from_name('efficientnet-b1', num_classes=num_classes).to(device)
model_dir = base_dir+'/models/'+model_name +'_'+ training +'_'+ dataset +'.pt'
print('Beginning of training : '+model_name)

#model_bc = bc.BC(model).to(device) ### use this to prepare your model for binarization 

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD( model.parameters(),
                        lr=learning_rate,
                        # weight_decay = weight_decay, # if no weight decay it means we are regularizing
                        momentum = momentum)  
scheduler = CosineAnnealingLR(optimizer,
                              T_max = num_epochs, # Maximum number of iterations.
                              eta_min = learning_rate/100) # Minimum learning rate.
# scheduler = CyclicLR(optimizer, 
#                      base_lr = learning_rate/10, # Initial learning rate which is the lower boundary in the cycle for each parameter group
#                      max_lr = learning_rate, # Upper learning rate boundaries in the cycle for each parameter group
#                      step_size_up = 5, # Number of training iterations in the increasing half of a cycle
#                      mode = "exp_range")
lets_regul = False
regu_incr = (weight_decay*100-weight_decay)/(num_epochs-end_sched)

# Mixups
def mixup_data(x, y, alpha=0.2, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Early stopping
patience = 3
delta_loss = 0.002
early_stopper = EarlyStopper(patience, delta_loss)
stopping_list = []

# To plot the accruacy
epoch_list = list(range(num_epochs))

running_loss = 0.
train_losses = []
val_losses = []

correct = 0
total = 0
train_acc = []
val_acc = []

# Train the model
total_step_train = len(trainloader)
total_step_val = len(testloader)
number_batch = len(trainloader)

for epoch in range(num_epochs):

    print(f'Epoch [{epoch+1}/{num_epochs}]')
    if epoch> end_sched:
        lets_regul=True
          
    running_loss = 0.
    correct = 0
    total = 0

    model.train()
    for i, (images, labels) in enumerate(trainloader) :  
        # Move tensors to the configured device
        images = images.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

        # mixup function
        images, label_a, label_b, lam = mixup_data(images, labels, 0.2, 10)
        images, label_a, label_b = map(Variable, (images, label_a, label_b))
            
        # This binarizes all weights in the model
        #model.binarization()

        # Forward pass
        outputs = model(images)

        # Compute loss
        #loss = criterion(outputs, labels)
        loss = mixup_criterion(criterion, outputs, label_a, label_b, lam)
        running_loss += loss.item() # pour calculer sur une moyenne d'epoch
        
        l2_regu = weight_decay * sum([(p**2).sum() for p in model.parameters()])
        loss_with_penalty = loss + l2_regu

        # Backward and optimize
        optimizer.zero_grad()
        #loss.backward()
        loss_with_penalty.backward() # for l2 regularization
        optimizer.step()

        # For accuracy (up : classic, down : mixup)
        _, predicted = torch.max(outputs.data, 1)
        #correct += (predicted == labels).float().sum().item()
        correct += (lam * predicted.eq(label_a.data).cpu().sum().item()
                    + (1 - lam) * predicted.eq(label_b.data).cpu().sum().item())
        total += labels.size(0)

        print("\r"+"Batch training : ",i+1,"/",number_batch ,end="")

        torch.cuda.empty_cache()
    
    # Schedulers
    scheduler.step() # updates the lr value
    if lets_regul:
        weight_decay += regu_incr # update the regularization value

    # del images, labels, outputs
    # torch.cuda.empty_cache()

    train_losses.append(running_loss / total)
    train_acc.append(100*correct/total)
    print('\r'+f'Train Loss : {train_losses[-1]:.4f} , Train accuracy : {train_acc[-1]:.4f}')


    with torch.no_grad():

        running_loss = 0.
        correct = 0
        total = 0

        model.eval()
        for i, (images, labels) in enumerate(testloader) :
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() # pour calculer sur une moyenne d'epoch

            # For accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).float().sum().item()
            total += labels.size(0)

        val_losses.append(running_loss / total)
        val_acc.append(100*correct/total)

        # del images, labels, outputs
        # torch.cuda.empty_cache()

    print(f'Validation loss: {val_losses[-1]:.4f} , Validation accuracy: {val_acc[-1]:.4f}')

    # Early stopping in case of overfitting
    if early_stopper.early_stop(running_loss):

        lets_regul = True
        regu_incr *= 1.01 # increase regularization

        model_dir_early = base_dir+'/models/'+ model_name +'_'+ training +'_'+ dataset +'_epoch'+str(epoch)+'_early.pt'
        model_state = {'model name': model_name,
                       'model': model,
                       'optimizer': optimizer,
                       'epoch': epoch,
                       'training': training,
                       'dataset': dataset,
                       'accuracy': 100*correct/total,
                       'loss': running_loss/total}
        torch.save(model_state, model_dir_early)
        print("\n"+"Training stop early at epoch ",epoch+1,"/",num_epochs," with a loss of : ",running_loss/total,", and accuracy of : ",100*correct/total)
        stopping_list.append(epoch+1)
    
    print('---------------------------------------')

if len(stopping_list) == 0:
    print("Pas d'overfiting !")
else :
    print("\n"+"Les epoch d'early stop sont : ",stopping_list)

# ------------------------------------------
# save the model and weights
model_state = {'model name': model_name,
        'model': model,
        'optimizer': optimizer,
        'epoch': num_epochs,
        'training': training,
        'dataset': dataset,
        'accuracy': 100*correct/total,
        'loss': running_loss/total}
torch.save(model_state, model_dir)
print("Modèle sauvegardé dans le chemin : ",model_dir)

# # ------------------------------------------
# # save the metrics

# my_file = base_dir+'/metrics/'+model_name+'_'+training+'_'+dataset+'.txt'
# file_dir = base_dir+'/metrics/'+model_name+'_'+training+'_'+dataset+'.png'

# with open(my_file, 'wb') as f:
#     pickle.dump(train_losses, f)
#     pickle.dump(val_losses, f)
#     pickle.dump(train_acc, f)
#     pickle.dump(val_acc, f)

# plt.subplot(121)
# plt.plot(epoch_list, train_losses, label = "Training loss")
# plt.plot(epoch_list, val_losses, label = "Validation loss")
# plt.xlabel("Epochs")
# plt.ylabel("%")
# plt.title("Losses")

# plt.subplot(122)
# plt.plot(epoch_list, train_acc, label = "Training accuracy")
# plt.plot(epoch_list, val_acc, label = "Validation accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("%")
# plt.title("Accuracy")
# plt.legend()

# plt.rcParams['figure.figsize'] = [10, 5] #size of plot
# plt.suptitle("Effectivness of training")
# plt.savefig(file_dir)
# plt.show()

# print("Metriques sauvegardées dans le chemin : ",file_dir)