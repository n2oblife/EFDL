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

from torchvision import models
from efficientnet_pytorch import EfficientNet

from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
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
    transforms.RandomRotation(90),
    AddGaussianNoise(0., 0.01),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,])

##
# Parameters in original paper for optim
# batch = 64 / 90 epochs / Learning rate: 0.1, decreased by a factor of 10 at epochs 30 and 60 / Weight decay and momentum: 0.00004 and 0.9
# Utiliser le dropout pour éviter un gros overfitting et le mixup
##

## Hyperparameters
num_classes = 100
num_epochs = 90
batch_size = 64
learning_rate = 0.01
weight_decay = 0.00004
momentum = 0.9
end_sched = int(num_epochs/2)

### The data from CIFAR100 will be downloaded in the following folder
rootdir = '../data/cifar100'

# adapt the set for test
c100train = CIFAR100(rootdir,train=True,download=True,transform=transform_train)
c100test = CIFAR100(rootdir,train=False,download=True,transform=transform_test)

trainloader = DataLoader(c100train,batch_size=batch_size,shuffle=True)
testloader = DataLoader(c100test,batch_size=batch_size) 

## number of target samples for the final dataset
num_train_examples = len(c100train)
num_samples_subset = 15000

# Model definition
model = EfficientNet.from_name('efficientnet-b1', num_classes=num_classes).to(device)
model_dir = './models/efficientnet-b1_base_cifar100.pth'
model_dir_early = './models/efficientnet-b1_base_cifar100_early.pth'

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                            lr=learning_rate,
                            weight_decay = weight_decay,
                            momentum = momentum)  
scheduler = CosineAnnealingLR(optimizer,
                              T_max = end_sched, # Maximum number of iterations.
                              eta_min = learning_rate/1000) # Minimum learning rate.
# scheduler = CyclicLR(optimizer, 
#                      base_lr = learning_rate/10, # Initial learning rate which is the lower boundary in the cycle for each parameter group
#                      max_lr = learning_rate, # Upper learning rate boundaries in the cycle for each parameter group
#                      step_size_up = 5, # Number of training iterations in the increasing half of a cycle
#                      mode = "exp_range")

# Early stopping
patience = 2
delta_loss = 1
early_stopper = EarlyStopper(patience, delta_loss)

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

    model.train()
    running_loss = 0.
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(trainloader) :  
        # Move tensors to the configured device
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)
        running_loss += loss.item() # pour calculer sur une moyenne d'epoch

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # For accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).float().sum().item()

        print("\r"+"Batch training : ",i,"/",number_batch ,end="")

        torch.cuda.empty_cache()

    if early_stopper.early_stop(running_loss):
        torch.save(model.state_dict(), model_dir_early)
        print("Training stop early at epoch ",epoch,"/",num_epochs," with a loss of : ",running_loss)

    # del images, labels, outputs
    # torch.cuda.empty_cache()

    train_losses.append(running_loss / total_step_train)
    train_acc.append(100*correct/total_step_train)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss : {train_losses[-1]:.4f}, Train accuracy : {train_acc[-1]:.4f}')


    model.eval()
    with torch.no_grad():
        running_loss = 0.
        correct = 0
        total = 0
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

        val_losses.append(running_loss / total)
        val_acc.append(100*correct/total)

        # del images, labels, outputs
        # torch.cuda.empty_cache()

    print(f'Epoch [{epoch+1}/{num_epochs}], Train accuracy: {train_acc[-1]:.4f}, Validation accuracy: {val_acc[-1]:.4f}')

# ------------------------------------------
# save the model and weights

torch.save(model.state_dict(), model_dir)

# ------------------------------------------
# write in the file the lists

my_file = "./models/efficientnet-b1_base_cifar100.txt"
file_dir = "./models/efficientnet-b1_base_cifar100.png"

with open(my_file, 'wb') as f:
    pickle.dump(train_losses, f)
    pickle.dump(val_losses, f)
    pickle.dump(train_acc, f)
    pickle.dump(val_acc, f)

plt.subplot(121)
plt.plot(epoch_list, train_losses, label = "Training loss")
plt.plot(epoch_list, val_losses, label = "Validation loss")
plt.xlabel("Epochs")
plt.ylabel("%")
plt.title("Losses")

plt.subplot(122)
plt.plot(epoch_list, train_acc, label = "Training accuracy")
plt.plot(epoch_list, val_acc, label = "Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("%")
plt.title("Accuracy")
plt.legend()

plt.rcParams['figure.figsize'] = [10, 5] #size of plot
plt.suptitle("Effectivness of training")
plt.savefig(file_dir)
plt.show()
