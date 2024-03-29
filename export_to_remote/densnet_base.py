import math
import pickle
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import numpy as np 
import torchvision.transforms as transforms
from torchvision import models
import torch 
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune


# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'

## Normalization adapted for CIFAR
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network. 
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,])

### The data from CIFAR100 will be downloaded in the following folder
rootdir = '../data/cifar10'

# adapt the set for test
c100train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c100test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

trainloader = DataLoader(c100train,batch_size=32,shuffle=True)
testloader = DataLoader(c100test,batch_size=32) 

print(f"CIFAR100 training dataset has {len(c100train)} samples")
print(f"CIFAR100 testing dataset has {len(c100test)} samples")


## number of target samples for the final dataset
num_train_examples = len(c100train)
num_samples_subset = 15000

## We set a seed manually so as to reproduce the results easily
seed  = 2147483647
torch.manual_seed(seed)

## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

## We define the Subset using the generated indices 
#c100train_subset = torch.utils.data.Subset(c100train,indices[:num_samples_subset])
#print(f"Subset of CIFAR100 dataset has {len(c100train_subset)} samples")


# Séparer les données d'entraînement en ensembles d'entraînement et de validation
train_ratio = 0.9
train_size = int(train_ratio * len(c100train))
val_size = len(c100train) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(c100train, [train_size, val_size])

trainloader_subset = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# ------------------------------------
# Model definition

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=100):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12, num_classes=10)

model = densenet_cifar().to(device)

# ------------------------------------

## Hyperparameters
num_classes = 10
num_epochs = 30
batch_size = 32
learning_rate = 0.001

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

# To plot the accruacy
epoch_list = list(range(num_epochs))

running_loss = 0
train_losses = []
val_losses = []

correct = 0
total = 0
train_acc = []
val_acc = []

# Train the model
total_step_train = len(trainloader)
total_step_val = len(testloader)

for epoch in range(num_epochs):

    model.train()
    running_loss = 0
    correct = 0
    total = 0
    # pred = torch.empty(0).to(device)
    # target_values = torch.empty(0).to(device)
    for i,(images, labels) in enumerate(trainloader):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        
        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)
        running_loss += loss.item() # pour calculer sur une moyenne d'epoch

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).float().sum().item()

        print("batch number : ",i)

    # del images, labels, outputs
    # torch.cuda.empty_cache()

    train_losses.append(running_loss / total_step_train)
    train_acc.append(100*correct/total_step_train)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation loss: {val_losses[-1]:.4f}')


    model.eval()
    with torch.no_grad():
        running_loss = 0
        correct = 0
        total = 0
        # pred = torch.empty(0).to(device)
        # target_values = torch.empty(0).to(device)
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).float().sum().item()

            loss = criterion(outputs, labels)
            running_loss += loss.item() # Ajout de la perte de validation
            
            #pred = torch.cat((pred,predicted),0)
            #target_values = torch.cat((target_values,labels),0)

        val_losses.append(running_loss / total)
        val_acc.append(100*correct/total)

        # del images, labels, outputs
        # torch.cuda.empty_cache()

    print(f'Epoch [{epoch+1}/{num_epochs}], Train accuracy: {train_acc[-1]:.4f}, Validation accuracy: {val_acc[-1]:.4f}')

# ------------------------------------------
# save the model and weights

torch.save(model.state_dict(), './models/densenet_base_cifar10.pth')

# ------------------------------------------
# write in the file the lists

my_file = "densenet_base_cifar10.txt"

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
plt.savefig("densenet_base_cifar10.png")
plt.show()