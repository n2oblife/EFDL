from torchvision.transforms.transforms import ToPILImage
from minicifar import c10train,c10test
import minicifar
from torch.utils.data.dataloader import DataLoader
import torch
from torch.autograd import Variable
import torch.nn as nn
#Pruning
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import sys

#Quantization
#import binaryconnect
import torch.quantization
import copy
#import models_cifar100
import time

#trying data augmentation 
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler


#Sets the seed for generating random numbers.
torch.manual_seed(0)


transform_train = transforms.Compose([ 
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1),                               
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


train_size = 0.8

rootdir = './data/cifar10'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)


def train_validation_split(train_size, num_train_examples):
    # obtain training indices that will be used for validation
    indices = list(range(num_train_examples))
    np.random.shuffle(indices)
    idx_split = int(np.floor(train_size * num_train_examples))
    train_index, valid_index = indices[:idx_split], indices[idx_split:]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    return train_sampler,valid_sampler


num_train_examples=len(c10train)
train_sampler,valid_sampler=train_validation_split(train_size, num_train_examples)


#trainloader = DataLoader(c10train,batch_size=4,shuffle=False) ### Shuffle to False so that we always see the same images
trainloader = DataLoader(c10train,batch_size=128,sampler=train_sampler)
validloader = DataLoader(c10train,batch_size=256,sampler=valid_sampler)
testloader = DataLoader(c10test,batch_size=256)


"""
def onehot(label, n_classes):
    return torch.zeros(label.size(0), n_classes).scatter_(
        1, label.view(-1, 1), 1)


def mixup(data, targets, alpha, n_classes):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    targets = onehot(targets, n_classes)
    targets2 = onehot(targets2, n_classes)

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets
"""

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,fmaps_repeat=64):
        super(ResNet, self).__init__()
        self.in_planes = fmaps_repeat
        self.fmaps_repeat = fmaps_repeat

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.fmaps_repeat, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*self.fmaps_repeat, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*self.fmaps_repeat, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*self.fmaps_repeat, num_blocks[3], stride=2)
        self.linear = nn.Linear((8*self.fmaps_repeat)*block.expansion, num_classes)
        #Reducing the neurones to avoid the overfitting
        #self.dropout = nn.Dropout(0.2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        #out = self.dropout(self.linear(out))
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


#Use the GPU if there is one available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device '+str(device))

net = ResNet18()
net.to(device=device)
""""
mymodelbc = binaryconnect.BC(net) ### use this to prepare your model for binarization 

mymodelbc = mymodelbc.to(device) # it has to be set for GPU training
"""
#print(net)
# DEFINE THE CRITERION
criterion =    nn.CrossEntropyLoss() 
# DEFINE THE OPTIMIZER
optimizer = torch.optim.SGD(net.parameters(),lr = 0.01, momentum=0.9, weight_decay=5e-4)
#optimizer = torch.optim.SGD(net.parameters(),lr = 0.01, momentum=0.9, weight_decay=1e-4) # The results are most adapted to the real life
#optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)#optimizer = torch.optim.AdamW(net.parameters(), lr=0.05, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
#optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
# typical strategy is to divide it by 10 when reaching a plateau in performance
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


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


#TRAINING PART
n_epochs = 150  # number of epochs to train the model

#def training(n_epochs, train_loader, valid_loader, model, criterion, optimizer, scheduler, mymodelbc):

def training(n_epochs, train_loader, valid_loader, model, criterion, optimizer, scheduler):

  epoch_ef = 0 # effective number of epochs
  train_losses, valid_losses = [], []
  train_acc, valid_acc = [], []  
  # initialize tracker for minimum validation loss
  valid_loss_min = np.Inf  # set initial "min" to infinity

  #patience = 30  #Patience to avoid overfitting

  for epoch in range(n_epochs):
      train_loss, valid_loss = 0, 0 # monitor losses
      
      #Calculate the accuracy per epoch
      train_correct = 0
      valid_correct = 0
      train_total = 0
      valid_total = 0

      # train the model
      model.train() # prep model for training
      epoch_ef += 1
      for data, label in train_loader:
          data = data.to(device=device, dtype=torch.float32)
          label = label.to(device=device, dtype=torch.long)
          
          data, label_a, label_b, lam = mixup_data(data, label, 0.2, 10)

          data, label_a, label_b = map(Variable, (data, label_a, label_b))
            
          # This binarizes all weights in the model
          #model.binarization()
          output = model(data) # forward pass: compute predicted outputs by passing inputs to the model
          loss = mixup_criterion(criterion, output, label_a, label_b, lam) # calculate the loss
          
          _, predicted = torch.max(output.data, 1)
          #_, label = label.max(dim=1)
          train_total += label.size(0)
          train_correct += (lam * predicted.eq(label_a.data).cpu().sum().item()
                    + (1 - lam) * predicted.eq(label_b.data).cpu().sum().item())
          
          #train_correct += (predicted == label).sum().item()

          optimizer.zero_grad() # clear the gradients of all optimized variables
          loss.backward() # backward pass: compute gradient of the loss with respect to model parameters
        
          #  This reloads the full precision weights
          #model.restore()
        
          optimizer.step() # perform a single optimization step (parameter update) 
          
          # Clip the weights 
          #model.clip()
           
          train_loss += loss.item() * data.size(0) # update running training loss
      
      # validate the model
      model.eval()
      
      #model.binarization()
      
      for data, label in valid_loader:
          data = data.to(device=device, dtype=torch.float32)
          label = label.to(device=device, dtype=torch.long)
          with torch.no_grad():
            output = model(data)

            _, predicted = torch.max(output.data, 1)
            valid_total += label.size(0)
            valid_correct += (predicted == label).sum().item()

          loss = criterion(output,label)
          valid_loss += loss.item() * data.size(0)
      scheduler.step()
      
      #model.restore() 
      
      # calculate average loss over an epoch
      train_loss /= len(train_loader.sampler)
      valid_loss /= len(valid_loader.sampler)
      train_losses.append(train_loss)
      valid_losses.append(valid_loss)
      train_acc.append(100 * train_correct // train_total)
      valid_acc.append(100 * valid_correct // valid_total)
      
      print('epoch: {} \ttraining Loss: {:.6f} \tvalidation Loss: {:.6f}'.format(epoch+1, train_loss, valid_loss))
      print('The training accuracy is :', 100 * train_correct // train_total)
      print('The validation accuracy is :', 100 * valid_correct // valid_total)
      # save model if validation loss has decreased
      if valid_loss <= valid_loss_min:
          #patience = 30  #Patience to avoid overfitting
          print('validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
          valid_loss_min,
          valid_loss))
          torch.save(model.state_dict(), 'mybestmodel4_DA_Mixup_SGD.pth')
          valid_loss_min = valid_loss
      """
      else:
          patience -= 1

      if patience == 0:
          print("No improvement of the validation loss")
          break 
      """     
  return train_losses, valid_losses, train_acc, valid_acc, epoch_ef    

#Time 1

t1=time.time()

# Finally we can load the state_dict in order to load the trained parameters 
#net.load_state_dict(torch.load('/homes/k20amani/ai-optim/mybestmodel_DA_Mixup_SGD_94_Accuracy.pth'))

  # RUN THE TRAINING FUNCTION
train_losses_1, valid_losses_1, train_acc_1, valid_acc_1, epoch = training(n_epochs, trainloader, validloader, net, criterion, optimizer, scheduler)

#time 2

t2 = time.time()

t = t2 - t1

#plotting the evolution of the loss function for training and validation sets with respects to epochs
plt.plot(range(n_epochs), train_losses_1)
plt.plot(range(n_epochs), valid_losses_1)
plt.legend(['train', 'validation'], prop={'size': 10})
plt.title('loss function', size=10)
plt.xlabel('epoch', size=10)
plt.ylabel('loss value', size=10)
plt.show()
plt.savefig("loss_DA_Mixup_resnet18_150e_SGD.png")

plt.clf()

#Accuracy plotting

plt.plot(range(epoch), train_acc_1)
plt.plot(range(epoch), valid_acc_1)
plt.legend(['train_acc', 'validation_acc'], prop={'size': 10})
plt.title('Accuracy function', size=10)
plt.xlabel('Epochs', size=10)
plt.ylabel('Accuracy value', size=10)
plt.show()
plt.savefig("acc_DA_Mixup_resnet18_150e_SGD.png")


####################### Save the weights for the best model accuracy (threshold) ######################

state = {
            'net': net.state_dict(),
            'epoch': epoch
    }

#torch.save(net.state_dict(), 'stateDict4q_AdamW.pth')
#torch.save(state, 'mybestmodel4_Mixup_AdamW.pth')

net.load_state_dict(torch.load('mybestmodel4_DA_Mixup_SGD.pth', map_location=device))

# We load the dictionnary
#loaded_cpt = torch.load('mybestmodel4_Mixup_AdamW.pth')

# Fetch the hyperparam value
#hparam_bestvalue = loaded_cpt['epoch']

# Finally we can load the state_dict in order to load the trained parameters 
#net.load_state_dict(loaded_cpt['net'])


##TESTING
def evaluation(model, test_loader, criterion):

  # initialize lists to monitor test loss and accuracy
  test_loss = 0.0
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))

  model.eval() # prep model for evaluation
  for data, label in test_loader:
      data = data.to(device=device, dtype=torch.float32)
      label = label.to(device=device, dtype=torch.long)
      with torch.no_grad():
          output = model(data) # forward pass: compute predicted outputs by passing inputs to the model
      loss = criterion(output, label)
      test_loss += loss.item()*data.size(0)
      _, pred = torch.max(output,1) # convert output probabilities to predicted class
      correct = np.squeeze(pred.eq(label.data.view_as(pred))) # compare predictions to true label
      # calculate test accuracy for each object class
      for i in range(len(label)):
          digit = label.data[i]
          class_correct[digit] += correct[i].item()
          class_total[digit] += 1

  # calculate and print avg test loss
  test_loss = test_loss/len(test_loader.sampler)
  print('test Loss: {:.6f}\n'.format(test_loss))
  for i in range(10):
      print('test accuracy of %1s: %2d%% (%2d/%2d)' % (str(i), 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
  print('\ntest accuracy (overall): %2.2f%% (%2d/%2d)' % (100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))

evaluation(net,testloader,criterion) 

print("The training time in second with DA & Mixup is : ", t)