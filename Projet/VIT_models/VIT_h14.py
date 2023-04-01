# Importing file from ressources
import sys
sys.path.append('/users/local/ZacDL/EFDL/Projet/ressources/')

import torch 
import numpy as np

import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from torchvision.datasets import CIFAR10, CIFAR100
from torch.optim.lr_scheduler import CosineAnnealingLR

from EarlyStopper import EarlyStopper
from GaussianNoise import AddGaussianNoise
from InterruptHandler import keybInterrupt


try :
    # Device configuration
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda'

    ## We set a seed manually so as to reproduce the results easily
    seed  = 2147483647
    torch.manual_seed(seed)

    # ------------------------------------------

    ## Normalization adapted for CIFAR
    normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

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
        normalize_scratch])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize_scratch])


    ## Hyperparameters
    num_classes = 10
    num_epochs = 180
    batch_size = 64
    learning_rate = 0.001
    weight_decay = 10e-4
    momentum = 0.9

    ## Defining training and dataset
    training = 'finetunning'
    dataset = 'cifar10'

    ## Base directory from EFDL to EFDL_storage
    base_dir = '../EFDL_storage'

    # The data will be downloaded in the following folder
    rootdir = base_dir+'/data/'+dataset

    # adapt the set for test
    c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
    c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

    trainloader = DataLoader(c10train,batch_size=batch_size,shuffle=True)
    testloader = DataLoader(c10test,batch_size=batch_size) 

    ## number of target samples for the final dataset
    num_train_examples = len(c10train)
    num_samples_subset = 15000

    # Model definition
    model_name = 'vit_h_14'
    model = models.vit_h14(num_classes=10)
    state_dict = torch.hub.load_state_dict_from_url(
        url='https://dl.fbaipublicfiles.com/deit/vit_h14_384-d6e9f3c3.pth',
        map_location=torch.device('cpu')
        )
    model.load_state_dict(state_dict)
    model_dir = base_dir+'/models/'+model_name +'_'+ training +'_'+ dataset +'.pt'
   
    print('Beginning of training : '+model_name+' on '+dataset)

    # Loss and optimizer
    end_sched = max(int(4*num_epochs/5), 100)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD( model.parameters(),
                            lr=learning_rate,
                            weight_decay = weight_decay, # if no weight decay it means we are regularizing
                            momentum = momentum) 
    scheduler = CosineAnnealingLR(optimizer,
                                T_max = end_sched, # Maximum number of iterations.
                                eta_min = learning_rate/100) # Minimum learning rate. 
   
   
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

        running_loss = 0.
        correct = 0
        total = 0

        model.train()
        for i, (images, labels) in enumerate(trainloader) :  
            # Move tensors to the configured device
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() # pour calculer sur une moyenne d'epoch
            
            # Backward and optimize
            optimizer.zero_grad()
            #loss.backward()
            optimizer.step()

            # For accuracy (up : classic, down : mixup)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).float().sum().item()
            total += labels.size(0)

            print("\r"+"Batch training : ",i+1,"/",number_batch ,end="")

            torch.cuda.empty_cache()
        
        # del images, labels, outputs
        # torch.cuda.empty_cache()

        train_losses.append(running_loss / total)
        train_acc.append(100*correct/total)
        print('\n'+f'Train Loss : {train_losses[-1]:.4f} , Train accuracy : {train_acc[-1]:.4f}')


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

            model_dir_early = base_dir+'/models/'+ model_name +'_'+ training +'_'+ dataset +'_epoch'+str(epoch)+'_early.pt'
            model_state = {'model name': model_name,
                        'model': model,
                        'optimizer': optimizer,
                        'epoch': epoch,
                        'training': training,
                        'dataset': dataset,
                        'metrics' : {'train_loss' : train_losses,
                                     'val_loss' :val_losses,
                                     'train_acc' :train_acc,
                                     'val_acc' : val_acc}
                        }
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
            'metrics' : {'train_loss' : train_losses,
                                     'val_loss' :val_losses,
                                     'train_acc' :train_acc,
                                     'val_acc' : val_acc}
            }
    torch.save(model_state, model_dir)
    print("Modèle sauvegardé dans le chemin : ",model_dir)


except KeyboardInterrupt:
    print("\nKeyboard interrupt, we have saved the model and its metrics")

    model_state = {'model name': model_name,
        'model': model,
        'optimizer': optimizer,
        'epoch': num_epochs,
        'training': training,
        'dataset': dataset,
        'metrics' : {'train_loss' : train_losses,
                                     'val_loss' :val_losses,
                                     'train_acc' :train_acc,
                                     'val_acc' : val_acc}
        }
    torch.save(model_state, model_dir)
    print("Modèle sauvegardé dans le chemin : ",model_dir)

finally:
    print("end of training script of ",model_name)