#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:42:53 2019

@author: nibaran
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt  



transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
data_path_train= '/media/nibaran/STORAGE/SOHAM/PAP/Classification/train/'
train_dataset = torchvision.datasets.ImageFolder(root=data_path_train, transform=transform)

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=4,shuffle=True)

    
data_path_test='/media/nibaran/STORAGE/SOHAM/PAP/Classification/test'
test_dataset=torchvision.datasets.ImageFolder(
        root=data_path_test, transform=transform)
test_loader=torch.utils.data.DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False)
data_path_val='/media/nibaran/STORAGE/SOHAM/PAP/Classification/val'
val_dataset=torchvision.datasets.ImageFolder(
        root=data_path_val, transform=transform)
val_loader=torch.utils.data.DataLoader(
        val_dataset,batch_size=4,shuffle=True)

net=torchvision.models.resnet18(pretrained=False, progress=True)
net.fc = nn.Linear(512, 7)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
min_loss=0
a=1000
val_loss=[]
train_loss=[]
l=[]
for epoch in range(2):  # loop over the dataset multiple times
    l.append(epoch)
   
    running_loss_train = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss_train+= loss.item()
        if i %  len(train_loader) == 135:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss_train / len(train_loader)))
            train_loss.append(running_loss_train / len(train_loader))
            plt.plot(l,train_loss[epoch])
            
            running_loss_train = 0.0
            
    running_loss_val   =0.0     
    for i, data in enumerate(val_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss_val += loss.item()
        if i %  len(val_loader) == 48:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss_val / len(val_loader)))
            
            if(a>(running_loss_val / len(val_loader))):
                  PATH = '/media/nibaran/STORAGE/SOHAM/PAP/Classification/model.pt'
                  torch.save(net.state_dict(), PATH)
                  print("model saved")
                  a=running_loss_val / len(val_loader)
            val_loss.append(running_loss_val / len(val_loader))
            plt.plot(l,val_loss[epoch])       
            
            running_loss_val = 0.0
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('/media/nibaran/STORAGE/SOHAM/PAP/Classification/graph.png')
    
net=torchvision.models.resnet18(pretrained=False, progress=True)
net.fc = nn.Linear(512, 7)
net.load_state_dict(torch.load(PATH))

 

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on  test images: %d %%' % (
    100 * correct / total))



    
    