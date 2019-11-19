#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:07:26 2019

@author: nibaran
"""

from os import listdir
from os.path import isfile, join
import os
import shutil   
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


for i in os.listdir('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/New database pictures' ):
   
    for j in os.listdir('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/train/'+i ):
        shutil.copy2('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/train/'+i+'/'+j,'/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/New database pictures/'+i)
    for k in os.listdir('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/test/'+i ):
        shutil.copy2('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/test/'+i+'/'+k,'/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/New database pictures/'+i)          
    for m in os.listdir('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/val/'+i ):
        shutil.copy2('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/val/'+i+'/'+m,'/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/New database pictures/'+i)   









   
    

"""Creating directories for test train split"""
os.mkdir('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain')

for j in range(0,5):
    s=str(j+1)
    os.mkdir('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/'+'val_'+s)
    os.mkdir('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/'+'val_'+s+'/'+'test')
    os.mkdir('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/'+'val_'+s+'/'+'train')
    for k in os.listdir('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/train' ):
        os.mkdir('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/'+'val_'+s+'/'+'test'+'/'+k)
        os.mkdir('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/'+'val_'+s+'/'+'train'+'/'+k)
        
"""creating test train dataset"""


for i in os.listdir('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/New database pictures' ):
    l=len(os.listdir('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/New database pictures/'+i ))
    
    for m in range   (0,5):
        n=0
        for k in os.listdir('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/New database pictures/'+i ):
        
            if m==0:
                if n<(l/5):
                    shutil.copy2('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/New database pictures/'+i+'/'+k,'/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/val_1/test/'+i)
                else:
                     shutil.copy2('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/New database pictures/'+i+'/'+k,'/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/val_1/train/'+i)
                
            if m==1:  
                if n>=(l/5) and n<(2*l/5):
                    shutil.copy2('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/New database pictures/'+i+'/'+k,'/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/val_2/test/'+i)
                else:
                     shutil.copy2('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/New database pictures/'+i+'/'+k,'/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/val_2/train/'+i)
                
            if m==2:
                 if n>=(2*l/5) and n<(3*l/5):
                     shutil.copy2('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/New database pictures/'+i+'/'+k,'/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/val_3/test/'+i)
                 else:
                     shutil.copy2('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/New database pictures/'+i+'/'+k,'/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/val_3/train/'+i)
              
            if m==3:
                 if n>=(3*l/5) and n<(4*l/5):
                     shutil.copy2('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/New database pictures/'+i+'/'+k,'/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/val_4/test/'+i)
                 else:
                     shutil.copy2('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/New database pictures/'+i+'/'+k,'/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/val_4/train/'+i)
            
            if m==4:
                 if n>=(4*l/5):
                     shutil.copy2('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/New database pictures/'+i+'/'+k,'/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/val_5/test/'+i)
                 else:
                     shutil.copy2('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/New database pictures/'+i+'/'+k,'/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/val_5/train/'+i)
            n=n+1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)     
           
            
""" running the models"""
acc=0
accuracy=0
for i in range(0,5):
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    data_path_train= '/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/val'+'_'+str(i+1)+'/train'  
    train_dataset = torchvision.datasets.ImageFolder(root=data_path_train, transform=transform)

    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=4,shuffle=True)

  
    data_path_test= '/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/val'+'_'+str(i+1)+'/test'
    test_dataset=torchvision.datasets.ImageFolder(
        root=data_path_test, transform=transform)
    test_loader=torch.utils.data.DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False)
    net=torchvision.models.resnet34(pretrained=False, progress=True)
    net.fc = nn.Linear(512, 7)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    min_loss=0
    a=1000

    train_loss=[]
    l=[]
    for epoch in range(150): 
       # loop over the dataset multiple times
        l.append(epoch)
   
        running_loss_train = 0.0
        for i, data in enumerate(train_loader, 0):
          
        # get the inputs; data is a list of [inputs, labels]
        
        #inputs, labels = data
             inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
             optimizer.zero_grad()

        # forward + backward + optimize
             outputs = net(inputs)
             loss = criterion(outputs, labels)
             loss.backward()
             optimizer.step()

        # print statistics
             running_loss_train+= loss.item()
             if i %  len(train_loader) == 181: 
                 # print every 2000 mini-batches
                 print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss_train / len(train_loader)))
                 train_loss.append(running_loss_train / len(train_loader))
                 plt.plot(l,train_loss)
            
                 
                 if(a>(running_loss_train/ len(train_loader))):
                     
                     
                 
                     PATH = '/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/model.pt'
                     torch.save(net, PATH)
                     print("model saved")
                     a=running_loss_train / len(train_loader)

        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/graph.png')
      
    net=torch.load('/run/media/nibaran/STORAGE/SOHAM/PAP/Classification/testtrain/model.pt')
    net.to(device)
    correct = 0
    total = 0
      
    
    for data in test_loader:
           
         
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _,predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy=100 * (correct / total)
    acc=acc+accuracy

    print('Accuracy of the network on  test images: %d %%' % (
            100 * correct / total))

print('Average Accuracy of the network on  test images: %d %%' % (
           acc/5))
