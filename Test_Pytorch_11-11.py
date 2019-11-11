
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
data_path_train= '/media/nibaran/STORAGE/SOHAM/PAP/Classification/train/'
train_dataset = torchvision.datasets.ImageFolder(root=data_path_train, transform=transform)

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=4,shuffle=True)

    
data_path_test='/media/nibaran/STORAGE/SOHAM/PAP/Classification/test'
test_dataset=torchvision.datasets.ImageFolder(
        root=data_path_test, transform=transform)
test_loader=torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False)
data_path_val='/media/nibaran/STORAGE/SOHAM/PAP/Classification/val'
val_dataset=torchvision.datasets.ImageFolder(
        root=data_path_val, transform=transform)
val_loader=torch.utils.data.DataLoader(
        val_dataset,batch_size=4,shuffle=True)





def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(50):  # loop over the dataset multiple times

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
            running_loss_val = 0.0
