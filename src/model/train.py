import torch.nn as nn
import torch 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from dataloader import MyDataset
from model import BaseModel

# Hyper Parameters
K = 10
epochs = 5
lr = 0.001
feature_size = 128
hidden_size1 = 512 
hidden_size2 = 256

# Preparing train, val, test sets
# TODO()

# Training


def train(train_loader, model, opt, loss_fn, epochs, device):
    loss = []

    for ep in range(epochs):
        running_loss = 0
        for i, data in enumerate(train_loader):
            # print(i)
            t1, t2, y = data 
            t1, t2, y = t1.to(device), t2.to(device), y.to(device)
            y = F.one_hot(y, num_classes=3)

            opt.zero_grad()
            y_hat = model(t1, t2)
            training_loss = loss_fn(y_hat, y.float())
            running_loss += training_loss.item()
            training_loss.backward()
            opt.step()
        print(f'Epoch {ep}, loss becomes {running_loss}')
        loss.append(training_loss.item())
    correct = 0
    for i, data in enumerate(train_loader):
        t1, t2, y = data 
        y_hat = model(t1, t2)
        y_hat = torch.argmax(y_hat)
        if y_hat == y: correct += 1
    print(correct/len(train_loader))
    
    print(loss)
        

