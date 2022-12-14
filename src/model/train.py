

import torch.nn as nn
import torch 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader

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


def train(train_loader, val_loader, test_loader, model, opt, loss_fn, epochs, device):
    train_loss = []
    val_loss = []

    for ep in range(epochs):
        train_running_loss = 0
        num_train_instances = 0

        model.train()
        for i, data in enumerate(train_loader):
            # print(i)
            t1, t2, y = data 
            num_train_instances += y.shape[0]
            t1, t2, y = t1.to(device), t2.to(device), y.to(device)
            y = F.one_hot(y, num_classes=3)
            opt.zero_grad()
            y_hat = model(t1, t2)
            training_loss = loss_fn(y_hat, y.float())
            train_running_loss += training_loss.item()
            training_loss.backward()
            opt.step()
        print(f'Epoch {ep}, training loss becomes {train_running_loss/num_train_instances}')
        train_loss.append(training_loss.item())

        model.eval()
        num_val_instances = 0
        val_running_loss = 0
        correct = 0
        for i, data in enumerate(val_loader):
            t1, t2, y = data 
            t1, t2, y = t1.to(device), t2.to(device), y.to(device)
            # Val loss
            y_hat = model(t1, t2)
            y_hot = F.one_hot(y, num_classes=3)
            vloss = loss_fn(y_hat, y_hot.float())
            val_running_loss += vloss.item()

            # Val accuracy
            y_hat = torch.argmax(y_hat, axis=1)
            right = torch.sum(y_hat==y).item()
            correct += right
            num_val_instances += y.shape[0]
        print(f'Epoch {ep}: val loss becomes {val_running_loss/num_val_instances}')
        print(f'Epoch {ep}: val accuracy becomes {correct/num_val_instances}')
        
    correct = 0
    num_instance = 0
    for i, data in enumerate(test_loader):
        t1, t2, y = data 
        t1, t2, y = t1.to(device), t2.to(device), y.to(device)
        y_hat = model(t1, t2)
        y_hat = torch.argmax(y_hat, axis=1)
        right = torch.sum(y_hat==y).item()
        num_instance += y_hat.shape[0]
        correct += right
    
    accuracy = correct/num_instance
    print(accuracy)
    return train_loss, val_loss, accuracy