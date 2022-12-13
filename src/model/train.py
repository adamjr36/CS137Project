import torch.nn as nn
import torch 
import torch.nn.functional as F 
#import matplotlib.pyplot as plt 
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


def train(train_loader, val_loader, model, opt, loss_fn, epochs, device):
    train_loss = []
    val_loss = []
    val_acc = []

    for ep in range(epochs):
        #Training
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

       
        train_loss.append(running_loss / len(train_loader))

        #Validation
        correct = 0
        v_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                t1, t2, y = data 
                t1, t2, y = t1.to(device), t2.to(device), y.to(device)
                y_oh = F.one_hot(y, num_classes=3)

                y_hat = model(t1, t2)
                loss = loss_fn(y_hat, y_oh.float())
                v_loss += loss.item()

                y_hat = torch.argmax(y_hat, axis=1)
                right = torch.sum(y_hat==y)
                correct += right

        val_acc.append(correct / len(val_loader))
        val_loss.append(v_loss)

        print('Epoch {ep}, train loss {tloss}, val loss {vloss}, val acc {vacc}'.format(ep=ep, tloss=(running_loss / len(train_loader)), vloss=v_loss, vacc=(correct / len(val_loader))))
    
    return train_loss, val_loss, val_acc

        

