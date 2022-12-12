import torch.nn as nn
import torch 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from dataloader import MyDataset
from model import BaseModel

# Hyper Parameters
K = 10
epochs = 10
lr = 0.01
feature_size = 120
hidden_size1 = 100 
hidden_size2 = 100

# Preparing train, val, test sets
# TODO()

# Training
train_loader = DataLoader(MyDataset('x.csv', 'y.csv'), shuffle=True, batch_size=1)
model = BaseModel(106, feature_size, hidden_size1, hidden_size2, 3)
opt = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()


if __name__ == '__main__':
    for ep in range(epochs):
        running_loss = 0
        for i, data in enumerate(train_loader):
            t1, t2, y = data 
            y = F.one_hot(y, num_classes=3)

            opt.zero_grad()
            y_hat = model(t1, t2)
            training_loss = loss_fn(y_hat, y.float())
            running_loss += training_loss.item()
            training_loss.backward()
            opt.step()
        print(f'Epoch {ep}, loss becomes {running_loss}')
        

