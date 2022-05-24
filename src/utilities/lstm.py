import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from datetime import datetime


class LstmModel(nn.Module):
    def __init__(self):
        super(LstmModel, self).__init__()
        self.lstm = nn.LSTM(300, 100, num_layers=2, bidirectional=True, batch_first=True, dropout=.2)
        self.l1 = nn.Linear(200, 100)
        self.l2 = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(.2)
        
    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = self.drop(x)
        
        x_f = x[:, -1, :100]
        x_b = x[:, 0, 100:]
        x = torch.concat((x_f, x_b), axis=1)
        
        x = self.l1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.l2(x)
        
        x = self.sig(x)

        return x

class LSTM:

    def __init__(self, batches_print):
        self.model = LstmModel()
        self.batches_print = batches_print # print every x mini-batches
    
    def train(self, epochs, trainloader):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.BCELoss(reduction='mean')

        for epoch in range(epochs):  # loop over the dataset multiple times
            
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batches_print = self.batches_print # print every x mini-batches
                if i % batches_print == (batches_print - 1):
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batches_print:.3f}')
                    running_loss = 0.0

        filename = datetime.now().strftime("%b-%d-%Y-%H")
        filepath = f'data/trainedmodels/{filename}.pt'
        torch.save(self.model.state_dict(), filepath)
        print(f'> Saved the trained model to {filepath}\n')


def loss_calc(_range, td, dl):

    print("> Calculating the loss")
    loss_fn = nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    lossssss = []
    for epoch in range(_range):
        print(epoch)
        losses = []
        for x, y in dl:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()

            # nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
        print(np.mean(losses))
        lossssss.append(np.mean(losses))
    return lossssss




