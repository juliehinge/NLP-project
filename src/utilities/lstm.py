import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import torch.optim as optim

import numpy as np

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

class LstmModel(nn.Module):
    def __init__(self):
        super(LstmModel, self).__init__()
        self.lstm = nn.LSTM(300, 100, num_layers=2, bidirectional=True, batch_first=True, dropout=.2)
        self.l1 = nn.Linear(200, 100)
        self.l2 = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(.1)
        
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
        self.loss_fn = nn.BCELoss(reduction='mean')
    
    def train(self, epochs, trainloader, valoader):
        
        # turn training mode on
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)
        
        loss_train_hist = []
        loss_val_hist = []

        for epoch in range(epochs):  # loop over the dataset multiple times
            
            running_loss = 0
            loss_train_epoch = 0

            for i, data in enumerate(trainloader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                loss_train_epoch += loss.item()
                running_loss += loss.item()
                batches_print = self.batches_print # print every x mini-batches
                if i % batches_print == (batches_print - 1):
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batches_print:.3f}')
                    running_loss = 0.0
            
            # Add training loss of the epoch - average of batches loss
            loss_train_hist.append(loss_train_epoch/i)
            
            # turn evaluation mode on
            self.model.eval()

            # Add validation loss of the epoch
            inputs, labels = next(iter(valoader))
            outputs = self.model(inputs)
            val_loss = self.loss_fn(outputs, labels).item()
            loss_val_hist.append(val_loss)


        # Save the development of loss: train vs val
        fig, ax = plt.subplots()
        x = [i for i in range(1, epochs+1)]
        sns.lineplot(x, loss_train_hist, ax=ax, label='Training loss')
        sns.lineplot(x, loss_val_hist, ax=ax, label='Validation loss')
        #ax.set_ylim([0, 1])
        plt.xticks(x)
        plt.title('Training vs. validation loss',weight="bold")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')


        filename = datetime.now().strftime("%b-%d-%Y-%H--trainvsval_loss")
        fig.savefig(f'data/figures/{filename}')

        # Save model
        filename = datetime.now().strftime("%b-%d-%Y-%H")
        filepath = f'data/trainedmodels/{filename}.pt'
        torch.save(self.model.state_dict(), filepath)
        print(f'> Saved the trained model to {filepath}\n')
