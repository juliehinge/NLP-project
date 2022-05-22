import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np


class LSTM_test(nn.Module):
    def __init__(self):
        super(LSTM_test, self).__init__()
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

        print("> LSTM done")        
        return x




#td = TensorDataset(torch_emb, target)
#dl = DataLoader(td, batch_size=120, shuffle=True)

model = LSTM_test()


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


#loss = loss_calc(10)



#pred = torch.round(model(torch_emb[:100])[0])
#acc = sum(pred == target[:100]) / 100

#torch.sum(model(test_torch_emb) == test_target) / test_target.size(0)

