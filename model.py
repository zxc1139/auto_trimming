import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, activation_fun):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.activation_fun = activation_fun
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity=activation_fun, dropout=0.1)
        # self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.1)

        self.fc = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features = output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.1),
            )

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)  #self.hidden_dim
        # c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)

        out, hn = self.rnn(x, h0.detach())
        # out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :]) 
        # out = self.fc2(out)
        # out = self.fc_bn(out)
        out = F.sigmoid(out)
        return out
    

class FCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FCModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        #Multilayer Perceptron for regression
        self.net = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            # nn.Softmax(dim=0),
            nn.Sigmoid(),
            nn.BatchNorm1d(output_dim))
        
    def forward(self, x):
        # Encoder: affine function
        x = x.view(-1, self.input_dim)
        out = self.net(x)
        return out


class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, nfeed, nlayers, output_dim, dropout1, dropout2, hidden_dim):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nhead = nhead
    
        #Multilayer Perceptron for regression
        encoder_layers = TransformerEncoderLayer(input_dim, nhead, dim_feedforward=nfeed, dropout=dropout1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.Dropout(dropout2),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            )
        
    def forward(self, x):
        # Encoder: affine function
        x = x.view(-1, self.input_dim)
        output = self.transformer_encoder(x)
        output = self.decoder(output)
        # output = F.sigmoid(output)
        return output
