# src/model2.py

import torch
import torch.nn as nn

class GRUD(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.1):
        super(GRUD, self).__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, time, features]
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])  # Use last time step
        out = self.fc(out)
        return self.sigmoid(out).squeeze()
