# src/model4.py

import torch
import torch.nn as nn

class TransformerICU(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=128,
            dropout=0.1,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        x = self.input_proj(x)           # (batch, seq_len, model_dim)
        x = self.encoder(x)              # (batch, seq_len, model_dim)
        x = x.mean(dim=1)                # Global average pooling over time
        out = self.fc(x)                 # (batch, 1)
        return self.sigmoid(out).squeeze()  # (batch,)
