import torch
import torch.nn as nn

class CommunityEvolutionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0):
        super(CommunityEvolutionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)  # Binary output
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)                   # (B, T, H)
        normed = self.layer_norm(lstm_out)           # (B, T, H)
        logits = self.fc(normed).squeeze(-1)         # (B, T)
        return logits