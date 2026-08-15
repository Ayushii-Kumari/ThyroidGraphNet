import torch
import torch.nn as nn


class UGGF(nn.Module):

    def __init__(self, feature_dim=768, hidden_dim=256):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, cnn_feat, trans_feat, uncertainty):
        u = uncertainty.unsqueeze(-1)
        gate_input = torch.cat([cnn_feat, trans_feat, u], dim=-1)
        gate = self.gate_net(gate_input)

        fused = gate * cnn_feat + (1 - gate) * trans_feat
        fused = self.proj(fused)
        return fused, gate