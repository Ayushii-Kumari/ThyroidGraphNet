import torch
import torch.nn as nn
import torch.nn.functional as F


def build_adaptive_knn_adjacency(features, uncertainty, k_min=3, k_max=10):
    N = features.shape[0]
    device = features.device

    if N == 1:
        return torch.ones((1, 1), device=device), torch.zeros(1, dtype=torch.long, device=device)

    dist = torch.cdist(features, features, p=2)
    dist.fill_diagonal_(float("inf"))

    u_min, u_max = uncertainty.min(), uncertainty.max()
    if (u_max - u_min) < 1e-6:
        u_norm = torch.full_like(uncertainty, 0.5)
    else:
        u_norm = (uncertainty - u_min) / (u_max - u_min)

    k_per_node = (k_min + u_norm * (k_max - k_min)).round().long()
    k_per_node = k_per_node.clamp(min=1, max=N - 1)

    adjacency = torch.zeros((N, N), device=device)
    for i in range(N):
        k_i = int(k_per_node[i].item())
        nn_idx = torch.topk(dist[i], k_i, largest=False).indices
        adjacency[i, nn_idx] = 1.0
        adjacency[i, i] = 1.0

    return adjacency, k_per_node


def build_graph_with_support(query_feat, query_uncertainty, support_feats, k_min=3, k_max=10):
    dist = torch.cdist(query_feat, support_feats, p=2).squeeze(0)

    u = float(query_uncertainty) if not torch.is_tensor(query_uncertainty) else query_uncertainty.item()
    u = max(0.0, min(1.0, u))
    k = int(round(k_min + u * (k_max - k_min)))
    k = max(1, min(k, support_feats.shape[0]))

    nn_idx = torch.topk(dist, k, largest=False).indices
    neighbor_feats = support_feats[nn_idx]

    node_feats = torch.cat([query_feat, neighbor_feats], dim=0)
    n_nodes = node_feats.shape[0]

    adjacency = torch.zeros((n_nodes, n_nodes), device=node_feats.device)
    adjacency[0, :] = 1.0
    adjacency[:, 0] = 1.0
    for i in range(n_nodes):
        adjacency[i, i] = 1.0

    return node_feats, adjacency, k


class GATv2Layer(nn.Module):

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim

        self.W = nn.Linear(in_dim * 2, out_dim * heads, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.attn)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_dim * heads, in_dim)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x, adjacency):
        N = x.shape[0]
        x_i = x.unsqueeze(1).expand(N, N, -1)
        x_j = x.unsqueeze(0).expand(N, N, -1)
        pair = torch.cat([x_i, x_j], dim=-1)

        feat = self.leaky_relu(self.W(pair))
        feat = feat.view(N, N, self.heads, self.out_dim)

        scores = torch.einsum("ijho,ho->ijh", feat, self.attn)
        mask = (adjacency == 0).unsqueeze(-1)
        scores = scores.masked_fill(mask, float("-inf"))

        alpha = F.softmax(scores, dim=1)
        alpha = self.dropout(alpha)

        out = torch.einsum("ijh,ijho->iho", alpha, feat)
        out = out.reshape(N, self.heads * self.out_dim)
        out = self.out_proj(out)

        return self.norm(out + x)