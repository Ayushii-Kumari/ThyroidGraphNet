import torch
import torch.nn as nn
import timm

from utils.uncertainty import mc_dropout_predict
from utils.fusion import UGGF
from utils.graph import build_adaptive_knn_adjacency, build_graph_with_support, GATv2Layer


class ConvNeXtBranch(nn.Module):
    def __init__(self, model_name, feature_dim, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.proj = nn.Linear(self.backbone.num_features, feature_dim)

    def forward(self, x):
        return self.proj(self.backbone(x))


class SwinBranch(nn.Module):
    def __init__(self, model_name, feature_dim, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.proj = nn.Linear(self.backbone.num_features, feature_dim)

    def forward(self, x):
        return self.proj(self.backbone(x))


class MCHead(nn.Module):
    def __init__(self, in_dim, num_classes=2, dropout_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes=2, dropout_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ThyroidNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.cnn_branch = ConvNeXtBranch(cfg.CONVNEXT_MODEL, cfg.FEATURE_DIM)
        self.swin_branch = SwinBranch(cfg.SWIN_MODEL, cfg.FEATURE_DIM)

        self.u1_head = MCHead(cfg.FEATURE_DIM * 2, cfg.NUM_CLASSES, cfg.MC_DROPOUT_P)
        self.fusion = UGGF(cfg.FEATURE_DIM, hidden_dim=256)

        self.u2_head = MCHead(cfg.FEATURE_DIM, cfg.NUM_CLASSES, cfg.MC_DROPOUT_P)
        self.gat = GATv2Layer(cfg.FEATURE_DIM, cfg.GAT_HIDDEN, heads=cfg.GAT_HEADS)

        self.classifier = Classifier(cfg.FEATURE_DIM, cfg.NUM_CLASSES)

    def extract_features(self, x):
        cnn_feat = self.cnn_branch(x)
        trans_feat = self.swin_branch(x)

        concat = torch.cat([cnn_feat, trans_feat], dim=-1)
        _, u1 = mc_dropout_predict(self.u1_head, concat, self.cfg.MC_DROPOUT_SAMPLES)

        fused, gate = self.fusion(cnn_feat, trans_feat, u1.detach())
        return fused, u1, gate

    def forward_train_batch(self, x):
        cnn_feat = self.cnn_branch(x)
        trans_feat = self.swin_branch(x)
        concat = torch.cat([cnn_feat, trans_feat], dim=-1)

        u1_logits = self.u1_head(concat)
        _, u1 = mc_dropout_predict(self.u1_head, concat, self.cfg.MC_DROPOUT_SAMPLES)

        fused, gate = self.fusion(cnn_feat, trans_feat, u1.detach())

        u2_logits = self.u2_head(fused)
        _, u2 = mc_dropout_predict(self.u2_head, fused, self.cfg.MC_DROPOUT_SAMPLES)

        adjacency, k_per_node = build_adaptive_knn_adjacency(
            fused.detach(), u2.detach(), self.cfg.K_MIN, self.cfg.K_MAX
        )

        graph_out = self.gat(fused, adjacency)
        logits = self.classifier(graph_out)

        return {
            "logits": logits, "u1_logits": u1_logits, "u2_logits": u2_logits,
            "u1": u1, "u2": u2, "gate": gate, "k_per_node": k_per_node, "fused": fused,
        }

    @torch.no_grad()
    def forward_inference(self, x, support_feats):
        fused, u1, gate = self.extract_features(x)
        _, u2 = mc_dropout_predict(self.u2_head, fused, self.cfg.MC_DROPOUT_SAMPLES)

        node_feats, adjacency, k = build_graph_with_support(
            fused, u2[0], support_feats, self.cfg.K_MIN, self.cfg.K_MAX
        )

        graph_out = self.gat(node_feats, adjacency)
        logits = self.classifier(graph_out[0:1])

        mid = (self.cfg.K_MIN + self.cfg.K_MAX) / 2
        return {
            "logits": logits,
            "u1": u1.item(),
            "u2": u2.item(),
            "k": k,
            "graph_type": "sparse" if k <= mid else "dense",
        }