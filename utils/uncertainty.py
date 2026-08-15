import torch
import torch.nn.functional as F


def enable_mc_dropout(module):
    for m in module.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def mc_dropout_predict(head, features, n_samples=20):
    head.eval()
    enable_mc_dropout(head)

    probs_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = head(features)
            probs_list.append(F.softmax(logits, dim=-1).unsqueeze(0))

    probs_stack = torch.cat(probs_list, dim=0)
    mean_probs = probs_stack.mean(dim=0)

    eps = 1e-8
    entropy = -(mean_probs * torch.log(mean_probs + eps)).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(mean_probs.shape[-1]), device=features.device))
    normalized_entropy = (entropy / max_entropy).clamp(0.0, 1.0)

    head.eval()
    return mean_probs, normalized_entropy