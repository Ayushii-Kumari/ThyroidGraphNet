import os
import json
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from config import PROFILES
from dataset import ThyroidDataset, load_tn5000_split
from utils.preprocessing import get_train_transform, get_eval_transform
from network import ThyroidNet

# ---------------- recipe ----------------
SEED = 42
EPOCHS = 25
PATIENCE = 6              # early stopping on validation AUC
BACKBONE_LR = 2e-5        # pretrained ConvNeXt-V2 / Swin-V2
HEAD_LR = 2e-4            # fusion, heads, GATv2, classifier
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 1


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class ModelEMA:
    """Exponential moving average of weights; the EMA copy is what gets validated and saved."""

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


def compute_loss(out, labels, criterion, aux_weight, labels_b=None, lam=1.0):
    def ce(logits):
        loss = criterion(logits, labels)
        if labels_b is not None:
            loss = lam * loss + (1 - lam) * criterion(logits, labels_b)
        return loss
    return ce(out["logits"]) + aux_weight * (ce(out["u1_logits"]) + ce(out["u2_logits"]))


@torch.no_grad()
def build_bank(model, loader, device, bank_size):
    """Feature bank from TRAIN images (clean transform, shuffled so both classes appear)."""
    model.eval()
    feats, labels, n = [], [], 0
    for imgs, lab, _ in loader:
        fused, _, _ = model.extract_features(imgs.to(device))
        feats.append(fused); labels.append(lab); n += imgs.size(0)
        if n >= bank_size:
            break
    return torch.cat(feats)[:bank_size], torch.cat(labels)[:bank_size]


@torch.no_grad()
def validate(model, val_ds, bank_loader, device, bank_size):
    """Validate exactly like test time: query + support-bank graph, one image at a time."""
    model.eval()
    bank, _ = build_bank(model, bank_loader, device, bank_size)
    y, p = [], []
    for imgs, lab, _ in DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2):
        res = model.forward_inference(imgs.to(device), bank)
        p.append(torch.softmax(res["logits"], -1)[0, 1].item()); y.append(lab.item())
    y, p = np.array(y), np.array(p)
    pred = (p >= 0.5).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    sens, spec = tp / max((y == 1).sum(), 1), tn / max((y == 0).sum(), 1)
    return dict(acc=float((pred == y).mean()), sens=float(sens), spec=float(spec),
                auc=float(roc_auc_score(y, p))), y, p, bank.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=list(PROFILES), default="base")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = PROFILES[args.profile]()
    if args.profile != "base":
        tag = os.path.join(args.profile, f"seed{args.seed}")
        cfg.MODEL_DIR = os.path.join(cfg.MODEL_DIR, tag)
        cfg.RESULTS_DIR = os.path.join(cfg.RESULTS_DIR, tag)
    device = torch.device(cfg.DEVICE)
    use_amp = cfg.AMP and device.type == "cuda"
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    print(f"Device: {device} | profile {args.profile} | seed {args.seed} | AMP {use_amp}")

    train_files = load_tn5000_split(cfg.TN5000_DIR, "train")
    val_files = load_tn5000_split(cfg.TN5000_DIR, "val")
    n_ben = sum(1 for _, l in train_files if l == 0)
    n_mal = len(train_files) - n_ben
    n_vben = sum(1 for _, l in val_files if l == 0)
    print(f"TN5000 -> Train: {len(train_files)} (benign {n_ben} / malignant {n_mal}) | "
          f"Validation: {len(val_files)} (benign {n_vben} / malignant {len(val_files) - n_vben})")

    roi = cfg.ROI_MARGIN if cfg.USE_ROI else None
    train_ds = ThyroidDataset(train_files, get_train_transform(cfg.IMAGE_SIZE), roi, roi_jitter=True)
    bank_ds = ThyroidDataset(train_files, get_eval_transform(cfg.IMAGE_SIZE), roi)
    val_ds = ThyroidDataset(val_files, get_eval_transform(cfg.IMAGE_SIZE), roi)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    bank_loader = DataLoader(bank_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)

    model = ThyroidNet(cfg).to(device)
    ema = ModelEMA(model, cfg.EMA_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    bb = list(model.cnn_branch.backbone.parameters()) + list(model.swin_branch.backbone.parameters())
    bb_ids = {id(p) for p in bb}
    head = [p for p in model.parameters() if id(p) not in bb_ids]
    optimizer = torch.optim.AdamW(
        [{"params": bb, "lr": BACKBONE_LR}, {"params": head, "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(train_loader)
    warm = WARMUP_EPOCHS * steps_per_epoch
    total = EPOCHS * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warm),
         torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total - warm, 1))],
        milestones=[warm])

    w = torch.tensor([len(train_files) / (2 * n_ben), len(train_files) / (2 * n_mal)],
                     dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTHING)

    queue, _ = build_bank(model, bank_loader, device, cfg.SUPPORT_BANK_SIZE)   # initial feature queue

    best_auc, best_state, best_bank, best_val, bad, history = 0.0, None, None, None, 0, []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss, correct, seen = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            mixed = random.random() < cfg.MIXUP_PROB
            labels_b, lam = None, 1.0
            if mixed:
                lam = float(np.random.beta(cfg.MIXUP_ALPHA, cfg.MIXUP_ALPHA))
                perm = torch.randperm(imgs.size(0), device=device)
                imgs = lam * imgs + (1 - lam) * imgs[perm]
                labels_b = labels[perm]

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model.forward_train_bank(imgs, queue)
                loss = compute_loss(out, labels, criterion, cfg.AUX_LOSS_WEIGHT, labels_b, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            if not mixed:   # keep the support queue free of blended-image features
                queue = torch.cat([queue, out["fused"].detach().float()])[-cfg.SUPPORT_BANK_SIZE:]

            run_loss += loss.item() * imgs.size(0)
            correct += (out["logits"].argmax(1) == labels).sum().item()
            seen += imgs.size(0)
            pbar.set_postfix(loss=loss.item())

        v, yv, pv, bank_cpu = validate(ema.module, val_ds, bank_loader, device, cfg.SUPPORT_BANK_SIZE)
        row = {"epoch": epoch, "train_loss": run_loss / seen, "train_acc": correct / seen,
               **{f"val_{k}": x for k, x in v.items()}}
        history.append(row)
        print(f"Epoch {epoch}: train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} | "
              f"val_acc={v['acc']:.4f} sens={v['sens']:.4f} spec={v['spec']:.4f} auc={v['auc']:.4f}")

        if v["auc"] > best_auc:
            best_auc, bad = v["auc"], 0
            best_state = copy.deepcopy(ema.module.state_dict())
            best_bank, best_val = bank_cpu, (yv, pv)
            torch.save(best_state, os.path.join(cfg.MODEL_DIR, "thyroidnet_best.pth"))
            torch.save({"features": best_bank}, os.path.join(cfg.MODEL_DIR, "support_bank.pt"))
            print(f"  -> new best (val AUC {best_auc:.4f}); checkpoint + bank saved")
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping.")
                break

    # decision threshold chosen on VALIDATION only (Youden's J = sens + spec - 1)
    yv, pv = best_val
    fpr, tpr, thr = roc_curve(yv, pv)
    t = float(thr[int(np.argmax(tpr - fpr))])
    with open(os.path.join(cfg.RESULTS_DIR, "threshold.json"), "w") as f:
        json.dump({"threshold": t, "method": "Youden J on validation", "val_auc": best_auc}, f, indent=2)
    with open(os.path.join(cfg.RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best validation AUC {best_auc:.4f}; validation-chosen threshold {t:.3f}")


if __name__ == "__main__":
    main()
