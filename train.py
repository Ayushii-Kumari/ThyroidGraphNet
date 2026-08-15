import os
import json
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import ThyroidDataset, load_tn5000_split
from utils.preprocessing import get_train_transform, get_eval_transform
from network import ThyroidNet


def compute_loss(out, labels, criterion, aux_weight):
    loss_main = criterion(out["logits"], labels)
    loss_u1 = criterion(out["u1_logits"], labels)
    loss_u2 = criterion(out["u2_logits"], labels)
    return loss_main + aux_weight * (loss_u1 + loss_u2)


def evaluate_loader(model, loader, device, criterion, aux_weight):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model.forward_train_batch(imgs)
            loss = compute_loss(out, labels, criterion, aux_weight)
            loss_sum += loss.item() * imgs.size(0)
            preds = out["logits"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return loss_sum / total, correct / total


def build_support_bank(model, loader, device, bank_size):
    model.eval()
    feats_list, labels_list, collected = [], [], 0
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            fused, _, _ = model.extract_features(imgs)
            feats_list.append(fused.cpu())
            labels_list.append(labels)
            collected += imgs.size(0)
            if collected >= bank_size:
                break
    feats = torch.cat(feats_list, dim=0)[:bank_size]
    labels = torch.cat(labels_list, dim=0)[:bank_size]
    return feats, labels


def main():
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    print(f"Device: {device}")

    train_files = load_tn5000_split(
    cfg.TN5000_DIR,
    "train"
    )

    val_files = load_tn5000_split(
    cfg.TN5000_DIR,
    "val"
    )

    print(
    f"TN5000 -> "
    f"Train: {len(train_files)} | "
    f"Validation: {len(val_files)}"
    )

    train_ds = ThyroidDataset(train_files, get_train_transform(cfg.IMAGE_SIZE))
    val_ds = ThyroidDataset(val_files, get_eval_transform(cfg.IMAGE_SIZE))

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    model = ThyroidNet(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}")
        for imgs, labels, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model.forward_train_batch(imgs)
            loss = compute_loss(out, labels, criterion, cfg.AUX_LOSS_WEIGHT)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct += (out["logits"].argmax(dim=1) == labels).sum().item()
            total += imgs.size(0)
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        train_loss, train_acc = running_loss / total, correct / total
        val_loss, val_acc = evaluate_loader(model, val_loader, device, criterion, cfg.AUX_LOSS_WEIGHT)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            print(f"  -> New best model (val_acc={val_acc:.4f}), checkpoint updated.")

    torch.save(best_state, os.path.join(cfg.MODEL_DIR, "thyroidnet_best.pth"))

    model.load_state_dict(best_state)
    support_feats, support_labels = build_support_bank(model, train_loader, device, cfg.SUPPORT_BANK_SIZE)
    torch.save({"features": support_feats, "labels": support_labels},
               os.path.join(cfg.MODEL_DIR, "support_bank.pt"))

    with open(os.path.join(cfg.RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best internal validation accuracy: {best_val_acc:.4f}")
    print(f"Saved: {cfg.MODEL_DIR}/thyroidnet_best.pth")
    print(f"Saved: {cfg.MODEL_DIR}/support_bank.pt")


if __name__ == "__main__":
    main()