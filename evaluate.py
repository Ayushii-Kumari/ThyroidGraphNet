import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)

from config import Config
from dataset import ThyroidDataset, load_tn5000_split
from utils.preprocessing import get_eval_transform
from network import ThyroidNet

USE_TTA = True   # average original + horizontally flipped prediction


def metrics_at(y, s, thr):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / max(tp + fp, 1); sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    return dict(threshold=round(float(thr), 4), accuracy=(tp + tn) / len(y), precision=prec,
                sensitivity=sens, specificity=spec, balanced_accuracy=(sens + spec) / 2, f1_score=f1,
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp)), pred


def ece_score(correct, conf, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0; xs, ys = [], []
    for i in range(bins):
        m = (conf >= edges[i]) & ((conf < edges[i + 1]) if i < bins - 1 else (conf <= edges[i + 1]))
        if m.sum():
            xs.append(conf[m].mean()); ys.append(correct[m].mean()); e += m.mean() * abs(ys[-1] - xs[-1])
    return e, xs, ys


def main():
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(cfg.RESULTS_DIR, "preds"), exist_ok=True)

    model = ThyroidNet(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg.MODEL_DIR, "thyroidnet_best.pth"), map_location=device))
    model.eval()
    bank = torch.load(os.path.join(cfg.MODEL_DIR, "support_bank.pt"), map_location=device)["features"].to(device)
    thr = json.load(open(os.path.join(cfg.RESULTS_DIR, "threshold.json")))["threshold"] \
        if os.path.exists(os.path.join(cfg.RESULTS_DIR, "threshold.json")) else 0.5

    test_files = load_tn5000_split(cfg.TN5000_DIR, "test")
    ds = ThyroidDataset(test_files, get_eval_transform(cfg.IMAGE_SIZE))
    print(f"TN5000 test images: {len(ds)} | threshold {thr:.3f} | TTA {USE_TTA}")

    ids, y, s, u1s, u2s = [], [], [], [], []
    with torch.no_grad():
        for img, lab, path in DataLoader(ds, batch_size=1, shuffle=False, num_workers=0):
            img = img.to(device)
            outs = [model.forward_inference(img, bank)]
            if USE_TTA:
                outs.append(model.forward_inference(torch.flip(img, dims=[3]), bank))
            prob = np.mean([torch.softmax(o["logits"], -1)[0, 1].item() for o in outs])
            ids.append(path[0]); y.append(lab.item()); s.append(prob)
            u1s.append(np.mean([o["u1"] for o in outs])); u2s.append(np.mean([o["u2"] for o in outs]))
    y, s = np.array(y), np.array(s)

    pd.DataFrame({"image_id": ids, "y_true": y, "prob_malignant": s, "u1": u1s, "u2": u2s}).to_csv(
        os.path.join(cfg.RESULTS_DIR, "preds", "ThyroidGraphNet.csv"), index=False)

    m, pred = metrics_at(y, s, thr)
    m05, _ = metrics_at(y, s, 0.5)
    auc, ap = roc_auc_score(y, s), average_precision_score(y, s)
    conf = np.maximum(s, 1 - s); correct = (pred == y).astype(float)
    ece, bx, by = ece_score(correct, conf)

    out = {"dataset": "TN5000 Test", "n_samples": len(y), "prevalence_malignant": round(float(y.mean()), 4),
           **{k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in m.items()},
           "roc_auc": round(float(auc), 4), "pr_auc": round(float(ap), 4),
           "mean_u1": round(float(np.mean(u1s)), 4), "mean_u2": round(float(np.mean(u2s)), 4),
           "ece": round(float(ece), 4), "tta": USE_TTA,
           "at_threshold_0.5": {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in m05.items()}}
    json.dump(out, open(os.path.join(cfg.RESULTS_DIR, "metrics.json"), "w"), indent=4)

    R = cfg.RESULTS_DIR
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    fig, ax = plt.subplots(figsize=(5, 4)); im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], cfg.CLASS_NAMES); ax.set_yticks([0, 1], cfg.CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix — TN5000 Test")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(f"{R}/confusion_matrix.png", dpi=300); plt.close(fig)

    fpr, tpr, _ = roc_curve(y, s); plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}"); plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve — TN5000 Test")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{R}/roc_curve.png", dpi=300); plt.close()

    pr, rc, _ = precision_recall_curve(y, s); plt.figure(figsize=(5, 4))
    plt.plot(rc, pr, label=f"PR-AUC = {ap:.3f}"); plt.axhline(y.mean(), ls="--", color="k", label=f"Prevalence = {y.mean():.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve — TN5000 Test")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{R}/pr_curve.png", dpi=300); plt.close()

    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    ax.plot(bx, by, "o-", label="ThyroidGraphNet"); ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy"); ax.set_title("Reliability Diagram — TN5000 Test"); ax.legend(loc="upper left")
    ax.text(0.97, 0.05, f"ECE = {ece:.4f}", transform=ax.transAxes, ha="right")
    fig.tight_layout(); fig.savefig(f"{R}/calibration.png", dpi=300); plt.close(fig)

    print("=" * 60)
    for k in ["threshold", "accuracy", "precision", "sensitivity", "specificity", "balanced_accuracy", "f1_score",
              "roc_auc", "pr_auc", "mean_u1", "mean_u2", "ece"]:
        print(f"{k:18s}: {out[k]}")
    print("Confusion matrix:\n", cm)
    print("At threshold 0.5:", {k: out['at_threshold_0.5'][k] for k in ['accuracy', 'sensitivity', 'specificity']})
    print("=" * 60)


if __name__ == "__main__":
    main()
