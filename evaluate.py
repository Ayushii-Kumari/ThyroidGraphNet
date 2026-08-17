import os
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from config import Config
from dataset import ThyroidDataset, load_tn5000_split
from utils.preprocessing import get_eval_transform
from network import ThyroidNet


def plot_confusion_matrix(cm, class_names, save_path):

    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_title("Confusion Matrix — TN5000 Test")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    fig.colorbar(im)

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_roc(y_true, y_scores, auc, save_path):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_scores
    )

    plt.figure(figsize=(5, 4))

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {auc:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve — TN5000 Test")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_pr(y_true, y_scores, ap, save_path):

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_scores
    )

    plt.figure(figsize=(5, 4))

    plt.plot(
        recall,
        precision,
        label=f"PR-AUC = {ap:.3f}"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve — TN5000 Test")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_reliability_diagram(
    y_correct,
    y_confidence,
    save_path,
    n_bins=10
):

    y_confidence = np.asarray(
        y_confidence,
        dtype=float
    )

    y_correct = np.asarray(
        y_correct,
        dtype=float
    )

    y_confidence = np.clip(
        y_confidence,
        0.0,
        1.0
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    bin_confidence = []
    bin_accuracy = []
    bin_counts = []

    for b in range(n_bins):

        if b == n_bins - 1:

            mask = (
                (y_confidence >= bin_edges[b]) &
                (y_confidence <= bin_edges[b + 1])
            )

        else:

            mask = (
                (y_confidence >= bin_edges[b]) &
                (y_confidence < bin_edges[b + 1])
            )

        count = int(mask.sum())

        if count > 0:

            mean_conf = float(
                np.mean(
                    y_confidence[mask]
                )
            )

            accuracy = float(
                np.mean(
                    y_correct[mask]
                )
            )

            bin_confidence.append(
                mean_conf
            )

            bin_accuracy.append(
                accuracy
            )

            bin_counts.append(
                count
            )

    total_samples = len(
        y_confidence
    )

    ece = 0.0

    for conf, acc, count in zip(
        bin_confidence,
        bin_accuracy,
        bin_counts
    ):

        ece += (
            count / total_samples
        ) * abs(
            acc - conf
        )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration"
    )

    ax.plot(
        bin_confidence,
        bin_accuracy,
        marker="o",
        linewidth=2,
        markersize=6,
        label="ThyroidGraphNet"
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_xlabel(
        "Mean predicted confidence"
    )

    ax.set_ylabel(
        "Empirical accuracy"
    )

    ax.set_title(
        "Reliability Diagram — TN5000 Test",
        fontweight="bold"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        loc="upper left"
    )

    ax.text(
        0.97,
        0.05,
        f"ECE = {ece:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="0.7"
        )
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return ece


def main():

    cfg = Config()

    device = torch.device(
        cfg.DEVICE
    )

    print("=" * 60)
    print("TN5000 TEST SET EVALUATION")
    print("=" * 60)

    print(
        f"Device: {device}"
    )

    model_path = os.path.join(
        cfg.MODEL_DIR,
        "thyroidnet_best.pth"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"\nModel not found:\n{model_path}\n\n"
            "Run train.py first."
        )

    model = ThyroidNet(
        cfg
    ).to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    support_bank_path = os.path.join(
        cfg.MODEL_DIR,
        "support_bank.pt"
    )

    if not os.path.exists(
        support_bank_path
    ):

        raise FileNotFoundError(
            f"\nSupport bank not found:\n"
            f"{support_bank_path}\n\n"
            "Run train.py first."
        )

    bank = torch.load(
        support_bank_path,
        map_location=device
    )

    support_feats = bank[
        "features"
    ].to(device)

    test_files = load_tn5000_split(
        cfg.TN5000_DIR,
        "test"
    )

    test_dataset = ThyroidDataset(
        test_files,
        get_eval_transform(
            cfg.IMAGE_SIZE
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print(
        f"TN5000 test images: "
        f"{len(test_dataset)}"
    )

    y_true = []
    y_pred = []
    y_score = []

    y_confidence = []
    y_correct = []

    u1_list = []
    u2_list = []

    print(
        "\nRunning evaluation..."
    )

    with torch.no_grad():

        for imgs, labels, paths in test_loader:

            imgs = imgs.to(device)

            result = model.forward_inference(
                imgs,
                support_feats
            )

            probs = torch.softmax(
                result["logits"],
                dim=-1
            ).squeeze(0)

            true_label = labels.item()

            predicted_label = (
                probs.argmax().item()
            )

            class_1_probability = (
                probs[1].item()
            )

            confidence = (
                probs.max().item()
            )

            correct = int(
                predicted_label == true_label
            )

            y_true.append(
                true_label
            )

            y_pred.append(
                predicted_label
            )

            y_score.append(
                class_1_probability
            )

            y_confidence.append(
                confidence
            )

            y_correct.append(
                correct
            )

            u1_list.append(
                float(result["u1"])
            )

            u2_list.append(
                float(result["u2"])
            )

    y_true = np.array(
        y_true
    )

    y_pred = np.array(
        y_pred
    )

    y_score = np.array(
        y_score
    )

    y_confidence = np.array(
        y_confidence
    )

    y_correct = np.array(
        y_correct
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_score
    )

    pr_auc = average_precision_score(
        y_true,
        y_score
    )


    os.makedirs(
        cfg.RESULTS_DIR,
        exist_ok=True
    )


    reliability_path = os.path.join(
        cfg.RESULTS_DIR,
        "reliability_diagram.png"
    )

    ece = plot_reliability_diagram(
        y_correct,
        y_confidence,
        reliability_path,
        n_bins=10
    )


    metrics = {

        "dataset": "TN5000 Test",

        "n_samples": len(y_true),

        "accuracy": round(
            float(accuracy),
            4
        ),

        "precision": round(
            float(precision),
            4
        ),

        "sensitivity": round(
            float(sensitivity),
            4
        ),

        "specificity": round(
            float(specificity),
            4
        ),

        "f1_score": round(
            float(f1),
            4
        ),

        "roc_auc": round(
            float(roc_auc),
            4
        ),

        "pr_auc": round(
            float(pr_auc),
            4
        ),

        "mean_u1": round(
            float(np.mean(u1_list)),
            4
        ),

        "mean_u2": round(
            float(np.mean(u2_list)),
            4
        ),

        # NEW
        "ece": round(
            float(ece),
            4
        )
    }

    metrics_path = os.path.join(
        cfg.RESULTS_DIR,
        "metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    plot_confusion_matrix(
        cm,
        cfg.CLASS_NAMES,
        os.path.join(
            cfg.RESULTS_DIR,
            "confusion_matrix.png"
        )
    )

    plot_roc(
        y_true,
        y_score,
        roc_auc,
        os.path.join(
            cfg.RESULTS_DIR,
            "roc_curve.png"
        )
    )

    plot_pr(
        y_true,
        y_score,
        pr_auc,
        os.path.join(
            cfg.RESULTS_DIR,
            "pr_curve.png"
        )
    )

    print("\n")
    print("=" * 60)
    print("MODEL PERFORMANCE — TN5000 TEST")
    print("=" * 60)

    print(
        f"Accuracy       : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision      : {precision * 100:.2f}%"
    )

    print(
        f"Sensitivity    : {sensitivity * 100:.2f}%"
    )

    print(
        f"Specificity    : {specificity * 100:.2f}%"
    )

    print(
        f"F1-Score       : {f1 * 100:.2f}%"
    )

    print(
        f"ROC-AUC        : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC         : {pr_auc:.4f}"
    )

    print(
        f"Mean U1        : {np.mean(u1_list):.4f}"
    )

    print(
        f"Mean U2        : {np.mean(u2_list):.4f}"
    )

    print(
        f"ECE            : {ece:.4f}"
    )

    print("=" * 60)

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    print(
        f"\nReliability diagram saved to:\n"
        f"{reliability_path}"
    )

    print(
        f"\nResults saved to: "
        f"{cfg.RESULTS_DIR}/"
    )


if __name__ == "__main__":
    main()