import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.optim import Adam

from config import OUT_DIR, SEED, WEIGHTS
from data import get_loaders
from model import build_model, set_trainable


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_epoch(model, loader, criterion, optimizer, device, train, freeze_features=False):
    if train:
        model.train()
        if freeze_features:
            model.features.eval()
    else:
        model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    for x, labels in loader:
        x = x.to(device)
        y = (1 - labels).float().unsqueeze(1).to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
        pred = (torch.sigmoid(logits) >= 0.5).float()
        correct += (pred == y).sum().item()
    return total_loss / n, correct / n


def collect_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, folder_labels in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            y = (1 - folder_labels).numpy()
            probs.append(p)
            labels.append(y)
    return np.concatenate(probs), np.concatenate(labels)


def plot_curves(history, stage2_start, path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].axvline(stage2_start, color="gray", linestyle="--", label="stage 2")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].axvline(stage2_start, color="gray", linestyle="--", label="stage 2")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    set_seed()
    os.makedirs("weights", exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader, val_loader, test_loader = get_loaders()
    model = build_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_state = None
    best_val = float("inf")

    def record(tr_loss, tr_acc, va_loss, va_acc, tag):
        nonlocal best_state, best_val
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"{tag} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            print(f"  new best val_loss={best_val:.4f}")

    set_trainable(model, backbone=False)
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for epoch in range(1, 6):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, opt, device, True, freeze_features=True
        )
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device, False)
        record(tr_loss, tr_acc, va_loss, va_acc, f"stage1 epoch {epoch}/5")

    set_trainable(model, backbone=True, last_n_blocks=4)
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    for epoch in range(1, 11):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, opt, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device, False)
        record(tr_loss, tr_acc, va_loss, va_acc, f"stage2 epoch {epoch}/10")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), WEIGHTS)
    print("saved", WEIGHTS)

    plot_curves(history, stage2_start=5.5, path=os.path.join(OUT_DIR, "curves.png"))

    val_probs, val_labels = collect_probs(model, val_loader, device)
    np.save(os.path.join(OUT_DIR, "val_probs.npy"), val_probs)
    np.save(os.path.join(OUT_DIR, "val_labels.npy"), val_labels)

    test_probs, test_labels = collect_probs(model, test_loader, device)
    test_pred = (test_probs >= 0.5).astype(np.int64)
    y_true = test_labels.astype(np.int64)
    acc = float((test_pred == y_true).mean())
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, test_pred, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "n_test": int(len(y_true)),
        "best_val_loss": float(best_val),
        "history": history,
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("test metrics:", {k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "n_test")})

    cm = confusion_matrix(y_true, test_pred, labels=[0, 1])
    print("confusion matrix [[tn, fp], [fn, tp]]:\n", cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["not_hot_dog", "hot_dog"])
    ax.set_yticks([0, 1], ["not_hot_dog", "hot_dog"])
    ax.set_xlabel("predicted (hotdog=1)")
    ax.set_ylabel("true (hotdog=1)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
    plt.close(fig)

    test_ds = test_loader.dataset
    wrong = np.where(test_pred != y_true)[0]
    print(f"misclassified test images: {len(wrong)}")
    for i in wrong[:10]:
        path, folder_label = test_ds.samples[i]
        print(
            f"  {path} folder_label={folder_label} true_hotdog={int(y_true[i])} "
            f"prob_hotdog={test_probs[i]:.3f}"
        )


if __name__ == "__main__":
    main()
