"""
Training script for MIT-BIH beat-level model (optional).

IMPORTANT SAFETY NOTICE:
- Models trained with this script are for EDUCATIONAL AND RESEARCH use only.
- They must NOT be used for clinical or diagnostic purposes.
- Model predictions may be wrong. For any health concerns, consult a qualified medical professional.
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..data.mitbih_preprocessing import create_mitbih_dataloaders
from ..models.mitbih_model import MITBIHBeatModel
from ..training.trainer import EarlyStopping, evaluate, train_one_epoch
from ..utils.config import MITBIHConfig
from ..utils.metrics import compute_classification_metrics, save_confusion_matrix, save_classification_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train MIT-BIH beat classifier (research-only).")
    parser.add_argument("--data_dir", type=str, default="data/mitbih")
    parser.add_argument("--signals", type=str, default="data/mitbih/beats_signals.npy")
    parser.add_argument("--labels", type=str, default="data/mitbih/beats_labels.npy")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default="models")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    config = MITBIHConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.epochs,
        device=args.device,
    )
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, idx_to_label = create_mitbih_dataloaders(
        config, signals_path=args.signals, labels_path=args.labels
    )

    num_classes = len(idx_to_label)
    model = MITBIHBeatModel(num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    early_stopper = EarlyStopping(patience=7, mode="min")

    best_val_loss = float("inf")
    best_model_path = os.path.join(args.out_dir, "mitbih_best.pth")
    history = []

    for epoch in range(1, config.max_epochs + 1):
        print(f"\nEpoch {epoch}/{config.max_epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)

        metrics = compute_classification_metrics(
            val_targets.numpy(), val_preds.numpy(), labels=[idx_to_label[i] for i in range(num_classes)]
        )
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {metrics['f1']:.4f}")

        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": metrics["f1"],
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "idx_to_label": idx_to_label,
                    "config": config.__dict__,
                },
                best_model_path,
            )
            print(f"Saved best MIT-BIH model to {best_model_path}")

        if early_stopper.step(val_loss):
            print("Early stopping triggered.")
            break

    hist_path = os.path.join(args.out_dir, "mitbih_training_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {hist_path}")

    # Test
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
    test_metrics = compute_classification_metrics(
        test_targets.numpy(), test_preds.numpy(), labels=[idx_to_label[i] for i in range(num_classes)]
    )
    print(f"Test Loss: {test_loss:.4f}, Test F1: {test_metrics['f1']:.4f}, Acc: {test_metrics['accuracy']:.4f}")

    cm_path = os.path.join(args.out_dir, "mitbih_confusion_matrix.png")
    rep_path = os.path.join(args.out_dir, "mitbih_classification_report.txt")
    save_confusion_matrix(
        test_targets.numpy(),
        test_preds.numpy(),
        labels=[idx_to_label[i] for i in range(num_classes)],
        out_path=cm_path,
        title="MIT-BIH Test Confusion Matrix",
    )
    save_classification_report(
        test_targets.numpy(),
        test_preds.numpy(),
        labels=[idx_to_label[i] for i in range(num_classes)],
        out_path=rep_path,
    )
    print(f"Saved confusion matrix to {cm_path}")
    print(f"Saved classification report to {rep_path}")


if __name__ == "__main__":
    main()
