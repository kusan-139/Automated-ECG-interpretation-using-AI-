"""
Generic training utilities.

IMPORTANT SAFETY NOTICE:
- Any models trained with this code are for EDUCATIONAL AND RESEARCH use only.
- They must NOT be used for clinical or diagnostic decisions.
- Model predictions may be wrong. For health concerns, consult a qualified medical professional.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval", leave=False):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
    avg_loss = running_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return avg_loss, all_preds, all_targets


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "min", delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def step(self, metric: float) -> bool:
        if self.best_score is None:
            self.best_score = metric
            return False

        improve = (
            metric < self.best_score - self.delta
            if self.mode == "min"
            else metric > self.best_score + self.delta
        )

        if improve:
            self.best_score = metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False
