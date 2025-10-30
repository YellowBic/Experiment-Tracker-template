
import os, sys, math, argparse
from typing import Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from omegaconf import OmegaConf
from tracker.experiment import ExperimentTracker

# Simple MLP for 28x28 images
class MLP(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )
    def forward(self, x): return self.net(x)

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def train_epoch(model, loader, optimizer, tracker: ExperimentTracker):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(tracker.device), y.to(tracker.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=tracker.mixed_precision):
            logits = model(x)
            loss = ce(logits, y)
        tracker.scaler.scale(loss).backward()
        if tracker.cfg.get("trainer", {}).get("grad_clip_norm", None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), tracker.cfg["trainer"]["grad_clip_norm"])
        tracker.scaler.step(optimizer)
        tracker.scaler.update()

        acc = accuracy(logits.detach(), y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        n += bs
        tracker.global_step += 1
        if step % tracker.cfg["trainer"]["log_every_steps"] == 0:
            tracker.log(train_loss=total_loss/n, train_acc=total_acc/n)
    return total_loss/n, total_acc/n

@torch.no_grad()
def validate(model, loader, tracker: ExperimentTracker):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(tracker.device), y.to(tracker.device)
        logits = model(x)
        loss = ce(logits, y)
        acc = accuracy(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        n += bs
    return total_loss/n, total_acc/n

def load_cfg():
    # Load default.yaml and allow CLI overrides like key=value
    cfg = OmegaConf.load("configs/default.yaml")
    cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli)
    return OmegaConf.to_container(cfg, resolve=True)

def main():
    cfg = load_cfg()
    tracker = ExperimentTracker(cfg)
    device = tracker.device

    # Data
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tfm)
    val_ds   = datasets.FashionMNIST(root="./data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                              num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
                              num_workers=cfg["data"]["num_workers"], pin_memory=True)

    # Model/optim
    model = MLP(hidden_dim=cfg["model"]["hidden_dim"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["optim"]["lr"],
                             weight_decay=cfg["optim"]["weight_decay"],
                             betas=tuple(cfg["optim"]["betas"]))

    # Resume if requested
    resume_dir = cfg.get("resume")
    if resume_dir:
        tracker.try_resume(model, optim, ckpt_path=os.path.join(resume_dir, "checkpoints", "last.pt"))

    # Train
    best_val = None
    for epoch in range(tracker.epoch, cfg["trainer"]["max_epochs"]):
        tracker.epoch = epoch
        tr_loss, tr_acc = train_epoch(model, train_loader, optim, tracker)
        if (epoch + 1) % cfg["trainer"]["val_every_epochs"] == 0:
            val_loss, val_acc = validate(model, val_loader, tracker)
            tracker.log(val_loss=val_loss, val_acc=val_acc, train_loss=tr_loss, train_acc=tr_acc, lr=cfg["optim"]["lr"])

            # Early stopping & best checkpoint
            metric = val_loss if tracker.early.cfg.mode == "min" else val_acc
            is_best = (best_val is None) or ((metric < best_val) if tracker.early.cfg.mode == "min" else (metric > best_val))
            if is_best:
                best_val = metric
                tracker.best_metric = best_val
            tracker.save_checkpoint(model, optim, is_best=is_best)
            tracker.early.update(val_loss if tracker.early.cfg.mode == "min" else val_acc)
            if tracker.early.should_stop:
                print("[tracker] Early stopping fired.")
                break

    tracker.close()
    print(f"[tracker] Done. Run dir: {tracker.run_dir}")

if __name__ == "__main__":
    main()
