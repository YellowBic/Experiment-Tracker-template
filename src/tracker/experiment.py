
import os, sys, time, json, csv, random, shutil, socket, platform, hashlib, subprocess
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None

try:
    import shortuuid
    _gen_id = lambda: shortuuid.ShortUUID().random(length=6)
except Exception:
    import uuid
    _gen_id = lambda: uuid.uuid4().hex[:6]

def _now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def _to_yaml(cfg: Dict[str, Any]) -> str:
    if OmegaConf is not None and isinstance(cfg, (dict,)):
        try:
            return OmegaConf.to_yaml(OmegaConf.create(cfg))
        except Exception:
            pass
    # Fallback: basic YAML-ish
    import yaml
    return yaml.dump(cfg, sort_keys=False)

def seeds_everywhere(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(pref: str = "auto"):
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def git_info() -> Dict[str, Any]:
    info = {"is_git": False, "commit": None, "branch": None, "dirty": None}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        info.update({"is_git": True, "commit": commit, "branch": branch, "dirty": dirty})
    except Exception:
        pass
    return info

def env_info() -> Dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": sys.version.replace("\n", " "),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }

class CSVLogger:
    def __init__(self, path):
        self.path = path
        self.file = open(path, "w", newline="")
        self.writer = None

    def log(self, row: Dict[str, Any]):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row.keys()))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass

class JSONLLogger:
    def __init__(self, path):
        self.file = open(path, "w")

    def log(self, row: Dict[str, Any]):
        self.file.write(json.dumps(row) + "\n")
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass

class WandbLogger:
    def __init__(self, run_dir: str, project: str = "exp-tracker"):
        self.enabled = False
        try:
            import wandb  # noqa
            if os.environ.get("WANDB_API_KEY"):
                self.wandb = wandb
                self.wandb.init(project=project, dir=run_dir, config={}, reinit=True)
                self.enabled = True
        except Exception:
            self.enabled = False

    def log(self, row: Dict[str, Any], step: Optional[int] = None):
        if self.enabled:
            self.wandb.log(row, step=step)

    def close(self):
        if self.enabled:
            self.wandb.finish()

@dataclass
class EarlyStoppingCfg:
    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 5
    min_delta: float = 1e-4

class EarlyStopping:
    def __init__(self, cfg: EarlyStoppingCfg):
        self.cfg = cfg
        self.best = None
        self.count = 0
        self.should_stop = False
        self.cmp = (lambda a, b: a < b - cfg.min_delta) if cfg.mode == "min" else (lambda a, b: a > b + cfg.min_delta)

    def update(self, value: float):
        if self.best is None or self.cmp(value, self.best):
            self.best = value
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.cfg.patience:
                self.should_stop = True

class ExperimentTracker:
    def __init__(self, cfg: Dict[str, Any]):
        # resolve device & seeds
        self.cfg = cfg
        seeds_everywhere(cfg.get("seed", 0), cfg.get("deterministic", True))
        self.device = get_device(cfg.get("device", "auto"))

        # setup run dir
        runs_root = cfg.get("paths", {}).get("runs_root", "./runs")
        name = cfg.get("name", "exp")
        run_id = _gen_id()
        self.run_dir = os.path.abspath(os.path.join(runs_root, f"{_now()}_{name}_{run_id}"))
        os.makedirs(self.run_dir, exist_ok=True)

        # save config & metadata
        meta = {
            "name": name,
            "id": run_id,
            "time": _now(),
            "git": git_info(),
            "env": env_info(),
            "cfg": cfg,
        }
        with open(os.path.join(self.run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        with open(os.path.join(self.run_dir, "config.yaml"), "w") as f:
            f.write(_to_yaml(cfg))

        # loggers
        self.csv = CSVLogger(os.path.join(self.run_dir, "metrics.csv")) if cfg.get("loggers", {}).get("csv", True) else None
        self.jsonl = JSONLLogger(os.path.join(self.run_dir, "metrics.jsonl")) if cfg.get("loggers", {}).get("jsonl", True) else None
        self.tb = SummaryWriter(self.run_dir) if cfg.get("loggers", {}).get("tensorboard", True) else None
        self.wandb = WandbLogger(self.run_dir, project=cfg.get("loggers", {}).get("project", "exp-tracker")) if cfg.get("loggers", {}).get("wandb", False) else None

        # step counters
        self.global_step = 0
        self.epoch = 0

        # AMP
        self.mixed_precision = bool(cfg.get("mixed_precision", True) and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # Early stopping
        es_cfg = cfg.get("trainer", {}).get("early_stopping", {})
        self.early = EarlyStopping(EarlyStoppingCfg(**es_cfg))

        # checkpoint paths
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.best_metric = None

    def log(self, **kwargs):
        row = dict(step=self.global_step, epoch=self.epoch, **kwargs)
        if self.csv: self.csv.log(row)
        if self.jsonl: self.jsonl.log(row)
        if self.tb:
            for k, v in kwargs.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, self.global_step)
        if self.wandb: self.wandb.log(kwargs, step=self.global_step)

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, is_best: bool = False):
        payload = {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "best_metric": self.best_metric,
            "cfg": self.cfg,
        }
        last_path = os.path.join(self.ckpt_dir, "last.pt")
        torch.save(payload, last_path)
        if is_best:
            best_path = os.path.join(self.ckpt_dir, "best.pt")
            shutil.copy2(last_path, best_path)
        return last_path

    def try_resume(self, model: nn.Module, optimizer: torch.optim.Optimizer, ckpt_path: Optional[str] = None):
        """Resume from given path or from self.run_dir/checkpoints/last.pt if exists."""
        path = ckpt_path
        if path is None:
            path = os.path.join(self.ckpt_dir, "last.pt")
        if os.path.isfile(path):
            payload = torch.load(path, map_location="cpu")
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optim_state"])
            self.epoch = payload.get("epoch", 0)
            self.global_step = payload.get("global_step", 0)
            if self.scaler and payload.get("scaler_state"):
                self.scaler.load_state_dict(payload["scaler_state"])
            self.best_metric = payload.get("best_metric", None)
            print(f"[tracker] Resumed from {path} at epoch={self.epoch}, step={self.global_step}")
            return True
        return False

    def close(self):
        for lg in [self.csv, self.jsonl, self.tb, self.wandb]:
            if lg:
                try: lg.close()
                except Exception: pass

