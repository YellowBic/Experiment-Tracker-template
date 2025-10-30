# Experiment Tracker — minimal, fast, and actually useful

A lightweight experiment tracker + Colab-friendly template. No servers, no lock-in. 
It logs to plain files (CSV/JSONL), TensorBoard, and optionally Weights & Biases if you set `WANDB_API_KEY`.

## Highlights
- 💾 **Reproducible runs**: saves config, git hash, environment, seeds.
- 🗂️ **Run directory per experiment**: `runs/<date>_<time>_<name>_<id>/` with all logs & checkpoints.
- ✍️ **Metrics**: CSV and JSONL; TensorBoard scalars; optional W&B.
- ⚙️ **AMP + checkpointing + early stopping + resume** baked in.
- 🧩 **Simple API** (`ExperimentTracker`) and an example (`train_example.py`) you can copy into any project.

## Quick start (Colab or local)
```bash
pip install -r requirements.txt

# Train example (FashionMNIST) with default config
python src/train_example.py

# Override any config from CLI
python src/train_example.py trainer.max_epochs=5 optim.lr=0.0005 data.batch_size=128 name=mnist_amp

# Resume from last checkpoint in a run directory
python src/train_example.py resume=/path/to/runs/2025-01-01_12-00-00_mnist_xxx

# Enable Weights & Biases (optional)
export WANDB_API_KEY=your_key
python src/train_example.py loggers.wandb=true loggers.project=my-cool-proj
```
Open TensorBoard (optional):
```bash
tensorboard --logdir runs
```

## Structure
```
configs/default.yaml
src/tracker/experiment.py
src/train_example.py
```
