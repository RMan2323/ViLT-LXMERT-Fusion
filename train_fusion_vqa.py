# train_fusion_vqa.py
"""
Training script for ViLT + LXMERT fusion (debug mode).
- Uses pre-extracted Faster-RCNN features if present under extracted_feats/
- Fixed randomness (seed=42) and MAX_SAMPLES=100 for reproducible mini-experiments
- Robust to missing feature files (will skip bad batches and fallback to zeros)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Fix seeds for reproducibility (Priyanshu Rao requested deterministic sample selection)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Local imports (files must keep same names)
from vqa_dataset import VQADataset
from vilt_lxmert_fusion import ViLT_LXMERT_Fusion
from torch.serialization import add_safe_globals
import pytorch_lightning.callbacks.model_checkpoint as pl_ckpt

# ==== Setup ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Hyperparameters ===
BATCH_SIZE = 8
EPOCHS = 1
LR = 1e-4
DATASET_CSV = "Dataset/dataset_train2014_filtered.csv" # Changing to Dataset/dataset_train2014_filtered.csv from original Dataset/dataset_train2014_with_cp.csv
IMAGE_ROOT = "Dataset/train2014/"
FEATURE_DIR = "extracted_feats"   # where your .pt features are saved
NUM_ANSWERS = 1000
MAX_SAMPLES = 100   # keep small and reproducible for tests (Priyanshu wanted 100)

# === Checkpoint paths ===
os.makedirs("checkpoints", exist_ok=True)
PRETRAINED_CKPT = "checkpoints/vilt_vqa.ckpt"
LAST_EPOCH_CKPT = "checkpoints/last_epoch.ckpt"
BEST_MODEL_CKPT = "checkpoints/best_model.ckpt"

# === Collate function (robust) ===
def collate_fn(batch):
    # Remove None entries (dataset may return None for unreadable rows)
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    collated = {"vilt": {}, "lxmert": {}}
    try:
        # Stack dict tensors (vilt & lxmert)
        for k in batch[0]["vilt"].keys():
            collated["vilt"][k] = torch.stack([b["vilt"][k] for b in batch], dim=0)
        for k in batch[0]["lxmert"].keys():
            collated["lxmert"][k] = torch.stack([b["lxmert"][k] for b in batch], dim=0)

        # visual_feats and visual_pos are (36, feat_dim) per sample -> stack into (B, 36, feat_dim)
        collated["visual_feats"] = torch.stack([b["visual_feats"] for b in batch], dim=0)
        collated["visual_pos"] = torch.stack([b["visual_pos"] for b in batch], dim=0)

        # answers
        collated["answer_idx"] = torch.stack([b["answer_idx"] for b in batch], dim=0)
    except Exception as e:
        print(f"[Collate Error] {e}. Skipping this batch.")
        return None

    return collated

# === Load data ===
print("Loading Dataset...")
train_dataset = VQADataset(DATASET_CSV, IMAGE_ROOT,
                          max_samples=MAX_SAMPLES,
                          feature_dir=FEATURE_DIR)

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=2,
                          pin_memory=True)

print(f"Dataset size (samples used): {len(train_dataset)}")
if len(train_dataset) == 0:
    raise RuntimeError("No samples in dataset. Check CSV and paths.")

# === Model ===
# freeze_encoders=True keeps memory low (only projection & fusion trained). Change if you want to finetune.
model = ViLT_LXMERT_Fusion(num_answers=NUM_ANSWERS, freeze_encoders=True).to(device)
criterion = nn.CrossEntropyLoss()

# Train only parameters that require grad
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LR)

# === Load pretrained backbone if available ===
start_epoch = 0
best_loss = float("inf")

if os.path.exists(PRETRAINED_CKPT):
    print(f"🔹 Loading pretrained weights from {PRETRAINED_CKPT}")
    add_safe_globals([pl_ckpt.ModelCheckpoint])
    try:
        checkpoint = torch.load(PRETRAINED_CKPT, map_location=device, weights_only=False)
        if "state_dict" in checkpoint:
            state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Loaded pretrained backbone (ViLT + LXMERT)")
    except Exception as e:
        print(f"[Warning] Could not load pretrained backbone: {e}. Continuing with HF weights.")
else:
    print("⚠️ No pretrained checkpoint found. Training fusion from scratch")

# === Optionally resume training (guard optimizer mismatch) ===
if os.path.exists(LAST_EPOCH_CKPT):
    print(f"🔄 Found checkpoint {LAST_EPOCH_CKPT}. Attempting to resume...")
    try:
        checkpoint = torch.load(LAST_EPOCH_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        # Safe optimizer load: check param groups count match
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("✅ Loaded optimizer state from checkpoint")
            except Exception as e:
                print(f"[Optimizer load warning] {e}. Continuing with fresh optimizer.")
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", best_loss)
    except Exception as e:
        print(f"[Resume warning] Failed to resume fully: {e}. Starting from scratch.")

# === Training Loop ===
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0.0
    steps = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=120)
    for batch in pbar:
        # Skip empty/invalid batches from collate
        if batch is None:
            continue

        # Move inputs to device
        vilt_inputs = {k: v.to(device) for k, v in batch["vilt"].items()}
        lxmert_inputs = {k: v.to(device) for k, v in batch["lxmert"].items()}
        labels = batch["answer_idx"].long().to(device)

        # Visual features (CPU->GPU). Ensure float32
        visual_feats = batch.get("visual_feats", None)
        visual_pos = batch.get("visual_pos", None)
        if visual_feats is not None:
            visual_feats = visual_feats.to(device).float()   # [B, 36, 2048]
            visual_pos = visual_pos.to(device).float()       # [B, 36, 4]

        # Debug: print whether visual features are present (non-zero)
        if visual_feats is not None:
            # compute mean abs value to check if it's zero (fallback)
            mean_abs = visual_feats.abs().mean().item()
            uses_real_feats = mean_abs > 0.0
        else:
            uses_real_feats = False

        optimizer.zero_grad()

        # Try to forward with visual_feats if model accepts them. Otherwise, fallback.
        try:
            if visual_feats is not None:
                # many forward signatures expect (vilt_inputs, lxmert_inputs, visual_feats=..., visual_pos=...)
                logits = model(vilt_inputs, lxmert_inputs, visual_feats=visual_feats, visual_pos=visual_pos)
            else:
                logits = model(vilt_inputs, lxmert_inputs)
        except TypeError:
            # model.forward likely doesn't accept visual_feats named args -> call without them
            logits = model(vilt_inputs, lxmert_inputs)
            uses_real_feats = False
        except Exception as ex:
            print(f"[Forward Error] {ex}. Skipping batch.")
            continue

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

        pbar.set_postfix({
            "loss": f"{(total_loss/steps):.4f}",
            "uses_feats": "yes" if uses_real_feats else "no"
        })

    if steps == 0:
        print("[Warning] No valid training steps in epoch (all batches skipped).")
        continue

    avg_loss = total_loss / steps
    print(f"\n✅ Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # === Save last checkpoint (safe save) ===
    try:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "best_loss": best_loss
        }, LAST_EPOCH_CKPT)
        print(f"💾 Saved checkpoint: {LAST_EPOCH_CKPT}")
    except Exception as e:
        print(f"[Save Error] Could not save checkpoint: {e}")

    # === Save best model ===
    if avg_loss < best_loss:
        best_loss = avg_loss
        try:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, BEST_MODEL_CKPT)
            print(f"🏆 New Best Model Saved: {BEST_MODEL_CKPT}")
        except Exception as e:
            print(f"[Save Error] Could not save best model: {e}")

print("🎉 Training Complete!")
