# train_fusion_vqa.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import datetime
from torch.cuda.amp import autocast, GradScaler

from vqa_dataset import VQADataset
from vilt_lxmert_fusion import ViLT_LXMERT_Fusion

# ==== Setup ====
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Hyperparams ===
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-4
DATASET_CSV = "Dataset/dataset_train2014_with_cp.csv"
IMAGE_ROOT = "Dataset/train2014/"
FEATURE_DIR = "extracted_feats"
NUM_ANSWERS = 1000
MAX_SAMPLES = None
VAL_MAX_SAMPLE = None

os.makedirs("checkpoints_new_train_newer", exist_ok=True)
LAST_EPOCH_CKPT = "checkpoints_new_train_newer/last_epoch.ckpt"
BEST_MODEL_CKPT = "checkpoints_new_train_newer/best_model.ckpt"

def log_msg(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}\n"
    print(line.strip())
    with open("training_log_new_newer.txt", "a") as f:
        f.write(line)

# === Collate function (robust) ===
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    collated = {"vilt": {}, "lxmert": {}}
    try:
        for k in batch[0]["vilt"].keys():
            collated["vilt"][k] = torch.stack([b["vilt"][k] for b in batch], dim=0)
        for k in batch[0]["lxmert"].keys():
            collated["lxmert"][k] = torch.stack([b["lxmert"][k] for b in batch], dim=0)

        # If ANY sample is missing visual_feats -> set to None for whole batch (fusion will fallback to ViLT tokens)
        if any(b["visual_feats"] is None for b in batch):
            collated["visual_feats"] = None
            collated["visual_pos"] = None
        else:
            collated["visual_feats"] = torch.stack([b["visual_feats"] for b in batch], dim=0)
            collated["visual_pos"] = torch.stack([b["visual_pos"] for b in batch], dim=0)

        collated["answer_idx"] = torch.stack([b["answer_idx"] for b in batch], dim=0)
    except Exception as e:
        print(f"[Collate Error] {e}. Skipping this batch.")
        return None

    return collated

# === Load data ===
print("Loading Dataset...")
train_dataset = VQADataset(DATASET_CSV, IMAGE_ROOT, max_samples=MAX_SAMPLES, feature_dir=FEATURE_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=os.cpu_count(), pin_memory=True, prefetch_factor=4, persistent_workers=True)

VAL_CSV = "Dataset/dataset_Val2014_with_cp.csv"
VAL_IMAGE_ROOT = "Dataset/val2014/"
VAL_FEATURE_DIR = "extracted_feats_val"
val_dataset = VQADataset(VAL_CSV, VAL_IMAGE_ROOT, max_samples=VAL_MAX_SAMPLE, feature_dir=VAL_FEATURE_DIR)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=os.cpu_count(), pin_memory=True, prefetch_factor=4, persistent_workers=True)

print(f"Validation dataset size: {len(val_dataset)}")
if len(train_dataset) == 0:
    raise RuntimeError("No samples in dataset. Check CSV and paths.")

# === Model ===
model = ViLT_LXMERT_Fusion(num_answers=NUM_ANSWERS, freeze_encoders=True).to(device)
criterion = nn.CrossEntropyLoss()
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LR)

start_epoch = 0
best_loss = float("inf")

if os.path.exists(LAST_EPOCH_CKPT):
    print(f"🔄 Found checkpoint {LAST_EPOCH_CKPT}. Attempting to resume...")
    try:
        checkpoint = torch.load(LAST_EPOCH_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
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

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", ncols=100)
        for batch in pbar:
            if batch is None:
                continue

            vilt_inputs = {k: v.to(device) for k, v in batch["vilt"].items()}
            lxmert_inputs = {k: v.to(device) for k, v in batch["lxmert"].items()}
            labels = batch["answer_idx"].long().to(device)

            visual_feats = batch.get("visual_feats")
            visual_pos = batch.get("visual_pos")
            if visual_feats is not None:
                visual_feats = visual_feats.to(device).float()
                visual_pos = visual_pos.to(device).float()

            logits = model(vilt_inputs, lxmert_inputs,
                        visual_feats=visual_feats,
                        visual_pos=visual_pos)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({"val_loss": f"{(total_loss / steps):.4f}"})
    if steps == 0:
        return float("inf")

    avg_loss = total_loss / steps
    print(f"📊 Validation Loss: {avg_loss:.4f}")
    log_msg(f"📊 Validation Loss: {avg_loss:.4f}")
    return avg_loss

# === Training Loop ===
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0.0
    steps = 0
    scaler = GradScaler()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=120)
    for batch in pbar:
        if batch is None:
            continue

        vilt_inputs = {k: v.to(device) for k, v in batch["vilt"].items()}
        lxmert_inputs = {k: v.to(device) for k, v in batch["lxmert"].items()}
        labels = batch["answer_idx"].long().to(device)

        visual_feats = batch.get("visual_feats")
        visual_pos = batch.get("visual_pos")
        if visual_feats is not None:
            visual_feats = visual_feats.to(device).float()
            visual_pos = visual_pos.to(device).float()
            uses_real_feats = visual_feats.abs().mean().item() > 0.0
        else:
            uses_real_feats = False

        optimizer.zero_grad()
        with autocast():
            try:
                logits = model(vilt_inputs, lxmert_inputs,
                            visual_feats=visual_feats,
                            visual_pos=visual_pos)
            except Exception as ex:
                print(f"[Forward Error] {ex}. Skipping batch.")
                continue
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        steps += 1
        pbar.set_postfix({
            "loss": f"{(total_loss / steps):.4f}",
            "uses_feats": "yes" if uses_real_feats else "no"
        })

    if steps == 0:
        print("[Warning] No valid training steps in epoch (all batches skipped).")
        continue

    avg_train_loss = total_loss / steps
    print(f"\n✅ Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")
    log_msg(f"\n✅ Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")

    do_validation = (epoch + 1) % 2 == 0
    if do_validation:
        val_loss_to_save = evaluate(model, val_loader, criterion, device)
        log_msg(f"📊 Validation Loss (Epoch {epoch+1}): {val_loss_to_save:.4f}")
    else:
        val_loss_to_save = best_loss
        log_msg(f"⏭️ Skipping validation at epoch {epoch+1} for speed (keeping last val_loss={val_loss_to_save:.4f}).")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": val_loss_to_save,
        "best_loss": best_loss
    }, LAST_EPOCH_CKPT)
    log_msg(f"💾 Saved checkpoint: {LAST_EPOCH_CKPT}")

    if do_validation and val_loss_to_save < best_loss:
        best_loss = val_loss_to_save
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": val_loss_to_save,
            "best_loss": best_loss
        }, BEST_MODEL_CKPT)
        log_msg(f"🏆 New Best Model Saved (val_loss={val_loss_to_save:.4f})")

print("🎉 Training Complete!")
log_msg("🎉 Training Complete!")

for loader in [train_loader, val_loader]:
    if hasattr(loader, '_iterator') and loader._iterator is not None:
        loader._iterator._shutdown_workers()
