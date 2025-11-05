# train_fusion_vqa.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
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
DATASET_CSV = "Dataset/dataset_train2014_with_cp.csv"
IMAGE_ROOT = "Dataset/train2014/"
FEATURE_DIR = "extracted_feats"  # ✅ required for Faster-RCNN features
NUM_ANSWERS = 1000
MAX_SAMPLES = 800  # same as your previous setting

# === Checkpoint paths ===
os.makedirs("checkpoints", exist_ok=True)
PRETRAINED_CKPT = "checkpoints/vilt_vqa.ckpt"
LAST_EPOCH_CKPT = "checkpoints/last_epoch.ckpt"
BEST_MODEL_CKPT = "checkpoints/best_model.ckpt"

# ✅ Collate function added (needed for features stacking)
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    collated = {'vilt': {}, 'lxmert': {}}
    for k in batch[0]['vilt'].keys():
        collated['vilt'][k] = torch.stack([b['vilt'][k] for b in batch])
    for k in batch[0]['lxmert'].keys():
        collated['lxmert'][k] = torch.stack([b['lxmert'][k] for b in batch])

    collated['visual_feats'] = torch.stack([b['visual_feats'] for b in batch])
    collated['visual_pos'] = torch.stack([b['visual_pos'] for b in batch])
    collated['answer_idx'] = torch.stack([b['answer_idx'] for b in batch])
    return collated

# === Load data ===
print("Loading Dataset...")
train_dataset = VQADataset(DATASET_CSV, IMAGE_ROOT,
                           max_samples=MAX_SAMPLES,
                           feature_dir=FEATURE_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn,
                          num_workers=2, pin_memory=True)

# === Model ===
model = ViLT_LXMERT_Fusion(num_answers=NUM_ANSWERS).to(device)
criterion = nn.CrossEntropyLoss()

# ✅ Train only fusion model parameters (base encoders frozen)
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LR)

# === Load pretrained checkpoint if exists ===
start_epoch = 0
best_loss = float("inf")

if os.path.exists(PRETRAINED_CKPT):
    print(f"🔹 Loading pretrained weights from {PRETRAINED_CKPT}")
    add_safe_globals([pl_ckpt.ModelCheckpoint])

    checkpoint = torch.load(PRETRAINED_CKPT, map_location=device, weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    else:
        state_dict = checkpoint.get("model_state_dict", checkpoint)

    model.load_state_dict(state_dict, strict=False)
    print("✅ Loaded pretrained backbone (ViLT + LXMERT)")
else:
    print("⚠️ No pretrained checkpoint found. Training fusion from scratch")

# === Optionally resume full training ===
if os.path.exists(LAST_EPOCH_CKPT):
    print(f"🔄 Resuming from {LAST_EPOCH_CKPT}")
    checkpoint = torch.load(LAST_EPOCH_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint.get("best_loss", best_loss)

# === Training Loop ===
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        vilt_inputs = {k: v.to(device) for k, v in batch["vilt"].items()}
        lxmert_inputs = {k: v.to(device) for k, v in batch["lxmert"].items()}
        labels = batch["answer_idx"].long().to(device)

        optimizer.zero_grad()
        
        # ✅ Correct forwarding
        logits = model(vilt_inputs, lxmert_inputs)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()


        total_loss += loss.item()
        steps += 1

    avg_loss = total_loss / steps
    print(f"✅ Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # === Save last checkpoint ===
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "best_loss": best_loss
    }, LAST_EPOCH_CKPT)

    print(f"💾 Saved checkpoint: {LAST_EPOCH_CKPT}")

    # === Save best model ===
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }, BEST_MODEL_CKPT)
        print(f"🏆 New Best Model Saved: {BEST_MODEL_CKPT}")

print("🎉 Training Complete!")
