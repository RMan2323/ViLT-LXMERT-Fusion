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
NUM_ANSWERS = 1000

# === Checkpoint paths ===
os.makedirs("checkpoints", exist_ok=True)
PRETRAINED_CKPT = "checkpoints/vilt_vqa.ckpt"
LAST_EPOCH_CKPT = "checkpoints/last_epoch.ckpt"
BEST_MODEL_CKPT = "checkpoints/best_model.ckpt"

# === Load data ===
train_dataset = VQADataset(DATASET_CSV, IMAGE_ROOT, max_samples=80000)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Model ===
model = ViLT_LXMERT_Fusion(num_answers=NUM_ANSWERS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.fusion.parameters(), lr=LR)

# === Load pretrained weights if available ===
start_epoch = 0
best_loss = float("inf")

# if os.path.exists(PRETRAINED_CKPT):
#     print(f"🔹 Loading pretrained weights from {PRETRAINED_CKPT}")
#     checkpoint = torch.load(PRETRAINED_CKPT, map_location=device)
#     model.load_state_dict(checkpoint["model_state_dict"], strict=False)
# else:
#     print("⚠️ No pretrained checkpoint found. Starting from scratch.")

# === Load pretrained weights if available ===
start_epoch = 0
best_loss = float("inf")

if os.path.exists(PRETRAINED_CKPT):
    print(f"🔹 Loading pretrained weights from {PRETRAINED_CKPT}")

    # ✅ Allow Lightning class
    add_safe_globals([pl_ckpt.ModelCheckpoint])

    # ✅ Load even if it’s a Lightning checkpoint
    checkpoint = torch.load(PRETRAINED_CKPT, map_location=device, weights_only=False)

    # ✅ Handle Lightning format
    if "state_dict" in checkpoint:
        print("⚡ Detected PyTorch Lightning checkpoint — extracting weights...")
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    else:
        state_dict = checkpoint.get("model_state_dict", checkpoint)

    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded pretrained weights from {PRETRAINED_CKPT}")
else:
    print("⚠️ No pretrained checkpoint found. Starting from scratch.")

# === Optionally resume from last checkpoint ===
if os.path.exists(LAST_EPOCH_CKPT):
    print(f"🔄 Resuming training from {LAST_EPOCH_CKPT}")
    # checkpoint = torch.load(LAST_EPOCH_CKPT, map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    add_safe_globals([pl_ckpt.ModelCheckpoint])

    # Load checkpoint (may be from Lightning)
    checkpoint = torch.load(PRETRAINED_CKPT, map_location=device, weights_only=False)

    # If it's a Lightning checkpoint, extract weights properly
    if "state_dict" in checkpoint:
        print("⚡ Detected PyTorch Lightning checkpoint — extracting weights...")
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    else:
        state_dict = checkpoint.get("model_state_dict", checkpoint)

    model.load_state_dict(state_dict, strict=False)
    
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint.get("best_loss", best_loss)

# === Training Loop ===
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        vilt_inputs = {k: v.to(device) for k, v in batch["vilt"].items()}
        lxmert_inputs = {k: v.to(device) for k, v in batch["lxmert"].items()}
        labels = batch["answer_idx"].long().to(device)

        optimizer.zero_grad()
        logits = model(vilt_inputs, lxmert_inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # === Save last epoch checkpoint ===
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "best_loss": best_loss,
    }, LAST_EPOCH_CKPT)
    print(f"💾 Saved checkpoint: {LAST_EPOCH_CKPT}")

    # === Save best model if improved ===
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, BEST_MODEL_CKPT)
        print(f"🏆 New best model saved: {BEST_MODEL_CKPT} (loss={best_loss:.4f})")

print("🎉 Training complete.")
