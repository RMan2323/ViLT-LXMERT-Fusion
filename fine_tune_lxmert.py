# fine_tune_lxmert.py
"""
Fast fine-tuning LXMERT classification head (Priyanshu Rao)
with AMP, logging, validation toggle, checkpoint saving, and safe shutdown
"""

import os, torch, random, numpy as np, datetime
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LxmertModel
from vqa_dataset import VQADataset

# === Config ===
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4
DATASET_CSV = "Dataset/dataset_train2014_with_cp.csv"
IMAGE_ROOT = "Dataset/train2014/"
FEATURE_DIR = "extracted_feats"
VAL_CSV = "Dataset/dataset_Val2014_with_cp.csv"
VAL_IMAGE_ROOT = "Dataset/val2014/"
VAL_FEATURE_DIR = "extracted_feats_val"
NUM_ANSWERS = 1000
MAX_SAMPLES = None
VAL_MAX_SAMPLE = None
VAL_EVERY = 2  # validate every N epochs

LOG_FILE = "lxmert_training_log.txt"
LAST_EPOCH_CKPT = "checkpoints_lxmert/last_lxmert_head.ckpt"
BEST_MODEL_CKPT  = "checkpoints_lxmert/best_lxmert_head.ckpt"

# === Logger ===
def log_msg(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}\n"
    print(line.strip())
    with open(LOG_FILE, "a") as f:
        f.write(line)

# === Setup ===
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🟢 Using device: {device}")

# === Collate ===
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    collated = {"lxmert": {}}
    for k in batch[0]["lxmert"].keys():
        collated["lxmert"][k] = torch.stack([b["lxmert"][k] for b in batch])
    collated["visual_feats"] = torch.stack([b["visual_feats"] for b in batch])
    collated["visual_pos"] = torch.stack([b["visual_pos"] for b in batch])
    collated["answer_idx"] = torch.stack([b["answer_idx"] for b in batch])
    return collated

# === Data ===
train_dataset = VQADataset(DATASET_CSV, IMAGE_ROOT, max_samples=MAX_SAMPLES, feature_dir=FEATURE_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=os.cpu_count(),
                          pin_memory=True, prefetch_factor=4, persistent_workers=True)
val_dataset = VQADataset(VAL_CSV, VAL_IMAGE_ROOT, max_samples=VAL_MAX_SAMPLE, feature_dir=VAL_FEATURE_DIR)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn, num_workers=os.cpu_count(),
                        pin_memory=True, prefetch_factor=4, persistent_workers=True)

# === Model ===
lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased").to(device)
for p in lxmert.parameters(): p.requires_grad = False  # freeze encoder
head = nn.Sequential(
    nn.Linear(768, 1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1024, NUM_ANSWERS)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(head.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()

os.makedirs("checkpoints_lxmert", exist_ok=True)
best_loss = float("inf")

# === Evaluation ===
@torch.no_grad()
def evaluate():
    lxmert.eval(); head.eval()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(val_loader, desc="Evaluating", leave=False):
        if batch is None: continue
        inputs = {k: v.to(device, non_blocking=True) for k,v in batch["lxmert"].items()}
        feats = batch["visual_feats"].to(device, non_blocking=True).float()
        pos = batch["visual_pos"].to(device, non_blocking=True).float()
        labels = batch["answer_idx"].long().to(device, non_blocking=True)

        out = lxmert(**inputs, visual_feats=feats, visual_pos=pos, return_dict=True)
        logits = head(out.pooled_output)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100 * correct / total

# === Training ===
for epoch in range(EPOCHS):
    lxmert.eval(); head.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=120)
    for batch in pbar:
        if batch is None: continue
        inputs = {k: v.to(device, non_blocking=True) for k,v in batch["lxmert"].items()}
        feats = batch["visual_feats"].to(device, non_blocking=True).float()
        pos = batch["visual_pos"].to(device, non_blocking=True).float()
        labels = batch["answer_idx"].long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            out = lxmert(**inputs, visual_feats=feats, visual_pos=pos, return_dict=True)
            logits = head(out.pooled_output)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if total % (BATCH_SIZE * 100) == 0:
            pbar.set_postfix({"Loss": f"{total_loss/total:.4f}", "Acc": f"{100*correct/total:.2f}%"})

    avg_train_loss = total_loss / total
    train_acc = 100 * correct / total
    do_validation = ((epoch + 1) % VAL_EVERY == 0 or epoch == EPOCHS - 1)

    # --- Validation / Skip logic ---
    if do_validation:
        val_loss, val_acc = evaluate()
        log_msg(f"Epoch {epoch+1}: Train {avg_train_loss:.4f}/{train_acc:.2f}% | Val {val_loss:.4f}/{val_acc:.2f}%")
    else:
        val_loss, val_acc = best_loss, 0
        log_msg(f"Epoch {epoch+1}: Train {avg_train_loss:.4f}/{train_acc:.2f}% (Validation skipped, last best_loss={best_loss:.4f})")

    # --- Always save last epoch ---
    torch.save({
        "epoch": epoch,
        "lxmert_state": lxmert.state_dict(),
        "head_state": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
        "best_loss": best_loss
    }, LAST_EPOCH_CKPT)
    log_msg(f"💾 Saved checkpoint: {LAST_EPOCH_CKPT}")

    # --- Update and save best model ---
    if do_validation and val_loss < best_loss:
        best_loss = val_loss
        torch.save({
            "epoch": epoch,
            "lxmert_state": lxmert.state_dict(),
            "head_state": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "best_loss": best_loss
        }, BEST_MODEL_CKPT)
        log_msg(f"🏆 New Best Model Saved (val_loss={val_loss:.4f})")

# === Graceful Worker Shutdown ===
for loader in [train_loader, val_loader]:
    if hasattr(loader, "_iterator") and loader._iterator is not None:
        loader._iterator._shutdown_workers()

print("🎉 LXMERT fine-tuning complete! (All workers closed safely ✅)")
