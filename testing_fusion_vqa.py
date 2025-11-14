"""
Final Evaluation for Fine-Tuned ViLT, LXMERT, and Fusion Models (Priyanshu Rao)
Consistent preprocessing and evaluation pipeline.
"""

import os, torch, datetime
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from vqa_dataset import VQADataset
from transformers import ViltModel, ViltProcessor, LxmertModel
from vilt_lxmert_fusion import ViLT_LXMERT_Fusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Config ===
VAL_CSV = "Dataset/dataset_Val2014_with_cp.csv"
IMAGE_ROOT = "Dataset/val2014/"
FEATURE_DIR = "extracted_feats_val"
BATCH_SIZE = 64
NUM_ANSWERS = 1000
LOG_FILE = "evaluation_results.txt"
MAX_SAMPLES = 256

def log_msg(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}\n"
    print(line.strip())
    with open(LOG_FILE, "a") as f:
        f.write(line)

# === Compatibility Patch ===
import transformers
if hasattr(transformers.models.vilt.processing_vilt, "ViltProcessor"):
    old_call = transformers.models.vilt.processing_vilt.ViltProcessor.__call__
    def patched_call(self, *args, **kwargs):
        if "do_rescale" in kwargs:
            kwargs.pop("do_rescale")
        return old_call(self, *args, **kwargs)
    transformers.models.vilt.processing_vilt.ViltProcessor.__call__ = patched_call
    print("✅ Patched ViltProcessor to ignore 'do_rescale' argument")

# === Dataset ===
val_dataset = VQADataset(VAL_CSV, IMAGE_ROOT, feature_dir=FEATURE_DIR, max_samples = MAX_SAMPLES)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=os.cpu_count(), pin_memory=True)

criterion = nn.CrossEntropyLoss()

# === Models ===
# --- ViLT ---
vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm").to(device)
vilt_head = torch.load("checkpoints_vilt/best_vilt_head.ckpt", map_location=device)
vilt.load_state_dict(vilt_head["vilt_state"], strict=False)
vilt_classifier = nn.Sequential(
    nn.Linear(768, 1024), nn.ReLU(), nn.Dropout(0.3), nn.Linear(1024, NUM_ANSWERS)
).to(device)
vilt_classifier.load_state_dict(vilt_head["head_state"])

# Ensure ViLT processor consistency
vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
if hasattr(vilt_processor.image_processor, "do_rescale"):
    vilt_processor.image_processor.do_rescale = False
    print("✅ Set ViLT image processor do_rescale=False (align with training)")

# --- LXMERT ---
lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased").to(device)
lxmert_head = torch.load("checkpoints_lxmert/best_lxmert_head.ckpt", map_location=device)
lxmert.load_state_dict(lxmert_head["lxmert_state"], strict=False)
lxmert_classifier = nn.Sequential(
    nn.Linear(768, 1024), nn.ReLU(), nn.Dropout(0.3), nn.Linear(1024, NUM_ANSWERS)
).to(device)
lxmert_classifier.load_state_dict(lxmert_head["head_state"])

# --- Fusion ---
fusion = ViLT_LXMERT_Fusion(num_answers=NUM_ANSWERS, freeze_encoders=True).to(device)
fusion_ckpt = torch.load("checkpoints_new_train_newer/best_model.ckpt", map_location=device)
fusion.load_state_dict(fusion_ckpt["model_state_dict"], strict=False)
print("✅ Loaded fine-tuned fusion model checkpoint")

# === Evaluation Function ===
@torch.no_grad()
def evaluate(model_type):
    correct, total, total_loss = 0, 0, 0
    model_map = {"ViLT": (vilt, vilt_classifier),
                 "LXMERT": (lxmert, lxmert_classifier),
                 "Fusion": (fusion, None)}

    for batch in tqdm(val_loader, desc=f"Evaluating {model_type}", ncols=100):
        labels = batch["answer_idx"].long().to(device)
        if model_type == "ViLT":
            inputs = {k: v.to(device) for k, v in batch["vilt"].items()}
            out = model_map["ViLT"][0](**inputs, return_dict=True)
            logits = model_map["ViLT"][1](out.pooler_output)

        elif model_type == "LXMERT":
            inputs = {k: v.to(device) for k, v in batch["lxmert"].items()}
            feats = batch["visual_feats"].to(device).float()
            pos = batch["visual_pos"].to(device).float()
            out = model_map["LXMERT"][0](**inputs, visual_feats=feats, visual_pos=pos, return_dict=True)
            logits = model_map["LXMERT"][1](out.pooled_output)

        else:
            vilt_inputs = {k: v.to(device) for k, v in batch["vilt"].items()}
            lxmert_inputs = {k: v.to(device) for k, v in batch["lxmert"].items()}
            feats = batch["visual_feats"].to(device).float()
            pos = batch["visual_pos"].to(device).float()
            logits = model_map["Fusion"][0](vilt_inputs, lxmert_inputs, feats, pos)

        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

    return 100 * correct / total, total_loss / total

# === Run Evaluations ===
vilt_acc, vilt_loss = evaluate("ViLT")
lxmert_acc, lxmert_loss = evaluate("LXMERT")
fusion_acc, fusion_loss = evaluate("Fusion")

# === Log Results ===
log_msg("\n📊 Comparative Validation Results:")
log_msg(f"   ViLT   → {vilt_acc:.2f}% | Loss {vilt_loss:.4f}")
log_msg(f"   LXMERT → {lxmert_acc:.2f}% | Loss {lxmert_loss:.4f}")
log_msg(f"   Fusion → {fusion_acc:.2f}% | Loss {fusion_loss:.4f}")
