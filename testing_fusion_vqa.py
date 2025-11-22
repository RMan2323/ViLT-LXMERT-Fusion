# testing_fusion_vqa.py
"""
Final Evaluation for Fine-Tuned ViLT, Fine-Tuned LXMERT,
and Fusion Model (pretrained ViLT + pretrained LXMERT).
This version removes ALL rescale warnings by ensuring ViLT
always receives a tensor instead of a PIL image.
"""

import os
import torch
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from transformers import ViltModel, ViltProcessor
from transformers import LxmertModel, LxmertTokenizer

from vqa_dataset import VQADataset
from vilt_lxmert_fusion import ViLT_LXMERT_Fusion
import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_msg(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}\n"
    print(line.strip())
    with open("testing_test.txt", "a") as f:
        f.write(line)


# ============================================================
# Basic transform (same as vqa_dataset.py)
# ============================================================
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])


# ============================================================
# Dataset + DataLoader
# ============================================================
# VAL_CSV = "Dataset/dataset_test2015.csv" # Dataset/dataset_Val2014_with_cp.csv
# IMAGE_ROOT = "Dataset/test2015/" # Dataset/val2014/
# FEATURE_DIR = "extracted_feats_test" # extracted_feats_val

BATCH_SIZE = 64
NUM_ANSWERS = 1000
MAX_SAMPLES = None

# val_dataset = VQADataset(
#     VAL_CSV,
#     IMAGE_ROOT,
#     feature_dir=FEATURE_DIR,
#     max_samples=MAX_SAMPLES
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=os.cpu_count(),
#     prefetch_factor=4,
#     pin_memory=True,
#     persistent_workers = True
# )


criterion = nn.CrossEntropyLoss()


# ============================================================
# Load Fine-Tuned ViLT + Head
# ============================================================
vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm").to(device)
vilt_head_ckpt = torch.load("checkpoints_vilt/best_vilt_head.ckpt", map_location=device)

vilt.load_state_dict(vilt_head_ckpt["vilt_state"], strict=False)

vilt_classifier = nn.Sequential(
    nn.Linear(768, 1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1024, NUM_ANSWERS)
).to(device)

vilt_classifier.load_state_dict(vilt_head_ckpt["head_state"])


vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
if hasattr(vilt_processor.image_processor, "do_rescale"):
    vilt_processor.image_processor.do_rescale = False
    print("✅ Set ViLT image processor do_rescale=False")


# ============================================================
# Load Fine-Tuned LXMERT + Head
# ============================================================
lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased").to(device)
lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

lx_ckpt = torch.load("checkpoints_lxmert/best_lxmert_head.ckpt", map_location=device)
lxmert.load_state_dict(lx_ckpt["lxmert_state"], strict=False)

lxmert_classifier = nn.Sequential(
    nn.Linear(768, 1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1024, NUM_ANSWERS)
).to(device)

lxmert_classifier.load_state_dict(lx_ckpt["head_state"])

# ============================================================
# Pretrained (NON-FINE-TUNED) ViLT + simple classifier
# ============================================================
vilt_pre = ViltModel.from_pretrained("dandelin/vilt-b32-mlm").to(device)

vilt_pre_classifier = nn.Linear(768, NUM_ANSWERS).to(device)
# No weights loaded → stays random (pure pretrained backbone)
print("✅ Loaded pretrained ViLT (no fine-tuning)")

# ============================================================
# Pretrained (NON-FINE-TUNED) LXMERT + simple classifier
# ============================================================
lxmert_pre = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased").to(device)

lxmert_pre_classifier = nn.Linear(768, NUM_ANSWERS).to(device)
print("✅ Loaded pretrained LXMERT (no fine-tuning)")




# ============================================================
# Load Fusion Model (pretrained encoders)
# ============================================================
fusion = ViLT_LXMERT_Fusion(
    num_answers=NUM_ANSWERS,
    freeze_encoders=True
).to(device)

fusion_ckpt = torch.load("checkpoints_train_on_fine_tuned_deeper/best_model.ckpt", map_location=device) # checkpoints_train_on_fine_tuned_deeper # checkpoints_old_at_start
fusion.load_state_dict(fusion_ckpt["model_state_dict"], strict=False)
print("✅ Loaded fine-tuned fusion head")


# ============================================================
# Evaluation function
# ============================================================
@torch.no_grad()
def evaluate(model_type, loader):
    correct = 0
    total = 0
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Evaluating {model_type}", ncols=100)

    for batch in pbar:
        labels = batch["answer_idx"].to(device)

        # ====================================================
        # VI LT
        # ====================================================
        if model_type == "ViLT":
            vilt_inputs = {
                k: v.to(device)
                for k, v in batch["vilt"].items()
            }

            out = vilt(**vilt_inputs, return_dict=True)
            logits = vilt_classifier(out.pooler_output)

        # ====================================================
        # LXMERT
        # ====================================================
        elif model_type == "LXMERT":
            if batch["visual_feats"] is None:
                print("[⚠] Missing visual_feats for LXMERT batch → skip")
                continue

            lx_inputs = {
                k: v.to(device)
                for k, v in batch["lxmert"].items()
            }
            feats = batch["visual_feats"].to(device)
            pos = batch["visual_pos"].to(device)

            out = lxmert(
                **lx_inputs,
                visual_feats=feats.float(),
                visual_pos=pos.float(),
                return_dict=True
            )
            logits = lxmert_classifier(out.pooled_output)


        # ====================================================
        # PRETRAINED VILT
        # ====================================================
        elif model_type == "ViLT_pretrained":
            vilt_inputs = {
                k: v.to(device)
                for k, v in batch["vilt"].items()
            }
            out = vilt_pre(**vilt_inputs, return_dict=True)
            logits = vilt_pre_classifier(out.pooler_output)

        # ====================================================
        # PRETRAINED LXMERT
        # ====================================================
        elif model_type == "LXMERT_pretrained":
            lx_inputs = {
                k: v.to(device)
                for k, v in batch["lxmert"].items()
            }
            feats = batch["visual_feats"].to(device).float()
            pos = batch["visual_pos"].to(device).float()

            out = lxmert_pre(
                **lx_inputs,
                visual_feats=feats,
                visual_pos=pos,
                return_dict=True
            )
            logits = lxmert_pre_classifier(out.pooled_output)



        # ====================================================
        # FUSION (pretrained encoders)
        # ====================================================
        else:
            vilt_inputs = {
                k: v.to(device)
                for k, v in batch["vilt"].items()
            }
            lxmert_inputs = {
                k: v.to(device)
                for k, v in batch["lxmert"].items()
            }

            feats = batch["visual_feats"]
            pos = batch["visual_pos"]

            if feats is not None:
                feats = feats.to(device).float()
                pos = pos.to(device).float()

            logits = fusion(
                vilt_inputs,
                lxmert_inputs,
                feats,
                pos
            )

        # =========================
        # Loss + accuracy
        # =========================
        loss = criterion(logits, labels)
        running_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        return 0.0, float("inf")

    return 100 * correct / total, running_loss / total

def run_eval_for(csv_path, csv_name, image_root, feature_dir):
    print(f"\n========== Evaluating on {csv_name} ==========")
    log_msg(f"\n========== Evaluating on {csv_name} ==========")
    
        # Shutdown old workers (IMPORTANT)
    try:
        if 'val_loader' in globals() and val_loader is not None:
            if hasattr(val_loader, '_iterator') and val_loader._iterator is not None:
                val_loader._iterator._shutdown_workers()
    except:
        pass

    val_dataset = VQADataset(
        csv_path,
        image_root,
        feature_dir=feature_dir,
        max_samples=MAX_SAMPLES
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True
    )

    model_list = [
        # "ViLT",
        # "LXMERT",
        # "Fusion",
        "ViLT_pretrained",
        "LXMERT_pretrained"
    ]

    results = {}

    for m in model_list:
        acc, loss = evaluate(m, val_loader)
        results[m] = (acc, loss)
        print(f"{m}: {acc:.2f}% | Loss: {loss:.4f}")
        log_msg(f"{m}: {acc:.2f}% | Loss: {loss:.4f}")


    return results

# ============================================================
# RUN evaluations on BOTH CSV files
# ============================================================

results_test = run_eval_for(
    "Dataset/dataset_test2015.csv",
    "TEST2015",
    "Dataset/test2015/",
    "extracted_feats_test"
)

results_val = run_eval_for(
    "Dataset/dataset_Val2014_with_cp.csv",
    "VAL2014",
    "Dataset/val2014/",
    "extracted_feats_val"
)

# VAL_CSV = "Dataset/dataset_test2015.csv" # Dataset/dataset_Val2014_with_cp.csv
# IMAGE_ROOT = "Dataset/test2015/" # Dataset/val2014/
# FEATURE_DIR = "extracted_feats_test" # extracted_feats_val


log_msg("TEST RESULTS: " + str(results_test))
log_msg("VAL RESULTS: " + str(results_val))

# # ============================================================
# # RUN all evaluations
# # ============================================================
# vilt_acc, vilt_loss = evaluate("ViLT")
# lx_acc, lx_loss = evaluate("LXMERT")
# vilt_pre_acc, vilt_pre_loss = evaluate("ViLT_pretrained")
# lx_pre_acc, lx_pre_loss = evaluate("LXMERT_pretrained")

# fusion_acc, fusion_loss = evaluate("Fusion")


# print("\n📊 Comparative Results:")
# print(f"ViLT    → {vilt_acc:.2f}% | Loss: {vilt_loss:.4f}")
# print(f"LXMERT  → {lx_acc:.2f}% | Loss: {lx_loss:.4f}")
# print(f"ViLT (pretrained) → {vilt_pre_acc:.2f}% | Loss: {vilt_pre_loss:.4f}")
# print(f"LXMERT (pretrained) → {lx_pre_acc:.2f}% | Loss: {lx_pre_loss:.4f}")
# print(f"Fusion  → {fusion_acc:.2f}% | Loss: {fusion_loss:.4f}")


# log_msg("\n📊 Comparative Results:")
# log_msg(f"ViLT    → {vilt_acc:.2f}% | Loss: {vilt_loss:.4f}")
# log_msg(f"LXMERT  → {lx_acc:.2f}% | Loss: {lx_loss:.4f}")
# log_msg(f"ViLT (pretrained) → {vilt_pre_acc:.2f}% | Loss: {vilt_pre_loss:.4f}")
# log_msg(f"LXMERT (pretrained) → {lx_pre_acc:.2f}% | Loss: {lx_pre_loss:.4f}")
# log_msg(f"Fusion  → {fusion_acc:.2f}% | Loss: {fusion_loss:.4f}")
