# evaluate_test.py
import os
import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader

from vqa_dataset import VQADataset
from vilt_lxmert_fusion import ViLT_LXMERT_Fusion


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ================================
# Paths (EDIT THESE)
# ================================
TEST_CSV = "Dataset/dataset_test2015.csv"          # <-- your new csv
TEST_IMG_ROOT = "Dataset/test2015/"                # folder containing images
FEATURE_DIR = "extracted_feats_test"               # or same extracted_feats
BEST_MODEL = "checkpoints_train_on_fine_tuned_deeper/best_model.ckpt"
OUTPUT_CSV = "vqa_test_predictions.csv"
NUM_ANSWERS = 1000

# ================================
# Load Dataset
# ================================
test_dataset = VQADataset(
    TEST_CSV,
    TEST_IMG_ROOT,
    max_samples=None,
    feature_dir=FEATURE_DIR
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=os.cpu_count(),
    collate_fn=lambda b: b[0] if len(b) == 1 else VQA_collate(b)
)


# Collate function identical to train one
def VQA_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    collated = {"vilt": {}, "lxmert": {}}
    try:
        for k in batch[0]["vilt"].keys():
            collated["vilt"][k] = torch.stack([b["vilt"][k] for b in batch], dim=0)
        for k in batch[0]["lxmert"].keys():
            collated["lxmert"][k] = torch.stack([b["lxmert"][k] for b in batch], dim=0)

        # missing feats → fallback
        if any(b["visual_feats"] is None for b in batch):
            collated["visual_feats"] = None
            collated["visual_pos"] = None
        else:
            collated["visual_feats"] = torch.stack([b["visual_feats"] for b in batch], dim=0)
            collated["visual_pos"] = torch.stack([b["visual_pos"] for b in batch], dim=0)

        # In test, answer_idx = -1 → ignore
        collated["answer_idx"] = torch.stack([b["answer_idx"] for b in batch], dim=0)

    except Exception as e:
        print("[Collate Error]", e)
        return None

    return collated


# ================================
# Load Model
# ================================
model = ViLT_LXMERT_Fusion(
    num_answers=NUM_ANSWERS,
    freeze_encoders=True
).to(device)

print("Loading model checkpoint...")
ckpt = torch.load(BEST_MODEL, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

print("Model loaded!")

# ================================
# Evaluate (no labels)
# ================================
predictions = []
rows = test_dataset.data

with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader, desc="Predicting")):
        if batch is None:
            continue

        vilt_inputs = {k: v.to(device) for k, v in batch["vilt"].items()}
        lxmert_inputs = {k: v.to(device) for k, v in batch["lxmert"].items()}

        visual_feats = batch.get("visual_feats")
        visual_pos = batch.get("visual_pos")
        if visual_feats is not None:
            visual_feats = visual_feats.to(device).float()
            visual_pos = visual_pos.to(device).float()

        logits = model(
            vilt_inputs,
            lxmert_inputs,
            visual_feats=visual_feats,
            visual_pos=visual_pos
        )

        pred_idx = logits.argmax(dim=1).cpu().tolist()
        predictions.extend(pred_idx)

# ================================
# SAVE OUTPUT
# ================================
output_df = rows.copy()
output_df["pred"] = predictions

output_df.to_csv(OUTPUT_CSV, index=False)
print("Saved predictions to:", OUTPUT_CSV)
