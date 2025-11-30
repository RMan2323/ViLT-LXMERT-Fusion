import os
import torch
from extract_fasterrcnn_features_Test import load_fasterrcnn, extract_for_image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load FasterRCNN only once ✅
model = load_fasterrcnn(device)

# Paths
# images_list = "images_list.txt"
images_list = "images_list_full_test.txt"
out_dir = "extracted_feats_test"
os.makedirs(out_dir, exist_ok=True)

# Read image list
with open(images_list, "r") as f:
    paths = [line.strip() for line in f.readlines()]

print(f"Total images to process: {len(paths)}")

for img_path in tqdm(paths, desc="Extracting features"):
    base = os.path.splitext(os.path.basename(img_path))[0]
    save_file = os.path.join(out_dir, f"{base}.pt")
    
    if os.path.exists(save_file):
        continue

    try:
        out = extract_for_image(model, img_path, device, topk=36, feat_dim=2048)
        torch.save(out, save_file)
    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")
        continue


# for img_path in paths:
    # base = os.path.splitext(os.path.basename(img_path))[0]
    # save_file = os.path.join(out_dir, f"{base}.pt")

    # # Skip already extracted features ✅
    # if os.path.exists(save_file):
    #     print(f"[SKIP] Already exists: {save_file}")
    #     continue

    # print(f"[RUN] {img_path}")
    # out = extract_for_image(model, img_path, device, topk=36, feat_dim=2048)

    # torch.save(out, save_file)
    # print(f"[OK] Saved: {save_file} | {out['features'].shape}")
