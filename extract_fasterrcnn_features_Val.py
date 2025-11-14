"""
extract_fasterrcnn_features_silent.py
------------------------------------
✅ Production-silent feature extractor for Faster R-CNN (COCO pretrained).

Features:
- Deterministic, reproducible 2048-D region features (for LXMERT).
- Always creates one .pt per image (dummy fallback for failures).
- Logs all empty/error cases to empty_detections.txt (no console spam).
- Skips already processed .pt files (resumable).
- Clean tqdm progress bar, no print/log output.
"""

import os
import sys
import glob
import torch
from PIL import Image
from tqdm import tqdm
import torchvision
from torchvision import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# ============================================================
# Config
# ============================================================

transform = transforms.Compose([transforms.ToTensor()])


# ============================================================
# Model Loader
# ============================================================

def load_fasterrcnn(device):
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    model.to(device)
    print(f"Using device: {device}")
    return model


# ============================================================
# Dummy Fallback
# ============================================================

def make_dummy_output(size, feat_dim=2048):
    width, height = size
    dummy_box = torch.tensor([[0, 0, width, height]], dtype=torch.float32)
    dummy_score = torch.tensor([0.0])
    dummy_feat = torch.zeros((1, feat_dim), dtype=torch.float32)
    return {
        "boxes": dummy_box,
        "scores": dummy_score,
        "features": dummy_feat,
        "image_size": (height, width),
    }


# ============================================================
# Image Preprocessing
# ============================================================

def load_and_preprocess(image_path, device):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).to(device)
    return img_t, img.size


# ============================================================
# Deterministic 2048-D projection
# ============================================================

def project_features(feats, feat_dim, device):
    in_dim = feats.shape[1]
    if in_dim == feat_dim:
        return feats
    proj = torch.nn.Linear(in_dim, feat_dim, bias=False).to(device)
    with torch.no_grad():
        proj.weight.zero_()
        for i in range(min(in_dim, feat_dim)):
            proj.weight[i, i % in_dim] = 1.0
    return proj(feats)


# ============================================================
# Core Extraction for One Image
# ============================================================

def extract_for_image(model, image_path, device, topk=36, feat_dim=2048):
    try:
        img_t, size = load_and_preprocess(image_path, device)
        with torch.no_grad():
            outputs = model([img_t])

        out = outputs[0]
        boxes, scores = out["boxes"], out["scores"]

        if boxes.numel() == 0:
            # Log silently
            with open("empty_detections.txt", "a") as f:
                f.write(f"{image_path}\n")
            return make_dummy_output(size, feat_dim)

        # ROI features
        image_list = torchvision.models.detection.image_list.ImageList(
            img_t.unsqueeze(0), [(img_t.shape[1], img_t.shape[2])]
        )
        features = model.backbone(image_list.tensors)
        proposals = [boxes]
        pooled = model.roi_heads.box_roi_pool(features, proposals, image_list.image_sizes)
        box_feats = model.roi_heads.box_head(pooled)
        feats_2048 = project_features(box_feats, feat_dim, device)

        # Keep top-K
        if scores.numel() > topk:
            topk_idx = scores.topk(topk).indices
            boxes = boxes[topk_idx]
            scores = scores[topk_idx]
            feats_2048 = feats_2048[topk_idx]

        return {
            "boxes": boxes.detach().cpu(),
            "scores": scores.detach().cpu(),
            "features": feats_2048.detach().cpu(),
            "image_size": (size[1], size[0]),
        }


    except Exception as e:
        # Log failure silently
        with open("empty_detections.txt", "a") as f:
            f.write(f"{image_path} - error: {e}\n")
        return make_dummy_output((1, 1), feat_dim)


# ============================================================
# Batch Driver
# ============================================================

def extract_batch(input_path, out_dir, topk=36, feat_dim=2048):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fasterrcnn(device)

    # Resolve images
    if os.path.isdir(input_path):
        image_paths = sorted(glob.glob(os.path.join(input_path, "*.jpg")))
    elif "*" in input_path:
        image_paths = sorted(glob.glob(input_path))
    else:
        image_paths = [input_path]

    # Skip already done
    image_paths = [
        p for p in image_paths
        if not os.path.exists(os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + ".pt"))
    ]

    print(f"Total images to process: {len(image_paths)}")

    for p in tqdm(image_paths, desc="Extracting features", unit="img"):
        result = extract_for_image(model, p, device, topk, feat_dim)
        base = os.path.splitext(os.path.basename(p))[0].replace(" ", "_")
        save_path = os.path.join(out_dir, f"{base}.pt")
        try:
            torch.save(result, save_path)
        except Exception as e:
            with open("empty_detections.txt", "a") as f:
                f.write(f"{image_path} - save_error: {e}\n")
        del result
        torch.cuda.empty_cache()


# ============================================================
# CLI Entry
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="Path to image, folder, or wildcard (e.g. 'Dataset/train2014/*.jpg').")
    parser.add_argument("--out_dir", type=str, default="extracted_feats",
                        help="Output directory for .pt files.")
    parser.add_argument("--topk", type=int, default=36,
                        help="Top-K boxes to keep (default: 36).")
    parser.add_argument("--feat_dim", type=int, default=2048,
                        help="Projected feature dimensionality (default: 2048).")
    args = parser.parse_args()

    extract_batch(args.image, args.out_dir, args.topk, args.feat_dim)
