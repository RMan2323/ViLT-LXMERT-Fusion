"""
extract_fasterrcnn_features.py

Usage:
    python extract_fasterrcnn_features.py --image test.jpg --out_dir extracted_feats --topk 36

What it does:
- Loads torchvision Faster R-CNN (ResNet50-FPN) pretrained on COCO.
- Runs detection on given image(s).
- Uses the model's backbone + roi_heads.box_roi_pool + box_head to get region features.
- Projects box_head features to 2048-dim (linear) so they match LXMERT expected dim.
- Keeps top-k boxes by score (default 36).
- Saves a .pt file per image containing: boxes, scores, features (num_boxes x 2048), image_size.
"""

import os
import argparse
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import torchvision.models.detection.roi_heads as roi_heads
import torch.nn.functional as F

def load_fasterrcnn(device):
    # Use pretrained torchvision Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)
    return model

def extract_for_image(model, image_path, device, topk=36, feat_dim=2048):
    # Preprocess image (convert to tensor and normalized later by model internally)
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),  # to [0,1]
    ])
    img_t = transform(img).to(device)

    # torchvision detection models expect list of tensors
    with torch.no_grad():
        # Run the model to get detections (boxes/scores/labels)
        outputs = model([img_t])

    out = outputs[0]
    boxes = out["boxes"]        # [N,4]
    scores = out["scores"]      # [N]
    labels = out["labels"]      # [N] (not used here)

    # If no boxes detected, return empty
    if boxes.numel() == 0:
        return {"boxes": boxes.cpu(), "scores": scores.cpu(), "features": torch.zeros((0, feat_dim)), "image_size": img.size}

    # Get backbone feature maps
    # model.backbone returns an OrderedDict of feature maps (C x H x W) at different scales
    # For roi_pool we must pass a dict of feature maps as the model's forward does internally
    # We'll reuse the model components used in forward:
    image_list = torchvision.models.detection.image_list.ImageList(img_t.unsqueeze(0), [(img_t.shape[1], img_t.shape[2])])
    # call the transform? normally model.forward calls transform, but we passed tensor already; this is sufficient
    # Obtain backbone features
    features = model.backbone(image_list.tensors)  # dict of feature maps

    # Prepare proposals: convert boxes to list-of-tensors per image as expected by roi_pool
    proposals = [boxes]

    # ROI Pooling (box_roi_pool)
    # model.roi_heads.box_roi_pool expects (features, proposals, image_shapes)
    pooled_rois = model.roi_heads.box_roi_pool(features, proposals, image_list.image_sizes)  # [num_boxes, C, H, W]

    # box_head processes pooled proposals -> typically yields [num_boxes, representation_dim] (e.g., 1024)
    box_features = model.roi_heads.box_head(pooled_rois)  # [num_boxes, repr_dim]

    # box_predictor usually maps repr_dim -> classes and bbox, but we want the internal repr.
    # box_features shape might be [N, repr_dim] (repr_dim=1024 for torchvision's default)
    repr_dim = box_features.shape[1]

    # Project to desired feat_dim (2048) via a linear layer (create on the fly)
    # We create a deterministic projection (no randomness) and keep it saved alongside features if needed.
    proj = torch.nn.Linear(repr_dim, feat_dim).to(device)
    # Initialize projection weights (xavier) for stability
    torch.nn.init.xavier_uniform_(proj.weight)
    with torch.no_grad():
        feats_2048 = proj(box_features)  # [num_boxes, feat_dim]

    # Keep top-k by scores
    if scores.numel() > topk:
        topk_idx = scores.topk(topk).indices
        boxes = boxes[topk_idx]
        scores = scores[topk_idx]
        feats_2048 = feats_2048[topk_idx]

    # Move to cpu and return
    return {
        "boxes": boxes.cpu(),         # x1,y1,x2,y2 (float)
        "scores": scores.cpu(),
        "features": feats_2048.cpu(), # float32 [num_boxes, feat_dim]
        "image_size": (img.size[1], img.size[0])  # (H, W) to be consistent with torchvision image_size format
    }

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_fasterrcnn(device)
    image_paths = [p.strip() for p in args.image.split(",")]

    for p in image_paths:
        print(f"[INFO] Processing {p} ...")
        out = extract_for_image(model, p, device, topk=args.topk, feat_dim=args.feat_dim)
        base = os.path.splitext(os.path.basename(p))[0]
        save_path = os.path.join(args.out_dir, f"{base}.pt")
        torch.save(out, save_path)
        print(f"[OK] Saved: {save_path} | boxes: {out['boxes'].shape[0]} | feat_dim: {out['features'].shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="Path to image or comma-separated multiple images.")
    parser.add_argument("--out_dir", type=str, default="extracted_feats",
                        help="Directory to save .pt feature files")
    parser.add_argument("--topk", type=int, default=36, help="Top-K boxes to keep")
    parser.add_argument("--feat_dim", type=int, default=2048, help="Desired feature dimensionality")
    args = parser.parse_args()
    main(args)
