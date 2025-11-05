"""
verify_feature_forward.py

Loads an extracted features .pt file and runs a forward through:
- LXMERT (with real features)
- ViLT (with original image + question)
- Fusion MLP to produce logits

This verifies that:
- FasterRCNN features are compatible with LXMERT
- LXMERT loads and can consume real features of shape [1, num_boxes, feat_dim]
- Full fusion forward pass completes w/o shape errors
"""

import torch
from transformers import ViltProcessor, ViltModel, LxmertModel, LxmertTokenizer
from PIL import Image
import argparse
import os

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load processor/models
    vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    vilt_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm").to(device)
    lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    lxmert_model = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased").to(device)

    # Load saved features
    feat_path = args.feat
    assert os.path.exists(feat_path), f"{feat_path} not found"
    data = torch.load(feat_path)
    boxes = data["boxes"]            # [num_boxes,4]
    feats = data["features"]         # [num_boxes, feat_dim]
    image_size = data.get("image_size", None)

    print("Loaded features:", feats.shape, boxes.shape, "image_size:", image_size)

    # Move to device and add batch dim
    feats_b = feats.unsqueeze(0).to(device)   # [1, num_boxes, feat_dim]
    boxes_b = boxes.unsqueeze(0).to(device)   # [1, num_boxes, 4]

    # Create a sample question and a corresponding image for ViLT test (image can be any)
    image = Image.open(args.image).convert("RGB") if args.image else None
    question = args.question or "What is in the image?"

    # ViLT inputs (if image provided)
    if image is not None:
        vilt_inputs = vilt_processor(images=image, text=question, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
        vilt_inputs = {k: v.to(device) for k,v in vilt_inputs.items()}
        with torch.no_grad():
            vilt_out = vilt_model(**vilt_inputs)
        vilt_emb = vilt_out.pooler_output  # [1,768]
        print("ViLT embedding shape:", vilt_emb.shape)
    else:
        vilt_emb = torch.zeros((1,768), device=device)
        print("ViLT skipped (no image provided) -> using zero-vector")

    # Prepare LXMERT text inputs
    lxmert_inputs = lxmert_tokenizer(question, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
    lxmert_inputs = {k: v.to(device) for k,v in lxmert_inputs.items()}

    # Pass to LXMERT with real features
    with torch.no_grad():
        lxmert_out = lxmert_model(
            **lxmert_inputs,
            visual_feats=feats_b,
            visual_pos=boxes_b
        )
    lxmert_emb = lxmert_out.pooled_output  # [1,768]
    print("LXMERT embedding shape:", lxmert_emb.shape)

    # Fusion test (concat + small MLP)
    fusion_in = torch.cat([vilt_emb, lxmert_emb], dim=1)  # [1, 1536]
    mlp = torch.nn.Sequential(
        torch.nn.Linear(fusion_in.shape[1], 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1000)  # suppose 1000 answers
    ).to(device)
    with torch.no_grad():
        logits = mlp(fusion_in)
    print("Logits shape:", logits.shape)
    print("✅ Verify forward succeeded. Everything is compatible.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat", type=str, required=True, help="Path to saved .pt features (from extract_fasterrcnn_features.py)")
    parser.add_argument("--image", type=str, default=None, help="Optional image path to run ViLT on (for full test).")
    parser.add_argument("--question", type=str, default="What is in the image?", help="Test question")
    args = parser.parse_args()
    main(args)
