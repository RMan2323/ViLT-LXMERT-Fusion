# vqa_dataset.py
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from transformers import ViltProcessor, LxmertTokenizer
from torchvision import transforms
import torch

class VQADataset(Dataset):
    """
    VQADataset that loads:
      - question and answer idx from CSV
      - image (for ViLT processing)
      - pre-extracted Faster-RCNN features (.pt) located in `feature_dir`
        The .pt file must contain dict with keys: "features" (N x 2048), "boxes" (N x 4), "scores" (N)
    """
    def __init__(self, csv_path, image_root, max_samples=None, feature_dir="extracted_feats"):
        self.data = pd.read_csv(csv_path, low_memory=False)
        if max_samples:
            self.data = self.data.sample(max_samples).reset_index(drop=True)

        self.image_root = image_root
        self.feature_dir = feature_dir

        # Preload processors / tokenizers
        # (we keep vilt_processor here but ALWAYS pass a pre-converted tensor to it elsewhere)
        self.vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

        # Minimal transform to ensure consistent size for ViLT processor
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
        ])
        # compute a simple max length for padding/truncation
        self.max_question_length = max(len(str(q).split()) for q in self.data["question"])


    def __len__(self):
        return len(self.data)

    def _load_preextracted_feats(self, image_path):
        """
        Attempt to load pre-extracted features for given image_path.
        If not found -> return (None, None) and print a clear warning (user requested warnings instead of silent fallback).
        If present -> return (feats_tensor, boxes_tensor) padded/truncated to 36 boxes.
        """
        base = os.path.splitext(os.path.basename(image_path))[0]
        candidates = [
            f"{base}.pt",
            os.path.join(self.feature_dir, f"{base}.pt"),
            os.path.join(self.feature_dir, base + ".pt")
        ]
        tokens = base.split("_")
        if tokens and tokens[-1].isdigit():
            alt = tokens[-1] + ".pt"
            candidates.append(os.path.join(self.feature_dir, alt))

        feat_path = None
        for c in candidates:
            p = c if os.path.isabs(c) else os.path.join(self.feature_dir, os.path.basename(c))
            if os.path.exists(p):
                feat_path = p
                break

        if feat_path is None:
            # User asked: give warnings instead of silent zero fallback.
            print(f"[MISSING FEATS WARNING] Feature file not found for image {image_path}. Expected under {self.feature_dir}. Returning (None, None).")
            return None, None

        # Load and validate
        data = torch.load(feat_path)
        feats = data.get("features", None)
        boxes = data.get("boxes", None)
        if feats is None or boxes is None:
            raise ValueError(f"Feature file {feat_path} missing required keys 'features' and 'boxes'.")

        # Convert to torch tensors if not
        feats = feats if isinstance(feats, torch.Tensor) else torch.tensor(feats, dtype=torch.float32)
        boxes = boxes if isinstance(boxes, torch.Tensor) else torch.tensor(boxes, dtype=torch.float32)

        # Ensure shapes: pad/truncate to 36 boxes
        num_boxes = 36
        if feats.shape[0] >= num_boxes:
            feats = feats[:num_boxes]
            boxes = boxes[:num_boxes]
        else:
            pad_n = num_boxes - feats.shape[0]
            feat_dim = feats.shape[1]
            feats = torch.cat([feats, torch.zeros((pad_n, feat_dim), dtype=torch.float32)], dim=0)
            boxes = torch.cat([boxes, torch.zeros((pad_n, 4), dtype=torch.float32)], dim=0)

        return feats, boxes

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_root, os.path.basename(row["image_path"]))
        question = str(row["question"])
        answer_idx = int(row["answer_idx"]) if row["answer_idx"] >= 0 else 0

        # Load image as PIL (fallback to black image if load fails)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[Warning] Could not load image: {image_path}. Using blank image instead. ({e})")
            image = Image.new("RGB", (384, 384), (0, 0, 0))

        # IMPORTANT: convert to tensor BEFORE passing to processor to avoid HF rescale warnings.
        image_t = self.transform(image)  # 3x384x384 tensor (values in [0,1])

        vilt_inputs = self.vilt_processor(
            images=image_t,           # pass tensor to prevent processor rescale check/warning
            text=question,
            do_rescale=False,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_question_length
        )

        lxmert_inputs = self.lxmert_tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt"
        )

        # Load pre-extracted features (or None if missing)
        visual_feats, visual_pos = self._load_preextracted_feats(image_path)  # either tensors or (None, None)

        return {
            "vilt": {k: v.squeeze(0) for k, v in vilt_inputs.items()},
            "lxmert": {k: v.squeeze(0) for k, v in lxmert_inputs.items()},
            "visual_feats": visual_feats,   # either [36,2048] tensor or None
            "visual_pos": visual_pos,       # either [36,4] tensor or None
            "answer_idx": torch.tensor(answer_idx, dtype=torch.long)
        }
