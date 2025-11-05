# vqa_dataset.py
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from transformers import ViltProcessor, LxmertTokenizer
from torchvision import transforms
import torch

max_question_length = 32

class VQADataset(Dataset):
    """
    VQADataset that loads:
      - question and answer idx from CSV
      - image (for ViLT processing)
      - pre-extracted Faster-RCNN features (.pt) located in `feature_dir` (default: extracted_feats/)
        The .pt file must contain dict with keys: "features" (N x 2048), "boxes" (N x 4), "scores" (N)
    """
    def __init__(self, csv_path, image_root, max_samples=None, feature_dir="extracted_feats"):
        self.data = pd.read_csv(csv_path, low_memory=False)
        if max_samples:
            self.data = self.data.sample(max_samples).reset_index(drop=True)

        self.image_root = image_root
        self.feature_dir = feature_dir

        # Preload processors / tokenizers
        self.vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

        # Minimal transform to ensure consistent size for ViLT processor
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def _load_preextracted_feats(self, image_path):
        """
        Attempt to load pre-extracted features for given image_path.
        Feature file is expected at: os.path.join(self.feature_dir, <basename_no_ext> + ".pt")
        Returns (features, boxes) as torch.Tensor (N, feat_dim), (N,4)
        If not found, returns zeros with shape (36,2048) and (36,4) respectively.
        """
        base = os.path.splitext(os.path.basename(image_path))[0]
        # The dataset may have names like 'COCO_train2014_000000123456.jpg' or simply '000000123456.jpg'.
        # We'll try a few naming conventions to find the features, but priority is basename.pt
        candidates = [
            f"{base}.pt",
            os.path.join(self.feature_dir, f"{base}.pt"),
            os.path.join(self.feature_dir, base + ".pt")
        ]
        # If base has prefixes, also try last numeric token
        tokens = base.split("_")
        if tokens and tokens[-1].isdigit():
            alt = tokens[-1] + ".pt"
            candidates.append(os.path.join(self.feature_dir, alt))

        # Check existence
        feat_path = None
        for c in candidates:
            p = c if os.path.isabs(c) else os.path.join(self.feature_dir, os.path.basename(c))
            if os.path.exists(p):
                feat_path = p
                break

        if feat_path is None:
            # Not found -> fallback
            # Return zeros (36 boxes, 2048 dim) as default expected by LXMERT
            print(f"[Warning] Feature file not found for image {image_path}. Expected under {self.feature_dir}. Using zero-features fallback.")
            feat_dim = 2048
            num_boxes = 36
            return torch.zeros((num_boxes, feat_dim), dtype=torch.float32), torch.zeros((num_boxes, 4), dtype=torch.float32)

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
            # pad
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

        # Load image as PIL
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[Warning] Could not load image: {image_path}. Using blank image instead.")
            image = Image.new("RGB", (384, 384), (0, 0, 0))

        # ViLT: use processor (we will pass PIL image through a light transform to ensure size)
        image_t = self.transform(image)  # 3x384x384 tensor
        # vilt_processor can accept PIL image or tensor; we pass tensor here so processor will treat accordingly
        vilt_inputs = self.vilt_processor(
            images=image_t,
            text=question,
            do_rescale=False,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_question_length
        )

        # LXMERT text encoding
        lxmert_inputs = self.lxmert_tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=max_question_length,
            return_tensors="pt"
        )

        # Load pre-extracted features (or fallback zeros)
        visual_feats, visual_pos = self._load_preextracted_feats(image_path)  # tensors [36,2048], [36,4]

        return {
            "vilt": {k: v.squeeze(0) for k, v in vilt_inputs.items()},
            "lxmert": {k: v.squeeze(0) for k, v in lxmert_inputs.items()},
            "visual_feats": visual_feats,   # [36, 2048] (cpu tensor)
            "visual_pos": visual_pos,       # [36, 4] (cpu tensor)
            "answer_idx": torch.tensor(answer_idx, dtype=torch.long)
        }
