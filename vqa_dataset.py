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
      - pre-extracted Faster-RCNN features (.pt)
    """

    def __init__(self, csv_path, image_root, max_samples=None, feature_dir="extracted_feats"):
        self.data = pd.read_csv(csv_path, low_memory=False)
        if max_samples:
            self.data = self.data.sample(max_samples).reset_index(drop=True)

        self.image_root = image_root
        self.feature_dir = feature_dir

        # ================================
        #   PROCESSORS
        # ================================
        self.vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

        # 🔥 VERY IMPORTANT: Disable ALL image transforms
        ip = self.vilt_processor.image_processor
        ip.do_normalize = False
        ip.do_rescale = False
        ip.do_resize = False
        ip.do_center_crop = False
        ip.do_convert_rgb = False

        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

        # ================================
        #   MANUAL IMAGE -> TENSOR
        # ================================
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),     # produces tensor in [0,1]
        ])

        # max question length
        self.max_question_length = max(len(str(q).split()) for q in self.data["question"])


    def __len__(self):
        return len(self.data)


    # ============================================================
    #  LOAD .PT REGION FEATURES  (same code, no change)
    # ============================================================
    def _load_preextracted_feats(self, image_path):
        base = os.path.splitext(os.path.basename(image_path))[0]
        candidates = [
            f"{base}.pt",
            os.path.join(self.feature_dir, f"{base}.pt"),
            os.path.join(self.feature_dir, base + ".pt")
        ]
        tokens = base.split("_")
        if tokens and tokens[-1].isdigit():
            candidates.append(os.path.join(self.feature_dir, tokens[-1] + ".pt"))

        feat_path = None
        for c in candidates:
            p = c if os.path.isabs(c) else os.path.join(self.feature_dir, os.path.basename(c))
            if os.path.exists(p):
                feat_path = p
                break

        if feat_path is None:
            print(f"[MISSING FEATS WARNING] {image_path}: No .pt file found. Returning (None, None).")
            return None, None

        data = torch.load(feat_path)
        feats = data.get("features", None)
        boxes = data.get("boxes", None)
        if feats is None or boxes is None:
            raise ValueError(f"Feature file {feat_path} missing keys.")

        feats = feats if isinstance(feats, torch.Tensor) else torch.tensor(feats, dtype=torch.float32)
        boxes = boxes if isinstance(boxes, torch.Tensor) else torch.tensor(boxes, dtype=torch.float32)

        num_boxes = 36
        if feats.shape[0] >= num_boxes:
            feats = feats[:num_boxes]
            boxes = boxes[:num_boxes]
        else:
            pad_n = num_boxes - feats.shape[0]
            feat_dim = feats.shape[1]
            feats = torch.cat([feats, torch.zeros((pad_n, feat_dim))], dim=0)
            boxes = torch.cat([boxes, torch.zeros((pad_n, 4))], dim=0)

        return feats, boxes


    # ============================================================
    #  MAIN GETITEM (this is where we fixed everything)
    # ============================================================
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_root, os.path.basename(row["image_path"]))
        question = str(row["question"])
        answer_idx = int(row["answer_idx"]) if row["answer_idx"] >= 0 else 0

        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image = Image.new("RGB", (384, 384), (0, 0, 0))

        # real image
        image_t = self.transform(image)

        # dummy image for HF API
        dummy = torch.zeros_like(image_t)

        # ViLT processor (dummy image to satisfy HF API)
        vilt_inputs = self.vilt_processor(
            images=dummy,
            text=question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_question_length
        )

        # replace dummy pixel values with real tensor
        vilt_inputs["pixel_values"] = image_t.unsqueeze(0)

        # squeeze batch dim
        vilt_inputs = {k: v.squeeze(0) for k, v in vilt_inputs.items()}

        # LXMERT
        lxmert_inputs = self.lxmert_tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt"
        )
        lxmert_inputs = {k: v.squeeze(0) for k, v in lxmert_inputs.items()}

        # features
        visual_feats, visual_pos = self._load_preextracted_feats(image_path)

        return {
            "vilt": vilt_inputs,
            "lxmert": lxmert_inputs,
            "visual_feats": visual_feats,
            "visual_pos": visual_pos,
            "answer_idx": torch.tensor(answer_idx, dtype=torch.long)
        }
