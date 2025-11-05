# vqa_dataset.py
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from transformers import ViltProcessor, LxmertTokenizer
from torchvision import transforms

max_question_length = max(len(q.split()) for q in self.data["question"])

class VQADataset(Dataset):
    def __init__(self, csv_path, image_root, max_samples=None):
        self.data = pd.read_csv(csv_path, low_memory=False)
        if max_samples:
            self.data = self.data.sample(max_samples).reset_index(drop=True)

        self.image_root = image_root

        # Preload models
        self.vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

        # Define transform for backup / fusion image
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)
    

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

        # ✅ Let ViLT handle resizing and normalization automatically
        image = self.transform(image)  # ensures 3×384×384
        vilt_inputs = self.vilt_processor(
            images=image,
            text=question,
            do_rescale=False,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_question_length
        )

        # ✅ LXMERT text encoding
        lxmert_inputs = self.lxmert_tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=max_question_length,
            return_tensors="pt"
        )

        return {
            "vilt": {k: v.squeeze(0) for k, v in vilt_inputs.items()},
            "lxmert": {k: v.squeeze(0) for k, v in lxmert_inputs.items()},
            "answer_idx": answer_idx,
        }