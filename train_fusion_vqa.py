# train_fusion_vqa.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from vqa_dataset import VQADataset
from vilt_lxmert_fusion import ViLT_LXMERT_Fusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Hyperparameters ===
BATCH_SIZE = 4
EPOCHS = 1
LR = 1e-4
DATASET_CSV = "Dataset/dataset_train2014_with_cp.csv"
IMAGE_ROOT = "Dataset/train2014/"
NUM_ANSWERS = 1000  # based on your vocab size

# === Load data ===
train_dataset = VQADataset(DATASET_CSV, IMAGE_ROOT, max_samples=2000)  # small subset first
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Model ===
model = ViLT_LXMERT_Fusion(num_answers=NUM_ANSWERS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.fusion.parameters(), lr=LR)  # only train fusion head

# === Training ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        vilt_inputs = {k: v.to(device) for k, v in batch["vilt"].items()}
        lxmert_inputs = {k: v.to(device) for k, v in batch["lxmert"].items()}
        labels = torch.tensor(batch["answer_idx"], dtype=torch.long).to(device)

        optimizer.zero_grad()
        logits = model(vilt_inputs, lxmert_inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

print("Training complete.")
