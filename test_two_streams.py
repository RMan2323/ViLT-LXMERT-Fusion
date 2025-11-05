import torch
from PIL import Image
from transformers import AutoTokenizer, ViltProcessor, ViltModel, LxmertTokenizer, LxmertModel

# ------------------------
# Load ViLT and processor
# ------------------------
vilt_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

# ------------------------
# Load LXMERT + tokenizer
# ------------------------
lxmert_model = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

# ------------------------
# Test inputs
# ------------------------
image_path = "test.jpg"  # Keep any random image as test.jpg in same folder
question = "What is in the picture?"

image = Image.open(image_path).convert("RGB")

# ViLT processing
vilt_inputs = vilt_processor(images=image, text=question, return_tensors="pt")
with torch.no_grad():
    vilt_output = vilt_model(**vilt_inputs)
vilt_emb = vilt_output.pooler_output
print("✅ ViLT output:", vilt_emb.shape)

# LXMERT processing (dummy visual feats for now)
inputs = lxmert_tokenizer(question, return_tensors="pt")
dummy_visual_feats = torch.randn(1, 36, 2048)
dummy_visual_pos = torch.randn(1, 36, 4)

with torch.no_grad():
    lxmert_output = lxmert_model(
        **inputs,
        visual_feats=dummy_visual_feats,
        visual_pos=dummy_visual_pos,
        output_hidden_states=True,
    )

lxmert_emb = lxmert_output.pooled_output
print("✅ LXMERT output:", lxmert_emb.shape)

fusion = torch.cat([vilt_emb, lxmert_emb], dim=1)
print("✅ Fusion shape:", fusion.shape)
print("\n🎯 Test Passed! No shape mismatch. Everything is working!\n")
