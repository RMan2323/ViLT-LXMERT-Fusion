# vilt_lxmert_fusion.py

import torch
import torch.nn as nn
from transformers import ViltModel, LxmertModel

class ViLT_LXMERT_Fusion(nn.Module):
    def __init__(self, num_answers=1000, hidden_dim=768, fusion_dim=1024):
        super().__init__()

        # ✅ Load pretrained encoders
        self.vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")

        # ✅ Freeze encoders to train only fusion at first
        for p in self.vilt.parameters():
            p.requires_grad = False

        for p in self.lxmert.parameters():
            p.requires_grad = False

        # ✅ Fusion + Classification Head
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, num_answers)
        )

    def forward(self, vilt_inputs, lxmert_inputs):
        # ==========================
        #       ViLT Encoding
        # ==========================
        vilt_outputs = self.vilt(**vilt_inputs)
        vilt_emb = vilt_outputs.pooler_output  # Shape: [batch, 768]

        # ==========================
        #      LXMERT Encoding
        # (dummy visual input used)
        # ==========================
        batch_size = lxmert_inputs["input_ids"].shape[0]
        device = lxmert_inputs["input_ids"].device

        dummy_visual_feats = torch.zeros(
            (batch_size, 36, 2048), device=device
        )
        dummy_visual_pos = torch.zeros(
            (batch_size, 36, 4), device=device
        )

        lxmert_outputs = self.lxmert(
            **lxmert_inputs,
            visual_feats=dummy_visual_feats,
            visual_pos=dummy_visual_pos
        )
        lxmert_emb = lxmert_outputs.pooled_output  # Shape: [batch, 768]

        # ==========================
        #        Late Fusion
        # ==========================
        combined = torch.cat([vilt_emb, lxmert_emb], dim=1)
        logits = self.fusion(combined)
        return logits
