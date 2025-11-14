# vilt_lxmert_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViltModel, LxmertModel

class ViLT_LXMERT_Fusion(nn.Module):
    def __init__(self, num_answers=1000, hidden_dim=768, fusion_dim=1024,
                 num_regions=36, project_to_dim=2048, freeze_encoders=True):
        """
        Lightweight fusion: use ViLT patch embeddings as visual features for LXMERT.
        - num_regions: number of visual regions to pool to (default 36 for comparability)
        - project_to_dim: project ViLT (768) -> LXMERT expected (2048)
        """
        super().__init__()

        self.num_regions = num_regions
        self.project_to_dim = project_to_dim
        self.hidden_dim = hidden_dim

        # Load pretrained encoders
        # === Load pretrained encoders ===
        self.vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")

        # === Load Priyanshu's fine-tuned checkpoints ===
        try:
            vilt_ckpt = torch.load("checkpoints_vilt/best_vilt_head.ckpt", map_location="cpu")
            lxmert_ckpt = torch.load("checkpoints_lxmert/best_lxmert_head.ckpt", map_location="cpu")
            self.vilt.load_state_dict(vilt_ckpt["vilt_state"], strict=False)
            self.lxmert.load_state_dict(lxmert_ckpt["lxmert_state"], strict=False)
            print("✅ Loaded fine-tuned ViLT & LXMERT encoder weights.")
        except Exception as e:
            print(f"⚠️ Could not load fine-tuned encoder weights: {e}")
            print("→ Falling back to pretrained encoders.")


        # Freeze encoders by default for low memory usage
        if freeze_encoders:
            for p in self.vilt.parameters():
                p.requires_grad = False
            for p in self.lxmert.parameters():
                p.requires_grad = False

        # Projection from ViLT token dim (768) -> LXMERT visual dim (2048)
        self.vilt_to_lxmert_proj = nn.Linear(hidden_dim, self.project_to_dim)

        # Fusion + Classification Head
        # self.fusion = nn.Sequential(
        #     nn.Linear(2 * hidden_dim, fusion_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(fusion_dim, num_answers)
        # )

        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_answers)
        )



    def _make_grid_boxes(self, batch_size, device):
        """
        Create normalized grid boxes for num_regions (we use 6x6 grid when num_regions==36).
        Boxes in [x1, y1, x2, y2] normalized to [0,1].
        """
        # Choose grid size: prefer square grid if possible
        n = self.num_regions
        # try to find divisors close to sqrt(n)
        import math
        side = int(math.sqrt(n))
        if side * side != n:
            # fallback to rectangular approx (e.g., 6x6 for 36)
            side = int(round(math.sqrt(n)))
        H = side
        W = int(math.ceil(n / H))
        # compute box coordinates
        boxes = []
        for i in range(H):
            for j in range(W):
                if len(boxes) >= n:
                    break
                x1 = j / W
                y1 = i / H
                x2 = (j + 1) / W
                y2 = (i + 1) / H
                boxes.append([x1, y1, x2, y2])
        boxes = torch.tensor(boxes, dtype=torch.float32, device=device)  # [num_regions,4]
        boxes = boxes.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_regions, 4]
        return boxes

    def forward(self, vilt_inputs, lxmert_inputs, visual_feats=None, visual_pos=None):
        """
        Forward pass for ViLT + LXMERT fusion.
        Priority:
        1. Use external visual features (.pt) if provided.
        2. Otherwise, generate pseudo-regions from ViLT tokens.
        """

        # ===== ViLT forward (text + image) =====
        vilt_outputs = self.vilt(**vilt_inputs, return_dict=True, output_hidden_states=False)
        vilt_emb = vilt_outputs.pooler_output  # [B, 768]

        if visual_feats is None or visual_pos is None:
            # ===== ViLT token-level feature pooling (fallback mode) =====
            last_hidden = vilt_outputs.last_hidden_state  # [B, seq_len, 768]
            token_feats = last_hidden[:, 1:, :]  # drop CLS token

            B, T, C = token_feats.shape
            token_feats_t = token_feats.permute(0, 2, 1)
            pooled = F.adaptive_avg_pool1d(token_feats_t, self.num_regions)
            pooled = pooled.permute(0, 2, 1)
            proj_feats = self.vilt_to_lxmert_proj(pooled)
            visual_pos = self._make_grid_boxes(B, proj_feats.device)
        else:
            # ===== Use real pre-extracted features (.pt) =====
            proj_feats = visual_feats  # already [B, 36, 2048]
            visual_pos = visual_pos

        # ===== LXMERT multimodal forward =====
        lxmert_outputs = self.lxmert(
            **lxmert_inputs,
            visual_feats=proj_feats,
            visual_pos=visual_pos
        )
        lxmert_emb = lxmert_outputs.pooled_output  # [B, 768]

        # ===== Late Fusion (concat ViLT + LXMERT embeddings) =====
        combined = torch.cat([vilt_emb, lxmert_emb], dim=1)
        logits = self.fusion(combined)

        return logits