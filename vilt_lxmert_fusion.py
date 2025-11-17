# vilt_lxmert_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViltModel, LxmertModel

class ViLT_LXMERT_Fusion(nn.Module):
    def __init__(self, num_answers=1000, hidden_dim=768, fusion_dim=1024,
                 num_regions=36, project_to_dim=2048, freeze_encoders=True):
        super().__init__()

        self.num_regions = num_regions
        self.project_to_dim = project_to_dim
        self.hidden_dim = hidden_dim

        # === Load pretrained encoders (ONLY pretrained, as you requested) ===
        self.vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")

        # === REMOVE loading Priyanshu’s fine-tuned weights (you asked to disable this) ===
        """
        try:
            vilt_ckpt = torch.load("checkpoints_vilt/best_vilt_head.ckpt", map_location="cpu")
            lxmert_ckpt = torch.load("checkpoints_lxmert/best_lxmert_head.ckpt", map_location="cpu")
            self.vilt.load_state_dict(vilt_ckpt["vilt_state"], strict=False)
            self.lxmert.load_state_dict(lxmert_ckpt["lxmert_state"], strict=False)
            print("✅ Loaded fine-tuned ViLT & LXMERT encoder weights.")
        except Exception as e:
            print(f"⚠️ Could not load fine-tuned encoder weights: {e}")
            print("→ Falling back to pretrained encoders.")
        """

        print("🔵 Using PURE pretrained ViLT & pretrained LXMERT for fusion.")

        # === Freeze encoders if requested ===
        if freeze_encoders:
            for p in self.vilt.parameters(): 
                p.requires_grad = False
            for p in self.lxmert.parameters(): 
                p.requires_grad = False

        # Projection: 768 -> 2048 for LXMERT visual input
        self.vilt_to_lxmert_proj = nn.Linear(hidden_dim, self.project_to_dim)

        # === SMALL FUSION MLP (your requested network) ===
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, num_answers)
        )

        # === BIG MLP (commented out, as you asked) ===
        """
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_answers)
        )
        """

    def _make_grid_boxes(self, batch_size, device):
        n = self.num_regions
        import math
        side = int(math.sqrt(n))
        if side * side != n:
            side = int(round(math.sqrt(n)))
        H = side
        W = int(math.ceil(n / H))

        boxes = []
        for i in range(H):
            for j in range(W):
                if len(boxes) >= n:
                    break
                x1, y1 = j / W, i / H
                x2, y2 = (j + 1) / W, (i + 1) / H
                boxes.append([x1, y1, x2, y2])

        boxes = torch.tensor(boxes, dtype=torch.float32, device=device)
        return boxes.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, vilt_inputs, lxmert_inputs, visual_feats=None, visual_pos=None):
        # --- ViLT forward ---
        vilt_outputs = self.vilt(**vilt_inputs, return_dict=True)
        vilt_emb = vilt_outputs.pooler_output  # [B, 768]

        # --- Use external features if available, else fallback ---
        if visual_feats is None or visual_pos is None:
            last_hidden = vilt_outputs.last_hidden_state[:, 1:, :]    # drop CLS
            B, T, C = last_hidden.shape

            pooled = F.adaptive_avg_pool1d(last_hidden.permute(0,2,1), self.num_regions)
            pooled = pooled.permute(0,2,1)

            proj_feats = self.vilt_to_lxmert_proj(pooled)
            visual_pos = self._make_grid_boxes(B, proj_feats.device)
        else:
            proj_feats = visual_feats

        # --- LXMERT forward ---
        lxmert_out = self.lxmert(**lxmert_inputs, visual_feats=proj_feats, visual_pos=visual_pos)
        lxmert_emb = lxmert_out.pooled_output  # [B,768]

        # --- Fusion ---
        combined = torch.cat([vilt_emb, lxmert_emb], dim=1)
        logits = self.fusion(combined)

        return logits
