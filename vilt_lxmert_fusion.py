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
        self.vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")

        # Freeze encoders by default for low memory usage
        if freeze_encoders:
            for p in self.vilt.parameters():
                p.requires_grad = False
            for p in self.lxmert.parameters():
                p.requires_grad = False

        # Projection from ViLT token dim (768) -> LXMERT visual dim (2048)
        self.vilt_to_lxmert_proj = nn.Linear(hidden_dim, self.project_to_dim)

        # Fusion + Classification Head
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, num_answers)
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

    def forward(self, vilt_inputs, lxmert_inputs):
        """
        Forward does:
         - ViLT forward to get pooler_output (for ViLT embedding) and last_hidden_state (tokens)
         - Build visual features for LXMERT by pooling ViLT tokens into num_regions
         - Project pooled features to project_to_dim (e.g., 2048)
         - Build visual_pos grid and pass both to LXMERT
         - Late fusion of ViLT pooler_output and LXMERT pooled_output
        """
        # ===== ViLT forward (multimodal) =====
        # Ensure ViLT returns last_hidden_state
        vilt_outputs = self.vilt(**vilt_inputs, return_dict=True, output_hidden_states=False)
        # ViLT pooled multimodal embedding
        vilt_emb = vilt_outputs.pooler_output  # [B, 768]

        # Get ViLT token embeddings (B, seq_len, 768). seq_len ~ 1 + 14*14 = 197
        # Remove CLS token (index 0) to keep only patch tokens
        last_hidden = None
        if hasattr(vilt_outputs, "last_hidden_state"):
            last_hidden = vilt_outputs.last_hidden_state  # [B, seq_len, 768]
        else:
            # fallback: some ViLT variants might expose hidden_states; try to get final hidden state
            raise RuntimeError("ViLT output missing last_hidden_state; update transformers package.")

        # drop CLS token
        token_feats = last_hidden[:, 1:, :]  # [B, num_patches, 768], typically [B,196,768]

        # ===== Pool token_feats to num_regions (e.g., 36) =====
        # pool along token dimension using adaptive avg pool (via 1D pooling)
        B, T, C = token_feats.shape
        # reshape for pooling: (B, C, T)
        token_feats_t = token_feats.permute(0, 2, 1)  # [B, C, T]
        # Adaptive pool to num_regions -> (B, C, num_regions)
        pooled = F.adaptive_avg_pool1d(token_feats_t, self.num_regions)  # [B, C, num_regions]
        pooled = pooled.permute(0, 2, 1)  # [B, num_regions, C] where C==hidden_dim (768)

        # ===== Project pooled features to LXMERT visual dim (e.g., 2048) =====
        # Apply linear projection per region
        proj_feats = self.vilt_to_lxmert_proj(pooled)  # [B, num_regions, project_to_dim]

        # ===== Build visual_pos grid (normalized box coords) =====
        batch_size = proj_feats.shape[0]
        device = proj_feats.device
        visual_pos = self._make_grid_boxes(batch_size, device)  # [B, num_regions, 4]

        # ===== LXMERT forward with constructed visual_feats & visual_pos =====
        lxmert_outputs = self.lxmert(
            **lxmert_inputs,
            visual_feats=proj_feats,
            visual_pos=visual_pos,
        )
        lxmert_emb = lxmert_outputs.pooled_output  # [B, 768]

        # ===== Late fusion =====
        combined = torch.cat([vilt_emb, lxmert_emb], dim=1)  # [B, 1536]
        logits = self.fusion(combined)
        return logits
