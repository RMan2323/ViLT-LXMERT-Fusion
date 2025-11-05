┌───────────────────────────────┐
│        Input: Image           │
└─────────────┬─────────────────┘
              │
              ▼
┌───────────────────────────────┐
│ Faster R-CNN Feature Extractor │
│ (extract_topK_regions=36,     │
│  feature_dim=2048)            │
└─────────────┬─────────────────┘
              │
      ┌───────┴────────┐
      ▼                ▼
┌───────────────┐  ┌───────────────┐
│ Extracted .pt │  │ Zero-feature   │
│ files saved   │  │ fallback if    │
│ (extracted_feats/) │ .pt missing │
└───────────────┘  └───────────────┘
              │
              ▼
┌───────────────────────────────┐
│ Input: Question Text          │
└─────────────┬─────────────────┘
              │
              ▼
     ┌───────────────────┐
     │ Tokenization      │
     │ (word → token IDs)│
     └────────┬──────────┘
              │
              ▼
     ┌───────────────────┐
     │ Embedding Layer   │
     │ (word embeddings) │
     └────────┬──────────┘
              │
              ▼
┌─────────────────────────────────────┐
│      LXMERT / Fusion Model           │
│  - Takes: Visual Features (.pt)     │
│           Question Embeddings       │
│  - Cross-modality attention         │
│  - Output: fused representation     │
└─────────────┬───────────────────────┘
              │
              ▼
┌───────────────────────────────┐
│ Answer Classifier             │
│ (Top-1000 classes from CSV)  │
│ Output: Predicted Answer      │
└───────────────────────────────┘

==================== Optional ViLT Path ====================

┌───────────────────────────────┐
│ Input: Image + Question       │
└─────────────┬─────────────────┘
              │
              ▼
┌───────────────────────────────┐
│ ViLT Model                     │
│ - Vision Transformer (ViT)    │
│ - Text Transformer             │
│ - Cross-modality fusion inside │
│ - Direct end-to-end output     │
└─────────────┬─────────────────┘
              │
              ▼
┌───────────────────────────────┐
│ Output Answer                 │
│ (same format as LXMERT path) │
└───────────────────────────────┘

==================== Dataset / CSV Handling ====================

┌───────────────────────────────┐
│ dataset_train2014.csv /       │
│ dataset_train2014_with_cp.csv │
└─────────────┬─────────────────┘
              │
              ▼
┌───────────────────────────────┐
│ filter_annotations_by_sampled_images.py │
│ - Select 100 sampled images            │
│ - Match image_id → extracted_feats     │
│ - Save filtered CSV for training       │
└───────────────────────────────┘

==================== Training ====================

┌───────────────────────────────┐
│ train_fusion_vqa.py            │
│ - Loads filtered CSV           │
│ - Loads Faster-RCNN features   │
│ - Uses LXMERT (fusion) or ViLT │
│ - Backprop, optimizer, loss   │
│ - Save checkpoints & best model│
└───────────────────────────────┘





FURTHER MORE


─────────────────────────────────────────────────────────────
                          INPUT DATA
─────────────────────────────────────────────────────────────
Images: Dataset/train2014/*.jpg
Questions + Answers: Dataset CSV (dataset_train2014.csv / with complementary pairs)
─────────────────────────────────────────────────────────────
                     FEATURE EXTRACTION
─────────────────────────────────────────────────────────────
extract_feats_batch.py
┌─────────────────────────────────────────────────────────┐
│ Faster R-CNN (pretrained)                               │
│ - Extract top-K object regions (num_regions=36)        │
│ - Feature dim: 2048                                    │
│ - Output: dict with "features" (36x2048), "boxes" (36x4)│
│ - Save per-image .pt file in extracted_feats/          │
│ - If missing -> fallback zeros [36x2048], boxes [36x4] │
└─────────────────────────────────────────────────────────┘
─────────────────────────────────────────────────────────────
                 DATA FILTERING / CSV PREPARATION
─────────────────────────────────────────────────────────────
filter_annotations_by_sampled_images.py
┌─────────────────────────────────────────────────────────┐
│ - Load dataset CSV (dataset_train2014.csv / with CP)   │
│ - Select MAX_SAMPLES=100 for reproducible experiments │
│ - Keep only rows with available Faster-RCNN features   │
│ - Merge complementary pairs (is_cp=1)                 │
│ - Save filtered CSV: dataset_train2014_filtered.csv   │
└─────────────────────────────────────────────────────────┘
─────────────────────────────────────────────────────────────
                          DATASET
─────────────────────────────────────────────────────────────
vqa_dataset.py (PyTorch Dataset)
┌─────────────────────────────────────────────────────────┐
│ For each sample:                                       │
│ 1. Load question text + answer_idx from filtered CSV   │
│ 2. Load image (PIL) and apply minimal transform (384x384) │
│ 3. ViLT Processor: tokenizes question + processes image │
│ 4. LXMERT Tokenizer: tokenizes question                  │
│ 5. Load pre-extracted Faster-RCNN features (.pt)        │
│    - visual_feats: [36,2048]                             │
│    - visual_pos: [36,4]                                  │
│    - fallback zeros if missing                           │
└─────────────────────────────────────────────────────────┘
─────────────────────────────────────────────────────────────
                         MODEL ARCHITECTURE
─────────────────────────────────────────────────────────────
vilt_lxmert_fusion.py
┌─────────────────────────────────────────────────────────┐
│ ViLT Encoder (pretrained)                               │
│ - Patch embeddings → token embeddings                   │
│ - Pooled output: [B,768]                               │
│ - Token embeddings: [B, seq_len, 768]                  │
│                                                         │
│ LXMERT Encoder (pretrained)                             │
│ - Receives: projected ViLT visual features [B,36,2048] │
│ - Receives: visual_pos [B,36,4]                        │
│ - Pooled output: [B,768]                               │
│                                                         │
│ Fusion:                                                 │
│ - Concatenate ViLT pooled ([B,768]) + LXMERT pooled ([B,768]) → [B,1536] │
│ - Fully connected layers + ReLU + Dropout → num_answers (1000) logits │
└─────────────────────────────────────────────────────────┘
─────────────────────────────────────────────────────────────
                            TRAINING
─────────────────────────────────────────────────────────────
train_fusion_vqa.py
┌─────────────────────────────────────────────────────────┐
│ Dataset: VQADataset with filtered CSV                  │
│ DataLoader: batch_size=8, collate_fn handles stacking  │
│ Model: ViLT_LXMERT_Fusion (freeze_encoders=True)      │
│ Loss: CrossEntropyLoss                                  │
│ Optimizer: AdamW (only trainable projection + fusion) │
│ Epochs: 1 (for debug / test reproducibility)          │
│ Seed fixed (42) for deterministic sample selection     │
│                                                        │
│ Forward pass:                                         │
│ - ViLT → pooled embedding & token features           │
│ - Token features pooled → projected → LXMERT visual  │
│ - LXMERT → pooled output                               │
│ - Fusion (ViLT + LXMERT) → logits                     │
│                                                        │
│ Backprop → optimizer.step                               │
│ Checkpoints: last_epoch.ckpt, best_model.ckpt          │
└─────────────────────────────────────────────────────────┘
─────────────────────────────────────────────────────────────
                           OUTPUT
─────────────────────────────────────────────────────────────
- Predicted answer indices (answer_idx) for each (image, question) pair
- Optionally mapped to top-1000 answer vocabulary
- Training logs: loss, uses_real_feats flag per batch
─────────────────────────────────────────────────────────────
