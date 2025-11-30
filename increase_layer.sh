#!/bin/bash

# Define checkpoint directory and other parameters
CHECKPOINT_DIR="checkpoints_freeze_pretrained_inc_layers"
EPOCHS=20
NUM_WORKERS=8

# Train the model
echo "Starting training..."
python train_fusion_vqa.py --checkpoint_dir "$CHECKPOINT_DIR" --epochs $EPOCHS --num_workers $NUM_WORKERS

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully."
    # Now test the model
    echo "Starting testing..."
    python testing_fusion_vqa.py
else
    echo "Training failed. Skipping testing."
fi
