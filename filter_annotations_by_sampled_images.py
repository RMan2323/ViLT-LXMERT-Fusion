import os
import pandas as pd

csv_path = "Dataset/dataset_Val2014_with_cp.csv"  # ✅ Your merged CSV
features_dir = "extracted_feats_val_TEMP"         # ✅ Directory containing extracted features

print("Loading dataset CSV...")
df = pd.read_csv(csv_path, low_memory=False)
print(f"Total rows loaded: {len(df)}")

# Extract available feature IDs from .pt file names
feature_files = os.listdir(features_dir)
available_ids = {
    int(f.split("_")[-1].split(".")[0])  # Extract 012345678901 from COCO_train2014_XXXXXXXXXXXX.pt
    for f in feature_files if f.endswith(".pt")
}

print(f"Total extracted features available: {len(available_ids)}")

def has_feature(row):
    # Extract image_id from image_path instead of NaN column
    try:
        img_id = int(row["image_path"].split("_")[-1].split(".")[0])
        return img_id in available_ids
    except:
        return False

# Filter rows having feature extracted
filtered_df = df[df.apply(has_feature, axis=1)].reset_index(drop=True)
print(f"✅ Filtered rows: {len(filtered_df)}")

# Save filtered CSV
output_csv = "Dataset/dataset_val2014_filtered.csv"
filtered_df.to_csv(output_csv, index=False)

print(f"✅ Saved filtered CSV to {output_csv}")
print("🎯 Done!")
