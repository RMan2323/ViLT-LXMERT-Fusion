import pandas as pd
import random

random.seed(42)

csv_file = "Dataset/dataset_train2014_with_cp.csv"
df = pd.read_csv(csv_file)

# Only images that exist in training folder
df = df[df["image_path"].str.contains("train2014")]

print("Total training samples:", len(df))

# Pick 100 unique images ONLY
unique_imgs = df["image_path"].unique()
print("Unique images:", len(unique_imgs))

selected_imgs = random.sample(list(unique_imgs), 100)

output = "images_list.txt"
with open(output, "w") as f:
    for p in selected_imgs:
        # Full correct path
        full_path = f"Dataset/{p}"
        f.write(full_path + "\n")

print("✅ Saved 100 matched images:", output)
