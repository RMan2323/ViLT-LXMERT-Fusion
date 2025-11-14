import pandas as pd
import random

random.seed(42)

csv_file = "Dataset/dataset_Val2014_with_cp.csv"
df = pd.read_csv(csv_file)


print("Total validation samples:", len(df))

# Pick 100 unique images ONLY
unique_imgs = df["image_path"].unique()
print("Unique images:", len(unique_imgs))

# selected_imgs = random.sample(list(unique_imgs), 100)
selected_imgs = list(unique_imgs)

# output = "images_list.txt"
output = "images_list_full_val.txt"
with open(output, "w") as f:
    for p in selected_imgs:
        # Full correct path
        full_path = f"Dataset/{p}"
        f.write(full_path + "\n")

# print("✅ Saved 100 matched images:", output)
print("✅ Saved all matched images:", output)