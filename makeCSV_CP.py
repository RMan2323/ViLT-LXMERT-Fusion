import os
import json
import pandas as pd

base_dir = "Dataset"
train_csv_path = os.path.join(base_dir, "dataset_train2014.csv")
cp_zip_path = os.path.join(base_dir, "v2_Complementary_Pairs_Train_mscoco.zip")
cp_json_path = os.path.join(base_dir, "v2_mscoco_train2014_complementary_pairs.json")

# === Step 1: Load the existing training CSV ===
print("Loading existing training CSV...")
train_df = pd.read_csv(train_csv_path)
print(f"Loaded {len(train_df)} samples from {train_csv_path}")

# === Step 2: Ensure complementary pairs JSON is extracted ===
if not os.path.exists(cp_json_path):
    print("Extracting complementary pairs JSON...")
    import zipfile
    with zipfile.ZipFile(cp_zip_path, "r") as zip_ref:
        zip_ref.extractall(base_dir)

print(f"Found complementary pairs file: {cp_json_path}")

# === Step 3: Load complementary pairs ===
with open(cp_json_path, "r") as f:
    cp_pairs = json.load(f)

print(f"Total complementary pair entries: {len(cp_pairs)}")

# === Step 4: Load original questions/answers for lookup ===
questions_path = os.path.join(base_dir, "questions_train/v2_OpenEnded_mscoco_train2014_questions.json")
annotations_path = os.path.join(base_dir, "annotations_train/v2_mscoco_train2014_annotations.json")

with open(questions_path, "r") as f:
    questions_data = json.load(f)["questions"]
q_lookup = {q["question_id"]: (q["image_id"], q["question"]) for q in questions_data}

with open(annotations_path, "r") as f:
    ann_data = json.load(f)["annotations"]
a_lookup = {a["question_id"]: a["multiple_choice_answer"] for a in ann_data}

# === Step 5: Build complementary pairs DataFrame ===
cp_records = []

for pair in cp_pairs:
    # Each element is a [q1_id, q2_id] list
    if not isinstance(pair, list) or len(pair) != 2:
        continue
    for qid in pair:
        if qid not in q_lookup or qid not in a_lookup:
            continue
        img_id, question = q_lookup[qid]
        answer = a_lookup[qid]
        image_path = f"train2014/COCO_train2014_{img_id:012d}.jpg"
        cp_records.append({
            "image_path": image_path,
            "question": question,
            "answer": answer,
            "image_id": img_id,
            "question_id": qid,
            "is_cp": 1  # mark complementary pairs
        })

cp_df = pd.DataFrame(cp_records)
print(f"Built complementary pairs dataframe with {len(cp_df)} rows")

# === Step 6: Add answer_idx ===
answer_vocab_path = os.path.join(base_dir, "answer_vocab_top1000.json")
if os.path.exists(answer_vocab_path):
    with open(answer_vocab_path, "r") as f:
        answer_to_idx = json.load(f)
    cp_df["answer_idx"] = cp_df["answer"].map(lambda a: answer_to_idx.get(a, -1))
else:
    cp_df["answer_idx"] = -1

# === Step 7: Reorder columns to match training CSV ===
cp_df = cp_df[["image_path", "question", "answer_idx", "image_id", "question_id", "answer", "is_cp"]]

# Add is_cp=0 to train_df for clarity
train_df["is_cp"] = 0

# === Step 8: Merge and save ===
merged_df = pd.concat([train_df, cp_df], ignore_index=True)
merged_csv_path = os.path.join(base_dir, "dataset_train2014_with_cp.csv")
merged_df.to_csv(merged_csv_path, index=False)

print(f"✅ Saved merged CSV with complementary pairs to {merged_csv_path}")
print(f"Total rows: {len(merged_df)} (original {len(train_df)} + {len(cp_df)})")

