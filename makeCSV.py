import json
import pandas as pd
from pathlib import Path
from collections import Counter

# =============================
# CONFIGURATION
# =============================
DATASET_ROOT = Path("Dataset")  # change if your root folder is different
TOP_K_ANSWERS = 1000            # top N most frequent answers to keep

# train/val settings
splits = [
    ("train2014", 
     DATASET_ROOT / "questions_train" / "v2_OpenEnded_mscoco_train2014_questions.json",
     DATASET_ROOT / "annotations_train" / "v2_mscoco_train2014_annotations.json"),
    ("val2014",
     DATASET_ROOT / "questions_val" / "v2_OpenEnded_mscoco_val2014_questions.json",
     DATASET_ROOT / "annotations_val" / "v2_mscoco_val2014_annotations.json"),
]

# =============================
# LOAD ALL ANNOTATIONS (for vocab)
# =============================
all_answers = Counter()

for split_name, q_path, a_path in splits:
    print(f"Counting answers from {split_name}...")
    with open(a_path, "r") as f:
        anns = json.load(f)["annotations"]
    for ann in anns:
        ans = ann["multiple_choice_answer"].strip().lower()
        all_answers[ans] += 1

print(f"Total distinct answers: {len(all_answers)}")

# Keep most common answers
most_common = all_answers.most_common(TOP_K_ANSWERS)
answer_vocab = {ans: i for i, (ans, _) in enumerate(most_common)}

# Save vocab
vocab_path = DATASET_ROOT / "answer_vocab_top{}.json".format(TOP_K_ANSWERS)
with open(vocab_path, "w") as f:
    json.dump(answer_vocab, f, indent=2)
print(f"Saved answer vocab to {vocab_path} ({len(answer_vocab)} answers).")

# =============================
# BUILD CSVs
# =============================
for split_name, q_path, a_path in splits:
    print(f"\nProcessing {split_name}...")

    # load jsons
    with open(q_path, "r") as f:
        questions = {q["question_id"]: q for q in json.load(f)["questions"]}
    with open(a_path, "r") as f:
        anns = json.load(f)["annotations"]

    records = []
    skipped = 0

    for ann in anns:
        qid = ann["question_id"]
        if qid not in questions:
            continue

        question = questions[qid]["question"].strip()
        img_id = ann["image_id"]
        img_name = f"COCO_{split_name}_{img_id:012d}.jpg"
        img_path = Path(split_name) / img_name

        ans = ann["multiple_choice_answer"].strip().lower()
        if ans not in answer_vocab:
            skipped += 1
            continue
        ans_idx = answer_vocab[ans]

        records.append({
            "image_path": str(img_path),
            "question": question,
            "answer_idx": ans_idx
        })

    df = pd.DataFrame(records)
    out_csv = DATASET_ROOT / f"dataset_{split_name}.csv"
    df.to_csv(out_csv, index=False)

    print(f"{split_name}: {len(df)} rows saved to {out_csv}, skipped {skipped}.")

