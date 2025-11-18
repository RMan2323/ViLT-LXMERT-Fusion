import json
import pandas as pd
from pathlib import Path

# =============================
# CONFIGURATION
# =============================
DATASET_ROOT = Path("Dataset")

# Test set has NO annotations (no answers)
# Only questions are provided
TEST_QUESTIONS_PATH = DATASET_ROOT / "questions_test" / "v2_OpenEnded_mscoco_test2015_questions.json"

# Output CSV path
OUT_CSV = DATASET_ROOT / "dataset_test2015.csv"

# Load answer vocab (for consistent columns)
VOCAB_PATH = DATASET_ROOT / "answer_vocab_top1000.json"
with open(VOCAB_PATH, "r") as f:
    answer_vocab = json.load(f)

print(f"Loaded answer vocab: {len(answer_vocab)} entries")

# =============================
# LOAD TEST QUESTIONS
# =============================
print("Loading test questions...")
with open(TEST_QUESTIONS_PATH, "r") as f:
    questions_json = json.load(f)["questions"]

records = []

for q in questions_json:
    qid = q["question_id"]
    img_id = q["image_id"]
    question = q["question"].strip()

    img_name = f"COCO_test2015_{img_id:012d}.jpg"
    img_path = Path("test2015") / img_name

    # Since no answers exist → fill with empty or -1
    records.append({
        "image_path": str(img_path),
        "question": question,
        "answer_idx": -1,          # No answer in test split
        "is_cp": 0,                # No complementary pairs for test
        "image_id": "",            # kept empty for compatibility
        "question_id": "",         # kept empty
        "answer": ""               # kept empty
    })

# =============================
# SAVE CSV
# =============================
df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)

print(f"✅ Saved Test CSV with {len(df)} rows → {OUT_CSV}")