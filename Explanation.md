## `makeCSV.py`
Works on:
* `training_questions.json`
* `training_annotations.json`
* `validation_questions.json`
* `validation_annotations.json`

Picks the top `k` answers and saves them to `answer_vocab_top{k}.json` with `answer_text: answer_index`

Converts the dataset to `dataset_train.csv` and `dataset_val.csv` with columns:
* `image_path`
* `question`
* `answer_idx`

## `makeCSV_CP.py`
Works on:
* `dataset_train.csv`
* `training_annotations.json`
* `training_complementary_pairs.zip` (the script extracts it to `.json`)

Merges the dataset with complementary pairs to `dataset_train_with_cp.csv` with columns:
* `image_path`
* `question`
* `answer_idx`
* `is_cp`
* `image_id`
* `question_id`
* `answer`

## `list_images.py`
Works on:
* `dataset_train.csv`

Gets 100 unique images randomly with a seed and outputs their paths into `images_list.txt`

## `extract_fasterrcnn_features.py`
Works on:
* Image file
* Pretrained Faster R-CNN (ResNet-50 FPN) model from torchvision, trained on COCO (automatically downloaded and cached)

Selects the top-`K` detections by confidence score (default 36).

Saves one `.pt` file under the directory `extracted_feats/` per image containing:

* boxes: `[num_boxes, 4]` bounding boxes `(x1,y1,x2,y2)`
* scores: `[num_boxes]` detection confidences
* features: `[num_boxes, 2048]` region embeddings
* image_size: `(H, W)`

## `extract_feats_batch.py`
Wrapper for batch execution of `extract_fasterrcnn_features.py` with multiple images
Skips images for which features are already extracted

Works on:
* `images_list.txt`

Saves the `.pt` files for every image into `extracted_feats/`

##  `filter_annotations_by_sampled_images.py`
Works on:
* `dataset_train.csv`
* The directory `extracted_feats/`

What it does:
1) Loads the full dataset CSV.
2) Lists all `.pt` files in `extracted_feats/` and extracts their image IDs.
3) Compares those IDs with the image IDs referenced in the CSV.
4) Keeps only the rows that have a corresponding `.pt` feature file.

Saves the filtered subset as a new file `dataset_train_filtered.csv`.

## `vqa_dataset.py`
Works on:
* `dataset_train.csv`
* Training images
* Pre-extracted Faster R-CNN features `.pt` (used by LXMERT to handle image features)

The `ViltProcessor` prepares image + question for ViLT

The `LxmertTokenizer` prepares question text for LXMERT

Returns a dictionary with:
* ViLT inputs
* LXMERT inputs
* `visual_feats`: region-level features [36, 2048]
* `visual_pos`: bounding box coordinates [36, 4]
* `answer_idx`

## `vilt_lxmert_fusion.py`
Works on:
* ViLT inputs
* LXMERT inputs

Gives the `logits` of shape `[batch_size, num_answers]`

## `train_fusion_vqa.py`
Works on:
* `dataset_train.csv` (can be filtered or non-filtered)
* Training images directory
* The directory `extracted_feats/`
* Checkpoints (pretrained/last/best)

Calls the helper scripts `vqa_dataset.py` and `vilt_lxmert_fusion.py`

For every epoch, saves `checkpoints/last_epoch.ckpt` and `checkpoints/best_model.ckpt`