# datasource-rf-detr-keypoint-data

COCO val2017 person-keypoint images and annotations as Parquet, kept for validation only.

## What it is for

**Validation only. Never training.** Every val2017 image with a non-crowd keypointed person is here, which includes the whole blinded holdout `coco_person_commercial_val2017`. The `train-*` and `test` file names are the split `gen_split.py` made, not a permission: no file here is trained on. CLAUDE.md in manuals-weftspun states the holdout rule.

Each row holds the original JPEG bytes and that image's person boxes and keypoints in COCO's order.

## Rebuild it

`gen_split.py` regenerates the files deterministically from a local val2017 image directory and its keypoint annotations. It needs `pyarrow` and `pillow`.

```sh
python gen_split.py keypoint <val2017-dir> <person_keypoints_val2017.json>
```

## Licence

The annotations are CC BY 4.0, as LICENSE.txt states. Each image keeps the licence COCO records for it, and most of those forbid commercial use or derivatives.
