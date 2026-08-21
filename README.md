# rf-detr-keypoint-data

> ## Validation only. Never training.
>
> **This set is the blinded holdout.** It takes every val2017 image with a non-crowd
> keypointed person, and all 523 images of `dataflow-coco-gemx/coco_person_commercial_val2017`
> are inside it: 481 in `train-*.parquet` and 42 in `test.parquet`.
>
> The file names are misleading. `train-*.parquet` must not be trained on.
>
> **The licence line below is also wrong.** It says CC BY 4.0 for the whole set. Only 523 of
> val2017's 5,000 person images are commercial and derivatives safe, so 1,823 of these 2,346
> carry the NC, ND and share-alike terms `filter_coco_licenses.py` drops.
>
> The two faults do not overlap, they compound: the licence-clean images are exactly the
> holdout, so no subset of this is both trainable and clean. A keypoint training set has to be
> built rather than filtered out of this one. RFD 0122 records what builds it.

Train/test split for RF-DETR keypoint-detection finetuning
([rf-detr-cpp](https://github.com/weftspun/rf-detr-cpp)), sourced from
[COCO val2017](https://cocodataset.org) `person_keypoints_val2017.json` --
every image with at least one non-crowd, visibly-keypointed person
annotation (2346 of val2017's 5000 images). License: CC BY 4.0 (COCO
images and annotations).

## Contents

- `train-0.parquet` .. `train-3.parquet` -- 2112 images (90% split) total,
  sharded into 4 files (~528 rows / ~85MB each) since the unsharded file
  exceeded GitHub's 100MB per-file limit and this repo intentionally
  doesn't use Git LFS. Concatenate them (e.g. `pyarrow.concat_tables`) to
  reconstruct the full train split.
- `test.parquet` -- 234 images (10% split), zstd-compressed, single file
  (under the 100MB limit as-is).
- `gen_split.py` -- the script that produced these files from a local
  COCO val2017 image directory + `person_keypoints_val2017.json` (shards
  automatically whenever a split's raw image bytes would exceed ~90MB).

Split is deterministic: sorted by `image_id`, shuffled with a fixed seed
(0), first 10% held out as test. Regenerating from the same COCO files
reproduces the same split exactly.

## Schema (per row)

| column            | type              | meaning                                                                                               |
| ----------------- | ----------------- | ----------------------------------------------------------------------------------------------------- |
| `image_id`        | int64             | COCO image id                                                                                         |
| `file_name`       | string            | original COCO file name                                                                               |
| `width`, `height` | int32             | image dimensions (px)                                                                                 |
| `image_bytes`     | binary            | the JPEG file, byte-identical to the COCO val2017 release -- no re-encode/resize applied by this repo |
| `category_id`     | list<int64>       | one entry per person instance (always COCO category 1)                                                |
| `bbox_xywh`       | list<list<float>> | COCO-native `[x, y, w, h]`, absolute pixels                                                           |
| `keypoints`       | list<list<float>> | one flat `[x, y, v]` \* 17 array per instance, COCO's keypoint order                                  |
| `num_keypoints`   | list<int64>       | count of labeled (v>0) keypoints per instance                                                         |

## Loading

```python
import pyarrow.parquet as pq
table = pq.read_table("train.parquet")
row = table.to_pylist()[0]
```

Consumed by `rf-detr-cpp`'s dataset loader via
`gen_reference/gen_from_parquet_split.py` (converts each split into the
existing `write_arr`-based `.bin` format `src/dataset.cpp`/
`CocoKeypointDataset` reads).
