#!/usr/bin/env python3
"""Build a lossless train/test split of real COCO val2017 data (images +
annotations) as zstd-compressed Parquet, for one of three RF-DETR training
modalities: detection, segmentation, or keypoint.

Images are stored byte-identical to the original COCO JPEG (no re-encode,
no resize -- "lossless" in the sense of zero additional lossy
transformation; any resize/normalize happens downstream at train time, the
same convention rf-detr-cpp's existing gen_reference/*.py scripts use).
Annotations are stored in their original COCO representation (xywh boxes,
RLE/polygon segmentation, flat [x,y,v,...] keypoints) so nothing is lost to
a lossy intermediate encoding either.

Split: deterministic (sorted by image_id, fixed seed), 90% train / 10% test
by image count, out of every val2017 image that has at least one usable
annotation for the given modality. This project's demos previously only
used a 24-image hand-picked sample; this is meant to replace that with a
real, reproducible held-out split, generated once and committed as data
(not regenerated per training run) so train/test membership is stable.

Usage:
    uv run --with pyarrow --with pillow gen_split.py detection \
        /path/to/val2017 /path/to/instances_val2017.json .
    uv run --with pyarrow --with pillow gen_split.py segmentation \
        /path/to/val2017 /path/to/instances_val2017.json .
    uv run --with pyarrow --with pillow gen_split.py keypoint \
        /path/to/val2017 /path/to/person_keypoints_val2017.json .
"""
import json
import os
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def load_images(img_dir, coco):
    by_id = {im["id"]: im for im in coco["images"]}
    return by_id


def build_rows(modality, img_dir, coco):
    by_id = load_images(img_dir, coco)
    anns_by_image = {}
    for a in coco["annotations"]:
        if a.get("iscrowd", 0):
            continue
        if modality == "keypoint" and a.get("num_keypoints", 0) == 0:
            continue
        anns_by_image.setdefault(a["image_id"], []).append(a)

    rows = []
    for image_id in sorted(anns_by_image.keys()):
        im_meta = by_id.get(image_id)
        if im_meta is None:
            continue
        img_path = os.path.join(img_dir, im_meta["file_name"])
        if not os.path.isfile(img_path):
            continue
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        anns = anns_by_image[image_id]
        row = {
            "image_id": image_id,
            "file_name": im_meta["file_name"],
            "width": im_meta["width"],
            "height": im_meta["height"],
            "image_bytes": image_bytes,
            "category_id": [a["category_id"] for a in anns],
            "bbox_xywh": [a["bbox"] for a in anns],  # COCO's native [x,y,w,h], absolute pixels
        }
        if modality == "segmentation":
            # Store exactly as COCO encodes it: a polygon list-of-lists, or
            # an RLE dict (counts/size) -- JSON round-trips both losslessly.
            row["segmentation_json"] = [json.dumps(a["segmentation"]) for a in anns]
        if modality == "keypoint":
            row["keypoints"] = [a["keypoints"] for a in anns]  # flat [x,y,v]*17 per instance
            row["num_keypoints"] = [a["num_keypoints"] for a in anns]
        rows.append(row)
    return rows


# GitHub rejects any single committed file over 100MB (and we're
# deliberately not using Git LFS) -- shard any split whose raw image bytes
# alone would exceed a conservative 90MB-per-file budget into multiple
# numbered parquet files (train-0.parquet, train-1.parquet, ...) instead of
# one large file. Single-file splits keep their plain name (e.g. test.parquet).
MAX_SHARD_BYTES = 90_000_000


def write_split(rows, out_path):
    total_bytes = sum(len(r["image_bytes"]) for r in rows)
    n_shards = max(1, -(-total_bytes // MAX_SHARD_BYTES))  # ceil div
    shard_size = -(-len(rows) // n_shards)  # ceil div

    base, ext = os.path.splitext(out_path)
    for i in range(n_shards):
        lo, hi = i * shard_size, min(len(rows), (i + 1) * shard_size)
        if lo >= hi:
            continue
        shard_rows = rows[lo:hi]
        columns = {k: [r[k] for r in shard_rows] for k in shard_rows[0].keys()}
        table = pa.table(columns)
        shard_path = out_path if n_shards == 1 else f"{base}-{i}{ext}"
        pq.write_table(table, shard_path, compression="zstd", compression_level=19)
        size_mb = os.path.getsize(shard_path) / 1e6
        print(f"wrote {len(shard_rows)} rows -> {shard_path} ({size_mb:.1f} MB)")


def main():
    modality = sys.argv[1]
    img_dir = sys.argv[2]
    ann_path = sys.argv[3]
    out_dir = sys.argv[4] if len(sys.argv) > 4 else "."
    assert modality in ("detection", "segmentation", "keypoint")

    with open(ann_path) as f:
        coco = json.load(f)

    rows = build_rows(modality, img_dir, coco)
    print(f"{modality}: {len(rows)} usable images (out of {len(coco['images'])} total in {ann_path})")

    rng = random.Random(0)
    order = list(range(len(rows)))
    rng.shuffle(order)
    n_test = max(1, len(order) // 10)
    test_idx = set(order[:n_test])
    train_rows = [rows[i] for i in range(len(rows)) if i not in test_idx]
    test_rows = [rows[i] for i in range(len(rows)) if i in test_idx]

    os.makedirs(out_dir, exist_ok=True)
    write_split(train_rows, os.path.join(out_dir, "train.parquet"))
    write_split(test_rows, os.path.join(out_dir, "test.parquet"))


if __name__ == "__main__":
    main()
