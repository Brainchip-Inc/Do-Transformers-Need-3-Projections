"""
Download and split the FineWeb-Edu sample-10BT subset for the 300M QKV baseline
training run, matching the data source used for the existing Q!=K=V checkpoint
(checkpoints_llm/qkv_keqv_300m_fineweb_edu.pt, see hf_model_card_llm/README.md).

The training script tokenizes on the fly from a 'text' column (see
transformer_KQV_300M_fineweb.py's LocalTextDataset), so this script only needs to
download the raw documents and carve out a held-out validation slice -- no
pre-tokenization needed here.

The train/val split is a plain sequential slice, NOT a global shuffle-then-split:
shuffling ~9M rows before save_to_disk forces random-order reads scattered across
the whole ~46GB corpus, which is extremely slow on a network filesystem (observed:
~18 minutes to write a single one of 99 shards, i.e. days for the full split). A
pre-shuffle buys nothing here anyway -- the training script's DataLoader already
shuffles every epoch -- so a sequential slice is equally valid and writes at
close to raw copy speed instead.

Usage:
    conda run -n kv python download_fineweb_edu.py
"""

import argparse

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", default="./fineweb_edu_train")
    ap.add_argument("--val-out", default="./fineweb_edu_validation")
    ap.add_argument("--val-docs", type=int, default=10_000,
                     help="documents held out for validation (small vs. ~10B-token corpus)")
    args = ap.parse_args()

    print("Downloading HuggingFaceFW/fineweb-edu (sample-10BT)...", flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
    print(f"Loaded {len(ds):,} documents", flush=True)

    val_ds = ds.select(range(args.val_docs))
    train_ds = ds.select(range(args.val_docs, len(ds)))

    print(f"Saving {len(train_ds):,} train / {len(val_ds):,} val documents...", flush=True)
    val_ds.save_to_disk(args.val_out)
    train_ds.save_to_disk(args.train_out)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
