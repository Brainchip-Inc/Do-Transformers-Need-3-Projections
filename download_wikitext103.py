"""
Download and prepare a ~30M-token subset of WikiText-103 for the Gao & Xu
"Keyless Attention" reproduction (arXiv:2606.21848): their setup uses "a
30M-token subset [of WikiText-103] due to GPU memory constraints," 10 epochs.

wikitext-103-raw-v1's rows are individual lines (many empty, or Wikipedia
section-header artifacts like " = Title = \n"), not documents -- feeding that
directly into LocalTextDataset (transformer_KQV_300M_fineweb.py), which
tokenizes/truncates each row independently to seq_length, would produce mostly
degenerate near-empty "documents". Instead: drop empty/header-only lines and
concatenate consecutive lines into ~4000-character chunks (roughly one
seq_length=1024 GPT-2-BPE document each), then save with a 'text' column so
LocalTextDataset needs no changes.

Usage:
    conda run -n kv python download_wikitext103.py
"""

import argparse

from datasets import load_dataset, Dataset as HFDataset


def chunk_lines(lines, chunk_chars):
    chunks, buf, buf_len = [], [], 0
    for line in lines:
        line = line.strip()
        if not line or (line.startswith("=") and line.endswith("=")):
            continue  # skip blank lines and " = Section = " headers
        buf.append(line)
        buf_len += len(line) + 1
        if buf_len >= chunk_chars:
            chunks.append(" ".join(buf))
            buf, buf_len = [], 0
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", default="./wikitext103_train")
    ap.add_argument("--val-out", default="./wikitext103_validation")
    ap.add_argument("--target-train-tokens", type=int, default=30_000_000,
                     help="approximate token budget (chars/4 as a rough proxy pre-tokenization)")
    ap.add_argument("--chunk-chars", type=int, default=4000)
    args = ap.parse_args()

    print("Downloading wikitext-103-raw-v1...", flush=True)
    train_raw = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    val_raw = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    print(f"Loaded {len(train_raw):,} train lines, {len(val_raw):,} val lines", flush=True)

    train_chunks = chunk_lines(train_raw["text"], args.chunk_chars)
    val_chunks = chunk_lines(val_raw["text"], args.chunk_chars)

    # ~4 chars/token for GPT-2 BPE is a standard rough estimate; keep enough
    # chunks to comfortably exceed target_train_tokens (LocalTextDataset will
    # further truncate each chunk to seq_length at tokenization time, so the
    # eventual token count used in training is governed by total_tokens in
    # the training config, not this file -- this is just corpus sizing).
    approx_chars_needed = args.target_train_tokens * 4
    kept, total_chars = [], 0
    for c in train_chunks:
        kept.append(c)
        total_chars += len(c)
        if total_chars >= approx_chars_needed:
            break
    print(f"Keeping {len(kept):,} train chunks (~{total_chars/4:,.0f} tokens est.)", flush=True)

    HFDataset.from_dict({"text": kept}).save_to_disk(args.train_out)
    HFDataset.from_dict({"text": val_chunks}).save_to_disk(args.val_out)
    print(f"Wrote {args.train_out} ({len(kept)} docs) and {args.val_out} ({len(val_chunks)} docs)",
          flush=True)


if __name__ == "__main__":
    main()
