"""
Upload the Q!=K=V checkpoints and model card to the Hugging Face Hub.

Reads the HF token from the HF_TOKEN environment variable (never hardcode it).
Uploads everything in ./checkpoints and ./hf_model_card/README.md to the target repo.

    HF_TOKEN=... python upload_to_hf.py --repo BrainChip-AI/<name> [--private] [--dry-run]
"""

import os
import argparse
from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. BrainChip-AI/do-transformers-need-3-projections")
    ap.add_argument("--private", action="store_true", help="create the repo private (default: public)")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--card", default="hf_model_card/README.md")
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded, then stop")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set in environment.")

    files = sorted(f for f in os.listdir(args.ckpt_dir) if f.endswith(".pt"))
    print(f"repo={args.repo}  private={args.private}")
    print(f"model card: {args.card}")
    print(f"checkpoints ({len(files)}): " + ", ".join(files))
    if args.dry_run:
        print("dry run — nothing uploaded.")
        return

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    # model card at repo root
    api.upload_file(path_or_fileobj=args.card, path_in_repo="README.md",
                    repo_id=args.repo, repo_type="model")
    # checkpoints under checkpoints/
    api.upload_folder(folder_path=args.ckpt_dir, path_in_repo="checkpoints",
                      repo_id=args.repo, repo_type="model")
    print(f"Uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
