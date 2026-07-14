#!/usr/bin/env python3
"""
hf_backup.py — push/pull trained checkpoints to/from the Hugging Face Hub so a pod
crash never costs a 12h training run again.

Auth: set a WRITE token in the environment (never hard-code it, never commit it):
    export HF_TOKEN=hf_xxx        # from https://huggingface.co/settings/tokens (role: write)
Install once:
    pip install -U huggingface_hub

Usage (repo defaults to gravycrazy/temporal_straightening — our backup store):
  # after training, back the run folder up (repo auto-created, private):
  python hf_backup.py push checkpoints/repro/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05

  # ...or target a specific repo explicitly:
  python hf_backup.py push <run_dir> gravycrazy/temporal_straightening

  # on a fresh pod, restore everything into checkpoints/repro/test:
  python hf_backup.py pull gravycrazy/temporal_straightening checkpoints/repro/test
"""
import argparse
import os
import sys

# Default backup repo on the Hugging Face Hub (holds the expensive trained
# checkpoints so a lost pod never costs us a 12h PushT run again).
DEFAULT_REPO_ID = "gravycrazy/temporal_straightening"


def _token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not tok:
        sys.exit("ERROR: set a write token first:  export HF_TOKEN=hf_xxx  "
                 "(https://huggingface.co/settings/tokens, role: write)")
    return tok


def push(local_dir, repo_id):
    from huggingface_hub import HfApi, create_repo
    local_dir = os.path.normpath(local_dir)
    if not os.path.isdir(local_dir):
        sys.exit(f"ERROR: {local_dir} is not a directory")
    tok = _token()
    create_repo(repo_id, repo_type="model", private=True, exist_ok=True, token=tok)
    # store each run under its own folder name inside the repo
    path_in_repo = os.path.basename(local_dir)
    HfApi(token=tok).upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
        commit_message=f"backup {path_in_repo}",
    )
    print(f"OK: pushed {local_dir}  ->  https://huggingface.co/{repo_id}/tree/main/{path_in_repo}")


def pull(repo_id, local_dir):
    from huggingface_hub import snapshot_download
    tok = _token()
    os.makedirs(local_dir, exist_ok=True)
    out = snapshot_download(repo_id=repo_id, repo_type="model", local_dir=local_dir, token=tok)
    print(f"OK: pulled {repo_id}  ->  {out}")


def main():
    ap = argparse.ArgumentParser(description="Back up / restore checkpoints on the Hugging Face Hub.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="upload a local folder to an HF model repo")
    p.add_argument("local_dir")
    p.add_argument("repo_id", nargs="?", default=DEFAULT_REPO_ID,
                   help=f"HF repo id (default: {DEFAULT_REPO_ID})")
    q = sub.add_parser("pull", help="download an HF model repo into a local folder")
    q.add_argument("repo_id", nargs="?", default=DEFAULT_REPO_ID,
                   help=f"HF repo id (default: {DEFAULT_REPO_ID})")
    q.add_argument("local_dir")
    args = ap.parse_args()
    if args.cmd == "push":
        push(args.local_dir, args.repo_id)
    else:
        pull(args.repo_id, args.local_dir)


if __name__ == "__main__":
    main()
