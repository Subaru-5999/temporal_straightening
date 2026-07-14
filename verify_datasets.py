#!/usr/bin/env python3
"""
verify_datasets.py  --  structural + content verification of the datasets this repo
expects, so we can confirm they were extracted/reproduced perfectly before training.

It does NOT change any dataset logic; it only reads the same files the loaders in
datasets/*.py read, and reports:
  - required files present / missing
  - tensor shapes and rollout counts
  - obs (frame/video) file counts vs number of rollouts
  - a quick load-test of one sample per dataset

Usage:
    export DATASET_DIR=/temporal_s/data
    python verify_datasets.py                 # checks all datasets under DATASET_DIR
    python verify_datasets.py --root /temporal_s/data
    python verify_datasets.py --only point_maze wall_single pusht_noise deformable

Exit code is 0 only if every checked dataset passes.
"""
import os
import sys
import pickle
import argparse
from pathlib import Path

try:
    import torch
except Exception as e:  # torch should be present; if not, we still do file-presence checks
    torch = None
    print(f"[warn] torch not importable ({e}); will only do file-presence checks.")

OK = "OK  "
BAD = "FAIL"
WARN = "warn"


def _exists(p: Path):
    return p.exists()


def _load_tensor(p: Path):
    if torch is None:
        return None, "torch-unavailable"
    try:
        t = torch.load(p, map_location="cpu", weights_only=False)
        return t, None
    except TypeError:
        # older torch without weights_only kwarg
        try:
            t = torch.load(p, map_location="cpu")
            return t, None
        except Exception as e:
            return None, str(e)
    except Exception as e:
        return None, str(e)


def _load_pickle(p: Path):
    try:
        with open(p, "rb") as f:
            return pickle.load(f), None
    except Exception as e:
        return None, str(e)


def _shape(x):
    try:
        return tuple(x.shape)
    except Exception:
        try:
            return f"len={len(x)}"
        except Exception:
            return type(x).__name__


def _count(pattern_dir: Path, suffix: str):
    if not pattern_dir.exists():
        return -1
    return len([f for f in pattern_dir.iterdir() if f.name.endswith(suffix)])


class Report:
    def __init__(self, name):
        self.name = name
        self.lines = []
        self.passed = True

    def line(self, status, msg):
        if status == BAD:
            self.passed = False
        self.lines.append(f"    [{status}] {msg}")

    def dump(self):
        header = f"== {self.name} :: {'PASS' if self.passed else 'FAIL'} =="
        print(header)
        for l in self.lines:
            print(l)
        print()


def check_flat(root: Path, name: str, required_tensors, obs_suffix, seq_file=None,
               seq_is_pickle=False, optional_tensors=()):
    """Single-folder datasets: point_maze, wall_single."""
    r = Report(name)
    base = root / name
    if not base.exists():
        r.line(BAD, f"folder missing: {base}")
        return r

    n_rollout = None
    for fn in required_tensors:
        p = base / fn
        if not _exists(p):
            r.line(BAD, f"missing required file: {fn}")
            continue
        t, err = _load_tensor(p)
        if err:
            r.line(BAD, f"{fn}: could not load ({err})")
        else:
            r.line(OK, f"{fn}: shape {_shape(t)}")
            if fn == "states.pth":
                try:
                    n_rollout = t.shape[0]
                except Exception:
                    pass

    for fn in optional_tensors:
        p = base / fn
        if _exists(p):
            t, err = _load_tensor(p)
            r.line(OK if not err else WARN, f"{fn} (optional): "
                   + (f"shape {_shape(t)}" if not err else f"load error {err}"))

    # seq lengths
    if seq_file:
        p = base / seq_file
        if not _exists(p):
            r.line(BAD, f"missing seq-length file: {seq_file}")
        else:
            data, err = (_load_pickle(p) if seq_is_pickle else _load_tensor(p))
            if err:
                r.line(BAD, f"{seq_file}: could not load ({err})")
            else:
                r.line(OK, f"{seq_file}: {_shape(data)}")

    # obses
    obs_dir = base / "obses"
    if not obs_dir.exists():
        r.line(BAD, "missing obses/ directory")
    else:
        n_obs = _count(obs_dir, obs_suffix)
        msg = f"obses/*.{obs_suffix.lstrip('.')}: {n_obs} files"
        if n_rollout is not None:
            if n_obs == n_rollout:
                r.line(OK, msg + f" (matches {n_rollout} rollouts)")
            else:
                r.line(BAD, msg + f" (expected {n_rollout} to match states)")
        else:
            r.line(WARN, msg + " (could not compare to rollout count)")
        # try loading first obs
        first = sorted([f for f in obs_dir.iterdir() if f.name.endswith(obs_suffix)])
        if first and obs_suffix == ".pth":
            t, err = _load_tensor(first[0])
            r.line(OK if not err else BAD,
                   f"load {first[0].name}: " + (f"shape {_shape(t)}" if not err else err))
        elif first and obs_suffix == ".mp4":
            r.line(OK, f"first video present: {first[0].name}")
    return r


def check_pusht(root: Path, name="pusht_noise"):
    r = Report(name)
    base = root / name
    if not base.exists():
        r.line(BAD, f"folder missing: {base}")
        return r
    for split in ["train", "val"]:
        sp = base / split
        if not sp.exists():
            r.line(BAD, f"missing split folder: {split}/")
            continue
        n_rollout = None
        for fn in ["states.pth", "rel_actions.pth", "velocities.pth"]:
            p = sp / fn
            if not _exists(p):
                r.line(BAD, f"{split}/: missing {fn}")
                continue
            t, err = _load_tensor(p)
            if err:
                r.line(BAD, f"{split}/{fn}: load error {err}")
            else:
                r.line(OK, f"{split}/{fn}: shape {_shape(t)}")
                if fn == "states.pth":
                    try:
                        n_rollout = t.shape[0]
                    except Exception:
                        pass
        # abs_actions optional (only used if relative=False)
        if _exists(sp / "abs_actions.pth"):
            r.line(OK, f"{split}/abs_actions.pth (optional) present")
        # seq lengths pickle
        p = sp / "seq_lengths.pkl"
        if not _exists(p):
            r.line(BAD, f"{split}/: missing seq_lengths.pkl")
        else:
            data, err = _load_pickle(p)
            r.line(OK if not err else BAD,
                   f"{split}/seq_lengths.pkl: " + (f"{_shape(data)}" if not err else err))
        # obses mp4
        obs_dir = sp / "obses"
        if not obs_dir.exists():
            r.line(BAD, f"{split}/: missing obses/ directory")
        else:
            n_obs = _count(obs_dir, ".mp4")
            msg = f"{split}/obses/*.mp4: {n_obs} files"
            if n_rollout is not None:
                if n_obs == n_rollout:
                    r.line(OK, msg + f" (matches {n_rollout} rollouts)")
                else:
                    r.line(BAD, msg + f" (expected {n_rollout})")
            else:
                r.line(WARN, msg)
    return r


def check_deformable(root: Path, name="deformable", objects=("granular", "rope")):
    r = Report(name)
    base = root / name
    if not base.exists():
        r.line(BAD, f"folder missing: {base}")
        return r
    found_any = False
    for obj in objects:
        od = base / obj
        if not od.exists():
            r.line(WARN, f"object folder not present: {obj}/ (skip)")
            continue
        found_any = True
        n_rollout = None
        for fn in ["states.pth", "actions.pth"]:
            p = od / fn
            if not _exists(p):
                r.line(BAD, f"{obj}/: missing {fn}")
                continue
            t, err = _load_tensor(p)
            if err:
                r.line(BAD, f"{obj}/{fn}: load error {err}")
            else:
                r.line(OK, f"{obj}/{fn}: shape {_shape(t)}")
                if fn == "states.pth":
                    try:
                        n_rollout = t.shape[0]
                    except Exception:
                        pass
        # per-episode folders %06d/obses.pth
        ep_dirs = sorted([d for d in od.iterdir() if d.is_dir() and d.name.isdigit()])
        r.line(OK if ep_dirs else BAD, f"{obj}/: {len(ep_dirs)} episode folders")
        if ep_dirs:
            first_obs = ep_dirs[0] / "obses.pth"
            if not first_obs.exists():
                r.line(BAD, f"{obj}/{ep_dirs[0].name}/obses.pth missing")
            else:
                t, err = _load_tensor(first_obs)
                r.line(OK if not err else BAD,
                       f"{obj}/{ep_dirs[0].name}/obses.pth: "
                       + (f"shape {_shape(t)}" if not err else err))
            if n_rollout is not None and len(ep_dirs) != n_rollout:
                r.line(WARN, f"{obj}/: {len(ep_dirs)} episode folders vs {n_rollout} states rollouts")
    if not found_any:
        r.line(BAD, "no object subfolders (granular/rope) found")
    return r


CHECKS = {
    "point_maze": lambda root: check_flat(
        root, "point_maze",
        required_tensors=["states.pth", "actions.pth"],
        obs_suffix=".pth", seq_file="seq_lengths.pth", seq_is_pickle=False),
    "point_maze_medium": lambda root: check_flat(
        root, "point_maze_medium",
        required_tensors=["states.pth", "actions.pth"],
        obs_suffix=".pth", seq_file="seq_lengths.pth", seq_is_pickle=False),
    "wall_single": lambda root: check_flat(
        root, "wall_single",
        required_tensors=["states.pth", "actions.pth",
                          "door_locations.pth", "wall_locations.pth"],
        obs_suffix=".pth", seq_file=None),
    "pusht_noise": lambda root: check_pusht(root),
    "deformable": lambda root: check_deformable(root),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("DATASET_DIR"),
                    help="dataset root (defaults to $DATASET_DIR)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of datasets to check")
    args = ap.parse_args()

    if not args.root:
        print("ERROR: set DATASET_DIR or pass --root /path/to/data")
        sys.exit(2)
    root = Path(args.root)
    print(f"Dataset root: {root}\n")
    if torch is not None:
        print(f"torch {torch.__version__}\n")

    # point_maze_medium is not part of the default Table-1 reproduction and is not
    # shipped on every pod, so it is only checked when explicitly requested via --only.
    default_names = [n for n in CHECKS.keys() if n != "point_maze_medium"]
    names = args.only if args.only else default_names
    all_pass = True
    for name in names:
        if name not in CHECKS:
            print(f"[skip] unknown dataset '{name}'")
            continue
        rep = CHECKS[name](root)
        rep.dump()
        all_pass = all_pass and rep.passed

    print("=" * 40)
    print("OVERALL:", "ALL CHECKED DATASETS PASS" if all_pass else "SOME DATASETS FAILED")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
