# Temporal Straightening for Latent Planning

**Abstract:** Learning good representations is essential for latent planning with world models. While pretrained visual encoders produce strong semantic visual features, they are not tailored to planning and contain information irrelevant—or even detrimental—to planning. Inspired by the perceptual straightening hypothesis in human visual processing, we introduce temporal straightening to improve representation learning for latent planning. Using a curvature regularizer that encourages locally straightened latent trajectories, we jointly learn an encoder and a predictor. We show that reducing curvature this way makes the Euclidean distance in latent space a better proxy for the geodesic distance and improves the conditioning of the planning objective. We demonstrate empirically that temporal straightening makes gradient-based planning more stable and yields significantly higher success rates across a suite of goal-reaching tasks.

<p align="center">
  &#151; <a href="https://agenticlearning.ai/temporal-straightening/"><b>View Paper Website</b></a> &#151;
</p>

![teaser_figure](assets/architecture.png)


## Reproducing on a fresh NVIDIA B200 pod (this fork)

> This fork reproduces Table 1 on a single **NVIDIA B200** MIG slice (no SLURM, no conda,
> NGC PyTorch container). The upstream `environment.yaml` pins torch 2.3, which does **not**
> support Blackwell (sm_100) — use the scripted path below instead. The trained model is not
> stored in git; it lives on the Hugging Face Hub and is fetched with an API key.

**1. Clone + reproduce the environment (torch 2.7 cu128 + training + planning deps):**
```bash
git clone https://github.com/Subaru-5999/temporal_straightening
cd temporal_straightening

# env only:
bash reproduce_env.sh

# ...or env + fetch the trained model from Hugging Face in one go:
export HF_TOKEN=hf_xxx                     # https://huggingface.co/settings/tokens (read is enough to pull)
bash reproduce_env.sh --with-model
```
`reproduce_env.sh` chains `setup_b200.sh` (Blackwell torch + training deps) and
`setup_planning.sh` (MuJoCo 210 / gym / mujoco-py / d4rl / PushT). Override paths with
`DATASET_ROOT=...` and the model repo with `HF_REPO_ID=...` (default
`gravycrazy/temporal_straightening`).

**2. Fetch / back up the trained model on the Hugging Face Hub (API key):**
```bash
export HF_TOKEN=hf_xxx
# restore the model onto a fresh pod:
python hf_backup.py pull gravycrazy/temporal_straightening checkpoints/repro/test
# back a freshly trained run up (so a lost pod never costs a retrain):
python hf_backup.py push checkpoints/repro/test/<run_folder>
```

**3. Verify + evaluate:**
```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"  # expect 2.7.0+cu128 (10, 0)
export DATASET_DIR=/workspace/temporal_s/data
bash eval_pusht_3seeds.sh checkpoints/repro/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
```

See `REPRODUCTION.md` for the full guide (per-row settings, results, and pitfalls) and
`verify_datasets.py` to confirm the datasets are intact before training.

## Getting Started

1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Training](#training)
4. [Planning](#planning)

## Installation

```bash
git clone git@github.com:agentic-learning-ai-lab/temporal-straightening.git
cd temporal-straightening
conda env create -f environment.yaml
conda activate ts
```

### Mujoco
                    
Create the `.mujoco` directory and download Mujoco210 using `wget`:

```bash
mkdir -p ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/.mujoco/
cd ~/.mujoco
tar -xzvf mujoco210-linux-x86_64.tar.gz
```

Append the following lines to your `~/.bashrc`:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<username>/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

Reload your shell configuration to apply the environment variable changes:

```bash
source ~/.bashrc
```

For more details, check [DINO-WM](https://github.com/gaoyuezhou/dino_wm).

## Datasets

We use the datasets from [DINO-WM](https://github.com/gaoyuezhou/dino_wm), which can be downloaded [here](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28). Unzip the datasets and set an environment variable pointing to your dataset folder:
```bash
# Replace /path/to/data with the actual path to your dataset folder.
export DATASET_DIR=/path/to/data
```
Inside the dataset folder, you should find the following structure:
```
data
├── deformable
│   ├── granular
│   └── rope
├── point_maze
├── pusht_noise
└── wall_single
```

Improve data loading:
- For PointMaze (`umaze` and `medium`), you might want to use preprocessed data + per-frame loading (`use_preprocessed=true`, `use_frame_files=true`) in `conf/env/point_maze.yaml` and `conf/env/point_maze_medium.yaml` when available.
- This is often helpful for global-feature runs (for example `encoder=dino_global`) on HPC, where training can otherwise be I/O-bound.
- Since there can be many small frame files, you might need squashFS (for singularity) or HDF5. NYU HPC has a good summary on [handling large numbers of small files](https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/ai-at-hpc-tips/large-number-of-small-files).
- Make sure `dataset.data_path` points to your preprocessed folders.


## Training
In `conf/train.yaml` (paper setup), default is `frameskip=5`, `num_hist=3`, with frozen DINOv2(patch) backbone. We use stop grad by default to prevent collapse but also have support for vc reg. Don't forget to set output path in `conf/train.yaml`!

Base command:
```bash
python train.py --config-name train.yaml env=point_maze
```

Variant overrides (append to the base command):
```bash
# DINOv2(patch) baseline (no projector, no straightening)
encoder=dino training.straighten=False

# DINOv2(patch) + channel projector
encoder=dino_channel training.straighten=[False|aggcos1e-1]

# DINOv2(patch) + global projector
encoder=dino_global training.straighten=[False|cos1e-1]

# ResNet spatial features (from scratch)
encoder=scratch_resnet_spatial training.straighten=[False|aggcos1e-1]

# ResNet global features (from scratch)
encoder=scratch_resnet training.straighten=[False|cos1e-1]
```

Straightening options:
- `training.straighten=False` disables straightening.
- `training.straighten=cos1e-1` enables patch-wise curvature regularization.
- `training.straighten=aggcos1e-1` enables pooled-feature curvature regularization.

To change pooling head (agg_type can be `mlp|flatten|mean`), check 

- Channel projector config: `conf/encoder/dino_channel.yaml`
  - `agg_type`, `agg_out_dim`, `agg_mlp_hidden_dim`
- ResNet spatial config: `conf/encoder/scratch_resnet_spatial.yaml`
  - `agg_type`, `agg_out_dim`, `agg_mlp_hidden_dim`

You can edit these files directly or override from CLI, e.g.:
```bash
python train.py --config-name train.yaml env=point_maze \
  encoder=dino_channel \
  encoder.agg_type=mlp \
  encoder.agg_out_dim=128 \
  training.straighten=aggcos1e-1
```


# Planning
Use one of the three planning configs:
- `plan_gd.yaml` (open-loop GD)
- `plan_cem.yaml` (open-loop CEM)
- `plan_gd_mpc.yaml` (MPC + GD sub-planner)

Example commands:
```bash
python plan.py --config-name plan_gd.yaml ckpt_base_path=<ckpt_root> model_name=<model_name>
python plan.py --config-name plan_cem.yaml ckpt_base_path=<ckpt_root> model_name=<model_name>
python plan.py --config-name plan_gd_mpc.yaml ckpt_base_path=<ckpt_root> model_name=<model_name>
```

Notes:
- PushT: use the same configs, but set `objective.alpha=1` (and for GD-MPC also set `objective.mode=staged`).
- Frameskip: planning reads `frameskip` from the saved training config and `plan.py` handles it (`goal_H`, `n_taken_actions`, and `sub_planner.horizon` will be divided by `frameskip`). Keep horizons divisible by `frameskip` to avoid truncation or shape mismatch.


## Acknowledgement

This repository is adapted from the excellent [DINO-WM](https://github.com/gaoyuezhou/dino_wm) codebase. We are grateful to the DINO-WM authors for sharing a clean, well-structured, and highly useful open-source implementation.


## Citation

If you find this repo useful, please cite:

```
@article{wang2026temporal_straightening,
  title={Temporal Straightening for Latent Planning},
  author={Wang, Ying and Bounou, Oumayma and Zhou, Gaoyue and Balestriero, Randall and Rudner, Tim GJ and LeCun, Yann and Ren, Mengye},
  journal={arXiv preprint arXiv:2603.12231},
  year={2026}
}
```
