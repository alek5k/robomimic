# Overview

https://robomimic.github.io/study/
https://github.com/alexander-soare/little_experiments/blob/main/diffusion_spatial_softmax.md

### Installing
https://robomimic.github.io/docs/introduction/installation.html

conda create -n robomimic2 python=3.10
conda activate robomimic2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/alek5k/robomimic
cd robomimic
git switch TCDP
pip install -e .
pip install "robosuite==1.5.1"

### Download dataset:
https://robomimic.github.io/docs/datasets/robomimic_v0.1.html
https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/download_datasets.py
Robomimic "MH" - or mixed proficiency - was the one the reviewer recommended.

```
cd robomimic/scripts
python download_datasets.py --tasks sim --dataset_types mh --hdf5_types raw --dry_run
python download_datasets.py --tasks sim --dataset_types ph --hdf5_types raw 
```

### Postprocess dataset:
https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/scripts/extract_obs_from_raw_datasets.sh

```
chmod +x extract_obs_from_raw_datasets.sh
extract_obs_from_raw_datasets.sh
```

Assume datasets already exist in robomimic/../datasets folder. Configs will be generated under robomimic/exps/paper, and training results will be at /tmp/experiment_results when launching training runs.
python robomimic/scripts/generate_paper_configs.py

### Run experiment
python robomimic/scripts/train.py --config robomimic/exps/paper/core/can/mh/image/bc.json --dataset datasets/can/mh/image_v15.hdf5
python robomimic/scripts/train.py --config robomimic/exps/paper/core/can/mh/image/bc_rnn.json --dataset datasets/can/mh/image_v15.hdf5
python robomimic/scripts/train.py --config robomimic/exps/paper/core/can/mh/image/bc_rnn.json --dataset datasets/can/mh/image_v15.hdf5


# Experiments
Note: the datasets are specified in the config file, so no need to pass explicitly


## Training
BC, BC-RNN, DP and TCDP

### Interactive train / evaluation launcher
robomimic_cli

Run this from the appropriate conda environment. It discovers the available
TemporalDP configs, prompts for environment, algorithm, GPU, and seed, and
finds timestamped checkpoints for evaluation. After each job it offers to run
another one.

```bash
python robomimic/scripts/experiment_cli.py
```

Use `--dry-run` to inspect the constructed command without launching it.
Seeds may be entered as a comma-separated list (for example, `1,2,3`) and
will run sequentially on the selected GPU.
The same menu can run `check_progress.py` or `check_progress_rollout.py`, and
can show a numbered training-progress list for deletion. Active runs are
protected; stale or completed runs can be selected with comma-separated
numbers, then require retyping their comma-separated timestamps.

Single seed:
```bash
ENV=square
SPLIT=ph
ALGO=tc_diffusion_policy_mod
GPU=1
SEED=3
gorobomimic && CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_GL=egl python robomimic/scripts/train.py --config robomimic/exps/temporaldp/${ENV}/${SPLIT}/image/${ALGO}.json --seed=${SEED}
```

Multiple seeds:
```bash
ENV=square
SPLIT=ph
ALGO=bc
GPU=0
for SEED in 1 2 3; do
    gorobomimic && CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_GL=egl python robomimic/scripts/train.py --config robomimic/exps/temporaldp/${ENV}/${SPLIT}/image/${ALGO}.json --seed=${SEED}
done
---
ENV=square
SPLIT=ph
ALGO=bc_rnn
GPU=0
for SEED in 1 2 3; do
    gorobomimic && CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_GL=egl python robomimic/scripts/train.py --config robomimic/exps/temporaldp/${ENV}/${SPLIT}/image/${ALGO}.json --seed=${SEED}
done
---
ENV=transport
SPLIT=ph
ALGO=diffusion_policy
GPU=0
for SEED in 1 2 3; do
    gorobomimic && CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_GL=egl python robomimic/scripts/train.py --config robomimic/exps/temporaldp/${ENV}/${SPLIT}/image/${ALGO}.json --seed=${SEED}
done
---
ENV=transport
SPLIT=ph
ALGO=tc_diffusion_policy_mod
GPU=1
for SEED in 1 2 3; do
    gorobomimic && CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_GL=egl python robomimic/scripts/train.py --config robomimic/exps/temporaldp/${ENV}/${SPLIT}/image/${ALGO}.json --seed=${SEED}
done
---
```

## Evaluations
```bash
for RUN_ID in 20260511134003 20260512121109 20260512121147; do
    N_ROLLOUTS=100
    ENV=square
    ALGO=tc_diffusion_policy_mod
    GPU=1
    SPLIT=ph
    SEED=$((RUN_ID % 4294967295))
    BASE="trained_models/temporaldp/${ALGO}/${ENV}/${SPLIT}/image/trained_models/temporaldp_${ALGO}_${ENV}_${SPLIT}_image/${RUN_ID}"
    AGENT=$(find "$BASE/models" -name "*.pth" | awk 'match($0,/success_([0-9.]+)/){s=substr($0,RSTART+8,RLENGTH-8); print s,$0}' | sort -nr | head -n1 | cut -d" " -f2-)
    NAME=$(basename "$AGENT")
    DATASET="rollouts/${ENV}/${SPLIT}/${ALGO}/${RUN_ID}_${NAME}.hdf5"
    INFO="rollouts/${ENV}/${SPLIT}/${ALGO}/${RUN_ID}_${NAME}_info.txt"

    mkdir -p "$(dirname "$INFO")"

    printf "RUN_ID=%s\nSEED=%s\nN_ROLLOUTS=%s\nENV=%s\nALGO=%s\nGPU=%s\nSPLIT=%s\nBASE=%s\nAGENT=%s\nNAME=%s\nDATASET=%s\n" "$RUN_ID" "$SEED" "$N_ROLLOUTS" "$ENV" "$ALGO" "$GPU" "$SPLIT" "$BASE" "$AGENT" "$NAME" "$DATASET" > "$INFO"

    gorobomimic && CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_GL=egl python robomimic/scripts/run_trained_agent.py --agent="$AGENT" --dataset_path="$DATASET" --n_rollouts=${N_ROLLOUTS} --seed=${SEED} --dataset_obs
done
```

## WaitAtGoal and LiftQA
BC:
conda activate robomimic2_temporalenvs && cd /home/sydney1/Repos/robomimic && export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy NUMBA_DISABLE_JIT=1

python robomimic/scripts/train.py \
  --config robomimic/exps/temporaldp/temporal/waitatgoal/bc_mod_nocrop.json

```bash
ENV=liftqa
ALGO=bc_rnn_mod_nocrop
GPU=0
for SEED in 1 2 3; do
    conda activate robomimic2_temporalenvs && cd /home/sydney1/Repos/robomimic && export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy NUMBA_DISABLE_JIT=1 && CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_GL=egl python robomimic/scripts/train.py --config robomimic/exps/temporaldp/temporal/${ENV}/${ALGO}.json --seed=${SEED}
done
```

DP and TC-DP
```bash
ENV=waitatgoal
ALGO=tc_diffusion_policy_mod_nocrop
GPU=1
for SEED in 1 2 3; do
    conda activate robomimic2_temporalenvs && cd /home/sydney1/Repos/robomimic && export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy NUMBA_DISABLE_JIT=1 && CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_GL=egl python robomimic/scripts/train.py --config robomimic/exps/temporaldp/temporal/${ENV}/${ALGO}.json --seed=${SEED}
done
```


# Modification Notes:
HBC failed because of image layout.

robomimic/algo/algo.py needs to be patched, so `postprocess_batch_for_training` includes these keys:

obs_keys = ["obs", "next_obs", "goal_obs", "subgoals", "target_subgoals"]

segfault encountered.
set num_data_workers to 0










# older
MUJOCO_GL="egl" python robomimic/scripts/run_trained_agent.py \
--agent=trained_models/temporaldp/bc/can/mh/image/trained_models/temporaldp_bc_can_mh_image/20260429222537/models/model_epoch_520_image_v15_success_0.9.pth \
--dataset_path=rollouts/can/bc/20260429222537.hdf5 \
--n_rollouts=100 \
--seed=300
