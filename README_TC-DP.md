# Overview
This is a fork of the [Robomimic study](https://robomimic.github.io/study/) study and [repository](https://github.com/ARISE-Initiative/robomimic) for the Temporally-Conditioned Diffusion Policy Paper.


Temporal conditioning encodings are implemented in:
`robomimic/utils/temporal_encodings.py`

Temporal encoding wrapper is implemented in:
`robomimic/envs/wrappers.py`, class `TemporalEncodingWrapper`

Temporal analysis is implemented in `scripts/analysis_temporal.py`


### Installing
Specific installation instructions for robomimic can be found [here](https://robomimic.github.io/docs/introduction/installation.html).

```bash
conda create -n robomimic2 python=3.10
conda activate robomimic2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
cd robomimic
pip install -e .
pip install "robosuite==1.5.1"
```

### Download dataset
Dataset info can be found [here](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html).

```
cd robomimic/scripts
python download_datasets.py --tasks sim --dataset_types ph --hdf5_types raw 
```

Then postprocess the dataset to get image datasets:

```
./extract_obs_from_raw_datasets.sh
```

### Download TC-DP trained checkpoints and inference datasets
`./download_checkpoints_and_inference_datasets.sh`

### Config Setup
The configs for the paper are already included in the repository. Within these configs, the `train/data` config variables need to be set to correspond to your repo location in the following configs:

```
robomimic/robomimic/exps/temporaldp/can/ph/image/bc.json
robomimic/robomimic/exps/temporaldp/can/ph/image/bc_rnn.json
robomimic/robomimic/exps/temporaldp/can/ph/image/diffusion_policy.json
robomimic/robomimic/exps/temporaldp/can/ph/image/tc_diffusion_policy.json
```

### Training
Replace ALGO below with:
bc, bc_rnn, diffusion_policy and tc_diffusion_policy.

```bash
conda activate robomimic2 && cd ~/Repos/robomimic
ENV=lift
SPLIT=ph
ALGO=tc_diffusion_policy_mod
GPU=1
for SEED in 1 2 3; do
    CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_GL=egl python robomimic/scripts/train.py --config robomimic/exps/temporaldp/${ENV}/${SPLIT}/image/${ALGO}.json --seed=${SEED}
done
```
 
### Evaluations
Replace with your RUN_ID generated during training.
```bash
conda activate robomimic2 && cd ~/Repos/robomimic
for RUN_ID in 20260715120741 20260716000116 20260716105927; do
    N_ROLLOUTS=100
    ENV=lift
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

    CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_GL=egl python robomimic/scripts/run_trained_agent.py --agent="$AGENT" --dataset_path="$DATASET" --n_rollouts=${N_ROLLOUTS} --seed=${SEED} --dataset_obs
done
```
