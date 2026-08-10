# Temporal environment BC baselines

Install the optional environment dependencies before using train-time rollouts:

```bash
pip install -e ".[temporal-envs]"
```

Convert the source demonstrations before training. The converter preserves the
LDP split rule (2% validation, seed 42) and writes `mask/train` and
`mask/valid` filter keys.

```bash
python robomimic/scripts/conversion/zarr_to_temporal_hdf5.py \
  --task waitatgoal \
  --source /mnt/shared/TemporalDiffusionPolicy/data/demonstration/wait_at_goal_dataset_v4.zarr \
  --output datasets/temporal/waitatgoal_image.hdf5

python robomimic/scripts/conversion/zarr_to_temporal_hdf5.py \
  --task liftqa \
  --source /mnt/shared/TemporalDiffusionPolicy/data/demonstration/lift_qa_v2.zarr \
  --output datasets/temporal/liftqa_image.hdf5
```

The twelve configs enable robomimic's standard train-time rollout loop through
the native `EnvTemporal` adapter. It applies the LiftQA 3-D-to-7-D action map,
uses the tasks' final completion criterion for success, and supplies robomimic
with uint8 HWC images.

Every conversion verifies its schema, LDP-compatible split, actions, poses,
terminal flags, and pixel round-trip before reporting success. Re-check an
existing output without rewriting it by adding `--verify` to the same command.

For each environment, the following variants are available:

- `bc` / `bc_rnn`: the existing `can/ph/image` visual settings
  (SpatialSoftmax with a 76×76 random crop).
- `bc_mod` / `bc_rnn_mod`: SpatialMeanPool with the same crop.
- `bc_mod_nocrop` / `bc_rnn_mod_nocrop`: SpatialMeanPool with no crop.

Train from the repository root, for example:

```bash
python robomimic/scripts/train.py --config robomimic/exps/temporaldp/temporal/waitatgoal/bc.json
python robomimic/scripts/train.py --config robomimic/exps/temporaldp/temporal/liftqa/bc_rnn_mod_nocrop.json
```
