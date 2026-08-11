#!/usr/bin/env python3
"""Interactively launch TemporalDP training and checkpoint evaluations.

Run from the repository root, preferably from the conda environment required
by the selected experiment:

    python robomimic/scripts/experiment_cli.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from typing import Sequence, TypeVar


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "robomimic" / "exps" / "temporaldp"
PACKAGE_ROOT = REPO_ROOT / "robomimic"
BASE_CONDA_ENV = "robomimic2"
TEMPORAL_CONDA_ENV = "robomimic2_temporalenvs"

T = TypeVar("T")


@dataclass(frozen=True)
class Experiment:
    environment: str
    algorithm: str
    config_path: Path
    split: str | None = None

    @property
    def is_temporal(self) -> bool:
        return self.environment in {"waitatgoal", "liftqa"}

    @property
    def label(self) -> str:
        split = f" / {self.split}" if self.split else ""
        return f"{self.algorithm}{split}"


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    run_id: str
    success: float | None
    epoch: int | None

    @property
    def label(self) -> str:
        details = []
        if self.success is not None:
            details.append(f"best success={self.success:.2f}")
        if self.epoch is not None:
            details.append(f"epoch={self.epoch}")
        suffix = f" ({', '.join(details)})" if details else ""
        return f"{self.run_id}{suffix} / {self.path.name}"


def prompt_choice(title: str, options: Sequence[T], label) -> T | None:
    if not options:
        print(f"No {title.lower()} available.")
        return None

    print(f"\n{title}")
    for index, option in enumerate(options, start=1):
        print(f"  {index:>2}. {label(option)}")
    print("   0. Cancel")

    while True:
        answer = input("Select an option: ").strip()
        if answer == "0":
            return None
        try:
            selected = int(answer)
        except ValueError:
            print("Enter an option number.")
            continue
        if 1 <= selected <= len(options):
            return options[selected - 1]
        print(f"Choose a number from 0 to {len(options)}.")


def prompt_text(label: str, default: str | None = None, validator=None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        answer = input(f"{label}{suffix}: ").strip()
        value = answer or default
        if value is None:
            print("A value is required.")
            continue
        if validator is not None and not validator(value):
            print("That value is not valid.")
            continue
        return value


def prompt_yes_no(question: str, default: bool = False) -> bool:
    marker = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{question} [{marker}]: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def parse_seeds(value: str) -> list[str]:
    seeds = [seed.strip() for seed in value.split(",")]
    if not seeds or any(not seed.isdigit() for seed in seeds):
        raise ValueError("Seeds must be comma-separated non-negative integers.")
    return seeds


def prompt_seeds(label: str, default: str) -> list[str]:
    while True:
        value = prompt_text(label, default=default)
        try:
            return parse_seeds(value)
        except ValueError as exc:
            print(exc)


def discover_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []
    for config_path in CONFIG_ROOT.rglob("*.json"):
        relative = config_path.relative_to(CONFIG_ROOT)
        parts = relative.parts
        if len(parts) == 4 and parts[2] == "image":
            environment, split, _, filename = parts
            experiments.append(Experiment(environment, config_path.stem, config_path, split))
        elif len(parts) == 3 and parts[0] == "temporal":
            _, environment, filename = parts
            experiments.append(Experiment(environment, config_path.stem, config_path))
    return sorted(experiments, key=lambda item: (item.environment, item.split or "", item.algorithm))


def select_experiment(experiments: Sequence[Experiment]) -> Experiment | None:
    environments = sorted({experiment.environment for experiment in experiments})
    environment = prompt_choice("Environment", environments, str)
    if environment is None:
        return None
    candidates = [experiment for experiment in experiments if experiment.environment == environment]
    return prompt_choice("Algorithm and dataset split", candidates, lambda experiment: experiment.label)


def load_config(experiment: Experiment) -> dict:
    with experiment.config_path.open() as file:
        return json.load(file)


def checkpoint_action_normalization(checkpoint: Checkpoint) -> str | None:
    """Read the immutable run config without unpickling the checkpoint."""
    config_path = checkpoint.path.parent.parent / "config.json"
    try:
        with config_path.open() as file:
            return json.load(file)["train"]["action_config"]["actions"].get("normalization")
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return None


def checkpoint_root(experiment: Experiment) -> Path:
    config = load_config(experiment)
    output_dir = Path(config["train"]["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = PACKAGE_ROOT / output_dir
    return output_dir.resolve() / config["experiment"]["name"]


def discover_checkpoints(experiment: Experiment) -> list[Checkpoint]:
    root = checkpoint_root(experiment)
    checkpoints_by_run: dict[str, list[Checkpoint]] = {}
    for path in root.glob("*/models/*.pth"):
        success_match = re.search(r"success_([0-9]+(?:\.[0-9]+)?)", path.name)
        epoch_match = re.search(r"epoch_(\d+)", path.name)
        checkpoint = Checkpoint(
            path=path,
            run_id=path.parent.parent.name,
            success=float(success_match.group(1)) if success_match else None,
            epoch=int(epoch_match.group(1)) if epoch_match else None,
        )
        checkpoints_by_run.setdefault(checkpoint.run_id, []).append(checkpoint)

    def checkpoint_rank(checkpoint: Checkpoint) -> tuple[bool, float, int, float]:
        return (
            checkpoint.success is not None,
            checkpoint.success if checkpoint.success is not None else float("-inf"),
            checkpoint.epoch if checkpoint.epoch is not None else -1,
            checkpoint.path.stat().st_mtime,
        )

    best_checkpoints = [
        max(run_checkpoints, key=checkpoint_rank)
        for run_checkpoints in checkpoints_by_run.values()
    ]
    return sorted(best_checkpoints, key=lambda item: item.run_id, reverse=True)


def conda_executable() -> str:
    candidates = [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        str(Path.home() / "miniconda3" / "bin" / "conda"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("Could not find conda. Set CONDA_EXE or add conda to PATH.")


def launch(command: list[str], gpu: str, temporal: bool, dry_run: bool, confirm: bool = True) -> int:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["MUJOCO_GL"] = "egl"
    if temporal:
        environment.update({"SDL_VIDEODRIVER": "dummy", "NUMBA_DISABLE_JIT": "1"})
    conda_env = TEMPORAL_CONDA_ENV if temporal else BASE_CONDA_ENV
    runtime_command = [conda_executable(), "run", "--no-capture-output", "-n", conda_env, *command]

    env_prefix = " ".join(
        f"{name}={shlex.quote(environment[name])}"
        for name in ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "SDL_VIDEODRIVER", "NUMBA_DISABLE_JIT")
        if name in environment
    )
    print("\nCommand:")
    print(f"  {env_prefix} {shlex.join(runtime_command)}")
    if dry_run:
        print("Dry run: command not launched.")
        return 0
    if confirm and not prompt_yes_no("Launch this command?", default=False):
        print("Skipped.")
        return 0
    return subprocess.run(runtime_command, cwd=REPO_ROOT, env=environment, check=False).returncode


def run_progress_script(script_name: str) -> None:
    command = [
        conda_executable(),
        "run",
        "--no-capture-output",
        "-n",
        BASE_CONDA_ENV,
        "python",
        f"robomimic/scripts/{script_name}",
    ]
    print(f"\nRunning: {shlex.join(command)}")
    exit_code = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
    if exit_code:
        print(f"{script_name} exited with status {exit_code}.")


def run_training(experiments: Sequence[Experiment], dry_run: bool) -> None:
    experiment = select_experiment(experiments)
    if experiment is None:
        return
    gpu = prompt_text("GPU", default="0", validator=lambda value: bool(value))
    seeds = prompt_seeds("Training seed(s), comma-separated", default="1")
    if len(seeds) > 1 and not dry_run:
        if not prompt_yes_no(f"Launch {len(seeds)} training jobs sequentially?", default=False):
            print("Skipped.")
            return

    for seed in seeds:
        command = [
            "python",
            "robomimic/scripts/train.py",
            "--config",
            str(experiment.config_path.relative_to(REPO_ROOT)),
            "--seed",
            seed,
        ]
        exit_code = launch(command, gpu, experiment.is_temporal, dry_run, confirm=len(seeds) == 1)
        if exit_code:
            print(f"Training seed {seed} exited with status {exit_code}.")


def rollout_output_path(experiment: Experiment, checkpoint: Checkpoint, seed: str, multi_seed: bool) -> Path:
    if experiment.is_temporal:
        parent = REPO_ROOT / "rollouts" / "temporal" / experiment.environment / experiment.algorithm
    else:
        assert experiment.split is not None
        parent = REPO_ROOT / "rollouts" / experiment.environment / experiment.split / experiment.algorithm
    seed_suffix = f"_seed{seed}" if multi_seed else ""
    return parent / f"{checkpoint.run_id}_{checkpoint.path.stem}{seed_suffix}.hdf5"


def write_eval_info(
    info_path: Path,
    experiment: Experiment,
    checkpoint: Checkpoint,
    dataset_path: Path,
    gpu: str,
    seed: str,
    n_rollouts: str,
) -> None:
    info_path.parent.mkdir(parents=True, exist_ok=True)
    fields = {
        "RUN_ID": checkpoint.run_id,
        "SEED": seed,
        "N_ROLLOUTS": n_rollouts,
        "ENV": experiment.environment,
        "ALGO": experiment.algorithm,
        "GPU": gpu,
        "SPLIT": experiment.split or "temporal",
        "AGENT": str(checkpoint.path),
        "DATASET": str(dataset_path),
    }
    info_path.write_text("".join(f"{key}={value}\n" for key, value in fields.items()))


def run_evaluation(experiments: Sequence[Experiment], dry_run: bool) -> None:
    experiment = select_experiment(experiments)
    if experiment is None:
        return
    checkpoint = prompt_choice(
        "Timestamped runs (best checkpoint selected automatically)",
        discover_checkpoints(experiment),
        lambda item: item.label,
    )
    if checkpoint is None:
        return
    action_normalization = checkpoint_action_normalization(checkpoint)
    if experiment.is_temporal and action_normalization is None:
        print(
            "\nWarning: this temporal configuration uses unnormalised coordinate actions. "
            "Existing checkpoints trained from it commonly saturate near [1, 1] and "
            "collide after a few steps. This evaluation will still run and be written, "
            "but retrain with the updated min_max configuration for meaningful rollouts."
        )
    gpu = prompt_text("GPU", default="0", validator=lambda value: bool(value))
    default_seed = str(int(checkpoint.run_id) % 4_294_967_295) if checkpoint.run_id.isdigit() else "1"
    seeds = prompt_seeds("Evaluation seed(s), comma-separated", default=default_seed)
    default_n_rollouts = "200" if experiment.is_temporal else "100"
    n_rollouts = prompt_text(
        "Number of rollouts",
        default=default_n_rollouts,
        validator=lambda value: value.isdigit() and int(value) > 0,
    )
    if not dry_run and not prompt_yes_no(
        f"Write metadata and launch {len(seeds)} evaluation job(s) sequentially?", default=False
    ):
        print("Skipped.")
        return

    for seed in seeds:
        dataset_path = rollout_output_path(experiment, checkpoint, seed, multi_seed=len(seeds) > 1)
        info_path = dataset_path.with_name(f"{dataset_path.stem}_info.txt")
        command = [
            "python",
            "robomimic/scripts/run_trained_agent.py",
            "--agent",
            str(checkpoint.path),
            "--dataset_path",
            str(dataset_path),
            "--n_rollouts",
            n_rollouts,
            "--seed",
            seed,
            "--dataset_obs",
        ]
        if not dry_run:
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            write_eval_info(info_path, experiment, checkpoint, dataset_path, gpu, seed, n_rollouts)
        exit_code = launch(command, gpu, experiment.is_temporal, dry_run, confirm=False)
        if exit_code:
            print(f"Evaluation seed {seed} exited with status {exit_code}.")


def select_progress_runs_for_deletion():
    # Reuse the progress report's own collection, sorting, and status logic so
    # the deletion menu reports the same state as ``check_progress.py``.
    from check_progress import collect_progress, format_progress, run_status, sort_key

    root = REPO_ROOT / "trained_models"
    runs = sorted(collect_progress(root), key=sort_key)
    if not runs:
        print(f"No timestamped training runs found under {root}.")
        return None

    now = time.time()
    active_within_seconds = 300
    print("\nTimestamped training runs (active runs cannot be deleted)")
    for index, run in enumerate(runs, start=1):
        print(f"  {index:>2}. {format_progress(run, width=16, now=now, active_within_seconds=active_within_seconds)}")
    print("   0. Cancel")

    while True:
        answer = input("Select run(s) to delete, comma-separated: ").strip()
        if answer == "0":
            return None
        try:
            indices = [int(index.strip()) for index in answer.split(",")]
        except ValueError:
            print("Enter one or more comma-separated option numbers.")
            continue
        if not indices or len(set(indices)) != len(indices) or any(not 1 <= index <= len(runs) for index in indices):
            print(f"Choose unique numbers from 1 to {len(runs)}, or 0 to cancel.")
            continue
        selected_runs = [runs[index - 1] for index in indices]
        active_runs = [
            run for run in selected_runs
            if run_status(run, now=now, active_within_seconds=active_within_seconds) == "active"
        ]
        if active_runs:
            print(
                "Active run(s) cannot be deleted: "
                + ", ".join(run.run_dir.name for run in active_runs)
            )
            return None
        return selected_runs


def delete_run_directories(dry_run: bool) -> None:
    runs = select_progress_runs_for_deletion()
    if runs is None:
        return
    root = (REPO_ROOT / "trained_models").resolve()
    targets = [run.run_dir.resolve() for run in runs]
    for target in targets:
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Refusing to delete a path outside trained_models: {target}") from exc
        if not target.is_dir() or not target.name.isdigit() or len(target.name) != 14:
            raise ValueError(f"Refusing to delete a non-timestamped run directory: {target}")

    timestamps = ",".join(target.name for target in targets)
    confirmation = prompt_text(
        f"Type {timestamps} to permanently delete the selected runs",
        default=None,
    )
    if confirmation != timestamps:
        print("Timestamp did not match; nothing was deleted.")
        return
    if dry_run:
        print("Dry run: would delete " + ", ".join(str(target) for target in targets))
        return
    for target in targets:
        shutil.rmtree(target)
        print(f"Deleted {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching them or writing metadata.")
    args = parser.parse_args()

    experiments = discover_experiments()
    if not experiments:
        raise FileNotFoundError(f"No TemporalDP configs found under {CONFIG_ROOT}")

    try:
        while True:
            action = prompt_choice(
                "Action",
                ["Train", "Evaluate", "Check training progress", "Check rollout progress", "Delete timestamped training run"],
                str,
            )
            if action is None:
                return
            if action == "Train":
                run_training(experiments, args.dry_run)
            elif action == "Evaluate":
                run_evaluation(experiments, args.dry_run)
            elif action == "Check training progress":
                run_progress_script("check_progress.py")
            elif action == "Check rollout progress":
                run_progress_script("check_progress_rollout.py")
            else:
                delete_run_directories(args.dry_run)
            if not prompt_yes_no("Would you like to run another?", default=False):
                return
    except (EOFError, KeyboardInterrupt):
        print("\nExited.")


if __name__ == "__main__":
    main()
