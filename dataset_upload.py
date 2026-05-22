from __future__ import annotations

import argparse
import re
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "erdrsvhr9/temporally_conditioned_diffusion_policy_robomimic"
ROOT = Path(__file__).resolve().parent
ALGO_FOLDERS = ("tc_diffusion_policy", "bc", "bc_rnn", "diffusion_policy")
TASK_FOLDERS = ("can",)
ROLLOUT_TASK = "can"
CHECKPOINT_RE = re.compile(r"success_(\d+(?:\.\d+)?)\.pth$")
EPOCH_RE = re.compile(r"model_epoch_(\d+)_")


def log(message: str) -> None:
    print(f"[dataset_upload] {message}", flush=True)


def should_include_file(path: Path) -> bool:
    if path.name == "log.txt":
        return False
    if ".tfevents." in path.name:
        return False
    return True


def best_checkpoint(models_dir: Path) -> Path | None:
    best_file = None
    best_success = -1.0
    best_epoch = -1

    for path in models_dir.glob("*success_*.pth"):
        success_match = CHECKPOINT_RE.search(path.name)
        if success_match is None:
            continue
        success = float(success_match.group(1))

        epoch_match = EPOCH_RE.search(path.name)
        epoch = int(epoch_match.group(1)) if epoch_match else -1

        if success > best_success or (success == best_success and epoch > best_epoch):
            best_file = path
            best_success = success
            best_epoch = epoch

    return best_file


def build_allow_patterns() -> list[str]:
    allow_patterns: list[str] = []
    trained_base = ROOT / "trained_models" / "temporaldp"

    log(f"Algorithms: {' '.join(ALGO_FOLDERS)}")
    log(f"Tasks: {' '.join(TASK_FOLDERS)}")

    for algo in ALGO_FOLDERS:
        for task in TASK_FOLDERS:
            ph_root = trained_base / algo / task / "ph"
            if not ph_root.is_dir():
                continue

            rel_ph_root = ph_root.relative_to(ROOT).as_posix()
            log(f"Including files under {rel_ph_root}")
            for path in ph_root.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix == ".pth":
                    continue
                if not should_include_file(path):
                    continue
                allow_patterns.append(path.relative_to(ROOT).as_posix())

            for models_dir in ph_root.rglob("models"):
                if not models_dir.is_dir():
                    continue
                checkpoint = best_checkpoint(models_dir)
                if checkpoint is None:
                    log(f"Warning: no success checkpoint found in {models_dir.relative_to(ROOT)}")
                    continue
                rel_checkpoint = checkpoint.relative_to(ROOT).as_posix()
                log(f"Selected checkpoint: {rel_checkpoint}")
                allow_patterns.append(rel_checkpoint)

    rollout_root = ROOT / "rollouts" / ROLLOUT_TASK / "ph"
    if rollout_root.is_dir():
        rel_rollout_root = rollout_root.relative_to(ROOT).as_posix()
        log(f"Including rollout files under {rel_rollout_root}")
        for path in rollout_root.rglob("*"):
            if path.is_file() and should_include_file(path):
                allow_patterns.append(path.relative_to(ROOT).as_posix())
    else:
        log(f"Warning: missing rollout directory {rollout_root.relative_to(ROOT)}")

    return sorted(set(allow_patterns))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be uploaded without uploading anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allow_patterns = build_allow_patterns()
    ignore_patterns = [
        "**/*.zarr/**",
        "**/*.zarr",
        "**/mh/**",
    ]

    log(f"Matched files: {len(allow_patterns)}")
    if args.dry_run:
        log("Dry run enabled; no upload will be performed")
        for pattern in allow_patterns:
            print(pattern)
        return

    api = HfApi()
    log(f"Uploading from repo root {ROOT}")
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(ROOT),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    log("Upload complete")


if __name__ == "__main__":
    main()
