#!/usr/bin/env python3
"""Show training progress for all trained model log files."""

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path


TRAIN_EPOCH_RE = re.compile(r"\bTrain Epoch\s+(\d+)\b")
SUCCESS_RE = re.compile(r"_success_([0-9]+(?:\.[0-9]+)?)\.pth$")
TASK_ORDER = ["can", "square", "lift", "tool_hang", "transport"]
ALGO_ORDER = ["bc", "bc_rnn", "hbc", "diffusion_policy", "tc_diffusion_policy"]
DATASET_TYPE_ORDER = ["ph", "mh"]
ANSI_GREEN = "\033[32m"
ANSI_BLUE = "\033[34m"
ANSI_RESET = "\033[0m"


@dataclass
class RunProgress:
    run_dir: Path
    log_path: Path
    config_path: Path
    current_epoch: int | None
    total_epochs: int | None
    experiment_name: str
    algo: str
    task: str
    dataset_type: str
    log_mtime: float | None
    config_mtime: float | None
    last_pth_mtime: float | None
    best_success_rate: float | None


def find_log_paths(root: Path) -> list[Path]:
    return sorted(root.glob("**/log.txt"))


def latest_train_epoch(log_path: Path) -> int | None:
    latest_epoch = None
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = TRAIN_EPOCH_RE.search(line)
            if match:
                latest_epoch = int(match.group(1))
    return latest_epoch


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config if isinstance(config, dict) else {}


def total_epochs_from_config(config: dict) -> int | None:
    train_cfg = config.get("train", {})
    if isinstance(train_cfg, dict) and train_cfg.get("num_epochs") is not None:
        return int(train_cfg["num_epochs"])

    # Saved configs may also include optimizer metadata populated by train.py.
    algo_cfg = config.get("algo", {})
    if isinstance(algo_cfg, dict):
        optim_params = algo_cfg.get("optim_params", {})
        if isinstance(optim_params, dict):
            for optim_cfg in optim_params.values():
                if isinstance(optim_cfg, dict) and optim_cfg.get("num_epochs") is not None:
                    return int(optim_cfg["num_epochs"])

    return None


def experiment_name_from_config(config: dict, run_dir: Path) -> str:
    experiment_cfg = config.get("experiment", {})
    if isinstance(experiment_cfg, dict) and experiment_cfg.get("name"):
        return str(experiment_cfg["name"])
    return run_dir.parent.name


def run_parts_from_path(run_dir: Path) -> tuple[str, str, str]:
    parts = run_dir.parts
    try:
        index = parts.index("temporaldp")
        return parts[index + 1], parts[index + 2], parts[index + 3]
    except (ValueError, IndexError):
        return "?", "?", "?"


def collect_progress(root: Path) -> list[RunProgress]:
    runs = []
    for log_path in find_log_paths(root):
        run_dir = log_path.parent.parent
        config_path = run_dir / "config.json"
        last_pth_path = run_dir / "last.pth"
        models_dir = run_dir / "models"
        config = load_config(config_path) if config_path.exists() else {}
        algo, task, dataset_type = run_parts_from_path(run_dir)
        try:
            log_mtime = log_path.stat().st_mtime
        except OSError:
            log_mtime = None
        try:
            config_mtime = config_path.stat().st_mtime
        except OSError:
            config_mtime = None
        try:
            last_pth_mtime = last_pth_path.stat().st_mtime
        except OSError:
            last_pth_mtime = None
        runs.append(
            RunProgress(
                run_dir=run_dir,
                log_path=log_path,
                config_path=config_path,
                current_epoch=latest_train_epoch(log_path),
                total_epochs=total_epochs_from_config(config),
                experiment_name=experiment_name_from_config(config, run_dir),
                algo=algo,
                task=task,
                dataset_type=dataset_type,
                log_mtime=log_mtime,
                config_mtime=config_mtime,
                last_pth_mtime=last_pth_mtime,
                best_success_rate=best_success_rate_from_models_dir(models_dir),
            )
        )
    return runs


def best_success_rate_from_models_dir(models_dir: Path) -> float | None:
    if not models_dir.exists():
        return None

    best = None
    for path in models_dir.glob("model_epoch_*.pth"):
        match = SUCCESS_RE.search(path.name)
        if not match:
            continue
        success = float(match.group(1))
        if best is None or success > best:
            best = success
    return best


def progress_bar(current: int | None, total: int | None, width: int) -> str:
    if current is None or total is None or total <= 0:
        return "[" + ("?" * width) + "]"

    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(round(ratio * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def is_complete(run: RunProgress) -> bool:
    return (
        run.current_epoch is not None
        and run.total_epochs is not None
        and run.total_epochs > 0
        and run.current_epoch >= run.total_epochs
    )


def age_seconds(run: RunProgress, now: float) -> float | None:
    if run.log_mtime is None:
        return None
    return max(0.0, now - run.log_mtime)


def format_age(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    if seconds < 60:
        return "{:.0f}s ago".format(seconds)
    if seconds < 3600:
        return "{:.0f}m ago".format(seconds / 60)
    if seconds < 86400:
        return "{:.1f}h ago".format(seconds / 3600)
    return "{:.1f}d ago".format(seconds / 86400)


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "?"
    if seconds < 60:
        return "{:.0f}s".format(seconds)
    if seconds < 3600:
        return "{:.0f}m".format(seconds / 60)
    if seconds < 86400:
        return "{:.1f}h".format(seconds / 3600)
    return "{:.1f}d".format(seconds / 86400)


def run_status(run: RunProgress, now: float, active_within_seconds: int) -> str:
    if is_complete(run):
        return "done"
    age = age_seconds(run, now)
    if age is not None and age <= active_within_seconds:
        return "active"
    return "stale"


def colorize(text: str, status: str, color: bool) -> str:
    if not color:
        return text
    if status == "done":
        return "{}{}{}".format(ANSI_GREEN, text, ANSI_RESET)
    if status == "active":
        return "{}{}{}".format(ANSI_BLUE, text, ANSI_RESET)
    return text


def format_status(run: RunProgress, now: float, active_within_seconds: int, color: bool, width: int = 18) -> str:
    status = run_status(run, now, active_within_seconds)
    if status == "done":
        text = status
    else:
        text = "{} ({})".format(status, format_age(age_seconds(run, now)))
    return colorize(text.ljust(width), status=status, color=color)


def format_progress(run: RunProgress, width: int, now: float, active_within_seconds: int) -> str:
    current = run.current_epoch
    total = run.total_epochs
    if run.config_mtime is None or run.last_pth_mtime is None:
        elapsed = format_duration(None)
    else:
        elapsed = format_duration(run.last_pth_mtime - run.config_mtime)
    if current is None or total is None or total <= 0:
        pct = "  ??.?%"
        epoch_text = "{}/{}".format(current if current is not None else "?", total if total is not None else "?")
    else:
        pct = "{:6.1f}%".format(100.0 * min(current / total, 1.0))
        epoch_text = "{}/{}".format(current, total)

    success_text = (
        "{:.2f}".format(run.best_success_rate)
        if run.best_success_rate is not None
        else "?"
    )

    return "{} {} {:>11}  {:>7}  {}  {:<20}  {:<10}  ({:<2})  {}".format(
        progress_bar(current, total, width),
        pct,
        epoch_text,
        elapsed,
        format_status(run, now, active_within_seconds, color=True),
        run.algo,
        run.task,
        run.dataset_type,
        "{}  best={}".format(run.run_dir.name, success_text),
    )


def sort_key(run: RunProgress):
    task_index = TASK_ORDER.index(run.task) if run.task in TASK_ORDER else len(TASK_ORDER)
    algo_index = ALGO_ORDER.index(run.algo) if run.algo in ALGO_ORDER else len(ALGO_ORDER)
    dataset_type_index = (
        DATASET_TYPE_ORDER.index(run.dataset_type)
        if run.dataset_type in DATASET_TYPE_ORDER
        else len(DATASET_TYPE_ORDER)
    )
    return (dataset_type_index, run.dataset_type, task_index, algo_index, run.run_dir.name)


def main():
    parser = argparse.ArgumentParser(description="Show train epoch progress for trained_models/**/log.txt")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("trained_models"),
        help="root directory to scan for log.txt files",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=24,
        help="progress bar width in characters",
    )
    parser.add_argument(
        "--active-within",
        type=int,
        default=300,
        help="consider an incomplete run active if log.txt was modified within this many seconds",
    )
    args = parser.parse_args()

    root = args.root.expanduser()
    if not root.exists():
        raise FileNotFoundError("Root path not found: {}".format(root))

    runs = collect_progress(root)
    if not runs:
        print("No log.txt files found under {}".format(root))
        return

    print("Found {} run(s) under {}".format(len(runs), root))
    print("")
    now = time.time()
    last_dataset_type = None
    for run in sorted(runs, key=sort_key):
        if last_dataset_type is not None and run.dataset_type != last_dataset_type:
            print("")
        print(format_progress(run, width=args.width, now=now, active_within_seconds=args.active_within))
        last_dataset_type = run.dataset_type


if __name__ == "__main__":
    main()
