#!/usr/bin/env python3
"""Interactively clean up non-success checkpoints under temporaldp."""

from __future__ import annotations

import argparse
from pathlib import Path


def iter_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []

    all_models = list(root.rglob("model_epoch_*.pth"))
    all_models_parents = list(set(p.parent for p in all_models))

    for path in root.rglob("model_epoch_*.pth"):
        if path.name == "last_bak.pth":
            continue
        if "_success_" in path.name:
            continue
        candidates.append(path)

    for parent in all_models_parents:
        sister_models = list(parent.glob("model_epoch_*success_*.pth"))
        success_rates = [float(s.name.split("_success_")[-1].split(".pth")[0]) for s in sister_models]

        if success_rates:
            argmax_idx = max(range(len(success_rates)), key=success_rates.__getitem__)
            best_model = sister_models[argmax_idx]
            sister_models.remove(best_model)
            candidates.extend(sister_models)

    for path in root.rglob("last_bak.pth"):
        candidates.append(path)
    
    return sorted(set(candidates))


def confirm_delete(path: Path, allow_all: bool) -> str:
    print("Directory contents:")
    for item in path.parent.rglob("*"):
        print(f"  {item}")
    print()
    prompt = "Delete {}? [y/N/a/q] ".format(path)
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"", "n", "no"}:
            return "skip"
        if answer in {"y", "yes"}:
            return "delete"
        if allow_all and answer in {"a", "all"}:
            return "all"
        if answer in {"q", "quit"}:
            return "quit"
        print("Please enter y, n, a, or q.")


def delete_paths(paths: list[Path], dry_run: bool, assume_yes: bool) -> None:
    if not paths:
        print("No matching checkpoints found.")
        return

    print("Found {} candidate file(s).".format(len(paths)))
    delete_all = False
    deleted = 0
    for path in paths:
        if assume_yes:
            if dry_run:
                print("[dry-run] Would delete {}".format(path))
            else:
                path.unlink(missing_ok=True)
                print("Deleted {}".format(path))
                deleted += 1
            continue
        if delete_all:
            if dry_run:
                print("[dry-run] Would delete {}".format(path))
            else:
                path.unlink(missing_ok=True)
                print("Deleted {}".format(path))
                deleted += 1
            continue

        decision = confirm_delete(path, allow_all=True)
        if decision == "quit":
            break
        if decision == "skip":
            continue
        if decision == "all":
            delete_all = True
            if dry_run:
                print("[dry-run] Would delete {}".format(path))
            else:
                path.unlink(missing_ok=True)
                print("Deleted {}".format(path))
                deleted += 1
            continue
        if decision == "delete":
            if dry_run:
                print("[dry-run] Would delete {}".format(path))
            else:
                path.unlink(missing_ok=True)
                print("Deleted {}".format(path))
                deleted += 1

    if not dry_run:
        print("Deleted {} file(s).".format(deleted))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively delete model_epoch_*.pth files without _success_ "
            "and any last_bak.pth under temporaldp."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent.parent / "trained_models/temporaldp",
        help="root directory to scan (default: trained_models/temporaldp)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be deleted without removing files",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="delete all matching files without prompting",
    )
    args = parser.parse_args()

    root = args.root.expanduser()
    if not root.exists():
        raise FileNotFoundError("Root path not found: {}".format(root))

    candidates = iter_candidates(root)
    delete_paths(candidates, dry_run=args.dry_run, assume_yes=args.yes)


if __name__ == "__main__":
    main()
