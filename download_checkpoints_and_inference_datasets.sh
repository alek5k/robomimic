#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (directory containing this script)
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$ROOT_DIR"

huggingface-cli download erdrsvhr9/temporally_conditioned_diffusion_policy_robomimic \
    --repo-type dataset \
    --local-dir "$ROOT_DIR" \
