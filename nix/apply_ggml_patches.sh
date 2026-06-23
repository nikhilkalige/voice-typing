#!/usr/bin/env bash
# Replacement for parakeet.cpp's scripts/apply_ggml_patches.sh that uses
# `patch` instead of `git apply`, so it works in the Nix sandbox (no .git).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GGML_DIR="${PROJECT_ROOT}/third_party/ggml"
PATCH_DIR="${PROJECT_ROOT}/third_party/ggml-patches"

[[ -d "${GGML_DIR}" ]]  || { echo "error: ggml not found at ${GGML_DIR}" >&2; exit 1; }
[[ -d "${PATCH_DIR}" ]] || { echo "error: patch dir not found at ${PATCH_DIR}" >&2; exit 1; }

shopt -s nullglob
PATCHES=("${PATCH_DIR}"/*.patch)
shopt -u nullglob

if [[ ${#PATCHES[@]} -eq 0 ]]; then
    echo "ggml patches: no patches found (nothing to do)"
    exit 0
fi

IFS=$'\n' PATCHES=($(printf '%s\n' "${PATCHES[@]}" | sort)); unset IFS

applied=0; skipped=0
cd "${GGML_DIR}"

for patch in "${PATCHES[@]}"; do
    name="$(basename "${patch}")"
    if patch -p1 --dry-run --reverse --quiet < "${patch}" >/dev/null 2>&1; then
        echo "ggml patches: skipping ${name} (already applied)"
        skipped=$((skipped + 1))
        continue
    fi
    if patch -p1 --dry-run --quiet < "${patch}" >/dev/null 2>&1; then
        patch -p1 < "${patch}"
        echo "ggml patches: applied ${name}"
        applied=$((applied + 1))
        continue
    fi
    echo "error: cannot apply ${name}" >&2; exit 1
done

echo "ggml patches: applied ${applied}, skipped ${skipped}"
