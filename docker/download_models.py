#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-download every model Kimodo needs at runtime.

This script is executed while building the Docker image so the resulting image
is fully self-contained: no Hugging Face downloads happen at container start.

Models downloaded:
  * Every Kimodo and TMR checkpoint used by ``kimodo.model.registry``.
  * The LLM2Vec text encoder (adapter + supervised adapter) and the ungated
    ``NousResearch/Meta-Llama-3-8B-Instruct`` base weights.
  * The SOMA-X assets used by ``SOMASkin`` / ``SomaLayer``.

The two LLM2Vec adapter repos hardcode ``meta-llama/Meta-Llama-3-8B-Instruct``
as their PEFT base model. That repo is gated, so this script also writes local
copies of the adapters with ``base_model_name_or_path`` rewritten to the ungated
NousResearch mirror. ``TEXT_ENCODERS_DIR`` points at those copies at runtime.

Example::

    python docker/download_models.py --cache-dir /opt/hf-cache/hub
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Keep in sync with kimodo/model/registry.py. Duplicated here so the download
# script has no dependency on the source tree being importable.
KIMODO_REPOS = [
    "nvidia/Kimodo-SOMA-RP-v1",
    "nvidia/Kimodo-SOMA-RP-v1.1",
    "nvidia/Kimodo-SOMA-RP-v1.2",
    "nvidia/Kimodo-SOMA-RP-ArdyRep-v1.2",
    "camenduru/Kimodo-SMPLX-RP-v1",
    "nvidia/Kimodo-G1-RP-v1",
    "nvidia/Kimodo-SOMA-SEED-v1",
    "nvidia/Kimodo-SOMA-SEED-v1.1",
    "nvidia/Kimodo-G1-SEED-v1",
]
TMR_REPOS = [
    "nvidia/TMR-SOMA-RP-v1",
]

# Ungated mirror of meta-llama/Meta-Llama-3-8B-Instruct.
TEXT_ENCODER_BASE_REPO = "NousResearch/Meta-Llama-3-8B-Instruct"
TEXT_ENCODER_ADAPTER_REPOS = [
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
]
TEXT_ENCODER_REPOS = [TEXT_ENCODER_BASE_REPO, *TEXT_ENCODER_ADAPTER_REPOS]

# The Llama repo may ship both HF safetensors and a legacy
# ``original/consolidated.00.pth``. Transformers only reads the safetensors, so
# skip any redundant copy if present.
DOWNLOAD_ALLOW_PATTERNS = {
    TEXT_ENCODER_BASE_REPO: ["*.json", "*.safetensors", "*.model", "*.txt", "*.md"],
}

# Keep the revision in sync with py-soma-x's soma.assets.get_assets_dir().
# The actual default is read from the installed package at download time; this
# value is only a fallback for environments where soma is not importable.
SOMA_X_REPO = "nvidia/soma-x"
SOMA_X_REVISION = "v0.3.5"

# Files that must exist in every snapshot after download. A missing file means
# the image would lazily hit the network at runtime, which we never want.
REQUIRED_FILES = {
    **{repo: ("config.yaml", "model.safetensors") for repo in KIMODO_REPOS},
    **{
        repo: (
            "config.yaml",
            "last_weights/text_encoder.pt",
            "last_weights/motion_encoder.pt",
            "last_weights/motion_decoder.pt",
        )
        for repo in TMR_REPOS
    },
    TEXT_ENCODER_BASE_REPO: ("config.json", "model.safetensors.index.json", "tokenizer.json"),
    **{
        repo: ("adapter_config.json", "adapter_model.safetensors")
        for repo in TEXT_ENCODER_ADAPTER_REPOS
    },
}


def _detect_soma_revision(fallback: str) -> str:
    """Return the asset revision expected by the installed py-soma-x."""
    try:
        import inspect

        from soma.assets import get_assets_dir

        revision = inspect.signature(get_assets_dir).parameters["revision"].default
        if isinstance(revision, str) and revision:
            return revision
    except Exception:  # noqa: BLE001 - fall back to the pinned default
        pass
    return fallback


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _verify(repo_id: str, snapshot_dir: Path) -> None:
    required = REQUIRED_FILES.get(repo_id, ())
    missing = [name for name in required if not (snapshot_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Snapshot for {repo_id} is missing required file(s): {', '.join(missing)}")


def download(repo_id: str, cache_dir: Path, revision: str | None = None) -> Path:
    print(f"\n==> {repo_id}" + (f" @ {revision}" if revision else ""), flush=True)
    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache_dir),
            allow_patterns=DOWNLOAD_ALLOW_PATTERNS.get(repo_id),
        )
    )
    _verify(repo_id, snapshot_dir)
    print(f"    cached at {snapshot_dir} ({_human(_dir_size_bytes(snapshot_dir))})", flush=True)
    return snapshot_dir


def prepare_text_encoders(cache_dir: Path, output_dir: Path, base_repo: str) -> None:
    """Copy the LLM2Vec adapters locally and repoint them at an ungated base."""
    print(f"\n==> preparing local text-encoder adapters in {output_dir}", flush=True)
    for repo_id in TEXT_ENCODER_ADAPTER_REPOS:
        snapshot_dir = Path(
            snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir),
                local_files_only=True,
            )
        )
        dst = output_dir / repo_id
        if dst.exists():
            shutil.rmtree(dst)
        # Follow the cache's symlinks so the copy is self-contained.
        shutil.copytree(snapshot_dir, dst, symlinks=False)

        config_path = dst / "adapter_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        previous = config.get("base_model_name_or_path")
        config["base_model_name_or_path"] = base_repo
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"    {repo_id}: base {previous!r} -> {base_repo!r}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/opt/hf-cache")) / "hub",
        help="Hugging Face hub cache directory (default: $HF_HOME/hub).",
    )
    parser.add_argument(
        "--text-encoders-dir",
        type=Path,
        default=Path(os.environ.get("TEXT_ENCODERS_DIR", "/opt/text-encoders")),
        help="Where to write patched local LLM2Vec adapters (default: $TEXT_ENCODERS_DIR).",
    )
    parser.add_argument(
        "--skip-text-encoder",
        action="store_true",
        help="Do not download the LLM2Vec / Llama-3 text encoder.",
    )
    parser.add_argument(
        "--skip-soma",
        action="store_true",
        help="Do not download the SOMA-X visualization assets.",
    )
    parser.add_argument(
        "--soma-revision",
        default=None,
        help=f"SOMA-X asset revision (default: detected from py-soma-x, else {SOMA_X_REVISION}).",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading models into: {cache_dir}")

    repos = KIMODO_REPOS + TMR_REPOS
    if not args.skip_text_encoder:
        repos += TEXT_ENCODER_REPOS

    failed: list[str] = []
    for repo_id in repos:
        try:
            download(repo_id, cache_dir)
        except Exception as error:  # noqa: BLE001 - report all failures at the end
            print(f"    FAILED: {type(error).__name__}: {error}", flush=True)
            failed.append(repo_id)

    if not args.skip_text_encoder and not any(repo in failed for repo in TEXT_ENCODER_REPOS):
        prepare_text_encoders(cache_dir, args.text_encoders_dir, TEXT_ENCODER_BASE_REPO)

    if not args.skip_soma:
        soma_revision = args.soma_revision or _detect_soma_revision(SOMA_X_REVISION)
        print(f"SOMA-X revision: {soma_revision}")
        try:
            download(SOMA_X_REPO, cache_dir, revision=soma_revision)
        except Exception as error:  # noqa: BLE001
            print(f"    FAILED: {type(error).__name__}: {error}", flush=True)
            failed.append(SOMA_X_REPO)

    total = _dir_size_bytes(cache_dir)
    print(f"\nHF cache size: {_human(total)}")

    if failed:
        print("\nERROR: failed to pre-download:\n  - " + "\n  - ".join(failed))
        return 1

    print("All models are cached. The image can now run fully offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
