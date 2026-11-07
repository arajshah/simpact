"""Download the Section 8 pinned AnyGoal files without overwriting.

The file list and revisions are the memorandum's. Checksums are recorded
after the copy. This script does not download the rest of either repository.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

MODEL_REVISION = "b27f9dac9c750410d1331d258a95eff0d56570c2"
DATA_REVISION = "1719e7dab2fbeb07a0d5c3cf9cd881bc49510c88"
REPO_ID = "HazelNam/anygoal"

MODEL_FILES = (
    "sensorimotor_insertpeg.ckpt",
    "sensorimotor_insertpeg.json",
    "toy-adaln_insertpeg.pt",
    "toy-adaln_insertpeg.json",
    "toy-action-duplicate_insertpeg.pt",
    "toy-action-duplicate_insertpeg.json",
)
DATA_FILES = (
    "manipulator_insert_peg_weak_expert_train.h5",
    "extract_peg_test.h5",
    "avoid_peg_test.h5",
)


def target_dir(kind: str) -> Path:
    result = subprocess.check_output(
        ["anygoal", "config", "path", kind], text=True
    ).strip()
    directory = Path(result)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(overwrite_ok: bool) -> list[tuple[str, str, int]]:
    from huggingface_hub import hf_hub_download

    specs = [
        ("model", MODEL_REVISION, target_dir("checkpoints"), MODEL_FILES),
        ("dataset", DATA_REVISION, target_dir("data"), DATA_FILES),
    ]
    written: list[tuple[str, str, int]] = []
    for repo_type, revision, directory, filenames in specs:
        for filename in filenames:
            destination = directory / filename
            if destination.exists():
                if not overwrite_ok:
                    raise FileExistsError(
                        f"Refusing to overwrite {destination}. "
                        "Use an isolated AnyGoal home or pass --verify-existing."
                    )
            else:
                downloaded = hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type=repo_type,
                    revision=revision,
                    filename=filename,
                )
                shutil.copy2(downloaded, destination)
            digest = sha256(destination)
            written.append((str(destination), digest, destination.stat().st_size))
            print(f"{digest}  {destination.stat().st_size:12d}  {destination}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Hash files that are already present instead of refusing them.",
    )
    args = parser.parse_args()
    if not os.environ.get("ANYGOAL_HOME"):
        raise RuntimeError("ANYGOAL_HOME is unset. Source scripts/anygoal_contact_env.sh.")
    download(overwrite_ok=args.verify_existing)


if __name__ == "__main__":
    main()
