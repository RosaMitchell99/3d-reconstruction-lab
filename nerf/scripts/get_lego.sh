#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
archive="data/nerf_example_data.zip"
scene="data/lego"
mkdir -p data

if [[ ! -f "$archive" ]]; then
  curl -fL --retry 3 'https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip' -o "$archive.part"
  mv "$archive.part" "$archive"
fi
printf '%s  %s\n' 'ce4e94e031c099a19ef04cfb6c71f1e47225d97d365be610b476e379a386c25f' "$archive" | sha256sum -c -
uv run --no-sync python - "$archive" "$scene" <<'PY'
import sys
import shutil
from pathlib import Path
from zipfile import ZipFile

archive, destination = Path(sys.argv[1]), Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)
with ZipFile(archive) as zip_file:
    for entry in zip_file.infolist():
        prefix = "nerf_synthetic/lego/"
        if not entry.filename.startswith(prefix) or entry.is_dir():
            continue
        relative = Path(entry.filename.removeprefix(prefix))
        if ".." in relative.parts or relative.is_absolute():
            raise ValueError(f"unsafe archive path: {entry.filename}")
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        with zip_file.open(entry) as source, output.open("wb") as target:
            shutil.copyfileobj(source, target)
for split in ("train", "val", "test"):
    if not (destination / f"transforms_{split}.json").is_file():
        raise FileNotFoundError(f"missing {split} split")
print(f"Lego scene ready: {destination}")
PY
