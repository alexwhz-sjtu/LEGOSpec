"""Export inference metadata without modifying or copying trained tensors.

Run: python -m sglang.srt.tools.export_legospec SOURCE DESTINATION
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from sglang.srt.configs.legospec import LEGOSpecDraftConfig


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_checkpoint(source, destination):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    raw = json.loads((source / "config.json").read_text())
    if raw.get("model_type") != "litespec_composable":
        raise ValueError("Expected an unmodified composable final checkpoint")
    raw.pop("model_type", None)
    config = LEGOSpecDraftConfig(**raw)
    weights = sorted(source.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"No safetensors weights in {source}")
    # Refuse replacement, including partially exported directories.
    destination.mkdir(parents=True, exist_ok=False)
    config.save_pretrained(destination)
    files = weights + sorted(source.glob("*.safetensors.index.json"))
    for path in files:
        os.symlink(path, destination / path.name)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    receipt = {
        "source": str(source),
        "source_config_sha256": sha256(source / "config.json"),
        "inference_config_sha256": sha256(destination / "config.json"),
        "weights_sha256": {p.name: sha256(p) for p in weights},
        "sglang_base_commit": commit,
    }
    (destination / "legospec_provenance.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
