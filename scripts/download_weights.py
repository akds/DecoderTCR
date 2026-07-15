#!/usr/bin/env python3
"""Fetch DecoderTCR model weights from the HuggingFace release into the paths the registry expects.

After this runs, `DecoderTCR.utils.model_zoo.load("<name>")` and the predict CLIs work with no
manual path wiring. Destinations come straight from the registry (MODEL_ZOO), so this never drifts
from the loader.

    uv run python scripts/download_weights.py                       # all models
    uv run python scripts/download_weights.py -m DecoderTCR-ESMC_600M   # just the default
    uv run python scripts/download_weights.py --sha256              # also verify SHA-256 (slower)

The HuggingFace repo mirrors the local layout: each checkpoint lives under its model-line folder
(e.g. `DecoderTCR-ESMC-V0.3/6B.ckpt`), and this script writes it to the matching checkpoints/<line>/
path locally. The repo is public, so no token is needed. Pass `--token`/$HF_TOKEN only if it is
ever made private or gated.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

from DecoderTCR.utils.model_zoo import MODEL_ZOO, REPO_ROOT

HF_REPO = "biohub/DecoderTCR"
HF_FILES: dict[str, str] = {                 # registry name -> path within the HF repo (mirrors checkpoints/)
    name: Path(spec.ckpt).relative_to("checkpoints").as_posix() for name, spec in MODEL_ZOO.items()
}

# Expected on-disk size (bytes) of each released checkpoint, keyed by HF path (mirrors checkpoints/).
# Used to (a) skip only files that are already COMPLETE, so an interrupted earlier run cannot leave
# a truncated .ckpt that we trust forever, and (b) reject a short download/copy before it can
# silently load as a degraded model (load_state_dict runs with strict=False). Refresh if a
# checkpoint is ever re-released: `stat -c %s <file>`. Keep in sync with MODEL_ZOO.
EXPECTED_SIZES: dict[str, int] = {
    "DecoderTCR-ESMC-V0.3/300M.ckpt":               3996365253,
    "DecoderTCR-ESMC-V0.3/600M.ckpt":               2300241260,
    "DecoderTCR-ESMC-V0.3/6B.ckpt":                25408220584,
    "DecoderTCR-ESM2-V0.1/650M_DecoderTCR.ckpt":  2604318926,
    "DecoderTCR-ESM2-V0.1/3B_DecoderTCR.ckpt":   11356181292,
}


def _sha256(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(
        description="Download DecoderTCR weights from HuggingFace into checkpoints/")
    p.add_argument("-m", "--models", nargs="+", default=list(MODEL_ZOO),
                   help=f"Models to fetch (default: all). Choices: {list(MODEL_ZOO)}")
    p.add_argument("--force", action="store_true", help="overwrite existing destinations")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace token (default: $HF_TOKEN or your cached `hf auth login`). "
                        "Only needed if the repo is private/gated.")
    p.add_argument("--sha256", action="store_true",
                   help="also verify each file's SHA-256 against the Hub's recorded hash "
                        "(slower: hashes the full file)")
    args = p.parse_args()

    unknown = [m for m in args.models if m not in MODEL_ZOO]
    if unknown:
        sys.exit(f"Unknown model(s): {unknown}. Choices: {list(MODEL_ZOO)}")

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        RepositoryNotFoundError,
    )

    print(f"source=huggingface ({HF_REPO})  -> {REPO_ROOT}/checkpoints/\n")

    # Pull the Hub's recorded SHA-256 for the requested files up front (best effort, a metadata
    # failure must not block the size-based integrity path below).
    remote_sha: dict[str, str] = {}
    if args.sha256:
        try:
            api = HfApi(token=args.token)
            for rf in api.get_paths_info(HF_REPO, [HF_FILES[n] for n in args.models], expand=True):
                if getattr(rf, "lfs", None) and rf.lfs.sha256:
                    remote_sha[rf.path] = rf.lfs.sha256
        except Exception as e:
            print(f"  warn   could not fetch Hub checksums ({type(e).__name__}: {e}), "
                  f"continuing with size checks only\n")

    done = skipped = failed = 0
    for name in args.models:
        spec = MODEL_ZOO[name]
        fname = HF_FILES[name]
        dest = spec.ckpt_path
        expected = EXPECTED_SIZES.get(fname)

        if dest.exists() and not args.force:
            if expected is None or dest.stat().st_size == expected:
                print(f"  skip   {name:<22} (exists, --force to replace)")
                skipped += 1
                continue
            print(f"  warn   {name:<22} exists but size {dest.stat().st_size} != expected "
                  f"{expected}, re-downloading")

        try:
            got = hf_hub_download(repo_id=HF_REPO, filename=fname, token=args.token)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(got, dest)

            # Integrity: size must match the release manifest (or, absent one, the verified source).
            ref = expected if expected is not None else os.path.getsize(got)
            actual = dest.stat().st_size
            if actual != ref:
                dest.unlink(missing_ok=True)
                raise RuntimeError(f"size mismatch after copy: {actual} != {ref} bytes")

            if args.sha256 and fname in remote_sha:
                digest = _sha256(dest)
                if digest != remote_sha[fname]:
                    dest.unlink(missing_ok=True)
                    raise RuntimeError(f"sha256 mismatch: {digest} != {remote_sha[fname]}")

            print(f"  ok     {name:<22} {spec.version:<5} downloaded ({actual / 1e9:.1f} GB)")
            done += 1
        except (GatedRepoError, RepositoryNotFoundError) as e:
            print(f"  FAIL   {name:<22} {type(e).__name__}: repo is private/gated or not found.")
            print(f"         Run `hf auth login` (or set HF_TOKEN) and request access at "
                  f"https://huggingface.co/{HF_REPO}")
            failed += 1
        except EntryNotFoundError:
            print(f"  FAIL   {name:<22} '{fname}' not found in {HF_REPO}. Weights must keep the "
                  f"checkpoints/ layout on HF (e.g. DecoderTCR-ESMC-V0.3/6B.ckpt).")
            failed += 1
        except Exception as e:
            print(f"  FAIL   {name:<22} {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{done} fetched, {skipped} already present, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
