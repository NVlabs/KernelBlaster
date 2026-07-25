#!/usr/bin/env python3
"""Write the reviewed vector-add smoke inputs into the configured local CAS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kernelblaster.gpu_jobs import build_deterministic_bundle  # noqa: E402
from src.kernelblaster.storage import StateStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    smoke = ROOT / "portfolio" / "trusted_gpu_smoke"
    source = (smoke / "vector_add.cu").read_bytes()
    driver = (smoke / "driver.cpp").read_bytes()
    bundle = build_deterministic_bundle({"vector_add.cu": source})
    manifest = json.loads(
        (ROOT / "portfolio" / "trusted-gpu-bundles.json").read_text(encoding="utf-8")
    )
    expected_source = manifest["source_bundle_digests"][0]
    expected_driver = manifest["bundles"][0]["driver_digest"]
    if hashlib.sha256(bundle).hexdigest() != expected_source:
        raise SystemExit("trusted source bundle digest does not match its manifest")
    if hashlib.sha256(driver).hexdigest() != expected_driver:
        raise SystemExit("trusted driver digest does not match its manifest")
    store = StateStore(state_dir=args.state_dir)
    source_artifact = store.cas.put_bytes(
        bundle, media_type="application/x-tar", producer="trusted-smoke-registry"
    )
    driver_artifact = store.cas.put_bytes(
        driver, media_type="text/x-c++src", producer="trusted-smoke-registry"
    )
    store.repository.register_artifact(source_artifact)
    store.repository.register_artifact(driver_artifact)
    print(json.dumps({"source_bundle_digest": source_artifact.digest, "driver_digest": driver_artifact.digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
