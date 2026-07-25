#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess


SIGNOFF = re.compile(r"(?im)^Signed-off-by:\s+.+\s+<[^<>\s]+@[^<>\s]+>\s*$")


def main() -> int:
    parser = argparse.ArgumentParser(description="Require DCO sign-off on PR commits.")
    parser.add_argument("--base-ref", required=True)
    args = parser.parse_args()
    merge_base = subprocess.run(
        ["git", "merge-base", "HEAD", args.base_ref],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = subprocess.run(
        [
            "git",
            "log",
            "--no-merges",
            "--format=%H%x1f%B%x1e",
            f"{merge_base}..HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    missing: list[str] = []
    for record in payload.split("\x1e"):
        record = record.strip()
        if not record:
            continue
        commit, _, body = record.partition("\x1f")
        if SIGNOFF.search(body) is None:
            missing.append(commit[:12])
    if missing:
        print("Commits missing Signed-off-by: " + ", ".join(missing))
        return 1
    print("All pull-request commits contain a DCO sign-off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
