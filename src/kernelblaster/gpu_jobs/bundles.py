"""Deterministic trusted-source bundles with strict archive validation."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path, PurePosixPath
import tarfile
from typing import Mapping


def _validated_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("bundle members must be safe relative POSIX paths")
    return path.as_posix()


def build_deterministic_bundle(files: Mapping[str, bytes]) -> bytes:
    """Create a byte-stable tar bundle containing regular files only."""
    if not files:
        raise ValueError("source bundle may not be empty")
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for raw_name, payload in sorted(files.items()):
            name = _validated_name(raw_name)
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o444
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, BytesIO(payload))
    return output.getvalue()


def validate_bundle(payload: bytes, *, maximum_bytes: int = 64 * 1024 * 1024) -> tuple[str, ...]:
    if len(payload) > maximum_bytes:
        raise ValueError("source bundle exceeds configured size limit")
    names: list[str] = []
    with tarfile.open(fileobj=BytesIO(payload), mode="r:*") as archive:
        for member in archive.getmembers():
            name = _validated_name(member.name)
            if not member.isfile() or member.issym() or member.islnk() or member.isdev():
                raise ValueError("source bundle may contain regular files only")
            names.append(name)
    if not names:
        raise ValueError("source bundle may not be empty")
    return tuple(names)


def extract_bundle(payload: bytes, destination: str | Path) -> tuple[Path, ...]:
    names = validate_bundle(payload)
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with tarfile.open(fileobj=BytesIO(payload), mode="r:*") as archive:
        for name in names:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.parent.resolve().is_relative_to(root):
                raise ValueError("source bundle extraction escapes staging root")
            source = archive.extractfile(name)
            assert source is not None
            target.write_bytes(source.read())
            extracted.append(target)
    return tuple(extracted)
