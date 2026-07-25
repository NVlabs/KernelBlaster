"""A small, crash-safe SHA-256 content-addressed store."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path, PurePath
import shutil
import tempfile
from typing import BinaryIO


_SHA256_LENGTH = 64


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_digest(digest: str) -> str:
    normalized = digest.strip().lower()
    if len(normalized) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("artifact digest must be a lowercase SHA-256 hex value")
    return normalized


def _safe_relative_path(relative_path: str | Path) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in PurePath(candidate).parts:
        raise ValueError("artifact path must be a relative path inside its export root")
    return candidate


@dataclass(frozen=True)
class ArtifactMetadata:
    digest: str
    size_bytes: int
    media_type: str
    created_at: str
    producer: str | None = None
    source_digest: str | None = None
    schema: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ContentAddressedStore:
    """Store immutable payloads under ``<root>/<digest-prefix>/<digest>``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("CAS root may not be a symbolic link")
        self.root = self.root.resolve()

    def _path(self, digest: str) -> Path:
        digest = _validate_digest(digest)
        path = self.root / digest[:2] / digest
        resolved_parent = path.parent.resolve()
        if not resolved_parent.is_relative_to(self.root):
            raise ValueError("artifact path escapes CAS root")
        if path.exists() and path.is_symlink():
            raise ValueError("CAS payload may not be a symbolic link")
        return path

    def put_bytes(
        self,
        payload: bytes,
        *,
        media_type: str = "application/octet-stream",
        producer: str | None = None,
        source_digest: str | None = None,
        schema: str | None = None,
    ) -> ArtifactMetadata:
        return self._put_stream(
            stream=None,
            payload=payload,
            media_type=media_type,
            producer=producer,
            source_digest=source_digest,
            schema=schema,
        )

    def put_file(
        self,
        source: str | Path,
        *,
        media_type: str = "application/octet-stream",
        producer: str | None = None,
        source_digest: str | None = None,
        schema: str | None = None,
    ) -> ArtifactMetadata:
        source_path = Path(source)
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError("CAS source must be a regular, non-symbolic-link file")
        with source_path.open("rb") as stream:
            return self._put_stream(
                stream=stream,
                payload=None,
                media_type=media_type,
                producer=producer,
                source_digest=source_digest,
                schema=schema,
            )

    def _put_stream(
        self,
        *,
        stream: BinaryIO | None,
        payload: bytes | None,
        media_type: str,
        producer: str | None,
        source_digest: str | None,
        schema: str | None,
    ) -> ArtifactMetadata:
        if source_digest is not None:
            source_digest = _validate_digest(source_digest)
        digest = hashlib.sha256()
        size_bytes = 0
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.root, prefix=".cas-", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                if payload is not None:
                    chunks = (payload,)
                else:
                    assert stream is not None
                    chunks = iter(lambda: stream.read(1024 * 1024), b"")
                for chunk in chunks:
                    if not chunk:
                        continue
                    digest.update(chunk)
                    size_bytes += len(chunk)
                    temporary.write(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())

            digest_value = digest.hexdigest()
            destination = self._path(digest_value)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                temporary_path.unlink()
            else:
                os.replace(temporary_path, destination)
                temporary_path = None
                self._fsync_directory(destination.parent)
            return ArtifactMetadata(
                digest=digest_value,
                size_bytes=size_bytes,
                media_type=media_type,
                created_at=_utc_now(),
                producer=producer,
                source_digest=source_digest,
                schema=schema,
            )
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def get_path(self, digest: str) -> Path:
        path = self._path(digest)
        if not path.is_file():
            raise FileNotFoundError(f"CAS payload {digest} does not exist")
        return path

    def get_bytes(self, digest: str) -> bytes:
        return self.get_path(digest).read_bytes()

    def verify(self, digest: str) -> ArtifactMetadata:
        path = self.get_path(digest)
        actual = hashlib.sha256()
        size_bytes = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                actual.update(chunk)
                size_bytes += len(chunk)
        normalized = _validate_digest(digest)
        if actual.hexdigest() != normalized:
            raise ValueError(f"CAS payload {normalized} is corrupt")
        return ArtifactMetadata(
            digest=normalized,
            size_bytes=size_bytes,
            media_type="application/octet-stream",
            created_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        )

    def export(self, digest: str, export_root: str | Path, relative_path: str | Path) -> Path:
        target_root = Path(export_root).expanduser()
        target_root.mkdir(parents=True, exist_ok=True)
        if target_root.is_symlink():
            raise ValueError("CAS export root may not be a symbolic link")
        target_root = target_root.resolve()
        target = target_root / _safe_relative_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.parent.resolve().is_relative_to(target_root):
            raise ValueError("CAS export path escapes export root")
        if target.exists() and target.is_symlink():
            raise ValueError("CAS export target may not be a symbolic link")
        shutil.copyfile(self.get_path(digest), target)
        return target
