"""State-directory configuration and the combined SQLite/CAS facade."""

from __future__ import annotations

import os
from pathlib import Path
import platform

from .cas import ContentAddressedStore
from .repository import JobRepository


_NETWORK_FILESYSTEMS = {"cifs", "smbfs", "nfs", "nfs4", "drvfs"}
_STATE_ENVIRONMENT_KEYS = (
    "KERNELBLASTER_STATE_DIR",
    "KERNELBLASTER_SQLITE_PATH",
    "KERNELBLASTER_CAS_DIR",
)


def state_storage_requested() -> bool:
    return any(os.getenv(key) for key in _STATE_ENVIRONMENT_KEYS)


def _mount_filesystem_type(path: Path) -> str | None:
    if platform.system() != "Linux" or not Path("/proc/mounts").is_file():
        return None
    resolved = str(path.resolve())
    mounts: list[tuple[str, str]] = []
    for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 3:
            mounts.append((fields[1].replace("\\040", " "), fields[2]))
    matching = [item for item in mounts if resolved == item[0] or resolved.startswith(item[0].rstrip("/") + "/")]
    return max(matching, key=lambda item: len(item[0]))[1] if matching else None


def _prepare_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("state paths may not be symbolic links")
    resolved = path.resolve()
    filesystem_type = _mount_filesystem_type(resolved)
    if filesystem_type in _NETWORK_FILESYSTEMS:
        raise ValueError(f"state path uses unsupported filesystem type: {filesystem_type}")
    if platform.system() != "Windows" and resolved.stat().st_mode & 0o022:
        raise PermissionError("state directory must not be group- or world-writable")
    if not os.access(resolved, os.R_OK | os.W_OK | os.X_OK):
        raise PermissionError("state directory must be readable, writable, and searchable")
    return resolved


def resolve_state_paths(
    *,
    state_dir: str | Path | None = None,
    sqlite_path: str | Path | None = None,
    cas_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    root = Path(state_dir or os.getenv("KERNELBLASTER_STATE_DIR") or Path.home() / ".local" / "share" / "kernelblaster")
    root = _prepare_directory(root.expanduser())
    database = Path(sqlite_path or os.getenv("KERNELBLASTER_SQLITE_PATH") or root / "control.sqlite3").expanduser()
    cas = Path(cas_dir or os.getenv("KERNELBLASTER_CAS_DIR") or root / "cas").expanduser()
    database.parent.mkdir(parents=True, exist_ok=True)
    database_parent = _prepare_directory(database.parent)
    cas = _prepare_directory(cas)
    return root, (database_parent / database.name).resolve(), cas


class StateStore:
    """Own the paired SQLite repository and local content-addressed store."""

    def __init__(
        self,
        *,
        state_dir: str | Path | None = None,
        sqlite_path: str | Path | None = None,
        cas_dir: str | Path | None = None,
    ) -> None:
        self.state_dir, self.sqlite_path, self.cas_dir = resolve_state_paths(
            state_dir=state_dir, sqlite_path=sqlite_path, cas_dir=cas_dir
        )
        self.repository = JobRepository(self.sqlite_path)
        self.cas = ContentAddressedStore(self.cas_dir)

    def verify_artifact_index(self) -> None:
        """Verify that every indexed artifact still resolves to its CAS payload."""
        for indexed in self.repository.list_artifacts():
            verified = self.cas.verify(indexed["digest"])
            if verified.size_bytes != indexed["size_bytes"]:
                raise ValueError(f"artifact metadata size mismatch for {indexed['digest']}")
