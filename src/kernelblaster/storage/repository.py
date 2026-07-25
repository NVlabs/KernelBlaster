"""SQLite repository for control-owned runs, jobs, leases, and artifact indexes."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator
import uuid

from .cas import ArtifactMetadata


TERMINAL_JOB_STATUSES = {"succeeded", "failed", "blocked", "timed_out", "cancelled"}
ACTIVE_JOB_STATUSES = {"leased", "running"}
ALL_JOB_STATUSES = {"pending", *ACTIVE_JOB_STATUSES, *TERMINAL_JOB_STATUSES}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    return (value or _utc_now()).isoformat().replace("+00:00", "Z")


def _json(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"))


class JobRepository:
    """A short-lived-connection SQLite repository safe for a single control process."""

    def __init__(self, database_path: str | Path, *, busy_timeout_ms: int = 5_000) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout_ms = busy_timeout_ms
        self.migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=self.busy_timeout_ms / 1000)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        return connection

    @contextmanager
    def _transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def migrate(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )"""
            )
            applied = {
                row["version"] for row in connection.execute("SELECT version FROM schema_migrations")
            }
            if 1 not in applied:
                connection.executescript(
                    """
                    CREATE TABLE runs (
                        id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        finished_at TEXT,
                        metadata_json TEXT NOT NULL
                    );
                    CREATE TABLE jobs (
                        id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        idempotency_key TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        status TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        terminal_reason TEXT,
                        UNIQUE(run_id, idempotency_key)
                    );
                    CREATE TABLE job_attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        ordinal INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        worker_id TEXT,
                        started_at TEXT NOT NULL,
                        finished_at TEXT,
                        result_json TEXT,
                        UNIQUE(job_id, ordinal)
                    );
                    CREATE TABLE leases (
                        id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        attempt_id INTEGER NOT NULL REFERENCES job_attempts(id) ON DELETE CASCADE,
                        worker_id TEXT NOT NULL,
                        acquired_at TEXT NOT NULL,
                        renewed_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        released_at TEXT,
                        release_reason TEXT
                    );
                    CREATE UNIQUE INDEX one_active_lease_per_job
                    ON leases(job_id) WHERE released_at IS NULL;
                    CREATE TABLE artifacts (
                        digest TEXT PRIMARY KEY,
                        media_type TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        producer TEXT,
                        source_digest TEXT,
                        schema_name TEXT
                    );
                    CREATE TABLE job_artifacts (
                        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        attempt_id INTEGER REFERENCES job_attempts(id) ON DELETE CASCADE,
                        digest TEXT NOT NULL REFERENCES artifacts(digest),
                        role TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(job_id, attempt_id, digest, role)
                    );
                    CREATE TABLE run_artifacts (
                        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        digest TEXT NOT NULL REFERENCES artifacts(digest),
                        role TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(run_id, digest, role)
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO schema_migrations(version, name, applied_at) VALUES (1, ?, ?)",
                    ("initial_state_store", _timestamp()),
                )

    def create_run(self, run_id: str, *, status: str = "running", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        now = _timestamp()
        with self._transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO runs(id, status, created_at, metadata_json) VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING""",
                (run_id, status, now, _json(metadata)),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"run {run_id} does not exist")
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def finish_run(self, run_id: str, status: str) -> dict[str, Any]:
        with self._transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?",
                (status, _timestamp(), run_id),
            )
        return self.get_run(run_id)

    def delete_run_view(self, run_id: str) -> None:
        """Delete DB references only; immutable CAS payloads remain available."""
        with self._transaction(immediate=True) as connection:
            connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    def submit_job(
        self,
        *,
        run_id: str,
        idempotency_key: str,
        kind: str,
        payload: dict[str, Any] | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        if not idempotency_key or not kind:
            raise ValueError("idempotency_key and kind are required")
        job_id = job_id or uuid.uuid4().hex
        now = _timestamp()
        with self._transaction(immediate=True) as connection:
            if connection.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is None:
                raise KeyError(f"run {run_id} does not exist")
            connection.execute(
                """INSERT INTO jobs(id, run_id, idempotency_key, kind, status, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)
                ON CONFLICT(run_id, idempotency_key) DO NOTHING""",
                (job_id, run_id, idempotency_key, kind, _json(payload), now, now),
            )
            row = connection.execute(
                "SELECT * FROM jobs WHERE run_id = ? AND idempotency_key = ?",
                (run_id, idempotency_key),
            ).fetchone()
        assert row is not None
        return self._job_row(row)

    @staticmethod
    def _job_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def _recover_expired_leases(self, connection: sqlite3.Connection, now: str) -> None:
        expired = connection.execute(
            "SELECT id, job_id, attempt_id FROM leases WHERE released_at IS NULL AND expires_at <= ?",
            (now,),
        ).fetchall()
        for lease in expired:
            connection.execute(
                "UPDATE leases SET released_at = ?, release_reason = 'expired' WHERE id = ? AND released_at IS NULL",
                (now, lease["id"]),
            )
            connection.execute(
                "UPDATE job_attempts SET status = 'timed_out', finished_at = ? WHERE id = ? AND finished_at IS NULL",
                (now, lease["attempt_id"]),
            )
            connection.execute(
                "UPDATE jobs SET status = 'pending', updated_at = ? WHERE id = ? AND status IN ('leased', 'running')",
                (now, lease["job_id"]),
            )

    def acquire_lease(self, *, worker_id: str, ttl_seconds: int = 60) -> dict[str, Any] | None:
        if not worker_id or not 1 <= ttl_seconds <= 3_600:
            raise ValueError("worker_id and a TTL between 1 and 3600 seconds are required")
        now_value = _utc_now()
        now = _timestamp(now_value)
        expires_at = _timestamp(now_value + timedelta(seconds=ttl_seconds))
        with self._transaction(immediate=True) as connection:
            self._recover_expired_leases(connection, now)
            job = connection.execute(
                "SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at, id LIMIT 1"
            ).fetchone()
            if job is None:
                return None
            ordinal = connection.execute(
                "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM job_attempts WHERE job_id = ?",
                (job["id"],),
            ).fetchone()[0]
            attempt = connection.execute(
                """INSERT INTO job_attempts(job_id, ordinal, status, worker_id, started_at)
                VALUES (?, ?, 'leased', ?, ?) RETURNING id""",
                (job["id"], ordinal, worker_id, now),
            ).fetchone()
            lease_id = uuid.uuid4().hex
            connection.execute(
                """INSERT INTO leases(id, job_id, attempt_id, worker_id, acquired_at, renewed_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (lease_id, job["id"], attempt["id"], worker_id, now, now, expires_at),
            )
            connection.execute(
                "UPDATE jobs SET status = 'leased', updated_at = ? WHERE id = ?",
                (now, job["id"]),
            )
            result = self._job_row(job)
            result.update(
                {
                    "lease_id": lease_id,
                    "attempt_id": attempt["id"],
                    "worker_id": worker_id,
                    "expires_at": expires_at,
                    "status": "leased",
                }
            )
            return result

    def acquire_job_lease(
        self, *, job_id: str, worker_id: str, ttl_seconds: int = 60
    ) -> dict[str, Any]:
        if not worker_id or not 1 <= ttl_seconds <= 3_600:
            raise ValueError("worker_id and a TTL between 1 and 3600 seconds are required")
        now_value = _utc_now()
        now = _timestamp(now_value)
        expires_at = _timestamp(now_value + timedelta(seconds=ttl_seconds))
        with self._transaction(immediate=True) as connection:
            self._recover_expired_leases(connection, now)
            job = connection.execute(
                "SELECT * FROM jobs WHERE id = ? AND status = 'pending'", (job_id,)
            ).fetchone()
            if job is None:
                raise KeyError(f"pending job {job_id} does not exist")
            ordinal = connection.execute(
                "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM job_attempts WHERE job_id = ?",
                (job_id,),
            ).fetchone()[0]
            attempt = connection.execute(
                """INSERT INTO job_attempts(job_id, ordinal, status, worker_id, started_at)
                VALUES (?, ?, 'leased', ?, ?) RETURNING id""",
                (job_id, ordinal, worker_id, now),
            ).fetchone()
            lease_id = uuid.uuid4().hex
            connection.execute(
                """INSERT INTO leases(id, job_id, attempt_id, worker_id, acquired_at, renewed_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (lease_id, job_id, attempt["id"], worker_id, now, now, expires_at),
            )
            connection.execute(
                "UPDATE jobs SET status = 'leased', updated_at = ? WHERE id = ?",
                (now, job_id),
            )
        return {
            **self._job_row(job),
            "status": "leased",
            "lease_id": lease_id,
            "attempt_id": attempt["id"],
            "worker_id": worker_id,
            "expires_at": expires_at,
        }

    def cancel_pending_job(self, job_id: str, *, reason: str = "cancelled") -> dict[str, Any]:
        now = _timestamp()
        with self._transaction(immediate=True) as connection:
            job = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(f"job {job_id} does not exist")
            if job["status"] in TERMINAL_JOB_STATUSES:
                return self._job_row(job)
            if job["status"] != "pending":
                raise ValueError("only pending jobs may be cancelled without an active lease")
            connection.execute(
                "UPDATE jobs SET status = 'cancelled', updated_at = ?, terminal_reason = ? WHERE id = ?",
                (now, reason, job_id),
            )
            cancelled = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        assert cancelled is not None
        return self._job_row(cancelled)

    def heartbeat_lease(self, *, lease_id: str, worker_id: str, ttl_seconds: int = 60) -> dict[str, Any]:
        if not worker_id or not 1 <= ttl_seconds <= 3_600:
            raise ValueError("worker_id and a TTL between 1 and 3600 seconds are required")
        now_value = _utc_now()
        now = _timestamp(now_value)
        expires_at = _timestamp(now_value + timedelta(seconds=ttl_seconds))
        with self._transaction(immediate=True) as connection:
            self._recover_expired_leases(connection, now)
            lease = connection.execute(
                "SELECT * FROM leases WHERE id = ? AND worker_id = ? AND released_at IS NULL",
                (lease_id, worker_id),
            ).fetchone()
            if lease is None:
                raise KeyError("active lease does not exist")
            connection.execute(
                "UPDATE leases SET renewed_at = ?, expires_at = ? WHERE id = ?",
                (now, expires_at, lease_id),
            )
            connection.execute(
                "UPDATE jobs SET status = 'running', updated_at = ? WHERE id = ? AND status = 'leased'",
                (now, lease["job_id"]),
            )
        return {"lease_id": lease_id, "expires_at": expires_at, "status": "running"}

    def complete_job(
        self,
        *,
        job_id: str,
        lease_id: str,
        worker_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        reason: str | None = None,
        artifact_roles: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if status not in TERMINAL_JOB_STATUSES:
            raise ValueError(f"invalid terminal status: {status}")
        now = _timestamp()
        with self._transaction(immediate=True) as connection:
            job = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(f"job {job_id} does not exist")
            if job["status"] in TERMINAL_JOB_STATUSES:
                attempt = connection.execute(
                    "SELECT id FROM job_attempts WHERE job_id = ? ORDER BY ordinal DESC LIMIT 1",
                    (job_id,),
                ).fetchone()
                self._link_job_artifacts(
                    connection,
                    job_id=job_id,
                    attempt_id=attempt["id"] if attempt is not None else None,
                    artifact_roles=artifact_roles or {},
                )
                return {"job": self._job_row(job), "idempotent": True}
            lease = connection.execute(
                """SELECT * FROM leases WHERE id = ? AND job_id = ? AND worker_id = ?
                AND released_at IS NULL AND expires_at > ?""",
                (lease_id, job_id, worker_id, now),
            ).fetchone()
            if lease is None:
                raise KeyError("active lease does not exist")
            connection.execute(
                "UPDATE leases SET released_at = ?, release_reason = 'completed' WHERE id = ?",
                (now, lease_id),
            )
            connection.execute(
                "UPDATE job_attempts SET status = ?, finished_at = ?, result_json = ? WHERE id = ?",
                (status, now, _json(result), lease["attempt_id"]),
            )
            connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ?, terminal_reason = ? WHERE id = ?",
                (status, now, reason, job_id),
            )
            self._link_job_artifacts(
                connection,
                job_id=job_id,
                attempt_id=lease["attempt_id"],
                artifact_roles=artifact_roles or {},
            )
            completed = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        assert completed is not None
        return {"job": self._job_row(completed), "idempotent": False}

    @staticmethod
    def _link_job_artifacts(
        connection: sqlite3.Connection,
        *,
        job_id: str,
        attempt_id: int | None,
        artifact_roles: dict[str, str],
    ) -> None:
        for digest, role in artifact_roles.items():
            if connection.execute(
                "SELECT 1 FROM artifacts WHERE digest = ?", (digest,)
            ).fetchone() is None:
                raise KeyError(f"artifact {digest} does not exist")
            connection.execute(
                """INSERT OR IGNORE INTO job_artifacts(job_id, attempt_id, digest, role, created_at)
                VALUES (?, ?, ?, ?, ?)""",
                (job_id, attempt_id, digest, role, _timestamp()),
            )

    def register_artifact(self, artifact: ArtifactMetadata) -> dict[str, Any]:
        with self._transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO artifacts(digest, media_type, size_bytes, created_at, producer, source_digest, schema_name)
                VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(digest) DO NOTHING""",
                (
                    artifact.digest,
                    artifact.media_type,
                    artifact.size_bytes,
                    artifact.created_at,
                    artifact.producer,
                    artifact.source_digest,
                    artifact.schema,
                ),
            )
            row = connection.execute("SELECT * FROM artifacts WHERE digest = ?", (artifact.digest,)).fetchone()
        assert row is not None
        return dict(row)

    def link_job_artifact(self, *, job_id: str, digest: str, role: str, attempt_id: int | None = None) -> None:
        with self._transaction(immediate=True) as connection:
            connection.execute(
                """INSERT OR IGNORE INTO job_artifacts(job_id, attempt_id, digest, role, created_at)
                VALUES (?, ?, ?, ?, ?)""",
                (job_id, attempt_id, digest, role, _timestamp()),
            )

    def link_run_artifact(self, *, run_id: str, digest: str, role: str) -> None:
        with self._transaction(immediate=True) as connection:
            connection.execute(
                """INSERT OR IGNORE INTO run_artifacts(run_id, digest, role, created_at)
                VALUES (?, ?, ?, ?)""",
                (run_id, digest, role, _timestamp()),
            )

    def artifact_references(self, digest: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, role FROM run_artifacts WHERE digest = ? ORDER BY run_id, role", (digest,)
            ).fetchall()
        return [dict(row) for row in rows]

    def list_artifacts(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM artifacts ORDER BY digest").fetchall()
        return [dict(row) for row in rows]
