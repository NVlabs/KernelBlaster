from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.kernelblaster.observability import RunRecorder
from src.kernelblaster.storage import StateStore


def _store(tmp_path: Path) -> StateStore:
    return StateStore(state_dir=tmp_path / "state")


def test_pending_jobs_survive_restart_and_submission_is_idempotent(tmp_path):
    store = _store(tmp_path)
    store.repository.create_run("run-1", metadata={"suite": "unit"})
    first = store.repository.submit_job(
        run_id="run-1", idempotency_key="candidate-1", kind="compile", payload={"x": 1}
    )
    second = store.repository.submit_job(
        run_id="run-1", idempotency_key="candidate-1", kind="compile", payload={"x": 2}
    )

    restarted = StateStore(state_dir=tmp_path / "state")
    lease = restarted.repository.acquire_lease(worker_id="worker-a")

    assert first["id"] == second["id"]
    assert second["payload"] == {"x": 1}
    assert lease is not None
    assert lease["id"] == first["id"]
    assert lease["status"] == "leased"


def test_expired_lease_is_recovered_once_and_completion_is_idempotent(tmp_path):
    store = _store(tmp_path)
    store.repository.create_run("run-1")
    job = store.repository.submit_job(
        run_id="run-1", idempotency_key="candidate-1", kind="profile"
    )
    first = store.repository.acquire_lease(worker_id="worker-a")
    assert first is not None
    expired_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat().replace(
        "+00:00", "Z"
    )
    with store.repository._transaction(immediate=True) as connection:
        connection.execute("UPDATE leases SET expires_at = ? WHERE id = ?", (expired_at, first["lease_id"]))

    second = store.repository.acquire_lease(worker_id="worker-b")
    assert second is not None
    assert second["lease_id"] != first["lease_id"]
    completed = store.repository.complete_job(
        job_id=job["id"],
        lease_id=second["lease_id"],
        worker_id="worker-b",
        status="succeeded",
        result={"speedup": 1.1},
    )
    duplicate = store.repository.complete_job(
        job_id=job["id"],
        lease_id=second["lease_id"],
        worker_id="worker-b",
        status="succeeded",
    )

    with store.repository._connect() as connection:
        expired_count = connection.execute(
            "SELECT COUNT(*) FROM leases WHERE release_reason = 'expired'"
        ).fetchone()[0]
        terminal_attempts = connection.execute(
            "SELECT COUNT(*) FROM job_attempts WHERE job_id = ? AND finished_at IS NOT NULL",
            (job["id"],),
        ).fetchone()[0]
    assert completed["idempotent"] is False
    assert duplicate["idempotent"] is True
    assert expired_count == 1
    assert terminal_attempts == 2  # one expired attempt and one completed attempt


def test_only_one_worker_receives_a_lease_for_one_pending_job(tmp_path):
    store = _store(tmp_path)
    store.repository.create_run("run-1")
    store.repository.submit_job(run_id="run-1", idempotency_key="candidate-1", kind="compile")
    with ThreadPoolExecutor(max_workers=2) as pool:
        leases = list(pool.map(lambda worker: store.repository.acquire_lease(worker_id=worker), ("a", "b")))
    assert sum(lease is not None for lease in leases) == 1


def test_cas_deduplicates_detects_corruption_and_restricts_exports(tmp_path):
    store = _store(tmp_path)
    first = store.cas.put_bytes(b"same payload", media_type="text/plain", producer="test")
    second = store.cas.put_bytes(b"same payload", media_type="text/plain", producer="test")
    assert first.digest == second.digest
    assert first.size_bytes == second.size_bytes == len(b"same payload")
    export = store.cas.export(first.digest, tmp_path / "exports", "nested/payload.txt")
    assert export.read_bytes() == b"same payload"
    with pytest.raises(ValueError, match="relative path"):
        store.cas.export(first.digest, tmp_path / "exports", "../escape.txt")

    store.cas.get_path(first.digest).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        store.cas.verify(first.digest)


def test_deleting_a_run_view_does_not_delete_shared_cas_payload(tmp_path):
    store = _store(tmp_path)
    artifact = store.cas.put_bytes(b"shared")
    store.repository.register_artifact(artifact)
    for run_id in ("run-a", "run-b"):
        store.repository.create_run(run_id)
        store.repository.link_run_artifact(run_id=run_id, digest=artifact.digest, role="summary")

    store.repository.delete_run_view("run-a")

    assert store.cas.verify(artifact.digest).size_bytes == len(b"shared")
    assert store.repository.artifact_references(artifact.digest) == [
        {"run_id": "run-b", "role": "summary"}
    ]


def test_run_recorder_indexes_manifest_events_and_summary_in_cas(tmp_path):
    store = _store(tmp_path)
    recorder = RunRecorder(
        tmp_path / "run-view",
        model="gpt-5.6-terra",
        provider_config={"api_key_configured": True},
        run_id="recorded-run",
        repo_root=tmp_path,
        state_store=store,
    )
    recorder.record_event("portfolio_run_started", data={"task_count": 1})
    recorder.close("completed")

    with store.repository._connect() as connection:
        references = connection.execute(
            "SELECT digest, role FROM run_artifacts WHERE run_id = ? ORDER BY role",
            ("recorded-run",),
        ).fetchall()
    assert [reference["role"] for reference in references] == ["events", "manifest", "summary"]
    assert all(store.cas.verify(reference["digest"]).size_bytes > 0 for reference in references)
    assert store.repository.get_run("recorded-run")["status"] == "completed"
    store.verify_artifact_index()
