from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

from src.kernelblaster.observability import (
    RunRecorder,
    event_context,
    prompt_metadata,
    record_event,
    redact_secrets,
    set_run_recorder,
)
from src.kernelblaster.measurements import Measurement, MeasurementSource, MeasurementUnit


def _recorder(tmp_path) -> RunRecorder:
    return RunRecorder(
        tmp_path,
        model="gpt-5.6-terra",
        provider_config={
            "base_url": "https://example.test/v1?api_key=secret",
            "api_key_configured": True,
        },
        suite={"name": "unit"},
        gpu_target="rtx3080",
        repo_root=tmp_path,
    )


def test_secret_redaction_handles_keys_headers_bearer_and_urls():
    payload = {
        "api_key": "sk-secret",
        "prompt_tokens": 12,
        "authorization_header": "Authorization: Bearer token-value",
        "message": (
            "Bearer another-token "
            "https://user:password@example.test/v1?access_token=url-token"
        ),
    }
    redacted = redact_secrets(payload)
    serialized = json.dumps(redacted)
    for secret in ("sk-secret", "token-value", "another-token", "url-token", "password"):
        assert secret not in serialized
    assert redacted["prompt_tokens"] == 12


def test_prompt_content_is_hashed_by_default():
    metadata = prompt_metadata(
        [{"role": "user", "content": "private prompt"}],
        include_content=False,
    )
    assert metadata["message_count"] == 1
    assert "private prompt" not in repr(metadata)
    assert len(metadata["sha256"]) == 64


def test_event_sequence_is_thread_safe_and_summary_is_atomic(tmp_path):
    recorder = _recorder(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda index: recorder.record_event(
                    "llm_request_completed",
                    data={
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 2,
                            "total_tokens": 3,
                        },
                        "latency_seconds": 0.1,
                        "index": index,
                    },
                ),
                range(40),
            )
        )
    recorder.close()

    events = [json.loads(line) for line in recorder.events_path.read_text().splitlines()]
    assert [event["sequence"] for event in events] == list(range(1, 41))
    summary = json.loads(recorder.summary_path.read_text())
    assert summary["llm"]["requests_completed"] == 40
    assert summary["llm"]["total_tokens"] == 120
    assert not list(tmp_path.glob("*.tmp"))


def test_context_is_propagated_to_events(tmp_path):
    recorder = _recorder(tmp_path)
    set_run_recorder(recorder)
    try:
        with event_context(
            task_id="036",
            rollout_id="2",
            stage="rollout_step_1",
            candidate_id="candidate-1",
        ):
            record_event("cuda_compile_completed", data={"success": True})
    finally:
        set_run_recorder(None)
        recorder.close()

    event = json.loads(recorder.events_path.read_text().strip())
    assert event["task_id"] == "036"
    assert event["rollout_id"] == "2"
    assert event["stage"] == "rollout_step_1"
    assert event["candidate_id"] == "candidate-1"


def test_failed_request_is_counted_and_marks_smoke_failed(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.record_event("llm_request_started", attempt=1)
    recorder.record_event(
        "llm_request_failed",
        status="error",
        attempt=1,
        data={"error_type": "AuthenticationError", "status_code": 401},
    )
    recorder.close("failed")

    summary = json.loads(recorder.summary_path.read_text())
    manifest = json.loads(recorder.manifest_path.read_text())
    assert summary["llm"]["requests_started"] == 1
    assert summary["llm"]["requests_failed"] == 1
    assert manifest["budget"]["consumed"]["requests"] == 1
    assert manifest["validation"]["llm_smoke_test"] == "FAILED"
    assert manifest["validation"]["gates"]["api_smoke"] == "FAILED"


def test_task_outcome_records_a_structured_measurement(tmp_path):
    recorder = _recorder(tmp_path)
    measurement = Measurement(
        value=12.5,
        unit=MeasurementUnit.MICROSECONDS,
        source=MeasurementSource.CUDA_EVENTS,
        protocol_id="events-v1",
        hardware_fingerprint="gpu-a",
    )
    recorder.record_event(
        "task_outcome",
        data={
            "outcome": "no_improvement",
            "measurement": measurement,
            "timing_status": "measured",
        },
    )
    recorder.close()
    summary = json.loads(recorder.summary_path.read_text())
    assert summary["tasks"]["results"][0]["measurement"]["unit"] == "us"
    assert summary["tasks"]["results"][0]["measurement"]["source"] == "cuda_events"
