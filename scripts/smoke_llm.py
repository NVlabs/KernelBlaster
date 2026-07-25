#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from src.kernelblaster.llm import (  # noqa: E402
    OpenAICompatibleProvider,
    OpenAICompatibleSettings,
)
from src.kernelblaster.observability import (  # noqa: E402
    RunRecorder,
    record_event,
    set_run_recorder,
)


SMOKE_MODEL = "gpt-5.6-sol"


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT_DIR / "out" / "portfolio" / "smoke" / timestamp


def _atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


async def _run(args: argparse.Namespace) -> int:
    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"{args.api_key_env} is not configured; the smoke test will not run."
        )

    output_dir = args.output_dir.resolve()
    if any(
        (output_dir / name).exists()
        for name in ("run_manifest.json", "events.jsonl", "summary.json")
    ):
        raise RuntimeError(f"Refusing to overwrite existing run artifacts: {output_dir}")

    settings = OpenAICompatibleSettings(
        base_url=args.base_url,
        api_key=api_key,
        timeout_seconds=args.timeout_seconds,
        max_concurrency=1,
        max_retries=0,
        max_requests=1,
        max_total_tokens=args.max_total_tokens,
        max_completion_tokens=args.max_completion_tokens,
        reasoning_effort=args.reasoning_effort,
        stream=False,
        log_content=False,
    )
    provider = OpenAICompatibleProvider(settings)
    recorder = RunRecorder(
        output_dir,
        model=args.model,
        provider_config=provider.public_config(),
        suite={"name": "live-api-smoke", "requests": 1},
        dry_run=False,
        repo_root=ROOT_DIR,
    )
    set_run_recorder(recorder)
    try:
        response = await provider.generate(
            [
                {
                    "role": "user",
                    "content": (
                        "Return exactly the single word KERNELBLASTER_OK and no "
                        "additional text."
                    ),
                }
            ],
            model=args.model,
            n=1,
        )
        content = response.response
        result = {
            "schema_version": "2.0",
            "requested_model": args.model,
            "response_models": response.response_models,
            "provider": response.provider,
            "request_ids": response.request_ids,
            "usage": response.usage,
            "usage_source": response.usage_source,
            "attempts": response.attempts,
            "elapsed_time_seconds": response.elapsed_time,
            "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "response_character_count": len(content),
            "expected_content_matched": content.strip() == "KERNELBLASTER_OK",
        }
        record_event(
            "llm_smoke_verified",
            status="ok" if result["expected_content_matched"] else "error",
            stage="live_api_smoke",
            data=result,
        )
        _atomic_write(output_dir / "smoke_result.json", result)
        recorder.close(
            "completed" if result["expected_content_matched"] else "completed_with_errors"
        )
        return 0 if result["expected_content_matched"] else 2
    except Exception as error:
        record_event(
            "llm_smoke_failed",
            status="error",
            stage="live_api_smoke",
            data={"error_type": type(error).__name__},
        )
        recorder.close("failed")
        raise
    finally:
        set_run_recorder(None)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send one bounded OpenAI-compatible request without logging content."
    )
    parser.add_argument(
        "--model",
        default=SMOKE_MODEL,
        choices=(SMOKE_MODEL,),
        help="Fixed trusted-smoke model; MODEL and alternate CLI values are ignored/rejected.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("KERNELBLASTER_LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--api-key-env",
        default=(
            "KERNELBLASTER_LLM_API_KEY"
            if os.getenv("KERNELBLASTER_LLM_API_KEY")
            else "OPENAI_API_KEY"
        ),
    )
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--max-completion-tokens", type=int, default=64)
    parser.add_argument("--max-total-tokens", type=int, default=10_000)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()
    try:
        return asyncio.run(_run(args))
    except Exception as error:
        parser.exit(1, f"LLM smoke failed: {type(error).__name__}\n")


if __name__ == "__main__":
    raise SystemExit(main())
