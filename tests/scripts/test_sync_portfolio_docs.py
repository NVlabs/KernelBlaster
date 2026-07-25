from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "sync_portfolio_docs", ROOT / "scripts" / "sync_portfolio_docs.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _copy_documentation_root(tmp_path: Path) -> Path:
    for name in ("README.md", "README.zh-CN.md"):
        shutil.copy2(ROOT / name, tmp_path / name)
    shutil.copytree(ROOT / "docs", tmp_path / "docs")
    shutil.copytree(ROOT / "artifacts", tmp_path / "artifacts")
    (tmp_path / "portfolio").mkdir()
    shutil.copy2(ROOT / "portfolio" / "status.json", tmp_path / "portfolio" / "status.json")
    shutil.copytree(
        ROOT / "portfolio" / "case_studies",
        tmp_path / "portfolio" / "case_studies",
    )
    return tmp_path


def test_context_derives_published_nine_and_ten_task_metrics():
    context = MODULE.load_context(ROOT)
    assert context["new9"]["attempted_upstream"] == pytest.approx(5.0199631675)
    assert context["new9"]["strict_upstream"] == pytest.approx(3.3017281925)
    assert context["new9"]["attempted_pytorch"] == pytest.approx(1.4154586790)
    assert context["new9"]["strict_pytorch"] == pytest.approx(0.9309749235)
    assert context["new9"]["verified"] == 3
    assert context["all10"]["verified"] == 4
    assert context["core10_v2_summary"]["verified_improved_tasks"] == 4
    assert context["core10_v2_summary"]["no_improvement_tasks"] == 1
    assert context["core10_v2_summary"]["inconclusive_tasks"] == 5
    assert context["core10_v2_summary"]["pytorch_comparable_tasks"] == 9


def test_sync_is_idempotent(tmp_path):
    root = _copy_documentation_root(tmp_path)
    path = root / "README.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "This fork has completed", "STALE: This fork has completed", 1
        ),
        encoding="utf-8",
    )
    changed = MODULE.synchronize(root=root, write=True)
    assert Path("README.md") in changed
    assert MODULE.synchronize(root=root, write=True) == []
    assert MODULE.synchronize(root=root, write=False) == []


def test_generated_status_blocks_are_localized_and_evidence_is_labeled():
    context = MODULE.load_context(ROOT)
    chinese = MODULE._root_block(context, chinese=True)
    english = MODULE._root_block(context, chinese=False)
    index_chinese = MODULE._index_block(context, chinese=True)
    index_english = MODULE._index_block(context, chinese=False)
    architecture_chinese = MODULE._architecture_block(context, chinese=True)
    architecture_english = MODULE._architecture_block(context, chinese=False)
    expected_chinese = MODULE._validation_labels(
        context["status"]["validation"], chinese=True
    )
    expected_english = MODULE._validation_labels(
        context["status"]["validation"], chinese=False
    )

    assert expected_chinese["current_cpu_pytest"] in chinese
    assert "历史 10/10；schema v2 完整验证 10/10 通过" in chinese
    assert expected_chinese["live_api_smoke"] in chinese
    assert expected_chinese["cross_gpu"] in chinese
    assert expected_english["cross_gpu"] in english
    assert "10/10 candidates passed" not in chinese
    assert "Schema v2 定向验证" in chinese
    assert "Schema v2 完整 Core 10 验证" in chinese
    assert "Schema-v2 targeted validation" in english
    assert "Schema-v2 full Core 10 validation" in english
    assert "Full Chinese report" in english
    assert "English full report" not in english
    assert "9/10" in index_chinese
    assert "1.053×" in index_chinese
    assert expected_chinese["live_api_smoke"] in index_chinese
    assert expected_chinese["live_api_smoke"] in architecture_chinese
    assert "9/10" in index_english
    assert "1.053×" in index_english
    assert expected_english["live_api_smoke"] in index_english
    assert expected_english["live_api_smoke"] in architecture_english
    assert "strict full-ten ratio is 0.992×" not in index_english


@pytest.mark.parametrize(
    ("ncu_state", "cross_gpu_state", "english_marker", "chinese_marker"),
    (
        (
            "blocked: ERR_NVGPUCTRPERM",
            "NOT RUN (Day 11-14 out of scope)",
            "blocked: ERR_NVGPUCTRPERM",
            "阻塞：ERR_NVGPUCTRPERM",
        ),
        (
            "local-passed",
            "local-passed",
            "local-passed (RTX 3080 counter evidence complete; cross-GPU pending)",
            "本地通过（RTX 3080 计数器证据已完成；等待跨 GPU 复测）",
        ),
        (
            "cross-gpu-passed",
            "cross-gpu-passed",
            "cross-gpu-passed (local and A100/L40S counter evidence complete)",
            "跨 GPU 通过（本地与 A100/L40S 计数器证据均已完成）",
        ),
    ),
)
def test_validation_labels_support_ncu_lifecycle_states(
    ncu_state, cross_gpu_state, english_marker, chinese_marker
):
    validation = dict(MODULE.load_context(ROOT)["status"]["validation"])
    validation["ncu_counters"] = ncu_state
    validation["cross_gpu"] = cross_gpu_state

    assert MODULE._validation_labels(validation, chinese=False)[
        "ncu_counters"
    ] == english_marker
    assert MODULE._validation_labels(validation, chinese=True)[
        "ncu_counters"
    ] == chinese_marker


def test_generated_blocks_do_not_hardcode_old_ncu_blocker_after_local_pass():
    context = MODULE.load_context(ROOT)
    context["status"] = dict(context["status"])
    context["status"]["validation"] = dict(context["status"]["validation"])
    context["status"]["validation"].update(
        {"ncu_counters": "local-passed", "cross_gpu": "local-passed"}
    )

    rendered = "\n".join(
        (
            MODULE._validation_block(context, chinese=False),
            MODULE._architecture_block(context, chinese=False),
            MODULE._rmsnorm_block(context, chinese=False),
            MODULE._validation_block(context, chinese=True),
            MODULE._architecture_block(context, chinese=True),
            MODULE._rmsnorm_block(context, chinese=True),
        )
    )
    assert "ERR_NVGPUCTRPERM" not in rendered
    assert "LOCAL-PASSED" in rendered
    assert "本地通过" in rendered


def test_unknown_ncu_lifecycle_state_is_rejected():
    validation = dict(MODULE.load_context(ROOT)["status"]["validation"])
    validation["ncu_counters"] = "maybe"
    with pytest.raises(MODULE.DocumentationSyncError, match="Unsupported ncu_counters"):
        MODULE._validation_labels(validation, chinese=False)


def test_readme_confirmation_commands_use_five_sessions():
    for name in ("README.md", "README.zh-CN.md"):
        readme = (ROOT / name).read_text(encoding="utf-8")
        confirmation = readme.split("### ", 2)[1]
        assert confirmation.count("--sessions 5") == 2
        assert "--sessions 3" not in confirmation


def test_docs_markdown_explanations_have_chinese_and_english_pairs():
    english_documents = [
        path for path in (ROOT / "docs").rglob("*.md") if not path.name.endswith(".zh-CN.md")
    ]
    chinese_documents = list((ROOT / "docs").rglob("*.zh-CN.md"))
    assert english_documents
    assert len(english_documents) == len(chinese_documents)
    for english in english_documents:
        chinese = english.with_name(f"{english.stem}.zh-CN.md")
        assert chinese.is_file(), f"Missing Chinese documentation pair for {english}"
        assert chinese.name in english.read_text(encoding="utf-8")
        assert english.name in chinese.read_text(encoding="utf-8")


def test_replace_block_rejects_missing_markers():
    with pytest.raises(MODULE.DocumentationSyncError, match="exactly one"):
        MODULE.replace_block("no markers", "STATUS", "body")


def test_replace_block_rejects_duplicate_markers():
    text = (
        "<!-- STATUS:START -->\na\n<!-- STATUS:END -->\n"
        "<!-- STATUS:START -->\nb\n<!-- STATUS:END -->\n"
    )
    with pytest.raises(MODULE.DocumentationSyncError, match="exactly one"):
        MODULE.replace_block(text, "STATUS", "body")


def test_invalid_comparison_schema_fails_loudly(tmp_path):
    root = _copy_documentation_root(tmp_path)
    comparison = root / "artifacts/portfolio-v1.0/results/core10_rtx3080_comparison.json"
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    payload["schema_version"] = "broken"
    comparison.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MODULE.DocumentationSyncError, match="schema_version"):
        MODULE.load_context(root)


def test_missing_status_source_fails_loudly(tmp_path):
    root = _copy_documentation_root(tmp_path)
    status_path = root / "portfolio/status.json"
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload["sources"]["comparison"] = "artifacts/missing.json"
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MODULE.DocumentationSyncError, match="does not exist"):
        MODULE.load_context(root)


def test_artifact_hash_mismatch_is_rejected(tmp_path):
    root = _copy_documentation_root(tmp_path)
    context = MODULE.load_context(root)
    readme = root / "artifacts/portfolio-v1.0/README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(MODULE.DocumentationSyncError, match="SHA256"):
        MODULE.validate_artifact_hashes(root, context)


def test_change_policy_requires_documentation_for_results():
    with pytest.raises(MODULE.DocumentationSyncError, match="require README/docs"):
        MODULE.validate_change_policy(["portfolio/case_studies/core10/new.cu"])
    MODULE.validate_change_policy(
        ["portfolio/case_studies/core10/new.cu", "docs/portfolio/validation.md"]
    )


def test_broken_relative_markdown_link_is_rejected(tmp_path):
    with pytest.raises(MODULE.DocumentationSyncError, match="Broken link"):
        MODULE.validate_links(
            tmp_path,
            {Path("README.md"): "[missing](docs/not-there.md)"},
        )


@pytest.mark.parametrize(
    "machine_path",
    (
        "/home/example/src/KernelBlaster/out/results.json",
        "C:\\Users\\example\\KernelBlaster\\out\\results.json",
        "\\\\" + "wsl.localhost\\Ubuntu\\home\\example\\KernelBlaster",
    ),
)
def test_machine_specific_absolute_path_is_rejected(tmp_path, machine_path):
    with pytest.raises(MODULE.DocumentationSyncError, match="Machine-specific"):
        MODULE.validate_links(tmp_path, {Path("README.md"): machine_path})
