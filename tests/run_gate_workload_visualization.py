#!/usr/bin/env python3
"""Test Gate providers, raw receive folding, load profiles, and HTML output."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYSRC = ROOT / "pysrc"
sys.path.insert(0, str(PYSRC))
sys.path.insert(0, str(ROOT))

from moe_dag import (  # noqa: E402
    ModelSpec,
    Placement,
    emit_workload,
    make_contiguous_expert_placement,
)
from moe_dag.models import (  # noqa: E402
    TransformerWorkloadConfig,
    build_transformer_workload,
)
from workload.gate import (  # noqa: E402
    GATE_PROVIDER_NAMES,
    RawReceiveDataset,
    create_gate_provider,
)


RAW_PLACEMENT = (
    ROOT
    / "workload"
    / "raw_data"
    / "ET_4+4_32_9_gsm8k_r1_2k_2k_0417_al_0.json"
)
VISUALIZER = ROOT / "visualization" / "gate_load_profile.py"


@dataclass
class Result:
    name: str
    status: str
    detail: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = ROOT / "test_logs" / f"run_{timestamp}_gate_workload_visualization"
    run_dir.mkdir(parents=True)
    results: list[Result] = []

    try:
        dataset = RawReceiveDataset.load(RAW_PLACEMENT)
        require(dataset.num_ranks == 32, "raw rank count must be 32")
        require(dataset.num_layers == 58, "raw layer count must be 58")
        require(dataset.slots_per_rank == 9, "raw slots/rank must be 9")
        require(dataset.num_logical_experts == 256, "raw logical experts must be 256")
        require(
            set(dataset.total_receive_by_layer) == {2_868_128},
            "raw per-layer receive totals changed",
        )
        require(
            all(sum(row) == 2_868_128 for row in dataset.logical_loads),
            "physical-to-logical folding lost receive counts",
        )
        results.append(Result("raw_receive_fold", "passed", "32 rank x 58 layer x 9 slot 折叠为 256 logical experts；每层总量 2,868,128"))
    except Exception as exc:
        results.append(Result("raw_receive_fold", "failed", str(exc)))
        dataset = None

    placement = Placement(32, 8, make_contiguous_expert_placement(256, 32))
    tokens = (2,) * 32
    provider_summaries: dict[str, object] = {}
    try:
        for name in GATE_PROVIDER_NAMES:
            kwargs: dict[str, object] = {}
            if name == "raw_receive_cdf":
                kwargs["raw_placement_json"] = RAW_PLACEMENT
                kwargs["layer_map"] = (0, 1)
            provider = create_gate_provider(name, seed=17, **kwargs)
            first = provider.sample(
                layer_id=0,
                microbatch_id=0,
                tokens_per_source_rank=tokens,
                placement=placement,
                topk=8,
            )
            second = provider.sample(
                layer_id=0,
                microbatch_id=0,
                tokens_per_source_rank=tokens,
                placement=placement,
                topk=8,
            )
            require(len(first.assignments) == 512, f"{name}: route count mismatch")
            require(
                first.metadata["assignment_digest_sha256"]
                == second.metadata["assignment_digest_sha256"],
                f"{name}: same seed is not deterministic",
            )
            require(sum(first.metadata["logical_expert_loads"]) == 512, f"{name}: expert loads lost routes")
            if name == "raw_receive_cdf":
                target = first.metadata["target_expert_weights"]
                realized = first.metadata["logical_expert_loads"]
                require(
                    max(
                        abs(count - weight * 512)
                        for count, weight in zip(realized, target)
                    )
                    <= 1.0,
                    "raw receive exact quota exceeds one-route rounding error",
                )
                require(
                    first.metadata["routing_fidelity"]
                    == "quota_matched_global_receive_histogram",
                    "raw receive routing fidelity is stale",
                )
            provider_summaries[name] = {
                "digest": first.metadata["assignment_digest_sha256"],
                "rank_imbalance": first.metadata["rank_imbalance"],
                "target_total_variation": first.metadata[
                    "target_realized_total_variation"
                ],
            }
        results.append(Result("gate_providers", "passed", "5 种 provider 均满足固定 routes、同 seed 可复现和负载守恒；raw quota 误差不超过 1 route"))
    except Exception as exc:
        results.append(Result("gate_providers", "failed", str(exc)))

    try:
        model = ModelSpec(
            name="gate_visualization_test",
            hidden=256,
            ffn_hidden=512,
            num_attention_heads=4,
            num_kv_heads=4,
            head_dim=64,
            num_experts=256,
            topk=8,
            sequence_length=32,
            num_layers=2,
            micro_batches=2,
        )
        provider = create_gate_provider(
            "raw_receive_cdf",
            seed=17,
            raw_placement_json=RAW_PLACEMENT,
            layer_map=(0, 1),
        )
        built = build_transformer_workload(
            TransformerWorkloadConfig(
                model=model,
                placement=placement,
                tokens_per_rank=2,
                algorithm="eplb",
                chunk_tokens=32,
                gate_provider=provider,
            )
        )
        workload_dir = run_dir / "workload"
        emit_workload(built.graph, workload_dir, metadata=built.metadata)
        for record in built.metadata["micro_batch_algorithms"]:
            profile = record["expert_load_profile"]
            require(profile["before"]["total_routes"] == 512, "profile before route count mismatch")
            require(profile["after"]["total_routes"] == 512, "profile after route count mismatch")
            require(
                profile["before"]["logical_expert_loads"]
                == profile["after"]["logical_expert_loads"],
                "before/after changed logical Gate assignments",
            )
        output_dir = run_dir / "gate_load"
        command = [
            sys.executable,
            str(VISUALIZER),
            "--workload-dir", str(workload_dir),
            "--output-dir", str(output_dir),
            "--title", "Gate workload visualization functional test",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
        (run_dir / "命令.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        (run_dir / "可视化.log").write_text(completed.stdout, encoding="utf-8")
        require(completed.returncode == 0, f"visualizer returned {completed.returncode}")
        html = (output_dir / "gate_load_profile.html").read_text(encoding="utf-8")
        summary = json.loads((output_dir / "gate_load_profile_summary.json").read_text(encoding="utf-8"))
        require(len(summary["records"]) == 4, "HTML summary needs four layer/microbatch records")
        require((output_dir / "gate_load_profile.csv").is_file(), "expert instance CSV is missing")
        require(
            all(marker in html for marker in (
                "Rank load before / after",
                "Gate logical-expert distribution",
                "Expert instance details",
            )),
            "HTML controls or modules are incomplete",
        )
        results.append(Result("before_after_html", "passed", "EPLB 2 layer x 2 microbatch 生成同尺度 before/after rank 图、Gate expert 图和实例明细"))
    except Exception as exc:
        results.append(Result("before_after_html", "failed", str(exc)))

    passed = sum(item.status == "passed" for item in results)
    summary_payload = {
        "passed": passed,
        "total": len(results),
        "raw_dataset": str(RAW_PLACEMENT),
        "providers": provider_summaries,
        "cases": [asdict(item) for item in results],
    }
    (run_dir / "结果.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Gate 分布与 before/after 可视化测试报告",
        "",
        f"- 通过：`{passed}/{len(results)}`",
        "- workload：EP32、E256、top-k=8、2 tokens/rank smoke",
        "- raw 数据仅作为 decode 实测偏斜分布来源",
        "",
        "| 测试 | 状态 | 内容 |",
        "|---|---|---|",
    ]
    lines.extend(f"| {item.name} | {item.status} | {item.detail} |" for item in results)
    (run_dir / "测试报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Gate workload tests: {passed}/{len(results)} passed")
    print(f"log directory: {run_dir}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
