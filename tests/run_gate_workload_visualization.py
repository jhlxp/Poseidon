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
from moe_dag.cost import ComputeEstimate  # noqa: E402
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


class SlowVisualizationCostModel:
    communication_sms = 20

    def estimate(
        self,
        operation_flops: int,
        *,
        operation: str,
        overlaps_communication: bool = False,
        token_count: int | None = None,
    ) -> ComputeEstimate:
        tokens = token_count or 1
        return ComputeEstimate(
            operation_flops=operation_flops,
            duration_us=max(1.0, tokens * 1000.0),
            overlaps_communication=overlaps_communication,
            available_sms=112 if overlaps_communication else 132,
            peak_flops_per_second=1.0,
            source="gate_transport_test_slow_compute",
            token_count=token_count,
            us_per_token=1000.0,
            token_kind=operation,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "model": "gate_transport_test_slow_compute",
            "communication_sms": self.communication_sms,
        }


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
                "Rank MoE expert-token load: baseline / placement",
                "Server MoE expert-token load: baseline / placement",
                "MoE expert-token load definition",
                "Baseline: first-layer raw distribution",
                "Final-layer EPLB physical placement",
                "TopK-expanded expert-token count",
                "final server mean",
                "serverChart",
                "Gate logical-expert distribution",
                "First-layer Gate",
                "Final-layer Gate",
                "Expert instance details",
                "data-comparison-scope",
                "currentComparison",
                "const algorithm = 'eplb'",
                "Final physical instances",
                "Final replicas",
            )),
            "HTML controls or modules are incomplete",
        )
        require(
            '<h2>Cross-server expert copies</h2>' not in html
            and '<i style="background:#1098a3"></i>Expert Weight' not in html,
            "EPLB HTML exposed ProbeEP-only migration modules",
        )
        require('id="layer"' not in html, "HTML must compare layers within one microbatch")
        results.append(Result("before_after_html", "passed", "EPLB 每个 microbatch 比较首层 baseline 与末层 physical placement，不混入 ProbeEP migration 模块"))
    except Exception as exc:
        results.append(Result("before_after_html", "failed", str(exc)))

    try:
        expectations = {
            "nccl": (
                "Final-layer execution load",
                "NCCL uses the original logical expert placement",
            ),
            "deepep": (
                "Final-layer execution load",
                "DeepEP changes token payload aggregation",
            ),
            "moonep": (
                "Final-layer MoonEP local replica placement",
                "MoonEP final-layer load uses server-local master/replica placement",
            ),
        }
        for algorithm, markers in expectations.items():
            algorithm_provider = create_gate_provider(
                "raw_receive_cdf",
                seed=17,
                raw_placement_json=RAW_PLACEMENT,
                layer_map=(0, 1),
            )
            algorithm_built = build_transformer_workload(
                TransformerWorkloadConfig(
                    model=model,
                    placement=placement,
                    tokens_per_rank=2,
                    algorithm=algorithm,
                    chunk_tokens=32,
                    gate_provider=algorithm_provider,
                    replicas_per_rank=(10 if algorithm == "moonep" else 0),
                )
            )
            algorithm_workload = run_dir / f"{algorithm}_workload"
            emit_workload(
                algorithm_built.graph,
                algorithm_workload,
                metadata=algorithm_built.metadata,
            )
            algorithm_output = run_dir / f"{algorithm}_gate_load"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(VISUALIZER),
                    "--workload-dir",
                    str(algorithm_workload),
                    "--output-dir",
                    str(algorithm_output),
                    "--title",
                    f"{algorithm.upper()} Gate visualization test",
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60,
                check=False,
            )
            require(completed.returncode == 0, f"{algorithm} visualizer failed")
            algorithm_html = (
                algorithm_output / "gate_load_profile.html"
            ).read_text(encoding="utf-8")
            require(
                f"const algorithm = '{algorithm}'" in algorithm_html
                and all(marker in algorithm_html for marker in markers),
                f"{algorithm} HTML does not use its own placement semantics",
            )
            require(
                '<h2>Cross-server expert copies</h2>' not in algorithm_html
                and '<i style="background:#1098a3"></i>Expert Weight'
                not in algorithm_html,
                f"{algorithm} HTML exposed ProbeEP-only modules",
            )
        results.append(
            Result(
                "algorithm_specific_html",
                "passed",
                "NCCL/DeepEP 只展示原始执行负载；MoonEP 展示服务器内 replica placement；均无 ProbeEP 专属列",
            )
        )
    except Exception as exc:
        results.append(Result("algorithm_specific_html", "failed", str(exc)))

    try:
        probe_provider = create_gate_provider(
            "raw_receive_cdf",
            seed=17,
            raw_placement_json=RAW_PLACEMENT,
            layer_map=(0, 1),
        )
        probe_built = build_transformer_workload(
            TransformerWorkloadConfig(
                model=model,
                placement=placement,
                tokens_per_rank=2,
                algorithm="probeep",
                chunk_tokens=32,
                gate_provider=probe_provider,
            ),
            cost_model=SlowVisualizationCostModel(),
        )
        probe_workload = run_dir / "probeep_workload"
        emit_workload(
            probe_built.graph, probe_workload, metadata=probe_built.metadata
        )
        probe_output = run_dir / "probeep_gate_load"
        command = [
            sys.executable,
            str(VISUALIZER),
            "--workload-dir",
            str(probe_workload),
            "--output-dir",
            str(probe_output),
            "--title",
            "ProbeEP directed transport visualization test",
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
        (run_dir / "ProbeEP可视化.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        require(
            completed.returncode == 0,
            f"ProbeEP visualizer returned {completed.returncode}",
        )
        html = (probe_output / "gate_load_profile.html").read_text(
            encoding="utf-8"
        )
        summary = json.loads(
            (probe_output / "gate_load_profile_summary.json").read_text(
                encoding="utf-8"
            )
        )
        require(
            all(
                marker in html
                for marker in (
                    "Cross-server expert copies",
                    "Directed server-pair load",
                    "Per-NIC directed load",
                    "Directed server-pair data table",
                    "Per-NIC directed data table",
                    "directedLoadChart",
                    "Final-layer ProbeEP admitted placement",
                    "Final migration P/A/D",
                    "Final-layer server padded routes",
                    "Remote expert copies",
                    "Source TX total",
                    "Expert IDs",
                )
            ),
            "ProbeEP transport HTML modules are incomplete",
        )
        total_remote_copies = 0
        source_records = sorted(
            probe_built.metadata["micro_batch_algorithms"],
            key=lambda item: (item["layer"], item["micro_batch"]),
        )
        for source, rendered in zip(source_records, summary["records"]):
            transport = rendered["transport"]
            planning = rendered["planning"]
            remote_replicas = source["remote_replicas"]
            total_remote_copies += len(remote_replicas)
            require(
                planning["planned_intent_count"]
                == len(source["planned_migration_intents"])
                and planning["admitted_intent_count"]
                == len(source["admitted_migration_intents"])
                and planning["deferred_intent_count"]
                == len(source["deferred_migration_intents"]),
                "rendered planning/admission counts do not match manifest",
            )
            require(
                transport["remote_replica_count"] == len(remote_replicas),
                "remote expert copies do not match manifest replicas",
            )
            require(
                transport["expert_weight_bytes"]
                == sum(int(item["weight_bytes"]) for item in remote_replicas),
                "remote expert bytes do not match manifest replicas",
            )
            pair_rows = transport["server_pairs"]
            transfer = source["hierarchical_transfer"]["bytes_by_leg"]
            require(
                sum(int(item["dispatch_bytes"]) for item in pair_rows)
                == int(transfer.get("dispatch_fabric", 0)),
                "directed pair Dispatch bytes are not conserved",
            )
            require(
                sum(int(item["combine_bytes"]) for item in pair_rows)
                == int(transfer.get("combine_fabric", 0)),
                "directed pair Combine bytes are not conserved",
            )
            for pair in pair_rows:
                nics = pair["nics"]
                require(
                    sum(int(item["dispatch_bytes"]) for item in nics)
                    == int(pair["dispatch_bytes"])
                    and sum(int(item["combine_bytes"]) for item in nics)
                    == int(pair["combine_bytes"])
                    and sum(int(item["expert_weight_bytes"]) for item in nics)
                    == int(pair["expert_weight_bytes"]),
                    "per-NIC bytes do not sum to their directed server pair",
                )
        require(total_remote_copies > 0, "ProbeEP transport test has no remote copies")
        results.append(
            Result(
                "probeep_directed_transport_html",
                "passed",
                f"{total_remote_copies} 个 remote copies；server-pair 与 per-NIC Dispatch/Combine/Weight bytes 守恒",
            )
        )
    except Exception as exc:
        results.append(
            Result("probeep_directed_transport_html", "failed", str(exc))
        )

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
