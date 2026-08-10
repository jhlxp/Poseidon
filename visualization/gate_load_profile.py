#!/usr/bin/env python3
"""Render first-layer before vs final-layer after load for each microbatch."""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Gate / expert load profile")
    return parser.parse_args()


def build_transport_profile(invocation: dict[str, object]) -> dict[str, object]:
    rows: dict[tuple[int, int], dict[str, object]] = {}

    def ensure_row(source_server: int, destination_server: int) -> dict[str, object]:
        key = (source_server, destination_server)
        if key not in rows:
            rows[key] = {
                "source_server": source_server,
                "destination_server": destination_server,
                "dispatch_token_payloads": 0,
                "dispatch_expert_routes": 0,
                "dispatch_bytes": 0,
                "combine_token_payloads": 0,
                "combine_expert_routes": 0,
                "combine_bytes": 0,
                "token_bytes_total": 0,
                "expert_replica_count": 0,
                "expert_ids": set(),
                "expert_weight_bytes": 0,
                "moved_expert_routes": 0,
                "experts": [],
                "nics": {},
            }
        return rows[key]

    def ensure_nic(
        row: dict[str, object],
        rail: int,
        source_rank: int,
        destination_rank: int,
    ) -> dict[str, object]:
        nics = row["nics"]
        if rail not in nics:
            nics[rail] = {
                "rail": rail,
                "source_rank": source_rank,
                "destination_rank": destination_rank,
                "dispatch_token_payloads": 0,
                "dispatch_expert_routes": 0,
                "dispatch_bytes": 0,
                "combine_token_payloads": 0,
                "combine_expert_routes": 0,
                "combine_bytes": 0,
                "token_bytes_total": 0,
                "expert_weight_chunks": 0,
                "expert_weight_bytes": 0,
                "expert_ids": set(),
            }
        return nics[rail]

    token_rows = invocation.get("token_server_pair_transport", [])
    if not isinstance(token_rows, list):
        raise ValueError("token_server_pair_transport must be a list")
    for source in token_rows:
        if not isinstance(source, dict):
            raise ValueError("invalid token server-pair transport row")
        row = ensure_row(
            int(source["source_server"]), int(source["destination_server"])
        )
        for field in (
            "dispatch_token_payloads",
            "dispatch_expert_routes",
            "dispatch_bytes",
            "combine_token_payloads",
            "combine_expert_routes",
            "combine_bytes",
            "token_bytes_total",
        ):
            row[field] += int(source.get(field, 0))
        for source_nic in source.get("nics", []):
            nic = ensure_nic(
                row,
                int(source_nic["rail"]),
                int(source_nic["source_rank"]),
                int(source_nic["destination_rank"]),
            )
            for field in (
                "dispatch_token_payloads",
                "dispatch_expert_routes",
                "dispatch_bytes",
                "combine_token_payloads",
                "combine_expert_routes",
                "combine_bytes",
                "token_bytes_total",
            ):
                nic[field] += int(source_nic.get(field, 0))

    replicas = invocation.get("remote_replicas", [])
    if not isinstance(replicas, list):
        raise ValueError("remote_replicas must be a list")
    for replica in replicas:
        if not isinstance(replica, dict):
            raise ValueError("invalid remote replica row")
        row = ensure_row(
            int(replica["source_server"]),
            int(replica["destination_server"]),
        )
        expert_id = int(replica["expert_id"])
        weight_bytes = int(replica["weight_bytes"])
        row["expert_replica_count"] += 1
        row["expert_ids"].add(expert_id)
        row["expert_weight_bytes"] += weight_bytes
        row["moved_expert_routes"] += int(replica["moved_route_count"])
        row["experts"].append(
            {
                "expert_id": expert_id,
                "source_server": int(replica["source_server"]),
                "destination_server": int(replica["destination_server"]),
                "source_rank": int(replica["home_rank"]),
                "destination_rank": int(replica["destination_rank"]),
                "moved_route_count": int(replica["moved_route_count"]),
                "weight_bytes": weight_bytes,
            }
        )
        for chunk in replica.get("weight_chunks", []):
            nic = ensure_nic(
                row,
                int(chunk["rail"]),
                int(chunk["source_relay"]),
                int(chunk["destination_relay"]),
            )
            nic["expert_weight_chunks"] += 1
            nic["expert_weight_bytes"] += int(chunk["transfer_bytes"])
            nic["expert_ids"].add(expert_id)

    pair_rows: list[dict[str, object]] = []
    all_expert_ids: set[int] = set()
    for row in rows.values():
        row["expert_ids"] = sorted(row["expert_ids"])
        row["distinct_expert_count"] = len(row["expert_ids"])
        all_expert_ids.update(row["expert_ids"])
        row["experts"] = sorted(
            row["experts"], key=lambda item: item["expert_id"]
        )
        nic_rows = []
        for nic in row.pop("nics").values():
            nic["expert_ids"] = sorted(nic["expert_ids"])
            nic["distinct_expert_count"] = len(nic["expert_ids"])
            nic["total_bytes"] = (
                nic["token_bytes_total"] + nic["expert_weight_bytes"]
            )
            nic_rows.append(nic)
        row["nics"] = sorted(nic_rows, key=lambda item: item["rail"])
        row["total_bytes"] = (
            row["token_bytes_total"] + row["expert_weight_bytes"]
        )
        pair_rows.append(row)
    pair_rows.sort(
        key=lambda item: (item["source_server"], item["destination_server"])
    )
    return {
        "remote_replica_count": sum(
            int(row["expert_replica_count"]) for row in pair_rows
        ),
        "distinct_expert_count": len(all_expert_ids),
        "expert_ids": sorted(all_expert_ids),
        "expert_weight_bytes": sum(
            int(row["expert_weight_bytes"]) for row in pair_rows
        ),
        "expert_server_pair_count": sum(
            bool(row["expert_replica_count"]) for row in pair_rows
        ),
        "server_pairs": pair_rows,
    }


def load_records(workload_dir: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest_path = workload_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing workload manifest: {manifest_path}")
    root = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = root.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("manifest metadata is missing")
    invocations = metadata.get("micro_batch_algorithms")
    if not isinstance(invocations, list) or not invocations:
        raise ValueError("manifest has no micro_batch_algorithms")

    records: list[dict[str, object]] = []
    for invocation in invocations:
        if not isinstance(invocation, dict):
            raise ValueError("invalid micro_batch_algorithms record")
        gate = invocation.get("gate")
        profile = invocation.get("expert_load_profile")
        if not isinstance(gate, dict):
            raise ValueError("invocation is missing gate metadata")
        if not isinstance(profile, dict) or profile.get("schema") != "expert_load_profile_v1":
            raise ValueError("invocation is missing expert_load_profile_v1")
        records.append(
            {
                "layer": int(invocation["layer"]),
                "micro_batch": int(invocation["micro_batch"]),
                "gate": gate,
                "profile": profile,
                "planning": {
                    "planned_intent_count": len(
                        invocation.get("planned_migration_intents", [])
                    ),
                    "admitted_intent_count": len(
                        invocation.get("admitted_migration_intents", [])
                    ),
                    "deferred_intent_count": len(
                        invocation.get("deferred_migration_intents", [])
                    ),
                    "baseline_server_padded_routes": invocation.get(
                        "baseline_server_padded_routes", {}
                    ),
                    "planned_server_padded_routes": invocation.get(
                        "planned_server_padded_routes", {}
                    ),
                    "admitted_server_padded_routes": invocation.get(
                        "admitted_server_padded_routes", {}
                    ),
                },
                "transport": build_transport_profile(invocation),
            }
        )
    records.sort(key=lambda item: (item["layer"], item["micro_batch"]))
    return metadata, records


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    columns = (
        "layer",
        "micro_batch",
        "gate_provider",
        "state",
        "instance_id",
        "kind",
        "logical_expert",
        "physical_expert",
        "replica_index",
        "rank",
        "load",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            profile = record["profile"]
            gate = record["gate"]
            for state in ("before", "after"):
                for instance in profile[state]["instances"]:
                    writer.writerow(
                        {
                            "layer": record["layer"],
                            "micro_batch": record["micro_batch"],
                            "gate_provider": gate["name"],
                            "state": state,
                            "instance_id": instance["instance_id"],
                            "kind": instance["kind"],
                            "logical_expert": instance["logical_expert"],
                            "physical_expert": instance["physical_expert"],
                            "replica_index": instance["replica_index"],
                            "rank": instance["rank"],
                            "load": instance["load"],
                        }
                    )


def write_html(
    path: Path,
    title: str,
    metadata: dict[str, object],
    records: list[dict[str, object]],
) -> None:
    payload = json.dumps(records, separators=(",", ":"), ensure_ascii=True).replace(
        "</", "<\\/"
    )
    model = metadata.get("model", {})
    placement = metadata.get("placement", {})
    gpus_per_server = placement.get("gpus_per_server")
    if (
        isinstance(gpus_per_server, bool)
        or not isinstance(gpus_per_server, int)
        or gpus_per_server <= 0
    ):
        raise ValueError("placement.gpus_per_server must be a positive integer")
    algorithm = str(metadata.get("algorithm", "unknown")).lower()
    if algorithm not in {"nccl", "deepep", "eplb", "moonep", "probeep"}:
        raise ValueError(f"unsupported algorithm for Gate visualization: {algorithm}")
    algorithm_label = {
        "nccl": "NCCL",
        "deepep": "DeepEP",
        "eplb": "EPLB",
        "moonep": "MoonEP",
        "probeep": "ProbeEP",
    }[algorithm]
    placement_label = {
        "eplb": "EPLB physical placement",
        "moonep": "MoonEP local replica placement",
        "probeep": "ProbeEP admitted placement",
    }.get(algorithm, "original placement")
    has_placement = algorithm in {"eplb", "moonep", "probeep"}
    if has_placement:
        load_title = "MoE expert-token load: baseline / placement"
        first_load_title = "Baseline: first-layer raw distribution"
        final_load_title = f"Final-layer {placement_label}"
        first_state_title = "Baseline"
        final_state_title = f"{algorithm_label} placement"
    else:
        load_title = "MoE expert-token execution load by layer"
        first_load_title = "First-layer execution load"
        final_load_title = "Final-layer execution load"
        first_state_title = "First layer"
        final_state_title = "Final layer"
    subtitle = (
        f"{algorithm.upper()} | "
        f"{placement.get('num_ranks', '?')} ranks | "
        f"{gpus_per_server} ranks/server | "
        f"{model.get('num_experts', '?')} logical experts"
    )
    migration_panel = ""
    expert_weight_legend = ""
    if algorithm == "probeep":
        migration_panel = """<section class="panel"><h2>Cross-server expert copies</h2><div class="copy-summary" id="copySummary"></div><div class="migration-note" id="copyExperts"></div></section>"""
        expert_weight_legend = """<span><i style="background:#1098a3"></i>Expert Weight</span>"""
    document = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { color-scheme: light; font-family: Inter, Arial, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; color: #17212b; background: #f4f6f8; font-size: 13px; }
header { padding: 16px 20px 12px; background: #fff; border-bottom: 1px solid #cbd3dc; }
h1 { margin: 0 0 5px; font-size: 20px; letter-spacing: 0; }
.subtitle { color: #5a6877; }
main { padding: 14px; display: grid; gap: 12px; min-width: 0; }
.toolbar, .metrics, .panel { background: #fff; border: 1px solid #c8d0da; min-width: 0; width: 100%; }
.toolbar { padding: 10px 12px; display: flex; flex-wrap: wrap; align-items: end; gap: 12px; }
label { display: grid; gap: 4px; color: #596777; font-size: 11px; }
select { height: 31px; min-width: 125px; border: 1px solid #abb6c2; background: #fff; padding: 0 8px; }
.provider { margin-left: auto; color: #354454; align-self: center; min-width: 0; overflow-wrap: anywhere; }
.metrics { display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); }
.metric { padding: 11px 13px; border-right: 1px solid #dde2e8; }
.metric:last-child { border-right: 0; }
.metric span { display: block; color: #667483; font-size: 11px; margin-bottom: 4px; }
.metric strong { font-size: 18px; font-variant-numeric: tabular-nums; }
.copy-summary { display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); border-top: 1px solid #dde2e8; }
.copy-summary .metric { background: #fff; }
.migration-note { padding: 9px 12px; color: #526171; border-top: 1px solid #e1e5ea; }
.placement-note { padding: 0 12px 11px; color: #526171; overflow-wrap: anywhere; }
.load-definition { padding: 10px 12px; color: #354454; border-bottom: 1px solid #e1e5ea; font-variant-numeric: tabular-nums; }
.expert-ids { color: #1e5f74; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; }
.panel > h2, details > summary { margin: 0; padding: 10px 12px; font-size: 14px; font-weight: 650; }
.scope-title { color: #5d6976; font-size: 12px; font-weight: 500; }
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 0 12px 12px; }
.chart { min-width: 0; border: 1px solid #d4dae1; overflow: auto; }
.chart h3 { position: sticky; left: 0; margin: 0; padding: 8px 10px; font-size: 13px; border-bottom: 1px solid #e0e5ea; background: #fafbfc; }
.chart svg { display: block; min-width: 620px; background: #fff; }
.logical { padding: 0 12px 12px; overflow-x: auto; }
.logical svg { display: block; min-width: 900px; width: 100%; height: 180px; border: 1px solid #d4dae1; }
.transport-chart { padding: 0 12px 12px; overflow-x: auto; }
.transport-chart svg { display: block; min-width: 900px; width: 100%; background: #fff; border: 1px solid #d4dae1; }
.transport-legend { display: flex; flex-wrap: wrap; gap: 14px; padding: 0 12px 9px; color: #526171; }
.transport-legend span { display: inline-flex; align-items: center; gap: 5px; }
.transport-legend i { width: 12px; height: 12px; display: inline-block; }
.transport-details { border-top: 1px solid #d7dde4; }
.transport-details > summary { cursor: pointer; padding: 8px 12px; color: #526171; }
.axis { fill: #5d6976; font-size: 10px; }
.grid { stroke: #e3e7eb; stroke-width: 1; }
.mean-line { stroke: #17212b; stroke-width: 1; stroke-dasharray: 4 3; }
.mean-label { fill: #17212b; font-size: 10px; font-weight: 650; }
.rank-label { fill: #42505e; font-size: 10px; }
details.panel > summary { cursor: pointer; list-style: revert; }
.details-controls { padding: 0 12px 10px; display: flex; gap: 12px; align-items: end; }
.table-wrap { width: 100%; max-width: 100%; max-height: 500px; overflow: auto; border-top: 1px solid #d7dde4; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th { position: sticky; top: 0; background: #f3f5f7; z-index: 1; }
th, td { padding: 7px 9px; text-align: right; border-bottom: 1px solid #e1e5ea; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
.transport-table th, .transport-table td { white-space: nowrap; }
.transport-table .ids { min-width: 160px; max-width: 320px; white-space: normal; text-align: left; }
.empty { padding: 20px; color: #687583; }
@media (max-width: 900px) { .charts { grid-template-columns: 1fr; } .metrics, .copy-summary { grid-template-columns: 1fr; } .metric { border-right: 0; border-bottom: 1px solid #dde2e8; } .metric:last-child { border-bottom: 0; } .provider { margin-left: 0; width: 100%; } }
</style>
</head>
<body>
<header><h1>__TITLE__</h1><div class="subtitle">__SUBTITLE__</div></header>
<main>
<section class="toolbar">
  <label>Microbatch<select id="microbatch"></select></label>
  <div class="provider" id="provider"></div>
</section>
<section class="metrics" id="metrics"></section>
__MIGRATION_PANEL__
<details class="panel" open><summary>Directed server-pair load</summary><div class="transport-legend"><span><i style="background:#e28200"></i>Dispatch</span><span><i style="background:#ca4057"></i>Combine</span>__EXPERT_WEIGHT_LEGEND__</div><div class="transport-chart" id="pairLoadChart"></div><details class="transport-details"><summary>Directed server-pair data table</summary><div class="table-wrap" id="pairLoadTable"></div></details></details>
<details class="panel" open><summary>Per-NIC directed load</summary><div class="transport-legend"><span><i style="background:#e28200"></i>Dispatch</span><span><i style="background:#ca4057"></i>Combine</span>__EXPERT_WEIGHT_LEGEND__</div><div class="transport-chart" id="nicLoadChart"></div><details class="transport-details"><summary>Per-NIC directed data table</summary><div class="table-wrap" id="nicLoadTable"></div></details></details>
<section class="panel"><h2>MoE expert-token load definition <span class="scope-title" data-comparison-scope></span></h2><div class="load-definition" id="loadDefinition"></div></section>
<section class="panel"><h2>Server __LOAD_TITLE__ <span class="scope-title" data-comparison-scope></span></h2><div class="charts"><div class="chart"><h3>__FIRST_LOAD_TITLE__ <span class="scope-title" data-before-scope></span></h3><div id="serverBefore"></div></div><div class="chart"><h3>__FINAL_LOAD_TITLE__ <span class="scope-title" data-after-scope></span></h3><div id="serverAfter"></div></div></div></section>
<section class="panel"><h2>Rank __LOAD_TITLE__ <span class="scope-title" data-comparison-scope></span></h2><div class="placement-note" id="placementNote"></div><div class="charts"><div class="chart"><h3>__FIRST_LOAD_TITLE__ <span class="scope-title" data-before-scope></span></h3><div id="before"></div></div><div class="chart"><h3>__FINAL_LOAD_TITLE__ <span class="scope-title" data-after-scope></span></h3><div id="after"></div></div></div></section>
<section class="panel"><h2>Gate logical-expert distribution <span class="scope-title" data-comparison-scope></span></h2><div class="charts"><div class="chart"><h3>First-layer Gate <span class="scope-title" data-before-scope></span></h3><div class="logical" id="logicalBefore"></div></div><div class="chart"><h3>Final-layer Gate <span class="scope-title" data-after-scope></span></h3><div class="logical" id="logicalAfter"></div></div></div></section>
<details class="panel"><summary>Expert instance details</summary><div class="details-controls"><label>State<select id="state"><option value="before">__FIRST_STATE_TITLE__</option><option value="after">__FINAL_STATE_TITLE__</option></select></label><label>Rank<select id="rank"><option value="all">All ranks</option></select></label></div><div class="table-wrap" id="table"></div></details>
</main>
<script>
const records = __PAYLOAD__;
const mbSelect = document.getElementById('microbatch');
const rankSelect = document.getElementById('rank');
const stateSelect = document.getElementById('state');
const NS = 'http://www.w3.org/2000/svg';
const gpusPerServer = __GPUS_PER_SERVER__;
const algorithm = '__ALGORITHM__';
const algorithmLabel = '__ALGORITHM_LABEL__';
const hasPlacement = __HAS_PLACEMENT__;
const hasRemoteMigration = __HAS_REMOTE_MIGRATION__;
const placementLabel = '__PLACEMENT_LABEL__';
const colors = ['#3574b9','#e28200','#4e946c','#ca4057','#8167a9','#1098a3','#bc7438','#607d3b','#ad4d91','#63788c','#d0a21d','#3f8c88'];
const microbatches = [...new Set(records.map(r => r.micro_batch))].sort((a,b)=>a-b);
for (const value of microbatches) mbSelect.add(new Option(`MB ${value}`, value));

function currentComparison() {
  const selected=records.filter(r=>r.micro_batch===Number(mbSelect.value)).sort((a,b)=>a.layer-b.layer);
  if(!selected.length)return null;
  return {beforeRecord:selected[0],afterRecord:selected[selected.length-1]};
}
function layerLabel(r) { return `Layer ${r.layer + 1} (id ${r.layer}) / MB ${r.micro_batch}`; }
function comparisonLabel(c) { return hasPlacement ? `· MB ${c.beforeRecord.micro_batch}: Layer ${c.beforeRecord.layer + 1} baseline -> Layer ${c.afterRecord.layer + 1} ${placementLabel}` : `· MB ${c.beforeRecord.micro_batch}: Layer ${c.beforeRecord.layer + 1} execution -> Layer ${c.afterRecord.layer + 1} execution`; }
function fmt(value, digits=3) { return Number(value).toLocaleString(undefined, {maximumFractionDigits: digits}); }
function fmtBytes(value) { const n=Number(value); if(n>=1024**3)return `${fmt(n/1024**3,2)} GiB`; if(n>=1024**2)return `${fmt(n/1024**2,2)} MiB`; if(n>=1024)return `${fmt(n/1024,2)} KiB`; return `${fmt(n,0)} B`; }
function ids(values) { return values && values.length ? values.map(value=>`E${value}`).join(', ') : '-'; }
function serverVector(value) { return Object.entries(value||{}).sort((a,b)=>Number(a[0])-Number(b[0])).map(([server,load])=>`S${server}:${fmt(load,0)}`).join(' / '); }
function svgElement(tag, attrs={}) { const node=document.createElementNS(NS,tag); for (const [k,v] of Object.entries(attrs)) node.setAttribute(k,String(v)); return node; }
function rankChart(snapshot, commonMax) {
  const byRank = new Map();
  snapshot.instances.forEach(item => { if (!byRank.has(item.rank)) byRank.set(item.rank, []); if (item.load > 0) byRank.get(item.rank).push(item); });
  const rankCount = snapshot.rank_loads.length, rowH=21, left=56, right=18, top=22, width=700, inner=width-left-right, height=top+rankCount*rowH+24;
  const svg=svgElement('svg',{viewBox:`0 0 ${width} ${height}`,width:'100%',height});
  for (let i=0;i<=4;i++){ const x=left+inner*i/4; svg.append(svgElement('line',{x1:x,y1:top-8,x2:x,y2:height-20,class:'grid'})); const t=svgElement('text',{x,y:12,'text-anchor':'middle',class:'axis'}); t.textContent=fmt(commonMax*i/4,1); svg.append(t); }
  const mean=snapshot.rank_loads.reduce((sum,value)=>sum+Number(value),0)/rankCount,meanX=left+mean/commonMax*inner;svg.append(svgElement('line',{x1:meanX,y1:top-8,x2:meanX,y2:height-20,class:'mean-line'}));const meanLabel=svgElement('text',{x:meanX+3,y:12,class:'mean-label'});meanLabel.textContent=`mean ${fmt(mean,1)}`;svg.append(meanLabel);
  for (let rank=0;rank<rankCount;rank++) { const y=top+rank*rowH; const label=svgElement('text',{x:left-8,y:y+13,'text-anchor':'end',class:'rank-label'}); label.textContent=`R${rank}`; svg.append(label); let offset=0; const sorted=(byRank.get(rank)||[]).sort((a,b)=>b.load-a.load||a.logical_expert-b.logical_expert); for(const item of sorted){ const w=commonMax ? item.load/commonMax*inner : 0; const rect=svgElement('rect',{x:left+offset,y:y+3,width:Math.max(0,w),height:14,fill:colors[item.logical_expert%colors.length]}); const title=svgElement('title'); title.textContent=`rank ${rank} | logical E${item.logical_expert} | ${item.instance_id} | load ${item.load}`; rect.append(title); svg.append(rect); offset+=w; } const total=svgElement('text',{x:Math.min(left+offset+4,width-2),y:y+13,class:'axis'}); total.textContent=fmt(snapshot.rank_loads[rank],0); svg.append(total); }
  return svg;
}
function serverChart(snapshot, commonMax) {
  const serverCount=Math.ceil(snapshot.rank_loads.length/gpusPerServer), loads=Array(serverCount).fill(0), grouped=new Map();
  snapshot.rank_loads.forEach((load,rank)=>{loads[Math.floor(rank/gpusPerServer)]+=Number(load);});
  snapshot.instances.forEach(item=>{if(item.load<=0)return;const server=Math.floor(item.rank/gpusPerServer),key=`${server}:${item.logical_expert}`;if(!grouped.has(key))grouped.set(key,{server,logical_expert:item.logical_expert,load:0});grouped.get(key).load+=Number(item.load);});
  const byServer=new Map();for(const item of grouped.values()){if(!byServer.has(item.server))byServer.set(item.server,[]);byServer.get(item.server).push(item);}
  const rowH=34,left=56,right=18,top=22,width=700,inner=width-left-right,height=top+serverCount*rowH+24;
  const svg=svgElement('svg',{viewBox:`0 0 ${width} ${height}`,width:'100%',height,role:'img','aria-label':'Server aggregate expert load'});
  for(let i=0;i<=4;i++){const x=left+inner*i/4;svg.append(svgElement('line',{x1:x,y1:top-8,x2:x,y2:height-20,class:'grid'}));const t=svgElement('text',{x,y:12,'text-anchor':'middle',class:'axis'});t.textContent=fmt(commonMax*i/4,1);svg.append(t);}
  const mean=loads.reduce((sum,value)=>sum+value,0)/serverCount,meanX=left+mean/commonMax*inner;svg.append(svgElement('line',{x1:meanX,y1:top-8,x2:meanX,y2:height-20,class:'mean-line'}));const meanLabel=svgElement('text',{x:meanX+3,y:12,class:'mean-label'});meanLabel.textContent=`mean ${fmt(mean,1)}`;svg.append(meanLabel);
  for(let server=0;server<serverCount;server++){const y=top+server*rowH,label=svgElement('text',{x:left-8,y:y+18,'text-anchor':'end',class:'rank-label'});label.textContent=`S${server}`;svg.append(label);let offset=0;const sorted=(byServer.get(server)||[]).sort((a,b)=>b.load-a.load||a.logical_expert-b.logical_expert);for(const item of sorted){const w=commonMax?item.load/commonMax*inner:0;const rect=svgElement('rect',{x:left+offset,y:y+3,width:Math.max(0,w),height:22,fill:colors[item.logical_expert%colors.length]});const title=svgElement('title');title.textContent=`server ${server} | logical E${item.logical_expert} | load ${item.load}`;rect.append(title);svg.append(rect);offset+=w;}const total=svgElement('text',{x:Math.min(left+offset+4,width-2),y:y+18,class:'axis'});total.textContent=fmt(loads[server],0);svg.append(total);}
  return svg;
}
function logicalChart(loads) {
  const width=Math.max(900,loads.length*5), height=180, top=20, bottom=28, max=Math.max(1,...loads); const svg=svgElement('svg',{viewBox:`0 0 ${width} ${height}`,width:'100%',height}); const inner=height-top-bottom, barW=width/loads.length;
  for(let i=0;i<=4;i++){const y=top+inner*(1-i/4);svg.append(svgElement('line',{x1:0,y1:y,x2:width,y2:y,class:'grid'}));const t=svgElement('text',{x:3,y:y-2,class:'axis'});t.textContent=fmt(max*i/4,0);svg.append(t);}
  loads.forEach((load,expert)=>{const h=load/max*inner;const rect=svgElement('rect',{x:expert*barW+0.5,y:top+inner-h,width:Math.max(1,barW-1),height:h,fill:colors[expert%colors.length]});const title=svgElement('title');title.textContent=`logical expert ${expert} | load ${load}`;rect.append(title);svg.append(rect);});
  return svg;
}
function directedLoadChart(rows, labelFor) {
  const components=[['dispatch_bytes','#e28200','Dispatch'],['combine_bytes','#ca4057','Combine']];
  if(hasRemoteMigration)components.push(['expert_weight_bytes','#1098a3','Expert Weight']);
  const rowH=23, left=220, right=88, top=25, width=980, inner=width-left-right, height=top+Math.max(1,rows.length)*rowH+25;
  const max=Math.max(1,...rows.map(row=>Number(row.total_bytes)||0));
  const svg=svgElement('svg',{viewBox:`0 0 ${width} ${height}`,width:'100%',height,role:'img','aria-label':'Directed communication byte load'});
  for(let i=0;i<=4;i++){const x=left+inner*i/4;svg.append(svgElement('line',{x1:x,y1:top-8,x2:x,y2:height-20,class:'grid'}));const t=svgElement('text',{x,y:13,'text-anchor':'middle',class:'axis'});t.textContent=fmtBytes(max*i/4);svg.append(t);}
  rows.forEach((row,index)=>{const y=top+index*rowH;const label=svgElement('text',{x:left-8,y:y+14,'text-anchor':'end',class:'rank-label'});label.textContent=labelFor(row);svg.append(label);let offset=0;for(const [field,color,name] of components){const value=Number(row[field])||0;const w=value/max*inner;if(value>0){const rect=svgElement('rect',{x:left+offset,y:y+3,width:Math.max(1,w),height:15,fill:color});const title=svgElement('title');title.textContent=`${labelFor(row)} | ${name} ${fmtBytes(value)} | total ${fmtBytes(row.total_bytes)}`;rect.append(title);svg.append(rect);}offset+=w;}const total=svgElement('text',{x:Math.min(left+offset+5,width-right+4),y:y+14,class:'axis'});total.textContent=fmtBytes(row.total_bytes);svg.append(total);});
  return svg;
}
function renderTable() {
  const c=currentComparison(), state=stateSelect.value, filter=rankSelect.value; if(!c)return; const snapshot=state==='before'?c.beforeRecord.profile.before:c.afterRecord.profile.after; const rows=snapshot.instances.filter(x=>filter==='all'||x.rank===Number(filter)).sort((a,b)=>b.load-a.load||a.rank-b.rank||a.logical_expert-b.logical_expert);
  let html='<table><thead><tr><th>Instance</th><th>Kind</th><th>Logical expert</th><th>Physical expert</th><th>Replica</th><th>Rank</th><th>Load</th></tr></thead><tbody>';
  for(const x of rows) html+=`<tr><td>${x.instance_id}</td><td>${x.kind}</td><td>${x.logical_expert}</td><td>${x.physical_expert ?? '-'}</td><td>${x.replica_index ?? '-'}</td><td>${x.rank}</td><td>${fmt(x.load,0)}</td></tr>`;
  document.getElementById('table').innerHTML=html+'</tbody></table>';
}
function renderTransport(r) {
  const t=r.transport;
  if(hasRemoteMigration){const copyMetrics=[['Remote expert copies',fmt(t.remote_replica_count,0)],['Distinct experts',fmt(t.distinct_expert_count,0)],['Expert weight',fmtBytes(t.expert_weight_bytes)],['Expert server pairs',fmt(t.expert_server_pair_count,0)],['All directed pairs',fmt(t.server_pairs.length,0)]];document.getElementById('copySummary').innerHTML=copyMetrics.map(([k,v])=>`<div class="metric"><span>${k}</span><strong>${v}</strong></div>`).join('');document.getElementById('copyExperts').innerHTML=`Copied expert IDs: <span class="expert-ids">${ids(t.expert_ids)}</span>`;}
  if(!t.server_pairs.length){document.getElementById('pairLoadChart').innerHTML='<div class="empty">No cross-server fabric traffic in this invocation.</div>';document.getElementById('nicLoadChart').innerHTML='<div class="empty">No active cross-server NICs in this invocation.</div>';document.getElementById('pairLoadTable').replaceChildren();document.getElementById('nicLoadTable').replaceChildren();return;}
  document.getElementById('pairLoadChart').replaceChildren(directedLoadChart(t.server_pairs,x=>`S${x.source_server} -> S${x.destination_server}`));
  const pairExpertHeaders=hasRemoteMigration?'<th>Expert copies</th><th>Moved routes</th><th>Expert data</th><th class="ids">Expert IDs</th>':'';
  let pair=`<table class="transport-table"><thead><tr><th>Direction</th><th>Dispatch tokens</th><th>Dispatch routes</th><th>Dispatch data</th><th>Combine tokens</th><th>Combine routes</th><th>Combine data</th>${pairExpertHeaders}<th>Source TX total</th><th>Destination RX total</th></tr></thead><tbody>`;
  for(const x of t.server_pairs){const expertCells=hasRemoteMigration?`<td>${fmt(x.expert_replica_count,0)}</td><td>${fmt(x.moved_expert_routes,0)}</td><td>${fmtBytes(x.expert_weight_bytes)}</td><td class="ids">${ids(x.expert_ids)}</td>`:'';pair+=`<tr><td>S${x.source_server} -> S${x.destination_server}</td><td>${fmt(x.dispatch_token_payloads,0)}</td><td>${fmt(x.dispatch_expert_routes,0)}</td><td>${fmtBytes(x.dispatch_bytes)}</td><td>${fmt(x.combine_token_payloads,0)}</td><td>${fmt(x.combine_expert_routes,0)}</td><td>${fmtBytes(x.combine_bytes)}</td>${expertCells}<td>${fmtBytes(x.total_bytes)}</td><td>${fmtBytes(x.total_bytes)}</td></tr>`;}
  document.getElementById('pairLoadTable').innerHTML=pair+'</tbody></table>';
  const nics=t.server_pairs.flatMap(pair=>pair.nics.filter(nic=>nic.total_bytes>0).map(nic=>({...nic,source_server:pair.source_server,destination_server:pair.destination_server}))).sort((a,b)=>a.source_server-b.source_server||a.destination_server-b.destination_server||a.rail-b.rail);
  document.getElementById('nicLoadChart').replaceChildren(directedLoadChart(nics,x=>`S${x.source_server}->S${x.destination_server} / NIC${x.rail} / R${x.source_rank}->R${x.destination_rank}`));
  const nicExpertHeaders=hasRemoteMigration?'<th>Expert chunks</th><th>Expert data</th><th class="ids">Expert IDs</th>':'';
  let nic=`<table class="transport-table"><thead><tr><th>Direction</th><th>Rail / NIC</th><th>Source rank</th><th>Destination rank</th><th>Dispatch tokens</th><th>Dispatch routes</th><th>Dispatch data</th><th>Combine tokens</th><th>Combine routes</th><th>Combine data</th>${nicExpertHeaders}<th>NIC TX total</th><th>Peer RX total</th></tr></thead><tbody>`;
  for(const x of nics){const expertCells=hasRemoteMigration?`<td>${fmt(x.expert_weight_chunks,0)}</td><td>${fmtBytes(x.expert_weight_bytes)}</td><td class="ids">${ids(x.expert_ids)}</td>`:'';nic+=`<tr><td>S${x.source_server} -> S${x.destination_server}</td><td>NIC ${x.rail}</td><td>R${x.source_rank}</td><td>R${x.destination_rank}</td><td>${fmt(x.dispatch_token_payloads,0)}</td><td>${fmt(x.dispatch_expert_routes,0)}</td><td>${fmtBytes(x.dispatch_bytes)}</td><td>${fmt(x.combine_token_payloads,0)}</td><td>${fmt(x.combine_expert_routes,0)}</td><td>${fmtBytes(x.combine_bytes)}</td>${expertCells}<td>${fmtBytes(x.total_bytes)}</td><td>${fmtBytes(x.total_bytes)}</td></tr>`;}
  document.getElementById('nicLoadTable').innerHTML=nic+'</tbody></table>';
}
function render() {
  const c=currentComparison(); if(!c)return; const first=c.beforeRecord, final=c.afterRecord, before=first.profile.before, after=final.profile.after; const commonMax=Math.max(1,...before.rank_loads,...after.rank_loads); const serverLoads=snapshot=>Array.from({length:Math.ceil(snapshot.rank_loads.length/gpusPerServer)},(_,server)=>snapshot.rank_loads.slice(server*gpusPerServer,(server+1)*gpusPerServer).reduce((sum,value)=>sum+Number(value),0)); const serverMax=Math.max(1,...serverLoads(before),...serverLoads(after)); const beforeI=before.rank_imbalance.max_mean, afterI=after.rank_imbalance.max_mean;
  document.querySelectorAll('[data-comparison-scope]').forEach(node=>{node.textContent=comparisonLabel(c);});
  document.querySelectorAll('[data-before-scope]').forEach(node=>{node.textContent=`\u00b7 ${layerLabel(first)}`;});
  document.querySelectorAll('[data-after-scope]').forEach(node=>{node.textContent=`\u00b7 ${layerLabel(final)}`;});
  const beforeTotal=before.rank_loads.reduce((sum,value)=>sum+Number(value),0),afterTotal=after.rank_loads.reduce((sum,value)=>sum+Number(value),0),rankMean=afterTotal/after.rank_loads.length,serverCount=Math.ceil(after.rank_loads.length/gpusPerServer),serverMean=afterTotal/serverCount;
  const scope=hasPlacement?`Baseline uses the first layer; ${algorithmLabel} placement uses the final layer.`:`${algorithmLabel} does not remap expert execution; the charts show its first- and final-layer execution loads.`;
  document.getElementById('loadDefinition').textContent=`Load is TopK-expanded expert-token count, not unique input-token count and not execution time. ${scope} Totals: first ${fmt(beforeTotal,0)}, final ${fmt(afterTotal,0)} expert-tokens; final rank mean ${fmt(rankMean,1)}; server load is the sum of ${gpusPerServer} ranks; final server mean ${fmt(serverMean,1)}.`;
  document.getElementById('provider').textContent=`First ${layerLabel(first)}: ${first.gate.name} | Final ${layerLabel(final)}: ${final.gate.name}`;
  const p=final.planning; const admission=`${p.planned_intent_count}/${p.admitted_intent_count}/${p.deferred_intent_count}`;
  const metrics=[['First logical routes',fmt(before.total_routes,0)],['First rank max/mean',fmt(beforeI)],['Final rank max/mean',fmt(afterI)],['First -> final delta',`${afterI-beforeI>=0?'+':''}${fmt(afterI-beforeI)}`]];
  if(algorithm==='eplb'||algorithm==='moonep'){metrics.push(['Final physical instances',fmt(after.instances.length,0)],['Final replicas',fmt(after.instances.filter(item=>String(item.kind).includes('replica')).length,0)]);}
  if(hasRemoteMigration)metrics.push(['Final migration P/A/D',admission]);
  document.getElementById('metrics').innerHTML=metrics.map(([k,v])=>`<div class="metric"><span>${k}</span><strong>${v}</strong></div>`).join('');
  if(hasRemoteMigration)document.getElementById('placementNote').textContent=`Final-layer server padded routes | before ${serverVector(p.baseline_server_padded_routes)} | planned ${serverVector(p.planned_server_padded_routes)} | admitted/after ${serverVector(p.admitted_server_padded_routes)}.`;
  else if(algorithm==='eplb')document.getElementById('placementNote').textContent='EPLB final-layer load uses persistent physical primary/replica placement.';
  else if(algorithm==='moonep')document.getElementById('placementNote').textContent='MoonEP final-layer load uses server-local master/replica placement.';
  else if(algorithm==='deepep')document.getElementById('placementNote').textContent='DeepEP changes token payload aggregation and hierarchical transport, not expert execution placement.';
  else document.getElementById('placementNote').textContent='NCCL uses the original logical expert placement without payload deduplication or expert replication.';
  renderTransport(final);
  document.getElementById('serverBefore').replaceChildren(serverChart(before,serverMax)); document.getElementById('serverAfter').replaceChildren(serverChart(after,serverMax));
  document.getElementById('before').replaceChildren(rankChart(before,commonMax)); document.getElementById('after').replaceChildren(rankChart(after,commonMax)); document.getElementById('logicalBefore').replaceChildren(logicalChart(first.gate.logical_expert_loads)); document.getElementById('logicalAfter').replaceChildren(logicalChart(final.gate.logical_expert_loads));
  const rankCount=Math.max(before.rank_loads.length,after.rank_loads.length), keep=rankSelect.value; rankSelect.replaceChildren(new Option('All ranks','all')); for(let rank=0;rank<rankCount;rank++)rankSelect.add(new Option(`Rank ${rank}`,rank)); if([...rankSelect.options].some(o=>o.value===keep))rankSelect.value=keep; renderTable();
}
mbSelect.addEventListener('change',render); stateSelect.addEventListener('change',renderTable); rankSelect.addEventListener('change',renderTable); render();
</script>
</body>
</html>
"""
    document = document.replace("__TITLE__", escape(title), 2)
    document = document.replace("__SUBTITLE__", escape(subtitle))
    document = document.replace("__PAYLOAD__", payload)
    document = document.replace("__GPUS_PER_SERVER__", str(gpus_per_server))
    document = document.replace("__ALGORITHM__", algorithm)
    document = document.replace("__ALGORITHM_LABEL__", algorithm_label)
    document = document.replace(
        "__HAS_PLACEMENT__", str(has_placement).lower()
    )
    document = document.replace(
        "__HAS_REMOTE_MIGRATION__", str(algorithm == "probeep").lower()
    )
    document = document.replace("__PLACEMENT_LABEL__", placement_label)
    document = document.replace("__LOAD_TITLE__", load_title)
    document = document.replace("__FIRST_LOAD_TITLE__", first_load_title)
    document = document.replace("__FINAL_LOAD_TITLE__", final_load_title)
    document = document.replace("__FIRST_STATE_TITLE__", first_state_title)
    document = document.replace("__FINAL_STATE_TITLE__", final_state_title)
    document = document.replace("__MIGRATION_PANEL__", migration_panel)
    document = document.replace(
        "__EXPERT_WEIGHT_LEGEND__", expert_weight_legend
    )
    path.write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    metadata, records = load_records(args.workload_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "gate_load_profile_summary.json"
    csv_path = args.output_dir / "gate_load_profile.csv"
    html_path = args.output_dir / "gate_load_profile.html"
    summary_path.write_text(
        json.dumps(
            {
                "schema": "gate_load_profile_summary_v2",
                "algorithm": metadata.get("algorithm"),
                "routing_provider": metadata.get("routing_provider"),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, records)
    write_html(html_path, args.title, metadata, records)
    print(f"wrote {html_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
