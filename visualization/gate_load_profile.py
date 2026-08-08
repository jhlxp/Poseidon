#!/usr/bin/env python3
"""Render per-layer Gate and expert-instance load before/after placement."""

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
    subtitle = (
        f"{str(metadata.get('algorithm', 'unknown')).upper()} | "
        f"{placement.get('num_ranks', '?')} ranks | "
        f"{model.get('num_experts', '?')} logical experts"
    )
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
main { padding: 14px; display: grid; gap: 12px; }
.toolbar, .metrics, .panel { background: #fff; border: 1px solid #c8d0da; }
.toolbar { padding: 10px 12px; display: flex; flex-wrap: wrap; align-items: end; gap: 12px; }
label { display: grid; gap: 4px; color: #596777; font-size: 11px; }
select { height: 31px; min-width: 125px; border: 1px solid #abb6c2; background: #fff; padding: 0 8px; }
.provider { margin-left: auto; color: #354454; align-self: center; }
.metrics { display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); }
.metric { padding: 11px 13px; border-right: 1px solid #dde2e8; }
.metric:last-child { border-right: 0; }
.metric span { display: block; color: #667483; font-size: 11px; margin-bottom: 4px; }
.metric strong { font-size: 18px; font-variant-numeric: tabular-nums; }
.panel > h2, details > summary { margin: 0; padding: 10px 12px; font-size: 14px; font-weight: 650; }
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 0 12px 12px; }
.chart { min-width: 0; border: 1px solid #d4dae1; overflow: auto; }
.chart h3 { position: sticky; left: 0; margin: 0; padding: 8px 10px; font-size: 13px; border-bottom: 1px solid #e0e5ea; background: #fafbfc; }
.chart svg { display: block; min-width: 620px; background: #fff; }
.logical { padding: 0 12px 12px; overflow-x: auto; }
.logical svg { display: block; min-width: 900px; width: 100%; height: 180px; border: 1px solid #d4dae1; }
.axis { fill: #5d6976; font-size: 10px; }
.grid { stroke: #e3e7eb; stroke-width: 1; }
.rank-label { fill: #42505e; font-size: 10px; }
details.panel > summary { cursor: pointer; list-style: revert; }
.details-controls { padding: 0 12px 10px; display: flex; gap: 12px; align-items: end; }
.table-wrap { max-height: 500px; overflow: auto; border-top: 1px solid #d7dde4; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th { position: sticky; top: 0; background: #f3f5f7; z-index: 1; }
th, td { padding: 7px 9px; text-align: right; border-bottom: 1px solid #e1e5ea; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
.empty { padding: 20px; color: #687583; }
@media (max-width: 900px) { .charts { grid-template-columns: 1fr; } .metrics { grid-template-columns: 1fr 1fr; } .provider { margin-left: 0; width: 100%; } }
</style>
</head>
<body>
<header><h1>__TITLE__</h1><div class="subtitle">__SUBTITLE__</div></header>
<main>
<section class="toolbar">
  <label>Layer<select id="layer"></select></label>
  <label>Microbatch<select id="microbatch"></select></label>
  <div class="provider" id="provider"></div>
</section>
<section class="metrics" id="metrics"></section>
<section class="panel"><h2>Rank load before / after</h2><div class="charts"><div class="chart"><h3>Before: original logical-expert placement</h3><div id="before"></div></div><div class="chart"><h3>After: algorithm execution placement</h3><div id="after"></div></div></div></section>
<section class="panel"><h2>Gate logical-expert distribution</h2><div class="logical" id="logical"></div></section>
<details class="panel"><summary>Expert instance details</summary><div class="details-controls"><label>State<select id="state"><option value="before">Before</option><option value="after">After</option></select></label><label>Rank<select id="rank"><option value="all">All ranks</option></select></label></div><div class="table-wrap" id="table"></div></details>
</main>
<script>
const records = __PAYLOAD__;
const layerSelect = document.getElementById('layer');
const mbSelect = document.getElementById('microbatch');
const rankSelect = document.getElementById('rank');
const stateSelect = document.getElementById('state');
const NS = 'http://www.w3.org/2000/svg';
const colors = ['#3574b9','#e28200','#4e946c','#ca4057','#8167a9','#1098a3','#bc7438','#607d3b','#ad4d91','#63788c','#d0a21d','#3f8c88'];
const layers = [...new Set(records.map(r => r.layer))];
for (const value of layers) layerSelect.add(new Option(value, value));

function currentRecordsForLayer() { return records.filter(r => r.layer === Number(layerSelect.value)); }
function populateMicrobatches() {
  const keep = mbSelect.value;
  mbSelect.replaceChildren();
  for (const r of currentRecordsForLayer()) mbSelect.add(new Option(r.micro_batch, r.micro_batch));
  if ([...mbSelect.options].some(o => o.value === keep)) mbSelect.value = keep;
}
function current() { return records.find(r => r.layer === Number(layerSelect.value) && r.micro_batch === Number(mbSelect.value)); }
function fmt(value, digits=3) { return Number(value).toLocaleString(undefined, {maximumFractionDigits: digits}); }
function svgElement(tag, attrs={}) { const node=document.createElementNS(NS,tag); for (const [k,v] of Object.entries(attrs)) node.setAttribute(k,String(v)); return node; }
function rankChart(snapshot, commonMax) {
  const byRank = new Map();
  snapshot.instances.forEach(item => { if (!byRank.has(item.rank)) byRank.set(item.rank, []); if (item.load > 0) byRank.get(item.rank).push(item); });
  const rankCount = snapshot.rank_loads.length, rowH=21, left=56, right=18, top=22, width=700, inner=width-left-right, height=top+rankCount*rowH+24;
  const svg=svgElement('svg',{viewBox:`0 0 ${width} ${height}`,width:'100%',height});
  for (let i=0;i<=4;i++){ const x=left+inner*i/4; svg.append(svgElement('line',{x1:x,y1:top-8,x2:x,y2:height-20,class:'grid'})); const t=svgElement('text',{x,y:12,'text-anchor':'middle',class:'axis'}); t.textContent=fmt(commonMax*i/4,1); svg.append(t); }
  for (let rank=0;rank<rankCount;rank++) { const y=top+rank*rowH; const label=svgElement('text',{x:left-8,y:y+13,'text-anchor':'end',class:'rank-label'}); label.textContent=`R${rank}`; svg.append(label); let offset=0; const sorted=(byRank.get(rank)||[]).sort((a,b)=>b.load-a.load||a.logical_expert-b.logical_expert); for(const item of sorted){ const w=commonMax ? item.load/commonMax*inner : 0; const rect=svgElement('rect',{x:left+offset,y:y+3,width:Math.max(0,w),height:14,fill:colors[item.logical_expert%colors.length]}); const title=svgElement('title'); title.textContent=`rank ${rank} | logical E${item.logical_expert} | ${item.instance_id} | load ${item.load}`; rect.append(title); svg.append(rect); offset+=w; } const total=svgElement('text',{x:Math.min(left+offset+4,width-2),y:y+13,class:'axis'}); total.textContent=fmt(snapshot.rank_loads[rank],0); svg.append(total); }
  return svg;
}
function logicalChart(loads) {
  const width=Math.max(900,loads.length*5), height=180, top=20, bottom=28, max=Math.max(1,...loads); const svg=svgElement('svg',{viewBox:`0 0 ${width} ${height}`,width:'100%',height}); const inner=height-top-bottom, barW=width/loads.length;
  for(let i=0;i<=4;i++){const y=top+inner*(1-i/4);svg.append(svgElement('line',{x1:0,y1:y,x2:width,y2:y,class:'grid'}));const t=svgElement('text',{x:3,y:y-2,class:'axis'});t.textContent=fmt(max*i/4,0);svg.append(t);}
  loads.forEach((load,expert)=>{const h=load/max*inner;const rect=svgElement('rect',{x:expert*barW+0.5,y:top+inner-h,width:Math.max(1,barW-1),height:h,fill:colors[expert%colors.length]});const title=svgElement('title');title.textContent=`logical expert ${expert} | load ${load}`;rect.append(title);svg.append(rect);});
  return svg;
}
function renderTable() {
  const r=current(), state=stateSelect.value, filter=rankSelect.value; if(!r)return; const rows=r.profile[state].instances.filter(x=>filter==='all'||x.rank===Number(filter)).sort((a,b)=>b.load-a.load||a.rank-b.rank||a.logical_expert-b.logical_expert);
  let html='<table><thead><tr><th>Instance</th><th>Kind</th><th>Logical expert</th><th>Physical expert</th><th>Replica</th><th>Rank</th><th>Load</th></tr></thead><tbody>';
  for(const x of rows) html+=`<tr><td>${x.instance_id}</td><td>${x.kind}</td><td>${x.logical_expert}</td><td>${x.physical_expert ?? '-'}</td><td>${x.replica_index ?? '-'}</td><td>${x.rank}</td><td>${fmt(x.load,0)}</td></tr>`;
  document.getElementById('table').innerHTML=html+'</tbody></table>';
}
function render() {
  const r=current(); if(!r)return; const before=r.profile.before, after=r.profile.after; const commonMax=Math.max(1,...before.rank_loads,...after.rank_loads); const beforeI=before.rank_imbalance.max_mean, afterI=after.rank_imbalance.max_mean;
  document.getElementById('provider').textContent=`${r.gate.name} | seed ${r.gate.seed} | ${r.gate.routing_fidelity}`;
  const metrics=[['Logical routes',fmt(before.total_routes,0)],['Before rank max/mean',fmt(beforeI)],['After rank max/mean',fmt(afterI)],['Delta',`${afterI-beforeI>=0?'+':''}${fmt(afterI-beforeI)}`],['Gate target TV',r.gate.target_realized_total_variation==null?'-':fmt(r.gate.target_realized_total_variation,4)]];
  document.getElementById('metrics').innerHTML=metrics.map(([k,v])=>`<div class="metric"><span>${k}</span><strong>${v}</strong></div>`).join('');
  document.getElementById('before').replaceChildren(rankChart(before,commonMax)); document.getElementById('after').replaceChildren(rankChart(after,commonMax)); document.getElementById('logical').replaceChildren(logicalChart(r.gate.logical_expert_loads));
  const rankCount=before.rank_loads.length, keep=rankSelect.value; rankSelect.replaceChildren(new Option('All ranks','all')); for(let rank=0;rank<rankCount;rank++)rankSelect.add(new Option(`Rank ${rank}`,rank)); if([...rankSelect.options].some(o=>o.value===keep))rankSelect.value=keep; renderTable();
}
layerSelect.addEventListener('change',()=>{populateMicrobatches();render();}); mbSelect.addEventListener('change',render); stateSelect.addEventListener('change',renderTable); rankSelect.addEventListener('change',renderTable); populateMicrobatches(); render();
</script>
</body>
</html>
"""
    document = document.replace("__TITLE__", escape(title), 2)
    document = document.replace("__SUBTITLE__", escape(subtitle))
    document = document.replace("__PAYLOAD__", payload)
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
                "schema": "gate_load_profile_summary_v1",
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
