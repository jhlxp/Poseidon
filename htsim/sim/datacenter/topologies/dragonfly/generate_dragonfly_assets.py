#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def get_group_switch(src_group: int, dst_group: int, a: int, h: int, no_groups: int) -> int:
    if src_group < dst_group:
        right_steps = dst_group - src_group
    else:
        right_steps = (no_groups + dst_group) - src_group
    return (src_group * a) + (right_steps - 1) // h


def get_target_switch(src_switch: int, global_link: int, a: int, h: int, no_groups: int) -> int:
    src_group = src_switch // a
    src_switch_index = src_switch - (src_group * a)
    right_steps = (src_switch_index * h) + global_link + 1
    dst_group = (src_group + right_steps) % no_groups
    return (dst_group * a) + (a - 1) - src_switch_index


def get_src_pairs(src_switch: int,
                  dst_switch: int,
                  a: int,
                  h: int,
                  no_groups: int) -> List[Tuple[int, int]]:
    src_group = src_switch // a
    dst_group = dst_switch // a

    group_start = src_group * a
    group_switches = list(range(group_start, group_start + a))

    pairs: List[Tuple[int, int]] = []

    if src_group == dst_group:
        for hop_one in group_switches:
            if hop_one != src_switch:
                pairs.append((hop_one, 0))
        return pairs

    for link in range(h):
        hop_one = get_target_switch(src_switch, link, a, h, no_groups)
        pairs.append((hop_one, 0))

    for hop_one in group_switches:
        if hop_one == src_switch:
            continue
        for link in range(h):
            hop_two = get_target_switch(hop_one, link, a, h, no_groups)
            pairs.append((hop_one, hop_two))

    deduped: List[Tuple[int, int]] = []
    seen = set()
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            deduped.append(pair)
    return deduped


def write_dragonfly_topo(path: Path,
                         p: int,
                         a: int,
                         h: int,
                         switch_latency_ns: int,
                         link_speed_gbps: int,
                         link_latency_global_ns: int,
                         link_latency_local_ns: int,
                         link_latency_host_ns: int) -> None:
    content = (
        "# Dragonfly\n"
        f"p {p}\n"
        f"a {a}\n"
        f"h {h}\n\n"
        f"Switch_latency_ns {switch_latency_ns}\n\n"
        "# Default - link speed\n"
        f"Link_speed_global_Gbps {link_speed_gbps}\n"
        f"Link_speed_local_Gbps {link_speed_gbps}\n"
        f"Link_speed_host_Gbps {link_speed_gbps}\n\n"
        "# Default - link latency\n"
        f"Link_latency_global_ns {link_latency_global_ns}\n"
        f"Link_latency_local_ns {link_latency_local_ns}\n"
        f"Link_latency_host_ns {link_latency_host_ns}\n"
    )
    (path / "dragonfly.topo").write_text(content)


def write_dragonfly_adjlist(path: Path, a: int, h: int, no_groups: int, no_switches: int) -> None:
    lines: List[str] = []
    for src in range(no_switches):
        src_group = src // a
        group_start = src_group * a
        locals_ = [sw for sw in range(group_start, group_start + a) if sw != src]
        globals_ = [get_target_switch(src, link, a, h, no_groups) for link in range(h)]
        neighbors = locals_ + globals_
        lines.append(" ".join([str(src)] + [str(x) for x in neighbors]))
    (path / "dragonfly.adjlist").write_text("\n".join(lines) + "\n")


def write_host_tables(path: Path, p: int, a: int, h: int, no_groups: int, no_switches: int) -> None:
    host_table_dir = path / "host_table"
    host_table_dir.mkdir(parents=True, exist_ok=True)

    for src in range(no_switches):
        out_lines: List[str] = []
        for dst in range(no_switches):
            if src == dst:
                continue

            out_lines.append(str(dst))
            pairs = get_src_pairs(src, dst, a, h, no_groups)

            direct = [(h1, h2) for (h1, h2) in pairs if h2 == 0]
            two_hop = [(h1, h2) for (h1, h2) in pairs if h2 != 0]

            if direct:
                payload = " ".join(f"{h1} {h2}" for (h1, h2) in direct)
                out_lines.append(f"1 {len(direct)} {payload}")
            if two_hop:
                payload = " ".join(f"{h1} {h2}" for (h1, h2) in two_hop)
                out_lines.append(f"2 {len(two_hop)} {payload}")

        (host_table_dir / f"{src}.lt").write_text("\n".join(out_lines) + "\n")


def parse_pairs_from_lt(path: Path) -> Dict[int, List[Tuple[int, int]]]:
    entries: Dict[int, List[Tuple[int, int]]] = {}
    current_dst = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) == 1:
            current_dst = int(tokens[0])
            entries[current_dst] = []
            continue
        if current_dst is None:
            continue
        for idx in range(2, len(tokens) - 1, 2):
            entries[current_dst].append((int(tokens[idx]), int(tokens[idx + 1])))
    return entries


def compare_with_reference(generated_dir: Path, reference_dir: Path) -> None:
    ref_host = reference_dir / "host_table"
    gen_host = generated_dir / "host_table"
    if not ref_host.exists() or not gen_host.exists():
        print("[compare] skipped: host_table missing in one of the directories")
        return

    strict_subset_ok = True
    coverage_stats = []

    for gen_file in sorted(gen_host.glob("*.lt")):
        ref_file = ref_host / gen_file.name
        if not ref_file.exists():
            strict_subset_ok = False
            continue

        gen_entries = parse_pairs_from_lt(gen_file)
        ref_entries = parse_pairs_from_lt(ref_file)

        for dst, gen_pairs in gen_entries.items():
            ref_pairs = set(ref_entries.get(dst, []))
            gen_set = set(gen_pairs)

            if not gen_set.issubset(ref_pairs):
                strict_subset_ok = False

            cov = 1.0 if not gen_set else len(gen_set & ref_pairs) / len(gen_set)
            coverage_stats.append(cov)

    avg_cov = sum(coverage_stats) / len(coverage_stats) if coverage_stats else 0.0
    print(f"[compare] generated pairs subset of reference: {'yes' if strict_subset_ok else 'no'}")
    print(f"[compare] average generated-pair coverage in reference: {avg_cov:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Dragonfly topology assets")
    parser.add_argument("-p", type=int, required=True, help="Hosts per switch")
    parser.add_argument("-a", type=int, required=True, help="Switches per group")
    parser.add_argument("-H", "--h", dest="h", type=int, required=True,
                        help="Global links per switch")
    parser.add_argument("--out", type=Path, required=True, help="Output topology directory")

    parser.add_argument("--switch-latency-ns", type=int, default=500)
    parser.add_argument("--link-speed-gbps", type=int, default=400)
    parser.add_argument("--link-latency-global-ns", type=int, default=500)
    parser.add_argument("--link-latency-local-ns", type=int, default=25)
    parser.add_argument("--link-latency-host-ns", type=int, default=25)

    parser.add_argument("--compare-ref", type=Path,
                        help="Optional reference topology dir (with host_table) to compare coverage")

    args = parser.parse_args()

    if args.p <= 0 or args.a <= 0 or args.h <= 0:
        raise ValueError("p, a, h must all be > 0")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    no_groups = args.a * args.h + 1
    no_switches = args.a * no_groups
    no_hosts = args.p * no_switches

    write_dragonfly_topo(out,
                         args.p,
                         args.a,
                         args.h,
                         args.switch_latency_ns,
                         args.link_speed_gbps,
                         args.link_latency_global_ns,
                         args.link_latency_local_ns,
                         args.link_latency_host_ns)

    write_dragonfly_adjlist(out, args.a, args.h, no_groups, no_switches)
    write_host_tables(out, args.p, args.a, args.h, no_groups, no_switches)

    print(f"Generated Dragonfly assets in: {out}")
    print(f"p={args.p} a={args.a} h={args.h}")
    print(f"switches={no_switches} hosts={no_hosts}")

    if args.compare_ref:
        compare_with_reference(out, args.compare_ref)


if __name__ == "__main__":
    main()
