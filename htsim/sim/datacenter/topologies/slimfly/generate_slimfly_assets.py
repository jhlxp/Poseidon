#!/usr/bin/env python3

import argparse
import itertools
import math
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
from sympy.ntheory.primetest import isprime
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_add, gf_gcdex, gf_irreducible_p, gf_mul, gf_rem, gf_sub


def is_prime(x: int) -> bool:
    return len([d for d in range(1, x + 1) if x % d == 0]) == 2


def get_power_of_prime(q: int) -> Tuple[int, int]:
    prime_divisors = [d for d in range(2, q + 1) if q % d == 0 and is_prime(d)]
    if len(prime_divisors) != 1:
        return 0, 0

    prime = prime_divisors[0]
    m = 0
    v = q
    while v != 1:
        if v % prime == 0:
            v = v // prime
            m += 1
        else:
            return 0, 0
    return prime, m


class GF:
    def __init__(self, p: int, n: int = 1):
        p, n = int(p), int(n)
        if not isprime(p):
            raise ValueError(f"p must be prime, got {p}")
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")

        self.p = p
        self.n = n

        if n == 1:
            self.reducing = [1, 0]
        else:
            for c in itertools.product(range(p), repeat=n):
                poly = (1, *c)
                if gf_irreducible_p(poly, p, ZZ):
                    self.reducing = poly
                    break

        self.elems = None

    def get_elems(self) -> List[List[int]]:
        if self.elems is None:
            self.elems = [[]]
            for c in range(1, self.p):
                self.elems.append([c])
            for deg in range(1, self.n):
                for c in itertools.product(range(self.p), repeat=deg):
                    for first in range(1, self.p):
                        self.elems.append(list((first, *c)))
        return self.elems

    def get_primitive_elem(self) -> List[int]:
        for primitive in self.get_elems():
            if self.is_primitive_elem(primitive):
                return primitive
        raise RuntimeError("No primitive element found")

    def is_primitive_elem(self, primitive: List[int]) -> bool:
        elems = [[]]
        tmp = [1]
        for _ in range(1, self.p**self.n):
            tmp = self.mul(tmp, primitive)
            elems.append(tmp)

        return len(elems) == self.p**self.n and all(elem in elems for elem in self.get_elems())

    def add(self, x: List[int], y: List[int]) -> List[int]:
        return gf_add(x, y, self.p, ZZ)

    def sub(self, x: List[int], y: List[int]) -> List[int]:
        return gf_sub(x, y, self.p, ZZ)

    def mul(self, x: List[int], y: List[int]) -> List[int]:
        product = gf_mul(x, y, self.p, ZZ)
        return gf_rem(product, self.reducing, self.p, ZZ)

    def inv(self, x: List[int]) -> List[int]:
        s, _, _ = gf_gcdex(x, self.reducing, self.p, ZZ)
        return s


def get_slimfly(q: int) -> nx.Graph:
    assert (q - 1) % 4 == 0 or (q + 1) % 4 == 0 or q % 4 == 0

    network = nx.Graph()
    prime, prime_power = get_power_of_prime(q)
    if prime == 0 or prime_power == 0 or prime**prime_power != q:
        raise ValueError(f"q={q} is not a prime power")

    if q % 4 == 3:
        delta = -1
    elif q % 4 == 2:
        raise ValueError("q is not of form 4w + delta")
    else:
        delta = q % 4

    if (q - delta) % 4 != 0:
        raise ValueError("invalid q, cannot derive w")
    w = int((q - delta) / 4)
    if w < 1:
        raise ValueError("invalid w")

    gf = GF(prime, prime_power)
    pe = gf.get_primitive_elem()

    pe_powers = [[1]]
    for i in range(1, q):
        pe_powers.append(gf.mul(pe_powers[i - 1], pe))

    if delta == 0:
        X = [pe_powers[i] for i in range(0, q - 1) if i % 2 == 0]
        Xp = [pe_powers[i] for i in range(1, q) if i % 2 == 1]
    elif delta == 1:
        X = [pe_powers[i] for i in range(0, q - 2) if i % 2 == 0]
        Xp = [pe_powers[i] for i in range(1, q - 1) if i % 2 == 1]
    else:
        X = [pe_powers[i] for i in range(0, 2 * w - 1) if i % 2 == 0]
        X.extend([pe_powers[i] for i in range(2 * w - 1, 4 * w - 2) if i % 2 == 1])
        Xp = [pe_powers[i] for i in range(1, 2 * w) if i % 2 == 1]
        Xp.extend([pe_powers[i] for i in range(2 * w, 4 * w - 1) if i % 2 == 0])

    labels = [(v, x, y) for v in [1, 0] for x in gf.get_elems() for y in gf.get_elems()]
    maps = list(zip(labels, [i for i in range(2 * q**2)]))

    for s in maps:
        for t in maps:
            if (
                s[0][0] == 0
                and t[0][0] == 0
                and s[0][1] == t[0][1]
                and gf.sub(s[0][2], t[0][2]) in X
            ):
                network.add_edge(s[1], t[1])

            if (
                s[0][0] == 1
                and t[0][0] == 1
                and s[0][1] == t[0][1]
                and gf.sub(s[0][2], t[0][2]) in Xp
            ):
                network.add_edge(s[1], t[1])

            if (
                s[0][0] == 0
                and t[0][0] == 1
                and s[0][2] == gf.add(gf.mul(t[0][1], s[0][1]), t[0][2])
            ):
                network.add_edge(s[1], t[1])

    return network


_network = None


def init_worker(network_edges: List[Tuple[int, int]]) -> None:
    global _network
    _network = nx.Graph()
    _network.add_edges_from(network_edges)


def latency_sort_key(key: Tuple[int, int]) -> int:
    return key[0] * 25 + key[1] * 500


def generate_host_table_for_switch(out_dir: Path, p: int, q: int, src_switch: int) -> None:
    global _network
    no_switches = 2 * q**2

    output_lines: List[str] = []
    for dst_switch in range(no_switches):
        if src_switch == dst_switch:
            continue

        paths = []
        paths_final = []

        for n in _network.neighbors(src_switch):
            for dst in _network.neighbors(n):
                if src_switch != dst:
                    paths.append([src_switch, n, dst])

        for path in paths:
            n = path[-1]
            shortest_path = nx.shortest_path(_network, source=n, target=dst_switch)
            full = path + shortest_path[1:]
            if len(full) == len(set(full)):
                paths_final.append(full)

        group = defaultdict(list)
        for path in paths_final:
            local_hops = 0
            global_hops = 0
            for u, v in zip(path, path[1:]):
                if u // q == v // q:
                    local_hops += 1
                else:
                    global_hops += 1
            group[(local_hops, global_hops)].append((path[1], path[2]))

        output_lines.append(f"{dst_switch}")
        for key in sorted(group.keys(), key=latency_sort_key):
            line = f"{key[0]} {key[1]}"
            for x, y in group[key]:
                line += f" {x} {y}"
            output_lines.append(line)

    (out_dir / "host_table" / f"{src_switch}.lt").write_text("\n".join(output_lines) + "\n")


def generate_fib_for_switch(out_dir: Path, q: int, src_switch: int) -> None:
    global _network
    no_switches = 2 * q**2

    output_lines: List[str] = []
    for dst_switch in range(no_switches):
        if src_switch == dst_switch:
            continue
        shortest_path = nx.shortest_path(_network, source=src_switch, target=dst_switch)
        output_lines.append(f"{dst_switch} {shortest_path[1]}")

    (out_dir / "fib" / f"{src_switch}.fib").write_text("\n".join(output_lines) + "\n")


def process_switch_slice(args: Tuple[Path, int, int, int, int]) -> None:
    out_dir, p, q, start, end = args
    for src_switch in range(start, end):
        generate_host_table_for_switch(out_dir, p, q, src_switch)
        generate_fib_for_switch(out_dir, q, src_switch)


def write_slimfly_topo(path: Path,
                       p: int,
                       q: int,
                       switch_latency_ns: int,
                       link_speed_gbps: int,
                       link_latency_global_ns: int,
                       link_latency_local_ns: int,
                       link_latency_host_ns: int) -> None:
    content = (
        "# Slim Fly\n"
        f"p {p}\n"
        f"q {q}\n\n"
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
    (path / "slimfly.topo").write_text(content)


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
    parser = argparse.ArgumentParser(description="Generate SlimFly topology assets")
    parser.add_argument("-p", type=int, required=True, help="Hosts per switch")
    parser.add_argument("-q", type=int, required=True, help="SlimFly q parameter")
    parser.add_argument("--out", type=Path, required=True, help="Output topology directory")
    parser.add_argument("-n", "--num-workers", type=int, default=mp.cpu_count())

    parser.add_argument("--switch-latency-ns", type=int, default=500)
    parser.add_argument("--link-speed-gbps", type=int, default=400)
    parser.add_argument("--link-latency-global-ns", type=int, default=500)
    parser.add_argument("--link-latency-local-ns", type=int, default=25)
    parser.add_argument("--link-latency-host-ns", type=int, default=25)

    parser.add_argument("--compare-ref", type=Path,
                        help="Optional reference topology dir (with host_table) to compare coverage")

    args = parser.parse_args()
    if args.p <= 0 or args.q <= 0:
        raise ValueError("p and q must be > 0")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "host_table").mkdir(parents=True, exist_ok=True)
    (out / "fib").mkdir(parents=True, exist_ok=True)

    network = get_slimfly(args.q)
    nx.write_adjlist(network, out / "slimfly.adjlist")
    edges = list(network.edges())

    write_slimfly_topo(out,
                       args.p,
                       args.q,
                       args.switch_latency_ns,
                       args.link_speed_gbps,
                       args.link_latency_global_ns,
                       args.link_latency_local_ns,
                       args.link_latency_host_ns)

    no_switches = 2 * args.q**2
    num_workers = max(1, min(args.num_workers, no_switches))
    slice_size = math.ceil(no_switches / num_workers)
    work_items = []
    for start in range(0, no_switches, slice_size):
        end = min(start + slice_size, no_switches)
        work_items.append((out, args.p, args.q, start, end))

    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(edges,)) as pool:
        pool.map(process_switch_slice, work_items)

    print(f"Generated SlimFly assets in: {out}")
    print(f"p={args.p} q={args.q} switches={no_switches} hosts={args.p * no_switches}")

    if args.compare_ref:
        compare_with_reference(out, args.compare_ref)


if __name__ == "__main__":
    main()
