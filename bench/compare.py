#!/usr/bin/env python3
"""Merge benchmark outputs from benchgo and bench_presidio.py into one table.

Speed comparison:
    ./compare.py speed go.json py-slim.json [more.json ...]

Detection parity (are the engines finding the same things?):
    ./compare.py parity go-detections.jsonl py-detections.jsonl

Pure stdlib; run with any python3, no venv needed.
"""

import json
import sys
from collections import defaultdict


def load_reports(paths):
    reports = []
    for p in paths:
        with open(p) as f:
            reports.append(json.load(f))
    return reports


def speed(paths):
    reports = load_reports(paths)
    engines = [r["engine"] for r in reports]

    # group key -> engine -> stats
    table = defaultdict(dict)
    order = []
    for r in reports:
        for g in r["groups"]:
            key = (g["size_class"], g["density"])
            if key not in order:
                order.append(key)
            table[key][r["engine"]] = g

    base = engines[0]
    hdr = f"{'group':<14}" + "".join(
        f"{e + ' µs/doc':>24}{e + ' MB/s':>18}" for e in engines
    )
    if len(engines) > 1:
        hdr += f"{'speedup vs ' + base:>20}"
    print(hdr)
    print("-" * len(hdr))

    for key in order:
        row = f"{key[0] + '/' + key[1]:<14}"
        base_mean = table[key].get(base, {}).get("mean_us_per_doc")
        for e in engines:
            g = table[key].get(e)
            if g is None:
                row += f"{'-':>24}{'-':>18}"
                continue
            row += f"{g['mean_us_per_doc']:>24.1f}{g['mb_per_sec']:>18.1f}"
        if len(engines) > 1:
            last = table[key].get(engines[-1])
            if base_mean and last:
                row += f"{last['mean_us_per_doc'] / base_mean:>19.1f}x"
        print(row)


def load_detections(path):
    dets = defaultdict(set)  # doc_id -> {(entity_type, start, end)}
    counts = defaultdict(int)  # entity_type -> n
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            dets[d["doc_id"]].add((d["entity_type"], d["start"], d["end"]))
            counts[d["entity_type"]] += 1
    return dets, counts


def parity(path_a, path_b):
    a, counts_a = load_detections(path_a)
    b, counts_b = load_detections(path_b)

    all_types = sorted(set(counts_a) | set(counts_b))
    print(f"{'entity_type':<28}{'A':>8}{'B':>8}{'exact overlap':>16}")
    print("-" * 60)
    total_a = total_b = total_common = 0
    for t in all_types:
        common = 0
        for doc_id in set(a) | set(b):
            sa = {(s, e) for (et, s, e) in a.get(doc_id, ()) if et == t}
            sb = {(s, e) for (et, s, e) in b.get(doc_id, ()) if et == t}
            common += len(sa & sb)
        na, nb = counts_a.get(t, 0), counts_b.get(t, 0)
        total_a += na
        total_b += nb
        total_common += common
        flag = "" if na == nb == common else "  <-- differs"
        print(f"{t:<28}{na:>8}{nb:>8}{common:>16}{flag}")
    print("-" * 60)
    print(f"{'TOTAL':<28}{total_a:>8}{total_b:>8}{total_common:>16}")
    if total_a and total_b:
        print(
            f"\nexact span agreement: {total_common}/{total_a} of A, "
            f"{total_common}/{total_b} of B"
        )
        print(
            "note: if coverage differs a lot, restrict both engines to the "
            "same entity set (-entity / --entity) before quoting speedups."
        )


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "speed":
        speed(sys.argv[2:])
    elif cmd == "parity":
        if len(sys.argv) != 4:
            print("parity takes exactly two detection files")
            sys.exit(1)
        parity(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
