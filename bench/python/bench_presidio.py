"""Presidio side of the benchmark.

Modes:
    --mode bench    time analyze() over the corpus, print JSON stats to stdout
    --mode detect   print every detection (for the parity diff), no timing

Engines:
    --engine slim   SlimSpacyNlpEngine: tokenization only, no NER model.
                    Closest apples-to-apples with Alcatraz's pattern-only path.
    --engine full   Default AnalyzerEngine pipeline (spaCy NER on every call).
                    What Presidio users get out of the box.

The JSON schema matches bench/cmd/benchgo so compare.py can consume either
engine's output interchangeably.
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict

SIZE_CLASSES = ["100B", "1KB", "10KB", "1MB"]
DENSITIES = ["none", "sparse", "dense"]


def load_corpus(path):
    docs = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def build_analyzer(engine_kind):
    from presidio_analyzer import AnalyzerEngine

    if engine_kind == "slim":
        from presidio_analyzer.nlp_engine import SlimSpacyNlpEngine

        nlp = SlimSpacyNlpEngine(
            models=[{"lang_code": "en", "model_name": "en_core_web_sm"}]
        )
        analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
        # spaCy caps input at 1M chars to guard parser/NER memory blowups.
        # The slim engine disables both, so raising it is safe — and needed
        # for the corpus's 1MB documents.
        for lang_nlp in nlp.nlp.values():
            lang_nlp.max_length = 20_000_000
        return analyzer

    # "full": whatever the default provider gives (spaCy NER per call).
    return AnalyzerEngine(supported_languages=["en"])


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    return sorted_vals[int(p * (len(sorted_vals) - 1))]


def run_bench(analyzer, docs, iterations, entities):
    # Warm up: one full pass so lazy init is outside the timed loop.
    for d in docs:
        analyzer.analyze(text=d["text"], language="en", entities=entities)

    by_group = defaultdict(list)
    for d in docs:
        by_group[(d["size_class"], d["density"])].append(d)

    groups = []
    for sc in SIZE_CLASSES:
        for den in DENSITIES:
            gdocs = by_group.get((sc, den))
            if not gdocs:
                continue
            nbytes = sum(len(d["text"]) for d in gdocs)
            samples = []  # per-doc microseconds
            total_start = time.perf_counter()
            for _ in range(iterations):
                for d in gdocs:
                    s = time.perf_counter()
                    analyzer.analyze(text=d["text"], language="en", entities=entities)
                    samples.append((time.perf_counter() - s) * 1e6)
            elapsed = time.perf_counter() - total_start

            samples.sort()
            groups.append(
                {
                    "size_class": sc,
                    "density": den,
                    "docs": len(gdocs),
                    "bytes": nbytes,
                    "iterations": iterations,
                    "mean_us_per_doc": statistics.fmean(samples),
                    "p50_us_per_doc": percentile(samples, 0.50),
                    "p99_us_per_doc": percentile(samples, 0.99),
                    "mb_per_sec": (nbytes * iterations) / 1e6 / elapsed,
                }
            )
    return groups


def run_detect(analyzer, docs, entities):
    for d in docs:
        text = d["text"]
        for r in analyzer.analyze(text=text, language="en", entities=entities):
            yield {
                "doc_id": d["id"],
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": r.score,
                "text": text[r.start : r.end],
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../corpus.jsonl")
    ap.add_argument("--mode", choices=["bench", "detect"], default="bench")
    ap.add_argument("--engine", choices=["slim", "full"], default="slim")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--entity", action="append", dest="entities", default=None,
                    help="restrict to entity type (repeatable)")
    args = ap.parse_args()

    docs = load_corpus(args.corpus)
    analyzer = build_analyzer(args.engine)
    out = sys.stdout

    if args.mode == "bench":
        report = {
            "engine": f"presidio-py-{args.engine}",
            "mode": "bench",
            "runtime": f"python{sys.version_info.major}.{sys.version_info.minor}",
            "groups": run_bench(analyzer, docs, args.iterations, args.entities),
        }
        json.dump(report, out)
        out.write("\n")
    else:
        for det in run_detect(analyzer, docs, args.entities):
            json.dump(det, out)
            out.write("\n")


if __name__ == "__main__":
    main()
