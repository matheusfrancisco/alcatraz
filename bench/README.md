# Benchmarks: Alcatraz vs Presidio

Compares analyze throughput between **Alcatraz** (this repo, Go) and the
local **presidio-analyzer** checkout (`../../presidio/presidio-analyzer`,
Python). Both harnesses read the same corpus and emit the same JSON schema,
so results merge into one table.

This directory is its own Go module (`replace`d to the parent) and its own
uv project — nothing here touches the Alcatraz core module or your global
Python environment.

## Layout

```
bench/
├── go.mod                    Separate module; replace github.com/hoophq/alcatraz => ../
├── corpus.jsonl              Generated shared corpus (deterministic, seed 42)
├── cmd/gencorpus/            Corpus generator (valid Luhn/CPF/IBAN/SSN seeds)
├── cmd/benchgo/              Alcatraz harness: -mode bench | detect
├── internal/corpus/          JSONL loader shared by the Go tools
├── analyze_bench_test.go     Plain `go test -bench` benchmarks (benchstat-friendly)
├── python/                   uv project; presidio-analyzer installed editable
│   ├── pyproject.toml
│   └── bench_presidio.py     Presidio harness: --mode bench | detect
└── compare.py                Merges outputs: speed table + detection parity
```

## One-time setup

```bash
cd bench
go run ./cmd/gencorpus -out corpus.jsonl          # regenerate the corpus
cd python && uv sync                              # venv + local presidio-analyzer
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

## Run the comparison

```bash
cd bench

# 1. Speed (Python side takes ~25 min at 2 iterations, dominated by the 1MB groups)
go run ./cmd/benchgo -mode bench -iterations 20 > results-go.json
(cd python && uv run bench_presidio.py --mode bench --iterations 2) > results-py-slim.json
./compare.py speed results-go.json results-py-slim.json

# 2. Detection parity (velocity numbers are only fair if coverage is comparable)
go run ./cmd/benchgo -mode detect > detections-go.jsonl
(cd python && uv run bench_presidio.py --mode detect) > detections-py.jsonl
./compare.py parity detections-go.jsonl detections-py.jsonl
```

Restrict both engines to the same entity set for a stricter comparison:

```bash
go run ./cmd/benchgo -mode bench -entity CREDIT_CARD -entity EMAIL_ADDRESS > results-go.json
(cd python && uv run bench_presidio.py --mode bench --entity CREDIT_CARD --entity EMAIL_ADDRESS) > results-py-slim.json
```

## Presidio engine modes

- `--engine slim` (default): `SlimSpacyNlpEngine` — spaCy tokenization only,
  **no NER model**. This is the closest apples-to-apples comparison with
  Alcatraz's pattern-only core.
- `--engine full`: the default Presidio pipeline, which runs spaCy NER on
  every `analyze()` call. This is what Presidio users get out of the box
  (requires `en_core_web_lg`; slower by design). Report both numbers — they
  answer different questions.

## Methodology notes

- **Engine construction is excluded** from the timed loop; a full warm-up
  pass runs first. `BenchmarkNewEngine` measures construction separately.
- The corpus is deterministic (seed 42): 186 docs across a
  {100B, 1KB, 10KB, 1MB} x {none, sparse, dense} matrix. Seeded PII passes
  real checksums (Luhn cards, mod-11 CPF, mod-97 IBAN), so validator paths
  are exercised the way real data exercises them. The 1MB groups carry
  fewer docs (2 per group by default) — Presidio needs seconds to minutes
  per megabyte-sized document, so a full-size group would take hours.
- The `none` density rows matter most for real workloads — most production
  text has no PII, so scan-and-reject cost usually dominates.
- Throughput is reported as **µs/doc and MB/s** so numbers transfer across
  corpus choices.
- Everything is **single-threaded**. Concurrency (goroutines vs Python
  multiprocessing) is a separate, also interesting, benchmark.
- Alcatraz-only regression tracking: `go test -bench=. -count=10 | benchstat -`.
