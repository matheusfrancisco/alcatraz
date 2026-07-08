package bench_test

// Standard testing.B benchmarks for use with `go test -bench` + benchstat.
// The cross-engine comparison uses cmd/benchgo instead; this file is for
// tracking Alcatraz-only regressions over time.

import (
	"testing"

	"github.com/hoophq/alcatraz"
	"github.com/hoophq/alcatraz/bench/internal/corpus"
)

func loadCorpus(b *testing.B) []corpus.Doc {
	b.Helper()
	docs, err := corpus.Load("corpus.jsonl")
	if err != nil {
		b.Skipf("corpus.jsonl not found — run `go run ./cmd/gencorpus` first: %v", err)
	}
	return docs
}

func BenchmarkAnalyze(b *testing.B) {
	docs := loadCorpus(b)
	eng := alcatraz.NewEngine()

	for _, sc := range corpus.SizeClasses {
		for _, den := range corpus.Densities {
			var group []corpus.Doc
			for _, d := range docs {
				if d.SizeClass == sc && d.Density == den {
					group = append(group, d)
				}
			}
			if len(group) == 0 {
				continue
			}
			b.Run(sc+"/"+den, func(b *testing.B) {
				bytes := 0
				for _, d := range group {
					bytes += len(d.Text)
				}
				b.SetBytes(int64(bytes / len(group)))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_ = eng.Analyze(group[i%len(group)].Text, alcatraz.Options{})
				}
			})
		}
	}
}

func BenchmarkNewEngine(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = alcatraz.NewEngine()
	}
}
