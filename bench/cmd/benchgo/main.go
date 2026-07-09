// Command benchgo runs the Alcatraz side of the benchmark.
//
// Modes:
//
//	-mode bench   time Analyze over the corpus, print JSON stats to stdout
//	-mode detect  print every detection (for the parity diff), no timing
//
// The JSON schema matches bench/python/bench_presidio.py so compare.py can
// consume either engine's output interchangeably.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"sort"
	"time"

	"github.com/hoophq/alcatraz"
	"github.com/hoophq/alcatraz/bench/internal/corpus"
)

type groupStat struct {
	SizeClass  string  `json:"size_class"`
	Density    string  `json:"density"`
	Docs       int     `json:"docs"`
	Bytes      int     `json:"bytes"`
	Iterations int     `json:"iterations"`
	MeanUsDoc  float64 `json:"mean_us_per_doc"`
	P50UsDoc   float64 `json:"p50_us_per_doc"`
	P99UsDoc   float64 `json:"p99_us_per_doc"`
	MBPerSec   float64 `json:"mb_per_sec"`
}

type report struct {
	Engine    string      `json:"engine"`
	Mode      string      `json:"mode"`
	GoVersion string      `json:"runtime"`
	Groups    []groupStat `json:"groups"`
}

type detection struct {
	DocID      string  `json:"doc_id"`
	EntityType string  `json:"entity_type"`
	Start      int     `json:"start"`
	End        int     `json:"end"`
	Score      float64 `json:"score"`
	Text       string  `json:"text"`
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(p * float64(len(sorted)-1))
	return sorted[idx]
}

func runBench(eng *alcatraz.Engine, docs []corpus.Doc, iterations int, entities []string) report {
	opts := alcatraz.Options{Entities: entities}

	// Warm up: one full pass so lazy initialization is outside the timed loop.
	for _, d := range docs {
		_ = eng.Analyze(d.Text, opts)
	}

	byGroup := map[[2]string][]corpus.Doc{}
	for _, d := range docs {
		k := [2]string{d.SizeClass, d.Density}
		byGroup[k] = append(byGroup[k], d)
	}

	var groups []groupStat
	for _, sc := range corpus.SizeClasses {
		for _, den := range corpus.Densities {
			gdocs := byGroup[[2]string{sc, den}]
			if len(gdocs) == 0 {
				continue
			}
			var samples []float64 // per-doc µs
			bytes := 0
			for _, d := range gdocs {
				bytes += len(d.Text)
			}
			totalStart := time.Now()
			for it := 0; it < iterations; it++ {
				for _, d := range gdocs {
					s := time.Now()
					_ = eng.Analyze(d.Text, opts)
					samples = append(samples, float64(time.Since(s).Nanoseconds())/1e3)
				}
			}
			elapsed := time.Since(totalStart).Seconds()

			sort.Float64s(samples)
			mean := 0.0
			for _, v := range samples {
				mean += v
			}
			mean /= float64(len(samples))

			groups = append(groups, groupStat{
				SizeClass:  sc,
				Density:    den,
				Docs:       len(gdocs),
				Bytes:      bytes,
				Iterations: iterations,
				MeanUsDoc:  mean,
				P50UsDoc:   percentile(samples, 0.50),
				P99UsDoc:   percentile(samples, 0.99),
				MBPerSec:   float64(bytes*iterations) / 1e6 / elapsed,
			})
		}
	}

	return report{
		Engine:    "alcatraz-go",
		Mode:      "bench",
		GoVersion: runtime.Version(),
		Groups:    groups,
	}
}

func runDetect(eng *alcatraz.Engine, docs []corpus.Doc, entities []string) []detection {
	opts := alcatraz.Options{Entities: entities}
	var out []detection
	for _, d := range docs {
		for _, r := range eng.Analyze(d.Text, opts) {
			out = append(out, detection{
				DocID: d.ID, EntityType: r.EntityType,
				Start: r.Start, End: r.End, Score: r.Score, Text: r.Text,
			})
		}
	}
	return out
}

func main() {
	corpusPath := flag.String("corpus", "corpus.jsonl", "corpus JSONL path")
	mode := flag.String("mode", "bench", "bench or detect")
	iterations := flag.Int("iterations", 20, "timed passes over each group")
	var entityList entityFlags
	flag.Var(&entityList, "entity", "restrict to entity type (repeatable)")
	flag.Parse()

	docs, err := corpus.Load(*corpusPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	eng := alcatraz.NewEngine()
	enc := json.NewEncoder(os.Stdout)

	switch *mode {
	case "bench":
		if err := enc.Encode(runBench(eng, docs, *iterations, entityList)); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case "detect":
		for _, d := range runDetect(eng, docs, entityList) {
			if err := enc.Encode(d); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
		}
	default:
		fmt.Fprintf(os.Stderr, "unknown mode %q\n", *mode)
		os.Exit(1)
	}
}

type entityFlags []string

func (e *entityFlags) String() string     { return fmt.Sprint([]string(*e)) }
func (e *entityFlags) Set(v string) error { *e = append(*e, v); return nil }
