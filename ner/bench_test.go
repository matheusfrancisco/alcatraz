package ner

import (
	"context"
	"os"
	"strconv"
	"strings"
	"testing"
)

// BenchmarkLiveProcessTexts measures end-to-end batched inference throughput
// (tokenization, windowing, model, span mapping) over a synthetic message
// corpus. Like TestLiveNER it needs the real model, so it is gated behind
// ALCATRAZ_NER_LIVE=1 and configured by the same environment variables,
// which makes backend comparisons one-liners:
//
//	ALCATRAZ_NER_LIVE=1 go test -bench LiveProcessTexts -benchtime 1x -run xxx .
//	ALCATRAZ_NER_LIVE=1 ALCATRAZ_NER_BACKEND=ort CGO_LDFLAGS=-L/path/to/libtokenizers \
//	  go test -tags ORT -bench LiveProcessTexts -benchtime 1x -run xxx .
//
// ALCATRAZ_NER_BENCH_BYTES sets the corpus size (default 300000). Reported
// MB/s is corpus bytes per wall-clock second, single inference stream.
func BenchmarkLiveProcessTexts(b *testing.B) {
	if os.Getenv("ALCATRAZ_NER_LIVE") != "1" {
		b.Skip("set ALCATRAZ_NER_LIVE=1 to run the live benchmark")
	}
	cfg := DefaultConfig()
	cfg.Backend = os.Getenv("ALCATRAZ_NER_BACKEND")
	cfg.ORTLibraryPath = os.Getenv("ALCATRAZ_NER_ORT_LIB")
	cfg.Accelerator = os.Getenv("ALCATRAZ_NER_ACCELERATOR")

	corpusBytes := 300_000
	if v := os.Getenv("ALCATRAZ_NER_BENCH_BYTES"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			b.Fatalf("ALCATRAZ_NER_BENCH_BYTES: %v", err)
		}
		corpusBytes = n
	}

	// Message-like prose with entities, in varied lengths (1-7 paragraphs)
	// so batching sees a realistic mix of short rows and windowed rows.
	const para = "Yesterday John Smith from Berlin deployed the payment service " +
		"and emailed maria.silva@example.com about the incident in Sao Paulo. " +
		"The on-call engineer, Alice Johnson, reviewed the production logs and " +
		"confirmed the fix before the Tuesday retrospective. "
	var texts []string
	total := 0
	for i := 0; total < corpusBytes; i++ {
		msg := strings.Repeat(para, 1+i%7)
		texts = append(texts, msg)
		total += len(msg)
	}

	nlp, err := New(context.Background(), cfg)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	defer nlp.Close()

	b.SetBytes(int64(total))
	b.ResetTimer()
	for b.Loop() {
		if _, err := nlp.ProcessTexts(texts, "en"); err != nil {
			b.Fatalf("ProcessTexts: %v", err)
		}
	}
}
