package ner

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/hoophq/alcatraz/analyzer"
	"github.com/hoophq/alcatraz/entities"
	"github.com/hoophq/alcatraz/recognizers"
	"github.com/knights-analytics/hugot/pipelines"
)

func TestDefaultConfigSupportedEntities(t *testing.T) {
	got := DefaultConfig().SupportedEntities()
	want := []string{entities.DateTime, entities.Location, entities.NRP, entities.Person}
	if len(got) != len(want) {
		t.Fatalf("supported entities = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("supported entities = %v, want %v", got, want)
		}
	}
}

func TestByteSpan(t *testing.T) {
	text := "José lives in São Paulo" // é and ã are 2 bytes each

	t.Run("valid span unchanged", func(t *testing.T) {
		start, end, ok := byteSpan(text, 0, 5) // "José" = 5 bytes
		if !ok || start != 0 || end != 5 {
			t.Fatalf("got (%d,%d,%v), want (0,5,true)", start, end, ok)
		}
	})

	t.Run("mid-rune boundary snaps back", func(t *testing.T) {
		// byte 4 is the second byte of é
		start, end, ok := byteSpan(text, 0, 4)
		if !ok || end != 3 {
			t.Fatalf("got (%d,%d,%v), want end snapped to 3", start, end, ok)
		}
	})

	t.Run("out of range clamps", func(t *testing.T) {
		start, end, ok := byteSpan(text, -3, len(text)+10)
		if !ok || start != 0 || end != len(text) {
			t.Fatalf("got (%d,%d,%v), want (0,%d,true)", start, end, ok, len(text))
		}
	})

	t.Run("empty span dropped", func(t *testing.T) {
		if _, _, ok := byteSpan(text, 5, 5); ok {
			t.Fatal("empty span should not be ok")
		}
	})
}

func TestFoldASCII(t *testing.T) {
	t.Run("pure ASCII is identity with nil table", func(t *testing.T) {
		folded, offsets := foldASCII("John Smith lives in Berlin")
		if folded != "John Smith lives in Berlin" || offsets != nil {
			t.Fatalf("got (%q, %v)", folded, offsets)
		}
	})

	t.Run("accents fold to base letters, one byte per rune", func(t *testing.T) {
		text := "José Núñez mora em São Paulo"
		folded, offsets := foldASCII(text)
		if folded != "Jose Nunez mora em Sao Paulo" {
			t.Fatalf("folded = %q", folded)
		}
		if len(offsets) != len(folded) {
			t.Fatalf("offset table length %d != folded length %d", len(offsets), len(folded))
		}
		// A span over folded "Sao Paulo" must map back to "São Paulo".
		start, end := remapSpan(offsets, len(text), 19, 28)
		if text[start:end] != "São Paulo" {
			t.Fatalf("remapped span = %q, want 'São Paulo'", text[start:end])
		}
	})

	t.Run("span ending at text end maps to text end", func(t *testing.T) {
		text := "città"
		folded, offsets := foldASCII(text)
		if folded != "citta" {
			t.Fatalf("folded = %q", folded)
		}
		start, end := remapSpan(offsets, len(text), 0, len(folded))
		if start != 0 || end != len(text) {
			t.Fatalf("got (%d,%d), want (0,%d)", start, end, len(text))
		}
	})

	t.Run("non-decomposable rune becomes placeholder", func(t *testing.T) {
		folded, _ := foldASCII("名前 John")
		if len(folded) != len([]rune("名前 John")) {
			t.Fatalf("folded = %q, want one byte per rune", folded)
		}
	})
}

func TestToNerSpan(t *testing.T) {
	eng := &Engine{cfg: DefaultConfig()}
	text := "John Smith visited Berlin"

	t.Run("maps model label to canonical entity", func(t *testing.T) {
		span, ok := eng.toNerSpan(text, nil, pipelines.Entity{
			Entity: "PER", Score: 0.99, Start: 0, End: 10,
		})
		if !ok || span.EntityType != entities.Person {
			t.Fatalf("got (%+v, %v), want PERSON span", span, ok)
		}
		if span.Start != 0 || span.End != 10 || span.Score != float64(float32(0.99)) {
			t.Fatalf("unexpected span %+v", span)
		}
	})

	t.Run("ignored label dropped", func(t *testing.T) {
		if _, ok := eng.toNerSpan(text, nil, pipelines.Entity{
			Entity: "ORG", Score: 0.9, Start: 19, End: 25,
		}); ok {
			t.Fatal("ORG should be dropped: it maps to ORGANIZATION which is ignored")
		}
		if _, ok := eng.toNerSpan(text, nil, pipelines.Entity{
			Entity: "MISC", Score: 0.9, Start: 19, End: 25,
		}); ok {
			t.Fatal("MISC should be dropped")
		}
	})

	t.Run("unmapped label kept as-is", func(t *testing.T) {
		span, ok := eng.toNerSpan(text, nil, pipelines.Entity{
			Entity: "CUSTOM", Score: 0.7, Start: 0, End: 4,
		})
		if !ok || span.EntityType != "CUSTOM" {
			t.Fatalf("got (%+v, %v), want CUSTOM kept", span, ok)
		}
	})

	t.Run("low score entity down-weighted", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.LowScoreEntities = []string{entities.Person}
		low := &Engine{cfg: cfg}
		span, ok := low.toNerSpan(text, nil, pipelines.Entity{
			Entity: "PER", Score: 1.0, Start: 0, End: 10,
		})
		if !ok || span.Score != 0.4 {
			t.Fatalf("got (%+v, %v), want score 0.4", span, ok)
		}
	})

	t.Run("score clamped to MaxScore", func(t *testing.T) {
		span, ok := eng.toNerSpan(text, nil, pipelines.Entity{
			Entity: "PER", Score: 1.2, Start: 0, End: 10,
		})
		if !ok || span.Score != analyzer.MaxScore {
			t.Fatalf("got (%+v, %v), want score clamped to 1.0", span, ok)
		}
	})
}

func TestRecognizerWithArtifacts(t *testing.T) {
	eng := &Engine{cfg: DefaultConfig()}
	rec := eng.Recognizer("en")

	artifacts := &analyzer.NlpArtifacts{Ents: []analyzer.NerSpan{
		{EntityType: entities.Person, Start: 0, End: 10, Score: 0.9},
		{EntityType: entities.Location, Start: 19, End: 25, Score: 0.8},
	}}

	t.Run("emits all artifact spans", func(t *testing.T) {
		got := rec.AnalyzeWithArtifacts("John Smith visited Berlin", nil, artifacts)
		if len(got) != 2 {
			t.Fatalf("want 2 results, got %+v", got)
		}
		if got[0].RecognizerName != "NERecognizer" {
			t.Errorf("recognizer name = %q", got[0].RecognizerName)
		}
	})

	t.Run("entity filter applies", func(t *testing.T) {
		got := rec.AnalyzeWithArtifacts("John Smith visited Berlin",
			[]string{entities.Location}, artifacts)
		if len(got) != 1 || got[0].EntityType != entities.Location {
			t.Fatalf("want single LOCATION, got %+v", got)
		}
	})
}

// TestNewRejectsNonPositiveBuckets guards ProcessTexts' batching loop, which
// advances by the largest batch bucket: a non-positive value would make it
// loop forever, and a non-positive sequence bucket would corrupt the window
// token budget. New must reject both up front.
func TestNewRejectsNonPositiveBuckets(t *testing.T) {
	ctx := context.Background()

	t.Run("zero batch bucket", func(t *testing.T) {
		_, err := New(ctx, Config{ModelPath: t.TempDir(), BatchBuckets: []int{0}})
		if err == nil || !strings.Contains(err.Error(), "BatchBuckets") {
			t.Fatalf("err = %v, want BatchBuckets validation error", err)
		}
	})

	t.Run("negative batch bucket among valid ones", func(t *testing.T) {
		_, err := New(ctx, Config{ModelPath: t.TempDir(), BatchBuckets: []int{8, -1, 32}})
		if err == nil || !strings.Contains(err.Error(), "BatchBuckets") {
			t.Fatalf("err = %v, want BatchBuckets validation error", err)
		}
	})

	t.Run("non-positive sequence bucket", func(t *testing.T) {
		_, err := New(ctx, Config{ModelPath: t.TempDir(), SequenceBuckets: []int{-16, 128}})
		if err == nil || !strings.Contains(err.Error(), "SequenceBuckets") {
			t.Fatalf("err = %v, want SequenceBuckets validation error", err)
		}
	})
}

// TestLiveNER downloads the default model (~250MB on first run) and checks
// the full pipeline end to end. Gated behind ALCATRAZ_NER_LIVE=1.
//
// The backend is selectable so the same assertions verify every backend:
// ALCATRAZ_NER_BACKEND=ort (with a -tags ORT build), ALCATRAZ_NER_ORT_LIB
// pointing at the ONNX Runtime library if it is not in a standard location,
// and ALCATRAZ_NER_ACCELERATOR=coreml/cuda for GPU execution providers.
func TestLiveNER(t *testing.T) {
	if os.Getenv("ALCATRAZ_NER_LIVE") != "1" {
		t.Skip("set ALCATRAZ_NER_LIVE=1 to run the live model test")
	}
	cfg := DefaultConfig()
	cfg.Backend = os.Getenv("ALCATRAZ_NER_BACKEND")
	cfg.ORTLibraryPath = os.Getenv("ALCATRAZ_NER_ORT_LIB")
	cfg.Accelerator = os.Getenv("ALCATRAZ_NER_ACCELERATOR")

	// Construct with a cancellable ctx and cancel it right after New: the
	// ctx bounds construction only, so inference below must be unaffected
	// (the engine's lifetime is governed by Close, not the ctx).
	ctx, cancel := context.WithCancel(context.Background())
	nlp, err := New(ctx, cfg)
	cancel()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer nlp.Close()

	reg := analyzer.NewRegistry("en")
	recognizers.LoadDefaults(reg, "en")
	reg.Add("en", nlp.Recognizer("en"))
	eng := analyzer.NewEngine(reg, []string{"en"})
	eng.SetNlpEngine(nlp)

	text := "My name is John Smith, I live in Berlin and my email is john@example.com"
	results := eng.Analyze(text, analyzer.Options{})

	byType := map[string]string{}
	for _, r := range results {
		byType[r.EntityType] = r.Text
	}
	if byType[entities.Person] != "John Smith" {
		t.Errorf("PERSON = %q, want 'John Smith' (all: %v)", byType[entities.Person], byType)
	}
	if byType[entities.Location] != "Berlin" {
		t.Errorf("LOCATION = %q, want 'Berlin' (all: %v)", byType[entities.Location], byType)
	}
	if byType[entities.EmailAddress] != "john@example.com" {
		t.Errorf("EMAIL_ADDRESS = %q, want 'john@example.com' (all: %v)", byType[entities.EmailAddress], byType)
	}

	// Byte-offset invariant on multi-byte input.
	utf8Text := "José Núñez mora em São Paulo"
	utf8Results := eng.Analyze(utf8Text, analyzer.Options{})
	for _, r := range utf8Results {
		if utf8Text[r.Start:r.End] != r.Text {
			t.Errorf("offset invariant broken: [%d:%d] = %q, Text = %q",
				r.Start, r.End, utf8Text[r.Start:r.End], r.Text)
		}
	}

	// Windowed inference: a text far beyond the model's 512-token limit
	// must still be analyzed in full — entities near the end used to be
	// unreachable (inference failed outright on over-long input).
	filler := strings.Repeat("the deployment pipeline ran and produced output logs without issues. ", 300)
	long := filler + "Finally John Smith arrived in Berlin."
	longArts, err := nlp.ProcessText(long, "en")
	if err != nil {
		t.Fatalf("ProcessText on long text: %v", err)
	}
	found := map[string]bool{}
	for _, span := range longArts.Ents {
		found[span.EntityType+":"+long[span.Start:span.End]] = true
	}
	if !found[entities.Person+":John Smith"] {
		t.Errorf("long text: PERSON 'John Smith' not found (got %v)", found)
	}
	if !found[entities.Location+":Berlin"] {
		t.Errorf("long text: LOCATION 'Berlin' not found (got %v)", found)
	}
}
