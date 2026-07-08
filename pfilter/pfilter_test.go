package pfilter

import (
	"errors"
	"os"
	"testing"
	"unsafe"

	"github.com/hoophq/alcatraz/analyzer"
	"github.com/hoophq/alcatraz/entities"
	"github.com/hoophq/alcatraz/recognizers"
)

// fakeClassifier stands in for the FFI layer so the mapping and engine
// logic are testable without libpf or a model.
type fakeClassifier struct {
	entities []rawEntity
	err      error
	calls    int
}

func (f *fakeClassifier) classify(text string, threshold float32) ([]rawEntity, error) {
	f.calls++
	return f.entities, f.err
}

func (f *fakeClassifier) close() error { return nil }

func newFakeEngine(c classifier, cfg Config) *Engine {
	if cfg.LabelMapping == nil {
		cfg.LabelMapping = DefaultConfig("x.gguf").LabelMapping
	}
	return &Engine{classifier: c, cfg: cfg}
}

func TestPfEntityLayout(t *testing.T) {
	// The FFI layer reads pf_entity arrays by pointer arithmetic; this pins
	// the assumed C struct layout (see pf.h).
	var e pfEntity
	if size := unsafe.Sizeof(e); size != 24 {
		t.Errorf("sizeof(pfEntity) = %d, want 24", size)
	}
	if off := unsafe.Offsetof(e.start); off != 0 {
		t.Errorf("offsetof(start) = %d, want 0", off)
	}
	if off := unsafe.Offsetof(e.end); off != 4 {
		t.Errorf("offsetof(end) = %d, want 4", off)
	}
	if off := unsafe.Offsetof(e.score); off != 8 {
		t.Errorf("offsetof(score) = %d, want 8", off)
	}
	if off := unsafe.Offsetof(e.label); off != 16 {
		t.Errorf("offsetof(label) = %d, want 16", off)
	}
}

func TestDefaultConfigSupportedEntities(t *testing.T) {
	got := DefaultConfig("x.gguf").SupportedEntities()
	want := []string{
		"ACCOUNT_NUMBER", entities.DateTime, entities.EmailAddress,
		entities.Location, entities.Person, entities.PhoneNumber,
		"SECRET", entities.URL,
	}
	// want is already sorted (ACCOUNT_NUMBER < DATE_TIME < EMAIL_ADDRESS <
	// LOCATION < PERSON < PHONE_NUMBER < SECRET < URL).
	if len(got) != len(want) {
		t.Fatalf("supported entities = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("supported entities = %v, want %v", got, want)
		}
	}
}

func TestToNerSpan(t *testing.T) {
	text := "My name is Alice Smith"
	eng := newFakeEngine(&fakeClassifier{}, Config{})

	t.Run("maps model label to canonical entity", func(t *testing.T) {
		span, ok := eng.toNerSpan(text, rawEntity{label: "private_person", start: 11, end: 22, score: 0.97})
		if !ok || span.EntityType != entities.Person || span.Start != 11 || span.End != 22 {
			t.Fatalf("got (%+v, %v), want PERSON [11,22)", span, ok)
		}
	})

	t.Run("unmapped label normalized and kept", func(t *testing.T) {
		span, ok := eng.toNerSpan(text, rawEntity{label: "crypto_wallet", start: 0, end: 2, score: 0.8})
		if !ok || span.EntityType != "CRYPTO_WALLET" {
			t.Fatalf("got (%+v, %v), want CRYPTO_WALLET kept", span, ok)
		}
	})

	t.Run("ignore list drops by model label and by mapped name", func(t *testing.T) {
		cfg := Config{LabelsToIgnore: []string{"secret", entities.URL}}
		ig := newFakeEngine(&fakeClassifier{}, cfg)
		if _, ok := ig.toNerSpan(text, rawEntity{label: "secret", start: 0, end: 2, score: 0.9}); ok {
			t.Fatal("model label 'secret' should be dropped")
		}
		if _, ok := ig.toNerSpan(text, rawEntity{label: "private_url", start: 0, end: 2, score: 0.9}); ok {
			t.Fatal("mapped entity URL should be dropped")
		}
	})

	t.Run("score clamped and zero dropped", func(t *testing.T) {
		span, ok := eng.toNerSpan(text, rawEntity{label: "private_person", start: 0, end: 2, score: 1.5})
		if !ok || span.Score != analyzer.MaxScore {
			t.Fatalf("got (%+v, %v), want score clamped to 1.0", span, ok)
		}
		if _, ok := eng.toNerSpan(text, rawEntity{label: "private_person", start: 0, end: 2, score: 0}); ok {
			t.Fatal("zero score should be dropped")
		}
	})

	t.Run("invalid span dropped, oversized span clamped", func(t *testing.T) {
		if _, ok := eng.toNerSpan(text, rawEntity{label: "private_person", start: 9, end: 9, score: 0.9}); ok {
			t.Fatal("empty span should be dropped")
		}
		span, ok := eng.toNerSpan(text, rawEntity{label: "private_person", start: -5, end: 999, score: 0.9})
		if !ok || span.Start != 0 || span.End != len(text) {
			t.Fatalf("got (%+v, %v), want span clamped to [0,%d)", span, ok, len(text))
		}
	})
}

func TestEngineWithAnalyzer(t *testing.T) {
	text := "Contact John Doe at jdoe@example.com"
	fake := &fakeClassifier{entities: []rawEntity{
		{label: "private_person", start: 8, end: 16, score: 0.96},
		{label: "private_email", start: 20, end: 36, score: 0.99},
	}}
	nlp := newFakeEngine(fake, Config{})

	reg := analyzer.NewRegistry("en")
	recognizers.LoadDefaults(reg, "en")
	reg.Add("en", nlp.Recognizer("en"))
	eng := analyzer.NewEngine(reg, []string{"en"})
	eng.SetNlpEngine(nlp)

	results := eng.Analyze(text, analyzer.Options{})

	if fake.calls != 1 {
		t.Errorf("model ran %d times, want 1 (shared artifacts)", fake.calls)
	}
	byType := map[string]string{}
	for _, r := range results {
		byType[r.EntityType] = r.Text
	}
	if byType[entities.Person] != "John Doe" {
		t.Errorf("PERSON = %q, want 'John Doe' (all: %v)", byType[entities.Person], byType)
	}
	// The model and the pattern recognizer both report the email; same-type
	// dedup must collapse them to one span.
	if byType[entities.EmailAddress] != "jdoe@example.com" {
		t.Errorf("EMAIL_ADDRESS = %q, want 'jdoe@example.com'", byType[entities.EmailAddress])
	}
	emails := 0
	for _, r := range results {
		if r.EntityType == entities.EmailAddress {
			emails++
		}
	}
	if emails != 1 {
		t.Errorf("EMAIL_ADDRESS results = %d, want 1 after dedup", emails)
	}
}

func TestEngineInferenceFailureDegrades(t *testing.T) {
	fake := &fakeClassifier{err: errors.New("model exploded")}
	nlp := newFakeEngine(fake, Config{})

	reg := analyzer.NewRegistry("en")
	recognizers.LoadDefaults(reg, "en")
	reg.Add("en", nlp.Recognizer("en"))
	eng := analyzer.NewEngine(reg, []string{"en"})
	eng.SetNlpEngine(nlp)

	results := eng.Analyze("mail jane@example.com", analyzer.Options{})
	found := false
	for _, r := range results {
		if r.EntityType == entities.EmailAddress {
			found = true
		}
		if r.RecognizerName == "PrivacyFilterRecognizer" {
			t.Errorf("unexpected model result %+v after inference failure", r)
		}
	}
	if !found {
		t.Error("pattern EMAIL_ADDRESS should survive an inference failure")
	}
}

func TestRecognizerEntityFilter(t *testing.T) {
	nlp := newFakeEngine(&fakeClassifier{}, Config{})
	rec := nlp.Recognizer("en")

	artifacts := &analyzer.NlpArtifacts{Ents: []analyzer.NerSpan{
		{EntityType: entities.Person, Start: 0, End: 4, Score: 0.9},
		{EntityType: "SECRET", Start: 10, End: 14, Score: 0.8},
	}}
	got := rec.AnalyzeWithArtifacts("John ..... key1", []string{"SECRET"}, artifacts)
	if len(got) != 1 || got[0].EntityType != "SECRET" {
		t.Fatalf("want single SECRET, got %+v", got)
	}
}

// TestLivePrivacyFilter runs the real shared library and model end to end.
// Gated: requires ALCATRAZ_PF_LIVE=1, $PF_LIBRARY pointing at libpf and
// $PF_MODEL pointing at a privacy-filter GGUF.
func TestLivePrivacyFilter(t *testing.T) {
	if os.Getenv("ALCATRAZ_PF_LIVE") != "1" {
		t.Skip("set ALCATRAZ_PF_LIVE=1 (plus PF_LIBRARY and PF_MODEL) to run the live test")
	}
	model := os.Getenv("PF_MODEL")
	if model == "" {
		t.Fatal("PF_MODEL must point at a privacy-filter GGUF file")
	}

	nlp, err := New(DefaultConfig(model))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer nlp.Close()

	reg := analyzer.NewRegistry("en")
	recognizers.LoadDefaults(reg, "en")
	reg.Add("en", nlp.Recognizer("en"))
	eng := analyzer.NewEngine(reg, []string{"en"})
	eng.SetNlpEngine(nlp)

	text := "Contact John Doe at jdoe@example.com or +1 212 555 0100"
	results := eng.Analyze(text, analyzer.Options{})

	byType := map[string]string{}
	for _, r := range results {
		byType[r.EntityType] = r.Text
		if text[r.Start:r.End] != r.Text {
			t.Errorf("offset invariant broken: [%d:%d] = %q, Text = %q",
				r.Start, r.End, text[r.Start:r.End], r.Text)
		}
	}
	if byType[entities.Person] == "" {
		t.Errorf("no PERSON detected (all: %v)", byType)
	}
	if byType[entities.EmailAddress] != "jdoe@example.com" {
		t.Errorf("EMAIL_ADDRESS = %q, want 'jdoe@example.com'", byType[entities.EmailAddress])
	}

	// Byte-offset invariant on multi-byte input.
	utf8Text := "Meu nome é José Núñez, moro em São Paulo"
	for _, r := range eng.Analyze(utf8Text, analyzer.Options{}) {
		if utf8Text[r.Start:r.End] != r.Text {
			t.Errorf("utf8 offset invariant broken: [%d:%d] = %q, Text = %q",
				r.Start, r.End, utf8Text[r.Start:r.End], r.Text)
		}
	}
}
