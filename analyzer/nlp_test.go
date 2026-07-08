package analyzer

import (
	"errors"
	"testing"
)

// fakeNlpEngine records how many times it ran and returns fixed artifacts.
type fakeNlpEngine struct {
	calls     int
	artifacts *NlpArtifacts
	err       error
}

func (f *fakeNlpEngine) ProcessText(text, language string) (*NlpArtifacts, error) {
	f.calls++
	return f.artifacts, f.err
}

// fakeArtifactRecognizer emits the artifacts' spans, or nothing via plain
// Analyze so tests can tell which path the engine took.
type fakeArtifactRecognizer struct {
	name          string
	entities      []string
	plainAnalyzed int
}

func (r *fakeArtifactRecognizer) Name() string                { return r.name }
func (r *fakeArtifactRecognizer) SupportedEntities() []string { return r.entities }
func (r *fakeArtifactRecognizer) SupportedLanguage() string   { return "en" }

func (r *fakeArtifactRecognizer) Analyze(text string, entities []string) []Result {
	r.plainAnalyzed++
	return nil
}

func (r *fakeArtifactRecognizer) AnalyzeWithArtifacts(text string, entities []string, a *NlpArtifacts) []Result {
	var out []Result
	for _, ent := range a.Ents {
		if entities != nil && !supportsAny([]string{ent.EntityType}, entities) {
			continue
		}
		out = append(out, Result{
			EntityType:     ent.EntityType,
			Start:          ent.Start,
			End:            ent.End,
			Score:          ent.Score,
			RecognizerName: r.name,
		})
	}
	return out
}

func newNlpTestEngine(recs ...Recognizer) *Engine {
	reg := NewRegistry("en")
	for _, r := range recs {
		reg.Add("en", r)
	}
	return NewEngine(reg, []string{"en"})
}

func TestEngineSharedArtifacts(t *testing.T) {
	text := "call John Smith now"
	person := NerSpan{EntityType: "PERSON", Start: 5, End: 15, Score: 0.85}

	t.Run("single NLP pass shared across artifact recognizers", func(t *testing.T) {
		nlp := &fakeNlpEngine{artifacts: &NlpArtifacts{Ents: []NerSpan{person}}}
		a := &fakeArtifactRecognizer{name: "ner-a", entities: []string{"PERSON"}}
		b := &fakeArtifactRecognizer{name: "ner-b", entities: []string{"LOCATION"}}
		eng := newNlpTestEngine(a, b)
		eng.SetNlpEngine(nlp)

		got := eng.Analyze(text, Options{})
		if nlp.calls != 1 {
			t.Errorf("NLP engine ran %d times, want 1", nlp.calls)
		}
		if len(got) != 1 || got[0].EntityType != "PERSON" || got[0].Text != "John Smith" {
			t.Fatalf("want single PERSON 'John Smith', got %+v", got)
		}
		if a.plainAnalyzed != 0 || b.plainAnalyzed != 0 {
			t.Error("plain Analyze should not run when artifacts are available")
		}
	})

	t.Run("no NLP engine falls back to plain Analyze", func(t *testing.T) {
		a := &fakeArtifactRecognizer{name: "ner", entities: []string{"PERSON"}}
		eng := newNlpTestEngine(a)

		got := eng.Analyze(text, Options{})
		if a.plainAnalyzed != 1 {
			t.Errorf("plain Analyze ran %d times, want 1", a.plainAnalyzed)
		}
		if len(got) != 0 {
			t.Fatalf("want no results, got %+v", got)
		}
	})

	t.Run("no artifact recognizer means no NLP pass", func(t *testing.T) {
		nlp := &fakeNlpEngine{artifacts: &NlpArtifacts{}}
		rec := NewPatternRecognizer("EmailRule", "EMAIL_ADDRESS", "en",
			[]*Pattern{MustPattern("email", `\S+@\S+\.\S+`, 0.5)})
		eng := newNlpTestEngine(rec)
		eng.SetNlpEngine(nlp)

		eng.Analyze("mail jane@example.com", Options{})
		if nlp.calls != 0 {
			t.Errorf("NLP engine ran %d times, want 0", nlp.calls)
		}
	})

	t.Run("entity filter skips artifact recognizer and NLP pass", func(t *testing.T) {
		nlp := &fakeNlpEngine{artifacts: &NlpArtifacts{Ents: []NerSpan{person}}}
		a := &fakeArtifactRecognizer{name: "ner", entities: []string{"PERSON"}}
		eng := newNlpTestEngine(a)
		eng.SetNlpEngine(nlp)

		got := eng.Analyze(text, Options{Entities: []string{"EMAIL_ADDRESS"}})
		if nlp.calls != 0 {
			t.Errorf("NLP engine ran %d times, want 0", nlp.calls)
		}
		if len(got) != 0 {
			t.Fatalf("want no results, got %+v", got)
		}
	})

	t.Run("inference failure degrades to plain Analyze", func(t *testing.T) {
		nlp := &fakeNlpEngine{err: errors.New("model exploded")}
		a := &fakeArtifactRecognizer{name: "ner", entities: []string{"PERSON"}}
		eng := newNlpTestEngine(a)
		eng.SetNlpEngine(nlp)

		got := eng.Analyze(text, Options{})
		if a.plainAnalyzed != 1 {
			t.Errorf("plain Analyze ran %d times, want 1", a.plainAnalyzed)
		}
		if len(got) != 0 {
			t.Fatalf("want no results, got %+v", got)
		}
	})

	t.Run("NER results flow through threshold and dedup", func(t *testing.T) {
		nlp := &fakeNlpEngine{artifacts: &NlpArtifacts{Ents: []NerSpan{
			person,
			{EntityType: "PERSON", Start: 5, End: 9, Score: 0.3}, // contained, lower score
		}}}
		a := &fakeArtifactRecognizer{name: "ner", entities: []string{"PERSON"}}
		eng := newNlpTestEngine(a)
		eng.SetNlpEngine(nlp)

		threshold := 0.5
		got := eng.Analyze(text, Options{Threshold: &threshold})
		if len(got) != 1 || got[0].Score != 0.85 {
			t.Fatalf("want single PERSON at 0.85 after dedup+threshold, got %+v", got)
		}
	})
}
