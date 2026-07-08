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

// fakeBatchNlpEngine implements BatchNlpEngine, returning per-text artifacts
// keyed by the text itself and counting batch calls.
type fakeBatchNlpEngine struct {
	fakeNlpEngine
	batchCalls int
	perText    map[string]*NlpArtifacts
	batchErr   error
}

func (f *fakeBatchNlpEngine) ProcessTexts(texts []string, language string) ([]*NlpArtifacts, error) {
	f.batchCalls++
	if f.batchErr != nil {
		return nil, f.batchErr
	}
	out := make([]*NlpArtifacts, len(texts))
	for i, t := range texts {
		if a, ok := f.perText[t]; ok {
			out[i] = a
		} else {
			out[i] = &NlpArtifacts{}
		}
	}
	return out, nil
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

	t.Run("AnalyzeBatch runs a batch engine once for all texts", func(t *testing.T) {
		texts := []string{"call John Smith now", "meet in Berlin"}
		nlp := &fakeBatchNlpEngine{perText: map[string]*NlpArtifacts{
			texts[0]: {Ents: []NerSpan{person}},
			texts[1]: {Ents: []NerSpan{{EntityType: "LOCATION", Start: 8, End: 14, Score: 0.9}}},
		}}
		a := &fakeArtifactRecognizer{name: "ner", entities: []string{"PERSON", "LOCATION"}}
		eng := newNlpTestEngine(a)
		eng.SetNlpEngine(nlp)

		got := eng.AnalyzeBatch(texts, Options{})
		if nlp.batchCalls != 1 || nlp.calls != 0 {
			t.Errorf("batchCalls=%d calls=%d, want 1 and 0", nlp.batchCalls, nlp.calls)
		}
		if len(got) != 2 {
			t.Fatalf("want 2 result slices, got %d", len(got))
		}
		if len(got[0]) != 1 || got[0][0].Text != "John Smith" {
			t.Errorf("text 0: want PERSON 'John Smith', got %+v", got[0])
		}
		if len(got[1]) != 1 || got[1][0].Text != "Berlin" {
			t.Errorf("text 1: want LOCATION 'Berlin', got %+v", got[1])
		}
	})

	t.Run("AnalyzeBatch matches per-text Analyze results", func(t *testing.T) {
		texts := []string{"call John Smith now", "nothing here"}
		nlp := &fakeBatchNlpEngine{perText: map[string]*NlpArtifacts{
			texts[0]: {Ents: []NerSpan{person}},
		}}
		nlp.artifacts = &NlpArtifacts{Ents: []NerSpan{person}}
		a := &fakeArtifactRecognizer{name: "ner", entities: []string{"PERSON"}}
		eng := newNlpTestEngine(a)
		eng.SetNlpEngine(nlp)

		batch := eng.AnalyzeBatch(texts, Options{})
		single := eng.Analyze(texts[0], Options{})
		if len(batch[0]) != len(single) || batch[0][0] != single[0] {
			t.Errorf("batch result %+v != single result %+v", batch[0], single)
		}
		if len(batch[1]) != 0 {
			t.Errorf("text without entities: want none, got %+v", batch[1])
		}
	})

	t.Run("AnalyzeBatch with non-batch engine falls back to per-text passes", func(t *testing.T) {
		nlp := &fakeNlpEngine{artifacts: &NlpArtifacts{Ents: []NerSpan{person}}}
		a := &fakeArtifactRecognizer{name: "ner", entities: []string{"PERSON"}}
		eng := newNlpTestEngine(a)
		eng.SetNlpEngine(nlp)

		got := eng.AnalyzeBatch([]string{text, text}, Options{})
		if nlp.calls != 2 {
			t.Errorf("NLP engine ran %d times, want 2", nlp.calls)
		}
		if len(got) != 2 || len(got[0]) != 1 || len(got[1]) != 1 {
			t.Fatalf("want one PERSON per text, got %+v", got)
		}
	})

	t.Run("AnalyzeBatch batch failure degrades to plain Analyze", func(t *testing.T) {
		nlp := &fakeBatchNlpEngine{batchErr: errors.New("model exploded")}
		a := &fakeArtifactRecognizer{name: "ner", entities: []string{"PERSON"}}
		eng := newNlpTestEngine(a)
		eng.SetNlpEngine(nlp)

		got := eng.AnalyzeBatch([]string{text, text}, Options{})
		if a.plainAnalyzed != 2 {
			t.Errorf("plain Analyze ran %d times, want 2", a.plainAnalyzed)
		}
		if len(got) != 2 || len(got[0]) != 0 || len(got[1]) != 0 {
			t.Fatalf("want empty results, got %+v", got)
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
