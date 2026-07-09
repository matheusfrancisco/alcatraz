package ner

import (
	"strings"
	"testing"

	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/hoophq/alcatraz/analyzer"
	"github.com/hoophq/alcatraz/entities"
)

// fakeTokenizer tokenizes on whitespace (one token per word) and adds two
// special tokens, mimicking a [CLS] ... [SEP] encoding. It implements just
// enough of api.Tokenizer for the windowing code.
type fakeTokenizer struct{}

func (fakeTokenizer) Encode(text string) []int {
	return make([]int, len(strings.Fields(text))+2)
}

func (fakeTokenizer) EncodeWithAnnotations(text string) api.AnnotatedEncoding {
	var spans []api.TokenSpan
	i := 0
	for i < len(text) {
		if text[i] == ' ' || text[i] == '\n' || text[i] == '\t' {
			i++
			continue
		}
		start := i
		for i < len(text) && text[i] != ' ' && text[i] != '\n' && text[i] != '\t' {
			i++
		}
		spans = append(spans, api.TokenSpan{Start: start, End: i})
	}
	enc := api.AnnotatedEncoding{
		IDs:               make([]int, len(spans)+2),
		Spans:             make([]api.TokenSpan, 0, len(spans)+2),
		SpecialTokensMask: make([]int, 0, len(spans)+2),
	}
	enc.Spans = append(enc.Spans, api.TokenSpan{})
	enc.SpecialTokensMask = append(enc.SpecialTokensMask, 1)
	for _, s := range spans {
		enc.Spans = append(enc.Spans, s)
		enc.SpecialTokensMask = append(enc.SpecialTokensMask, 0)
	}
	enc.Spans = append(enc.Spans, api.TokenSpan{})
	enc.SpecialTokensMask = append(enc.SpecialTokensMask, 1)
	return enc
}

func (fakeTokenizer) With(api.EncodeOptions) error                 { return nil }
func (fakeTokenizer) Decode([]int) string                          { return "" }
func (fakeTokenizer) SpecialTokenID(api.SpecialToken) (int, error) { return 0, api.ErrNotImplemented }
func (fakeTokenizer) Normalize(s string) string                    { return s }
func (fakeTokenizer) VocabSize() int                               { return 0 }
func (fakeTokenizer) Config() *api.Config                          { return nil }

func words(n int) string {
	w := make([]string, n)
	for i := range w {
		w[i] = "word"
	}
	return strings.Join(w, " ")
}

func TestTokenWindows(t *testing.T) {
	tk := fakeTokenizer{}

	t.Run("short text is a single full window", func(t *testing.T) {
		text := words(10)
		wins := tokenWindows(tk, text, 512)
		if len(wins) != 1 || wins[0].start != 0 || wins[0].end != len(text) {
			t.Fatalf("wins = %+v, want single window over full text", wins)
		}
	})

	t.Run("long text splits into overlapping windows covering everything", func(t *testing.T) {
		// 1000 words, budget 512 tokens (510 content after 2 specials).
		text := words(1000)
		wins := tokenWindows(tk, text, 512)
		if len(wins) < 2 {
			t.Fatalf("want multiple windows, got %+v", wins)
		}
		if wins[0].start != 0 {
			t.Errorf("first window starts at %d, want 0", wins[0].start)
		}
		if wins[len(wins)-1].end != len(text) {
			t.Errorf("last window ends at %d, want %d", wins[len(wins)-1].end, len(text))
		}
		for i, w := range wins {
			total := len(tk.Encode(text[w.start:w.end]))
			if total > 512 {
				t.Errorf("window %d re-encodes to %d tokens > 512", i, total)
			}
			if i > 0 && w.start >= wins[i-1].end {
				t.Errorf("gap between window %d (end %d) and %d (start %d)",
					i-1, wins[i-1].end, i, w.start)
			}
		}
	})

	t.Run("windows never cut words", func(t *testing.T) {
		text := words(600)
		for _, w := range tokenWindows(tk, text, 512) {
			body := text[w.start:w.end]
			if strings.HasPrefix(body, " ") || strings.HasSuffix(body, " ") ||
				len(body)%5 != 4 { // "word" repeated joins to length 5k+4
				t.Fatalf("window %+v does not align to word boundaries: %q…", w, body[:10])
			}
		}
	})
}

func TestByteWindows(t *testing.T) {
	t.Run("fits in one window", func(t *testing.T) {
		wins := byteWindows(100, 200, 20)
		if len(wins) != 1 || wins[0] != (textWindow{0, 100}) {
			t.Fatalf("wins = %+v", wins)
		}
	})

	t.Run("splits with overlap and full coverage", func(t *testing.T) {
		wins := byteWindows(1000, 300, 50)
		if wins[0].start != 0 || wins[len(wins)-1].end != 1000 {
			t.Fatalf("coverage broken: %+v", wins)
		}
		for i := 1; i < len(wins); i++ {
			if wins[i].start != wins[i-1].start+250 {
				t.Fatalf("step wrong at %d: %+v", i, wins)
			}
			if wins[i].start >= wins[i-1].end {
				t.Fatalf("gap at %d: %+v", i, wins)
			}
		}
	})

	t.Run("degenerate sizes never loop forever", func(t *testing.T) {
		wins := byteWindows(10, 1, 5)
		if wins[len(wins)-1].end != 10 {
			t.Fatalf("coverage broken: %+v", wins)
		}
	})
}

func TestMergeSpans(t *testing.T) {
	t.Run("exact duplicate keeps max score", func(t *testing.T) {
		got := mergeSpans([]analyzer.NerSpan{
			{EntityType: entities.Person, Start: 10, End: 20, Score: 0.7},
			{EntityType: entities.Person, Start: 10, End: 20, Score: 0.9},
		})
		if len(got) != 1 || got[0].Score != 0.9 {
			t.Fatalf("got %+v", got)
		}
	})

	t.Run("fragment contained in wider same-type span is dropped", func(t *testing.T) {
		got := mergeSpans([]analyzer.NerSpan{
			{EntityType: entities.Person, Start: 15, End: 20, Score: 0.9},
			{EntityType: entities.Person, Start: 10, End: 20, Score: 0.8},
		})
		if len(got) != 1 || got[0].Start != 10 || got[0].End != 20 {
			t.Fatalf("got %+v", got)
		}
	})

	t.Run("contained span of a different type is kept", func(t *testing.T) {
		got := mergeSpans([]analyzer.NerSpan{
			{EntityType: entities.Person, Start: 10, End: 30, Score: 0.9},
			{EntityType: entities.Location, Start: 15, End: 20, Score: 0.8},
		})
		if len(got) != 2 {
			t.Fatalf("got %+v", got)
		}
	})

	t.Run("disjoint spans pass through sorted", func(t *testing.T) {
		got := mergeSpans([]analyzer.NerSpan{
			{EntityType: entities.Person, Start: 50, End: 60, Score: 0.9},
			{EntityType: entities.Person, Start: 10, End: 20, Score: 0.8},
		})
		if len(got) != 2 || got[0].Start != 10 || got[1].Start != 50 {
			t.Fatalf("got %+v", got)
		}
	})
}
