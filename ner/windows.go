package ner

// This file implements windowed inference: texts whose token count exceeds
// the model's sequence limit are split into overlapping model-sized windows,
// each window is analyzed independently (batched with everything else), and
// the per-window spans are merged back. Without it, texts beyond the limit
// either fail inference outright (the pure-Go tokenizer applies no
// truncation for token-classification pipelines) or are silently truncated
// (the rust tokenizer), losing every entity past ~2KB of text.

import (
	"sort"

	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/hoophq/alcatraz/analyzer"
)

const (
	// windowOverlapTokens is how many content tokens consecutive windows
	// share. An entity that straddles a window boundary is cut in one
	// window but appears whole in the next as long as it is shorter than
	// the overlap; mergeSpans then drops the cut fragment.
	windowOverlapTokens = 32

	// fallbackBytesPerToken sizes byte-based windows when the tokenizer is
	// not introspectable (rust tokenizer builds). Three bytes per token is
	// conservative for prose (~4.5) without exploding the window count;
	// the rust tokenizer truncates over-long windows instead of failing,
	// so an underestimate degrades gracefully.
	fallbackBytesPerToken = 3
)

// textWindow is one model-sized slice of a folded text: byte offsets
// [start, end) into the folded string.
type textWindow struct {
	start, end int
}

// windows splits a folded text into inference-ready windows. Texts within
// the model's token budget yield a single window covering the whole text.
func (e *Engine) windows(folded string) []textWindow {
	tk := e.goTokenizer()
	if tk == nil {
		size := e.tokenBudget * fallbackBytesPerToken
		return byteWindows(len(folded), size, windowOverlapTokens*fallbackBytesPerToken)
	}
	return tokenWindows(tk, folded, e.tokenBudget)
}

// goTokenizer returns the pipeline's tokenizer when it is the pure-Go
// implementation (the default backend). Rust tokenizer builds return nil and
// fall back to byte-estimate windows.
func (e *Engine) goTokenizer() api.Tokenizer {
	tk := e.pipeline.Model.Tokenizer
	if tk == nil || tk.GoTokenizer == nil {
		return nil
	}
	return tk.GoTokenizer.Tokenizer
}

// tokenWindows builds windows by tokenizing the text once and slicing the
// content-token stream into runs that re-encode within totalBudget tokens
// (special tokens included). Each candidate window is verified by re-encoding
// its substring — cutting mid-word can inflate the token count, so the run is
// shrunk until it provably fits. Windows are built sequentially and each next
// window starts windowOverlapTokens before the previous one's actual end, so
// shrinking never opens a gap.
func tokenWindows(tk api.Tokenizer, folded string, totalBudget int) []textWindow {
	enc := tk.EncodeWithAnnotations(folded)
	spans := contentSpans(enc)
	if len(enc.IDs) <= totalBudget || len(spans) == 0 {
		return []textWindow{{0, len(folded)}}
	}
	// Special tokens ([CLS]/[SEP], model-dependent) occupy part of the
	// budget in every window.
	specials := len(enc.IDs) - len(spans)
	contentBudget := totalBudget - specials
	if contentBudget < 1 {
		contentBudget = 1
	}

	var wins []textWindow
	start := 0
	for start < len(spans) {
		take := min(contentBudget, len(spans)-start)
		for take > 1 {
			w := textWindow{spans[start].Start, spans[start+take-1].End}
			excess := len(tk.Encode(folded[w.start:w.end])) - totalBudget
			if excess <= 0 {
				break
			}
			take -= max(1, excess)
		}
		if take < 1 {
			take = 1
		}
		wins = append(wins, textWindow{spans[start].Start, spans[start+take-1].End})
		if start+take >= len(spans) {
			break
		}
		start += max(1, take-windowOverlapTokens)
	}
	return wins
}

// contentSpans returns the byte spans of non-special tokens, in text order.
// Special tokens are identified by the tokenizer's mask when present, and by
// their empty span otherwise.
func contentSpans(enc api.AnnotatedEncoding) []api.TokenSpan {
	spans := make([]api.TokenSpan, 0, len(enc.Spans))
	for i, s := range enc.Spans {
		if len(enc.SpecialTokensMask) == len(enc.Spans) && enc.SpecialTokensMask[i] == 1 {
			continue
		}
		if s.End <= s.Start {
			continue
		}
		spans = append(spans, s)
	}
	return spans
}

// byteWindows splits [0, n) into fixed-size windows with the given overlap.
// The folded text is always single-byte-per-rune ASCII, so any byte offset
// is a valid cut point.
func byteWindows(n, size, overlap int) []textWindow {
	if size < 1 {
		size = 1
	}
	if n <= size {
		return []textWindow{{0, n}}
	}
	step := size - overlap
	if step < 1 {
		step = 1
	}
	var wins []textWindow
	for s := 0; ; s += step {
		e := min(s+size, n)
		wins = append(wins, textWindow{s, e})
		if e == n {
			return wins
		}
	}
}

// mergeSpans resolves the duplicates that overlapping windows produce: exact
// duplicates collapse keeping the highest score, and a span contained within
// a wider span of the same entity type (a fragment cut at a window boundary)
// is dropped. The result is sorted by Start, then End.
func mergeSpans(spans []analyzer.NerSpan) []analyzer.NerSpan {
	if len(spans) < 2 {
		return spans
	}
	sort.Slice(spans, func(i, j int) bool {
		if spans[i].Start != spans[j].Start {
			return spans[i].Start < spans[j].Start
		}
		return spans[i].End > spans[j].End
	})
	// widest[type] is the largest End among kept spans of that type; since
	// spans are visited in ascending Start order, a span with End within it
	// is fully contained in an earlier, wider span.
	widest := map[string]int{}
	out := spans[:0]
	for _, s := range spans {
		if len(out) > 0 {
			last := &out[len(out)-1]
			if last.EntityType == s.EntityType && last.Start == s.Start && last.End == s.End {
				if s.Score > last.Score {
					last.Score = s.Score
				}
				continue
			}
		}
		if end, ok := widest[s.EntityType]; ok && s.End <= end {
			continue
		}
		out = append(out, s)
		if s.End > widest[s.EntityType] {
			widest[s.EntityType] = s.End
		}
	}
	return out
}
