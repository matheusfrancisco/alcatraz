package ner

import (
	"strings"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// foldASCII returns an ASCII-only rendering of text with exactly one byte per
// input rune, plus a table mapping each folded byte position to the byte
// offset of the originating rune in text. A nil table means text was already
// pure ASCII and was returned unchanged.
//
// The model runs on the folded text. This works around a span-tracking bug in
// the pure-Go tokenizer used by hugot's default backend (go-huggingface
// hftokenizer): its BertNormalizer builds the normalized→original offset
// table with one entry per rune instead of one per byte, so any multi-byte
// character shifts every subsequent token span and the reported entity spans
// overrun the text. On single-byte-per-rune input the two units coincide and
// the bug cannot trigger; accents are stripped via NFD so Latin text keeps
// its shape for the model ("José" → "Jose"), and runes with no ASCII
// decomposition become a letter placeholder.
func foldASCII(text string) (string, []int) {
	isASCII := true
	for i := 0; i < len(text); i++ {
		if text[i] >= utf8.RuneSelf {
			isASCII = false
			break
		}
	}
	if isASCII {
		return text, nil
	}

	var folded strings.Builder
	folded.Grow(len(text))
	offsets := make([]int, 0, len(text))
	for i, r := range text {
		folded.WriteByte(foldRune(r))
		offsets = append(offsets, i)
	}
	return folded.String(), offsets
}

// foldRune reduces a rune to a single ASCII byte: ASCII passes through,
// accented characters lose their marks via NFD decomposition, and everything
// else becomes a letter placeholder so words stay word-shaped for the model.
func foldRune(r rune) byte {
	if r < utf8.RuneSelf {
		return byte(r)
	}
	for _, d := range norm.NFD.String(string(r)) {
		if d < utf8.RuneSelf {
			return byte(d)
		}
	}
	return 'x'
}

// remapSpan converts a span reported against the folded text back to byte
// offsets in the original text. offsets is the table from foldASCII (nil
// means identity) and textLen is len(original text).
func remapSpan(offsets []int, textLen, start, end int) (int, int) {
	if offsets == nil {
		return start, end
	}
	if start < 0 || start >= len(offsets) || end <= 0 {
		return 0, 0
	}
	origStart := offsets[start]
	// A folded end is exclusive: it is the position after the span's last
	// byte, i.e. the start of the next rune in the original text.
	origEnd := textLen
	if end < len(offsets) {
		origEnd = offsets[end]
	}
	return origStart, origEnd
}

// byteSpan validates a span against the analyzed text and returns a safe
// byte span: bounds are clamped and boundaries snapped back to the start of
// the rune containing them, so a misaligned offset can never produce an
// invalid slice. The third return value is false when no usable span remains.
func byteSpan(text string, start, end int) (int, int, bool) {
	if start < 0 {
		start = 0
	}
	if end > len(text) {
		end = len(text)
	}
	for start > 0 && start < len(text) && !utf8.RuneStart(text[start]) {
		start--
	}
	for end > 0 && end < len(text) && !utf8.RuneStart(text[end]) {
		end--
	}
	if start >= end {
		return 0, 0, false
	}
	return start, end, true
}
