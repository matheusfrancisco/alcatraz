package pfilter

import "unicode/utf8"

// byteSpan validates a model-reported span against the analyzed text and
// returns a safe byte span: bounds are clamped and boundaries snapped back
// to the start of the rune containing them, so a misreported offset can
// never produce an invalid slice. privacy-filter.cpp guarantees byte
// offsets into the original UTF-8 text, so this is a defensive no-op on
// well-formed output. The third return value is false when no usable span
// remains.
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
