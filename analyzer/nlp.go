package analyzer

// This file defines the NLP seam: the types through which a statistical
// (ML/NER) backend plugs into the engine. The core stays dependency-free
// implementations of NlpEngine live in separate modules (e.g.
// github.com/hoophq/alcatraz/ner) so importers of the core never pull in a
// model runtime.
//
// The design follows the single-inference-pass model: the engine runs the
// NlpEngine at most once per Analyze call and shares the resulting
// NlpArtifacts with every recognizer that consumes them, so multiple
// artifact-aware recognizers (NER, context scoring, ...) never trigger
// duplicate model runs.

// Token is a single token of the analyzed text with its byte span.
// text[Start:End] yields the token.
type Token struct {
	// Text is the token's surface form.
	Text string
	// Start and End are byte offsets into the analyzed text: [Start, End).
	Start int
	End   int
}

// NerSpan is one model-detected entity span. EntityType is a canonical
// entities.* name (label mapping is the NlpEngine implementation's job) and
// Score is on the engine's [MinScore, MaxScore] scale.
type NerSpan struct {
	// EntityType is the canonical entity name, e.g. "PERSON".
	EntityType string
	// Start and End are byte offsets into the analyzed text: [Start, End).
	Start int
	End   int
	// Score is the detection confidence in [MinScore, MaxScore].
	Score float64
}

// NlpArtifacts is the output of one NLP pass over a text: tokens and
// model-detected entity spans. It is computed at most once per Analyze call
// and handed to every ArtifactRecognizer.
type NlpArtifacts struct {
	// Tokens holds the tokenization of the analyzed text, when the engine
	// provides one. It may be nil for engines that only do NER.
	Tokens []Token
	// Ents holds the model-detected entity spans.
	Ents []NerSpan
}

// NlpEngine produces NlpArtifacts for a text. Implementations wrap a model
// runtime and live outside the core so the core stays dependency-free.
type NlpEngine interface {
	// ProcessText runs the NLP pipeline over text. The language is the
	// engine-level analysis language (ISO 639-1).
	ProcessText(text, language string) (*NlpArtifacts, error)
}

// ArtifactRecognizer is an optional extension of Recognizer for detectors
// that consume precomputed NlpArtifacts. The engine detects it by type
// assertion: when an NlpEngine is configured (Engine.SetNlpEngine) and at
// least one applicable recognizer implements this interface, the engine runs
// the NLP pipeline once and calls AnalyzeWithArtifacts instead of Analyze.
type ArtifactRecognizer interface {
	Recognizer
	// AnalyzeWithArtifacts is Analyze with the shared artifacts of the
	// current Analyze call. The entities parameter has the same filter
	// semantics as Recognizer.Analyze. Returned offsets are byte indices.
	AnalyzeWithArtifacts(text string, entities []string, artifacts *NlpArtifacts) []Result
}
