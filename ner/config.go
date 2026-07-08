package ner

import (
	"sort"

	"github.com/hoophq/alcatraz/entities"
)

// Config configures the NER engine: which model to run and how its labels
// map onto alcatraz entity types.
//
// The label mapping plays the same role as Presidio's NerModelConfiguration:
// it is the single place where model-specific label names (PER, LOC, GPE, …)
// become canonical entities.* names, and where noisy labels are dropped or
// down-weighted.
type Config struct {
	// Model is a Hugging Face model id of an ONNX token-classification
	// export, e.g. "KnightsAnalytics/distilbert-NER". It is downloaded into
	// ModelsDir on first use. Ignored when ModelPath is set.
	Model string
	// ModelPath is a local directory holding the model files (model.onnx,
	// tokenizer.json, config.json). When set, no download happens.
	ModelPath string
	// ModelsDir is where downloaded models are stored. Defaults to
	// "alcatraz/models" under the user cache directory.
	ModelsDir string
	// OnnxFilename selects the .onnx file when the model repository holds
	// more than one. Empty means the single .onnx file in the repository.
	OnnxFilename string

	// LabelMapping maps model labels (after BIO-prefix stripping, so "PER"
	// not "B-PER") to canonical entity names. Unmapped labels are kept
	// as-is, mirroring Presidio's keep-but-warn behavior; add them to
	// LabelsToIgnore to drop them.
	LabelMapping map[string]string
	// LabelsToIgnore drops spans whose model label or mapped entity name is
	// in the list.
	LabelsToIgnore []string
	// LowScoreEntities lists mapped entity names whose confidence is
	// multiplied by LowScoreMultiplier, for labels the model is known to
	// over-predict.
	LowScoreEntities []string
	// LowScoreMultiplier is the factor applied to LowScoreEntities scores.
	// Zero disables the adjustment.
	LowScoreMultiplier float64
}

// DefaultConfig returns a configuration matching Presidio's default NER
// surface: it emits PERSON, LOCATION, NRP and DATE_TIME, and drops
// ORGANIZATION (many false positives) plus CoNLL's MISC (too broad to emit
// under a canonical name).
func DefaultConfig() Config {
	return Config{
		Model: "KnightsAnalytics/distilbert-NER",
		LabelMapping: map[string]string{
			// CoNLL-style labels.
			"PER": entities.Person,
			"LOC": entities.Location,
			"ORG": "ORGANIZATION",
			// OntoNotes-style labels, for models trained on that scheme.
			"PERSON":   entities.Person,
			"GPE":      entities.Location,
			"FAC":      entities.Location,
			"LOCATION": entities.Location,
			"NORP":     entities.NRP,
			"DATE":     entities.DateTime,
			"TIME":     entities.DateTime,
		},
		LabelsToIgnore:     []string{"ORGANIZATION", "MISC"},
		LowScoreMultiplier: 0.4,
	}
}

// SupportedEntities returns the sorted, de-duplicated entity names this
// configuration can emit: the mapping's values minus the ignore list.
func (c Config) SupportedEntities() []string {
	ignored := make(map[string]bool, len(c.LabelsToIgnore))
	for _, l := range c.LabelsToIgnore {
		ignored[l] = true
	}
	seen := map[string]bool{}
	var out []string
	for _, entity := range c.LabelMapping {
		if ignored[entity] || seen[entity] {
			continue
		}
		seen[entity] = true
		out = append(out, entity)
	}
	sort.Strings(out)
	return out
}
