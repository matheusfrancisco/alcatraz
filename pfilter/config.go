package pfilter

import (
	"sort"
	"strings"

	"github.com/hoophq/alcatraz/entities"
)

// Config configures the privacy-filter engine: which GGUF model to load, how
// to run it, and how its labels map onto alcatraz entity types.
type Config struct {
	// ModelPath is the path to a privacy-filter GGUF file (required). See
	// the pre-converted artifacts linked from
	// https://github.com/localai-org/privacy-filter.cpp (e.g.
	// LocalAI-io/privacy-filter-GGUF on Hugging Face).
	ModelPath string
	// Library is the path to the privacy-filter.cpp shared library
	// (libpf.so / libpf.dylib). Empty means: use $PF_LIBRARY if set, then
	// a library previously downloaded by EnsureLibrary, then let the
	// system loader search its default paths for "libpf".
	Library string
	// Device selects the compute device: "" or "cpu", "gpu", "cuda",
	// "vulkan" (optionally ":N" to pick the Nth matching GPU).
	Device string
	// Threads is the CPU thread count; <= 0 picks the runtime default.
	Threads int
	// PoolSize is the number of pf_ctx model contexts the engine keeps
	// (default 1). A pf_ctx serves one classify call at a time, so this
	// bounds in-engine inference concurrency: calls beyond PoolSize wait
	// for an idle context. Each context loads the model separately —
	// memory scales linearly — so raise it only when parallel Analyze
	// throughput matters more than RAM.
	PoolSize int
	// WindowTokens overrides the max tokens per forward pass (default
	// 4096). Longer inputs run as overlapping halo windows. Must be > 2048
	// to take effect; 0 keeps the runtime default.
	WindowTokens int
	// Threshold is the model-side score cutoff passed to pf_classify.
	// Keep it 0 and use alcatraz's Options.Threshold instead unless you
	// want to shed low-confidence spans before they reach the engine.
	Threshold float64

	// LabelMapping maps model span labels (BIOES already decoded, e.g.
	// "private_person") to canonical entity names. Unmapped labels are
	// kept, normalized to SCREAMING_SNAKE_CASE ("crypto_wallet" →
	// "CRYPTO_WALLET"); add them to LabelsToIgnore to drop them instead.
	LabelMapping map[string]string
	// LabelsToIgnore drops spans whose model label or mapped entity name
	// is in the list.
	LabelsToIgnore []string
}

// DefaultConfig returns a configuration for the base openai/privacy-filter
// model (8 categories) with its labels mapped onto the canonical entity
// names shared with the pattern recognizers, so model and pattern results
// de-duplicate against each other. modelPath is the GGUF file to load.
func DefaultConfig(modelPath string) Config {
	return Config{
		ModelPath: modelPath,
		LabelMapping: map[string]string{
			"private_person":  entities.Person,
			"private_address": entities.Location,
			"private_date":    entities.DateTime,
			"private_email":   entities.EmailAddress,
			"private_phone":   entities.PhoneNumber,
			"private_url":     entities.URL,
			"account_number":  "ACCOUNT_NUMBER",
			"secret":          "SECRET",
		},
	}
}

// SupportedEntities returns the sorted, de-duplicated entity names this
// configuration is known to emit: the mapping's values minus the ignore
// list. (Unmapped model labels can additionally surface, normalized to
// SCREAMING_SNAKE_CASE — relevant for the multilingual model's 54
// categories.)
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

// normalizeLabel converts an unmapped model label to the naming convention
// used by entities.*: "crypto_wallet" → "CRYPTO_WALLET".
func normalizeLabel(label string) string {
	return strings.ToUpper(strings.ReplaceAll(label, "-", "_"))
}
