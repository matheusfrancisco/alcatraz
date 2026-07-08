// Package corpus reads and describes the shared benchmark corpus.
package corpus

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
)

// Doc is one benchmark input text with its generation parameters.
type Doc struct {
	ID        string `json:"id"`
	SizeClass string `json:"size_class"`
	Density   string `json:"density"`
	Text      string `json:"text"`
}

// SizeClasses and Densities define the canonical group ordering used in
// reports, so every harness emits groups in the same order.
var (
	SizeClasses = []string{"100B", "1KB", "10KB", "1MB"}
	Densities   = []string{"none", "sparse", "dense"}
)

// Load reads a JSONL corpus file.
func Load(path string) ([]Doc, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var docs []Doc
	sc := bufio.NewScanner(f)
	// A 1MB-text doc serializes to a JSON line larger than 1MB; allow up to 8MB.
	sc.Buffer(make([]byte, 0, 1<<20), 8<<20)
	for sc.Scan() {
		var d Doc
		if err := json.Unmarshal(sc.Bytes(), &d); err != nil {
			return nil, fmt.Errorf("corpus line %d: %w", len(docs)+1, err)
		}
		docs = append(docs, d)
	}
	return docs, sc.Err()
}
