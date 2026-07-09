// Command gencorpus writes the shared benchmark corpus (corpus.jsonl).
//
// Every PII value it seeds passes the real validation scheme for its type
// (Luhn for cards, mod-11 for CPF, mod-97 for IBAN, area/group rules for
// SSN), so validator code paths are exercised the same way real data would
// exercise them. Generation is deterministic: the same seed always yields
// byte-identical output.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strings"

	"github.com/hoophq/alcatraz/bench/internal/corpus"
)

// filler is PII-free English text used as padding between seeded entities.
var filler = strings.Fields(`
the quarterly report shows steady growth across all regions and the team
expects continued momentum through next year pending final review of the
proposed budget adjustments which remain under discussion with stakeholders
meeting notes will be circulated after the session concludes and action
items assigned to owners with clear deadlines for follow up on open issues
`)

type generator struct{ rng *rand.Rand }

func (g *generator) words(n int) string {
	out := make([]string, n)
	for i := range out {
		out[i] = filler[g.rng.Intn(len(filler))]
	}
	return strings.Join(out, " ")
}

// luhnCard returns a 16-digit Visa-prefixed number with a valid Luhn check digit.
func (g *generator) luhnCard() string {
	digits := make([]int, 16)
	digits[0] = 4
	for i := 1; i < 15; i++ {
		digits[i] = g.rng.Intn(10)
	}
	sum := 0
	for i := 0; i < 15; i++ {
		d := digits[i]
		// Positions counted from the check digit: index 14 is doubled, 13 is not, ...
		if (15-i)%2 == 1 {
			d *= 2
			if d > 9 {
				d -= 9
			}
		}
		sum += d
	}
	digits[15] = (10 - sum%10) % 10
	var b strings.Builder
	for _, d := range digits {
		fmt.Fprintf(&b, "%d", d)
	}
	return b.String()
}

// cpf returns a valid Brazilian CPF (mod-11 check digits) formatted xxx.xxx.xxx-xx.
func (g *generator) cpf() string {
	d := make([]int, 11)
	for i := 0; i < 9; i++ {
		d[i] = g.rng.Intn(10)
	}
	sum := 0
	for i := 0; i < 9; i++ {
		sum += d[i] * (10 - i)
	}
	d[9] = (sum * 10 % 11) % 10
	sum = 0
	for i := 0; i < 10; i++ {
		sum += d[i] * (11 - i)
	}
	d[10] = (sum * 10 % 11) % 10
	return fmt.Sprintf("%d%d%d.%d%d%d.%d%d%d-%d%d",
		d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10])
}

// iban returns a valid German IBAN (ISO 7064 mod-97 check digits).
func (g *generator) iban() string {
	bban := make([]byte, 18)
	for i := range bban {
		bban[i] = byte('0' + g.rng.Intn(10))
	}
	// Check digits: rearrange to BBAN + "DE00", letters to numbers, mod 97.
	s := string(bban) + "131400" // D=13, E=14, 00
	rem := 0
	for _, c := range s {
		rem = (rem*10 + int(c-'0')) % 97
	}
	return fmt.Sprintf("DE%02d%s", 98-rem, bban)
}

// ssn returns a US SSN that satisfies Presidio/Alcatraz structural rules
// (no 000/666/9xx area, no 00 group, no 0000 serial).
func (g *generator) ssn() string {
	area := 100 + g.rng.Intn(500) // 100-599 avoids 000, 666 avoided below, 9xx excluded
	if area == 666 {
		area = 667
	}
	group := 1 + g.rng.Intn(99)
	serial := 1 + g.rng.Intn(9999)
	return fmt.Sprintf("%03d-%02d-%04d", area, group, serial)
}

func (g *generator) email() string {
	names := []string{"jane.doe", "carlos.silva", "wei.zhang", "amit.patel", "sofia.rossi"}
	domains := []string{"example.com", "corpmail.io", "acme-widgets.net"}
	return fmt.Sprintf("%s%d@%s", names[g.rng.Intn(len(names))], g.rng.Intn(1000), domains[g.rng.Intn(len(domains))])
}

func (g *generator) phone() string {
	return fmt.Sprintf("(212) 555-%04d", g.rng.Intn(10000))
}

func (g *generator) ip() string {
	return fmt.Sprintf("10.%d.%d.%d", g.rng.Intn(256), g.rng.Intn(256), 1+g.rng.Intn(254))
}

func (g *generator) url() string {
	return fmt.Sprintf("https://portal.example.com/accounts/%d/settings", g.rng.Intn(100000))
}

// pii returns one seeded entity with a natural-language lead-in, so context
// looks like real prose rather than a bare token list.
func (g *generator) pii() string {
	switch g.rng.Intn(8) {
	case 0:
		return "charged to card " + g.luhnCard()
	case 1:
		return "CPF do cliente " + g.cpf()
	case 2:
		return "wire to IBAN " + g.iban()
	case 3:
		return "SSN on file " + g.ssn()
	case 4:
		return "contact " + g.email()
	case 5:
		return "call " + g.phone()
	case 6:
		return "host at " + g.ip()
	default:
		return "see " + g.url()
	}
}

// doc builds one text of roughly targetBytes with the given PII density.
// Density: none = 0 entities, sparse = ~1 entity per KB, dense = ~1 per 100B.
func (g *generator) doc(targetBytes int, density string) string {
	var interval int
	switch density {
	case "none":
		interval = 0
	case "sparse":
		interval = 1000
	case "dense":
		interval = 100
	}

	var b strings.Builder
	for b.Len() < targetBytes {
		b.WriteString(g.words(12))
		b.WriteString(". ")
		if interval > 0 && b.Len()%interval < 90 {
			b.WriteString(g.pii())
			b.WriteString(". ")
		}
	}
	return b.String()[:targetBytes]
}

func main() {
	out := flag.String("out", "corpus.jsonl", "output path")
	seed := flag.Int64("seed", 42, "deterministic RNG seed")
	perGroup := flag.Int("per-group", 20, "documents per (size, density) group")
	flag.Parse()

	g := &generator{rng: rand.New(rand.NewSource(*seed))}
	sizes := map[string]int{"100B": 100, "1KB": 1024, "10KB": 10240, "1MB": 1 << 20}

	// Large docs get fewer instances per group: at 1MB each, the default 20
	// would balloon the corpus to ~60MB and make the Python side take hours.
	docsFor := func(sizeBytes int) int {
		if sizeBytes >= 1<<20 {
			n := *perGroup / 10
			if n < 2 {
				n = 2
			}
			return n
		}
		return *perGroup
	}

	f, err := os.Create(*out)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	n, total := 0, 0
	for _, sc := range corpus.SizeClasses {
		for _, den := range corpus.Densities {
			for i := 0; i < docsFor(sizes[sc]); i++ {
				d := corpus.Doc{
					ID:        fmt.Sprintf("%s-%s-%03d", sc, den, i),
					SizeClass: sc,
					Density:   den,
					Text:      g.doc(sizes[sc], den),
				}
				if err := enc.Encode(d); err != nil {
					fmt.Fprintln(os.Stderr, err)
					os.Exit(1)
				}
				n++
				total += len(d.Text)
			}
		}
	}
	fmt.Printf("wrote %d docs (%d bytes of text) to %s\n", n, total, *out)
}
