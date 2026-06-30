package recognizers

import (
	"github.com/hoophq/alcatraz/analyzer"
	"github.com/hoophq/alcatraz/entities"
)

// BRCPF detects Brazilian CPF numbers (Cadastro de Pessoas Físicas) and
// validates the two mod-11 check digits.
func BRCPF() analyzer.Recognizer {
	return analyzer.NewPatternRecognizer(
		"BrCpfRecognizer", entities.BRCPF, "pt",
		[]*analyzer.Pattern{
			analyzer.MustPattern("BR CPF (formatted)", `\b\d{3}\.\d{3}\.\d{3}-\d{2}\b`, 0.4),
			analyzer.MustPattern("BR CPF", `\b\d{11}\b`, 0.3),
		},
	).WithContext("cpf", "cadastro de pessoas físicas").WithValidator(validateBRCPF)
}

func validateBRCPF(s string) bool {
	ds, ok := digitsExactly(s, 11)
	if !ok || allEqual(ds) {
		return false
	}
	if mod11Weighted(ds[:9], []int{10, 9, 8, 7, 6, 5, 4, 3, 2}) != ds[9] {
		return false
	}
	return mod11Weighted(ds[:10], []int{11, 10, 9, 8, 7, 6, 5, 4, 3, 2}) == ds[10]
}

// BRCNPJ detects Brazilian CNPJ numbers (Cadastro Nacional da Pessoa Jurídica)
// and validates the two mod-11 check digits.
func BRCNPJ() analyzer.Recognizer {
	return analyzer.NewPatternRecognizer(
		"BrCnpjRecognizer", entities.BRCNPJ, "pt",
		[]*analyzer.Pattern{
			analyzer.MustPattern("BR CNPJ (formatted)", `\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b`, 0.4),
			analyzer.MustPattern("BR CNPJ", `\b\d{14}\b`, 0.3),
		},
	).WithContext("cnpj", "cadastro nacional").WithValidator(validateBRCNPJ)
}

func validateBRCNPJ(s string) bool {
	ds, ok := digitsExactly(s, 14)
	if !ok || allEqual(ds) {
		return false
	}
	if mod11Weighted(ds[:12], []int{5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2}) != ds[12] {
		return false
	}
	return mod11Weighted(ds[:13], []int{6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2}) == ds[13]
}

// BRRG detects Brazilian RG numbers (Registro Geral / identity card).
//
// RG check-digit rules are issued per-state and are not standardized
// nationally, so this is a shape-and-context recognizer with no validator: a
// single state's checksum would wrongly drop valid RGs from other states.
func BRRG() analyzer.Recognizer {
	return analyzer.NewPatternRecognizer(
		"BrRgRecognizer", entities.BRRG, "pt",
		[]*analyzer.Pattern{
			analyzer.MustPattern("BR RG (formatted)", `\b\d{1,2}\.\d{3}\.\d{3}-[0-9Xx]\b`, 0.4),
		},
	).WithContext("rg", "registro geral", "identidade", "carteira de identidade")
}

// BRCNH detects Brazilian driver's license numbers (Carteira Nacional de
// Habilitação) and validates the two check digits.
func BRCNH() analyzer.Recognizer {
	return analyzer.NewPatternRecognizer(
		"BrCnhRecognizer", entities.BRCNH, "pt",
		[]*analyzer.Pattern{analyzer.MustPattern("BR CNH", `\b\d{11}\b`, 0.3)},
	).WithContext("cnh", "habilitação", "carteira de motorista").WithValidator(validateBRCNH)
}

func validateBRCNH(s string) bool {
	ds, ok := digitsExactly(s, 11)
	if !ok || allEqual(ds) {
		return false
	}
	// First check digit: weights 9..1.
	sum, dsc := 0, 0
	for i := 0; i < 9; i++ {
		sum += ds[i] * (9 - i)
	}
	dv1 := sum % 11
	if dv1 >= 10 {
		dv1, dsc = 0, 2
	}
	if dv1 != ds[9] {
		return false
	}
	// Second check digit: weights 1..9, offset by dsc when the first rolled over.
	sum = 0
	for i := 0; i < 9; i++ {
		sum += ds[i] * (i + 1)
	}
	dv2 := sum % 11
	if dv2 >= 10 {
		dv2 = 0
	} else if dv2 -= dsc; dv2 < 0 {
		dv2 += 11
	}
	return dv2 == ds[10]
}

// BRPIS detects Brazilian PIS/PASEP/NIS numbers and validates the mod-11 check
// digit.
func BRPIS() analyzer.Recognizer {
	return analyzer.NewPatternRecognizer(
		"BrPisRecognizer", entities.BRPIS, "pt",
		[]*analyzer.Pattern{
			analyzer.MustPattern("BR PIS (formatted)", `\b\d{3}\.\d{5}\.\d{2}-\d\b`, 0.4),
			analyzer.MustPattern("BR PIS", `\b\d{11}\b`, 0.3),
		},
	).WithContext("pis", "pasep", "nis", "nit").WithValidator(validateBRPIS)
}

func validateBRPIS(s string) bool {
	ds, ok := digitsExactly(s, 11)
	if !ok || allEqual(ds) {
		return false
	}
	return mod11Weighted(ds[:10], []int{3, 2, 9, 8, 7, 6, 5, 4, 3, 2}) == ds[10]
}

// mod11Weighted computes a Brazilian mod-11 check digit: the dot product of ds
// with weights taken mod 11, where a remainder below 2 yields 0.
func mod11Weighted(ds, weights []int) int {
	sum := 0
	for i, d := range ds {
		sum += d * weights[i]
	}
	if r := sum % 11; r >= 2 {
		return 11 - r
	}
	return 0
}

// allEqual reports whether every digit in ds is identical (e.g. 111.111.111-11)
// — a sequence that satisfies the mod-11 math but is never a real document.
func allEqual(ds []int) bool {
	for i := 1; i < len(ds); i++ {
		if ds[i] != ds[0] {
			return false
		}
	}
	return len(ds) > 0
}
