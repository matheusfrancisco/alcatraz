//go:build !((darwin || linux) && (amd64 || arm64))

package pfilter

import (
	"fmt"
	"runtime"
)

// newFFIClassifier is a stub for platforms where the purego dlopen path is
// not wired up: non-darwin/linux systems, and 32-bit architectures (the
// pfEntity struct mirror in ffi.go assumes the 64-bit pf_entity layout).
// The rest of the package compiles everywhere so importers can gate usage
// at runtime.
func newFFIClassifier(cfg Config) (classifier, error) {
	return nil, fmt.Errorf("pfilter: privacy-filter.cpp binding is not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
}
