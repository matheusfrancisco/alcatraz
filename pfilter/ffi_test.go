//go:build (darwin || linux) && (amd64 || arm64)

package pfilter

import (
	"testing"
	"unsafe"
)

func TestPfEntityLayout(t *testing.T) {
	// The FFI layer reads pf_entity arrays by pointer arithmetic; this pins
	// the assumed 64-bit C struct layout (see pf.h). 32-bit platforms never
	// build this file — they get ffi_stub.go.
	var e pfEntity
	if size := unsafe.Sizeof(e); size != 24 {
		t.Errorf("sizeof(pfEntity) = %d, want 24", size)
	}
	if off := unsafe.Offsetof(e.start); off != 0 {
		t.Errorf("offsetof(start) = %d, want 0", off)
	}
	if off := unsafe.Offsetof(e.end); off != 4 {
		t.Errorf("offsetof(end) = %d, want 4", off)
	}
	if off := unsafe.Offsetof(e.score); off != 8 {
		t.Errorf("offsetof(score) = %d, want 8", off)
	}
	if off := unsafe.Offsetof(e.label); off != 16 {
		t.Errorf("offsetof(label) = %d, want 16", off)
	}
}
