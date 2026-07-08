package pfilter

import (
	"unsafe"
)

// abiVersion is the pf.h ABI this binding was written against.
const abiVersion = 1

// rawEntity is one span as reported by pf_classify, before label mapping.
type rawEntity struct {
	start int
	end   int
	score float64
	label string
}

// classifier is the seam between the engine and the FFI layer, so the
// engine's mapping logic is testable without the shared library.
type classifier interface {
	classify(text string, threshold float32) ([]rawEntity, error)
	close() error
}

// pfEntity mirrors pf.h's pf_entity on 64-bit platforms:
//
//	typedef struct {
//	    int32_t      start;
//	    int32_t      end;
//	    float        score;
//	    const char * label;
//	} pf_entity;
//
// The padding field covers the 4 bytes the compiler inserts so the label
// pointer is 8-byte aligned (sizeof == 24, alignof == 8).
type pfEntity struct {
	start int32
	end   int32
	score float32
	_     int32
	label *byte
}

// goString copies a NUL-terminated C string into a Go string. p may be nil.
func goString(p *byte) string {
	if p == nil {
		return ""
	}
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(p, n))
}

// ctxPool hands out pf_ctx handles to concurrent classify calls. pf.h does
// not document a single pf_ctx as thread-safe, so each handle serves one
// call at a time; independent contexts run in parallel. The channel is the
// pool: acquire receives an idle handle (blocking while all are busy),
// release returns it.
type ctxPool struct {
	ctxs chan unsafe.Pointer
	size int
}

func newCtxPool(ctxs []unsafe.Pointer) *ctxPool {
	p := &ctxPool{ctxs: make(chan unsafe.Pointer, len(ctxs)), size: len(ctxs)}
	for _, ctx := range ctxs {
		p.ctxs <- ctx
	}
	return p
}

// acquire returns an idle context, blocking while all are busy. ok is false
// after closeAll: the pool is closed and the context gone.
func (p *ctxPool) acquire() (ctx unsafe.Pointer, ok bool) {
	ctx, ok = <-p.ctxs
	return ctx, ok
}

func (p *ctxPool) release(ctx unsafe.Pointer) {
	p.ctxs <- ctx
}

// closeAll waits for every context to be idle (in-flight calls finish and
// release), frees each, and closes the pool. Every release happens before
// the drain completes, so closing the channel cannot race a send; acquire
// calls that arrive afterwards get ok == false.
func (p *ctxPool) closeAll(free func(unsafe.Pointer)) {
	for i := 0; i < p.size; i++ {
		free(<-p.ctxs)
	}
	close(p.ctxs)
}
