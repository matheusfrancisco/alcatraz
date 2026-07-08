// The pfEntity mirror below hardcodes the 64-bit pf_entity layout (8-byte
// label pointer, 4 bytes of padding before it), so this FFI path is
// restricted to 64-bit platforms; everything else gets ffi_stub.go.
//go:build (darwin || linux) && (amd64 || arm64)

package pfilter

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// pfLib holds the registered pf.h functions of one loaded shared library.
// Libraries are loaded once per path and never unloaded (dlclose while a
// pf_ctx lives would be unsafe).
type pfLib struct {
	abiVersion   func() int32
	load         func(ggufPath string, device string, threads int32) unsafe.Pointer
	free         func(ctx unsafe.Pointer)
	lastError    func(ctx unsafe.Pointer) *byte
	setWindow    func(ctx unsafe.Pointer, maxForwardTokens int32)
	classify     func(ctx unsafe.Pointer, text string, length uintptr, threshold float32, out *unsafe.Pointer, n *uintptr) int32
	entitiesFree func(ents unsafe.Pointer, n uintptr)
}

var (
	libsMu sync.Mutex
	libs   = map[string]*pfLib{}
)

// pfEntity mirrors pf.h's pf_entity on the 64-bit platforms this file is
// built for (see the build constraint above):
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

// defaultLibraryNames are tried in order when no explicit path is given.
func defaultLibraryNames() []string {
	if runtime.GOOS == "darwin" {
		return []string{"libpf.dylib", "libpf.so"}
	}
	return []string{"libpf.so"}
}

// loadLibrary loads (or reuses) the privacy-filter.cpp shared library and
// registers the pf.h symbols. path resolution: explicit argument, then
// $PF_LIBRARY, then a library previously downloaded by EnsureLibrary, then
// the platform's default library names via the system loader search path.
func loadLibrary(path string) (*pfLib, error) {
	candidates := []string{path}
	if path == "" {
		if env := os.Getenv("PF_LIBRARY"); env != "" {
			candidates = []string{env}
		} else {
			candidates = nil
			if cached, err := cachedLibraryPath(); err == nil {
				if _, err := os.Stat(cached); err == nil {
					candidates = append(candidates, cached)
				}
			}
			candidates = append(candidates, defaultLibraryNames()...)
		}
	}

	libsMu.Lock()
	defer libsMu.Unlock()

	var firstErr error
	for _, candidate := range candidates {
		if lib, ok := libs[candidate]; ok {
			return lib, nil
		}
		handle, err := purego.Dlopen(candidate, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		lib := &pfLib{}
		purego.RegisterLibFunc(&lib.abiVersion, handle, "pf_abi_version")
		purego.RegisterLibFunc(&lib.load, handle, "pf_load")
		purego.RegisterLibFunc(&lib.free, handle, "pf_free")
		purego.RegisterLibFunc(&lib.lastError, handle, "pf_last_error")
		purego.RegisterLibFunc(&lib.setWindow, handle, "pf_set_window")
		purego.RegisterLibFunc(&lib.classify, handle, "pf_classify")
		purego.RegisterLibFunc(&lib.entitiesFree, handle, "pf_entities_free")

		if got := lib.abiVersion(); got != abiVersion {
			return nil, fmt.Errorf("pfilter: %s reports pf ABI %d, this binding requires %d", candidate, got, abiVersion)
		}
		libs[candidate] = lib
		return lib, nil
	}
	return nil, fmt.Errorf("pfilter: could not load privacy-filter.cpp shared library (tried %v; set Config.Library or $PF_LIBRARY): %w", candidates, firstErr)
}

// ffiClassifier is the production classifier: a pool of pf_ctx handles
// (Config.PoolSize, default 1). pf.h does not document a single pf_ctx as
// thread-safe, so each handle serves one classify call at a time;
// independent handles run in parallel. Every context loads the model
// separately — ggml weights are not shared — so PoolSize > 1 multiplies
// model memory.
type ffiClassifier struct {
	lib       *pfLib
	pool      *ctxPool
	closeOnce sync.Once
}

// newFFIClassifier loads the library and PoolSize model contexts, failing
// fast (and freeing what was loaded) on any error.
func newFFIClassifier(cfg Config) (classifier, error) {
	lib, err := loadLibrary(cfg.Library)
	if err != nil {
		return nil, err
	}

	device := cfg.Device
	if device == "" {
		device = "cpu"
	}
	size := cfg.PoolSize
	if size <= 0 {
		size = 1
	}
	ctxs := make([]unsafe.Pointer, 0, size)
	for i := 0; i < size; i++ {
		ctx := lib.load(cfg.ModelPath, device, int32(cfg.Threads))
		if ctx == nil {
			err = fmt.Errorf("pfilter: pf_load returned no context for %s", cfg.ModelPath)
		} else if msg := goString(lib.lastError(ctx)); msg != "" {
			lib.free(ctx)
			err = fmt.Errorf("pfilter: loading %s: %s", cfg.ModelPath, msg)
		}
		if err != nil {
			for _, loaded := range ctxs {
				lib.free(loaded)
			}
			return nil, err
		}
		if cfg.WindowTokens > 0 {
			lib.setWindow(ctx, int32(cfg.WindowTokens))
		}
		ctxs = append(ctxs, ctx)
	}
	return &ffiClassifier{lib: lib, pool: newCtxPool(ctxs)}, nil
}

func (c *ffiClassifier) classify(text string, threshold float32) ([]rawEntity, error) {
	if text == "" {
		return nil, nil
	}
	ctx, ok := c.pool.acquire()
	if !ok {
		return nil, fmt.Errorf("pfilter: classify called after Close")
	}
	defer c.pool.release(ctx)

	var out unsafe.Pointer
	var n uintptr
	rc := c.lib.classify(ctx, text, uintptr(len(text)), threshold, &out, &n)
	if rc != 0 {
		msg := goString(c.lib.lastError(ctx))
		if msg == "" {
			msg = fmt.Sprintf("pf_classify returned %d", rc)
		}
		return nil, fmt.Errorf("pfilter: %s", msg)
	}
	if out == nil || n == 0 {
		return nil, nil
	}
	defer c.lib.entitiesFree(out, n)

	ents := unsafe.Slice((*pfEntity)(out), n)
	results := make([]rawEntity, 0, n)
	for i := range ents {
		results = append(results, rawEntity{
			start: int(ents[i].start),
			end:   int(ents[i].end),
			score: float64(ents[i].score),
			label: goString(ents[i].label),
		})
	}
	return results, nil
}

// close waits for in-flight classify calls to finish, then frees every
// context. Idempotent: a second close finds an already-closed pool.
func (c *ffiClassifier) close() error {
	c.closeOnce.Do(func() {
		c.pool.closeAll(c.lib.free)
	})
	return nil
}
