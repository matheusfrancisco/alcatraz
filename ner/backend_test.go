package ner

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/knights-analytics/hugot/options"
)

// apply runs the assembled options against a fresh hugot options struct for
// the given (hugot-normalized, uppercase) backend, mirroring what hugot's
// newSession does.
func apply(t *testing.T, backend string, opts []options.WithOption) *options.Options {
	t.Helper()
	o := options.Defaults()
	o.Backend = backend
	for _, opt := range opts {
		if err := opt(o); err != nil {
			t.Fatalf("applying option: %v", err)
		}
	}
	return o
}

func TestSessionOptionsBucketing(t *testing.T) {
	cfg := DefaultConfig()

	t.Run("go backend gets bucketing options", func(t *testing.T) {
		opts, err := sessionOptions(BackendGo, cfg)
		if err != nil {
			t.Fatal(err)
		}
		o := apply(t, "GO", opts)
		if len(o.GoMLXOptions.BatchBuckets) == 0 || len(o.GoMLXOptions.SequenceBuckets) == 0 {
			t.Fatalf("bucketing not applied: %+v", o.GoMLXOptions)
		}
	})

	t.Run("ort backend skips bucketing options", func(t *testing.T) {
		opts, err := sessionOptions(BackendORT, cfg)
		if err != nil {
			t.Fatal(err)
		}
		// Applying to an ORT options struct must not fail: hugot rejects
		// the gomlx bucketing options on ORT, so they must be absent.
		o := apply(t, "ORT", opts)
		if len(o.GoMLXOptions.BatchBuckets) != 0 {
			t.Fatalf("bucketing applied on ORT: %+v", o.GoMLXOptions)
		}
	})
}

func TestAcceleratorOption(t *testing.T) {
	t.Run("empty accelerator is a no-op", func(t *testing.T) {
		opt, err := acceleratorOption(BackendGo, Config{})
		if err != nil || opt != nil {
			t.Fatalf("got (%v, %v), want (nil, nil)", opt, err)
		}
	})

	t.Run("coreml requires ort", func(t *testing.T) {
		if _, err := acceleratorOption(BackendGo, Config{Accelerator: AcceleratorCoreML}); err == nil {
			t.Fatal("want error for coreml on go backend")
		}
		opt, err := acceleratorOption(BackendORT, Config{Accelerator: AcceleratorCoreML})
		if err != nil {
			t.Fatal(err)
		}
		o := apply(t, "ORT", []options.WithOption{opt})
		if o.ORTOptions.CoreMLOptions == nil {
			t.Fatal("CoreMLOptions not set: a nil map means 'not requested' to hugot")
		}
	})

	t.Run("cuda works on ort and xla, not go", func(t *testing.T) {
		if _, err := acceleratorOption(BackendGo, Config{Accelerator: AcceleratorCUDA}); err == nil {
			t.Fatal("want error for cuda on go backend")
		}
		opt, err := acceleratorOption(BackendXLA, Config{Accelerator: AcceleratorCUDA})
		if err != nil {
			t.Fatal(err)
		}
		o := apply(t, "XLA", []options.WithOption{opt})
		if !o.GoMLXOptions.Cuda {
			t.Fatal("XLA cuda flag not set")
		}
		opt, err = acceleratorOption(BackendORT, Config{Accelerator: AcceleratorCUDA})
		if err != nil {
			t.Fatal(err)
		}
		o = apply(t, "ORT", []options.WithOption{opt})
		if o.ORTOptions.CudaOptions == nil {
			t.Fatal("CudaOptions not set")
		}
	})

	t.Run("directml parses device_id", func(t *testing.T) {
		opt, err := acceleratorOption(BackendORT, Config{
			Accelerator:        AcceleratorDirectML,
			AcceleratorOptions: map[string]string{"device_id": "2"},
		})
		if err != nil {
			t.Fatal(err)
		}
		o := apply(t, "ORT", []options.WithOption{opt})
		if o.ORTOptions.DirectMLOptions == nil || *o.ORTOptions.DirectMLOptions != 2 {
			t.Fatalf("DirectMLOptions = %v, want 2", o.ORTOptions.DirectMLOptions)
		}

		if _, err := acceleratorOption(BackendORT, Config{
			Accelerator:        AcceleratorDirectML,
			AcceleratorOptions: map[string]string{"device_id": "gpu0"},
		}); err == nil {
			t.Fatal("want error for non-numeric device_id")
		}
	})

	t.Run("accelerator name is case-insensitive", func(t *testing.T) {
		if _, err := acceleratorOption(BackendORT, Config{Accelerator: "CoreML"}); err != nil {
			t.Fatalf("mixed-case accelerator rejected: %v", err)
		}
	})

	t.Run("unknown accelerator errors", func(t *testing.T) {
		if _, err := acceleratorOption(BackendORT, Config{Accelerator: "warp-drive"}); err == nil {
			t.Fatal("want error for unknown accelerator")
		}
	})
}

func TestWithORTLibrary(t *testing.T) {
	dir := t.TempDir()
	lib := filepath.Join(dir, ortLibraryName())
	if err := os.WriteFile(lib, []byte("not a real library"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Run("file path", func(t *testing.T) {
		o := options.Defaults()
		if err := withORTLibrary(lib)(o); err != nil {
			t.Fatal(err)
		}
		if *o.ORTOptions.LibraryPath != lib || *o.ORTOptions.LibraryDir != dir {
			t.Fatalf("got (%s, %s)", *o.ORTOptions.LibraryPath, *o.ORTOptions.LibraryDir)
		}
	})

	t.Run("directory path resolves the platform library name", func(t *testing.T) {
		o := options.Defaults()
		if err := withORTLibrary(dir)(o); err != nil {
			t.Fatal(err)
		}
		if *o.ORTOptions.LibraryPath != lib || *o.ORTOptions.LibraryDir != dir {
			t.Fatalf("got (%s, %s)", *o.ORTOptions.LibraryPath, *o.ORTOptions.LibraryDir)
		}
	})

	t.Run("missing path errors", func(t *testing.T) {
		if err := withORTLibrary(filepath.Join(dir, "nope"))(options.Defaults()); err == nil {
			t.Fatal("want error for missing library")
		}
	})

	t.Run("directory without the library errors", func(t *testing.T) {
		if err := withORTLibrary(t.TempDir())(options.Defaults()); err == nil {
			t.Fatal("want error for directory without library")
		}
	})
}

func TestNewBackendValidation(t *testing.T) {
	ctx := context.Background()

	t.Run("unknown backend fails fast", func(t *testing.T) {
		_, err := New(ctx, Config{ModelPath: t.TempDir(), Backend: "banana"})
		if err == nil || !strings.Contains(err.Error(), "unknown backend") {
			t.Fatalf("err = %v, want unknown backend error", err)
		}
	})

	t.Run("accelerator on wrong backend fails fast", func(t *testing.T) {
		_, err := New(ctx, Config{ModelPath: t.TempDir(), Accelerator: AcceleratorCoreML})
		if err == nil || !strings.Contains(err.Error(), "requires backend") {
			t.Fatalf("err = %v, want accelerator/backend mismatch error", err)
		}
	})

	t.Run("ort backend never falls back silently", func(t *testing.T) {
		// The failure mode depends on the build: hugot's stub reports the
		// missing ORT build tag, a -tags ORT binary without the shared
		// library reports the missing library, and a fully ORT-capable
		// binary fails on the empty model directory. In every case New
		// must return an error rather than degrade to another backend.
		_, err := New(ctx, Config{ModelPath: t.TempDir(), Backend: BackendORT})
		if err == nil {
			t.Fatal("want error: empty model dir can never yield a working ORT engine")
		}
	})
}
