package ner

// This file selects and configures the hugot inference backend. The default
// is hugot's pure-Go backend, which works in every build; the ORT and XLA
// backends trade portability for speed and are compiled in only under their
// hugot build tags ("-tags ORT", "-tags XLA" — both imply cgo). Selecting a
// backend that is not compiled in fails at construction with hugot's
// instructive error rather than at inference time.

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
)

// newBackendSession validates cfg's backend/accelerator combination,
// assembles the hugot session options and constructs the session for the
// selected backend.
func newBackendSession(ctx context.Context, cfg Config) (*hugot.Session, error) {
	backend := strings.ToLower(cfg.Backend)
	if backend == "" {
		backend = BackendGo
	}

	opts, err := sessionOptions(backend, cfg)
	if err != nil {
		return nil, err
	}

	switch backend {
	case BackendGo:
		return hugot.NewGoSession(ctx, opts...)
	case BackendORT:
		return hugot.NewORTSession(ctx, opts...)
	case BackendXLA:
		return hugot.NewXLASession(ctx, opts...)
	default:
		return nil, fmt.Errorf("unknown backend %q (want %q, %q or %q)",
			cfg.Backend, BackendGo, BackendORT, BackendXLA)
	}
}

// sessionOptions builds the hugot session options implied by cfg for the
// given (normalized) backend. cfg.SessionOptions are appended last so they
// can override anything derived here.
func sessionOptions(backend string, cfg Config) ([]options.WithOption, error) {
	var opts []options.WithOption

	// Shape bucketing exists to bound JIT compilation, a gomlx concept: it
	// applies to the Go and XLA backends only. ORT handles dynamic shapes
	// natively, and hugot rejects the bucketing options on it.
	if backend != BackendORT && len(cfg.BatchBuckets) > 0 && len(cfg.SequenceBuckets) > 0 {
		opts = append(opts,
			options.WithGoMLXBatchBuckets(cfg.BatchBuckets),
			options.WithGoMLXSequenceBuckets(cfg.SequenceBuckets),
		)
	}

	if backend == BackendORT {
		if cfg.ORTLibraryPath != "" {
			opts = append(opts, withORTLibrary(cfg.ORTLibraryPath))
		} else if lib := probeORTLibrary(); lib != "" {
			opts = append(opts, withORTLibrary(lib))
		}
	}

	accelOpt, err := acceleratorOption(backend, cfg)
	if err != nil {
		return nil, err
	}
	if accelOpt != nil {
		opts = append(opts, accelOpt)
	}

	return append(opts, cfg.SessionOptions...), nil
}

// acceleratorOption maps cfg.Accelerator to the hugot execution-provider
// option, validating that the selected backend supports it. A nil option
// with nil error means CPU (no accelerator requested).
func acceleratorOption(backend string, cfg Config) (options.WithOption, error) {
	accel := strings.ToLower(cfg.Accelerator)
	if accel == "" {
		return nil, nil
	}

	// hugot treats a nil provider-option map as "provider not requested",
	// so an explicit accelerator always passes a non-nil map.
	flags := cfg.AcceleratorOptions
	if flags == nil {
		flags = map[string]string{}
	}

	switch accel {
	case AcceleratorCoreML:
		if backend != BackendORT {
			return nil, acceleratorBackendError(accel, backend, BackendORT)
		}
		return options.WithCoreML(flags), nil
	case AcceleratorCUDA:
		if backend != BackendORT && backend != BackendXLA {
			return nil, acceleratorBackendError(accel, backend, BackendORT, BackendXLA)
		}
		return options.WithCuda(flags), nil
	case AcceleratorDirectML:
		if backend != BackendORT {
			return nil, acceleratorBackendError(accel, backend, BackendORT)
		}
		device := 0
		if raw, ok := flags["device_id"]; ok {
			d, err := strconv.Atoi(raw)
			if err != nil {
				return nil, fmt.Errorf("accelerator %q: invalid device_id %q", accel, raw)
			}
			device = d
		}
		return options.WithDirectML(device), nil
	default:
		return nil, fmt.Errorf("unknown accelerator %q (want %q, %q or %q)",
			cfg.Accelerator, AcceleratorCoreML, AcceleratorCUDA, AcceleratorDirectML)
	}
}

func acceleratorBackendError(accel, backend string, want ...string) error {
	quoted := make([]string, len(want))
	for i, w := range want {
		quoted[i] = strconv.Quote(w)
	}
	return fmt.Errorf("accelerator %q requires backend %s (Backend is %q)",
		accel, strings.Join(quoted, " or "), backend)
}

// withORTLibrary points hugot's ORT backend at the ONNX Runtime shared
// library. Unlike hugot's WithOnnxLibraryPath it accepts either the library
// file itself or the directory containing it, and validates existence up
// front so a wrong path fails with a path error instead of a dlopen failure.
func withORTLibrary(path string) options.WithOption {
	return func(o *options.Options) error {
		info, err := os.Stat(path)
		if err != nil {
			return fmt.Errorf("ONNX Runtime library: %w", err)
		}
		lib, dir := path, filepath.Dir(path)
		if info.IsDir() {
			lib, dir = filepath.Join(path, ortLibraryName()), path
			if _, err := os.Stat(lib); err != nil {
				return fmt.Errorf("ONNX Runtime library: %w", err)
			}
		}
		o.ORTOptions.LibraryPath = &lib
		o.ORTOptions.LibraryDir = &dir
		return nil
	}
}

// probeORTLibrary looks for the ONNX Runtime shared library in conventional
// install locations beyond hugot's single platform default — notably the
// Homebrew prefix on Apple Silicon, where "brew install onnxruntime" puts
// it. Empty means not found; hugot then applies its own default path (and
// its clear error when that is missing too).
func probeORTLibrary() string {
	var dirs []string
	switch runtime.GOOS {
	case "darwin":
		dirs = []string{"/opt/homebrew/lib", "/usr/local/lib"}
	case "windows":
		return "" // hugot's default (the working directory) is the convention
	default:
		dirs = []string{"/usr/local/lib", "/usr/lib"}
	}
	for _, dir := range dirs {
		lib := filepath.Join(dir, ortLibraryName())
		if _, err := os.Stat(lib); err == nil {
			return lib
		}
	}
	return ""
}

// ortLibraryName is the platform's ONNX Runtime shared library file name.
func ortLibraryName() string {
	switch runtime.GOOS {
	case "windows":
		return "onnxruntime.dll"
	case "darwin":
		return "libonnxruntime.dylib"
	default:
		return "libonnxruntime.so"
	}
}
