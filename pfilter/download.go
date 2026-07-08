package pfilter

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
)

// The privacy-filter GGUF artifacts published by LocalAI on Hugging Face.
// Each variant pins the upstream sha256 (the Git-LFS object id), so a
// download is verified before it is trusted.
const (
	// ModelQ8 is the base openai/privacy-filter model, Q8_0-quantized
	// (~1.6 GB): 8 PII categories, English. The recommended default.
	ModelQ8 = "q8"
	// ModelF16 is the base model in f16 (~2.8 GB): same categories,
	// reference precision.
	ModelF16 = "f16"
	// ModelMultilingualQ8 / ModelMultilingualF16 are the multilingual
	// fine-tune (54 categories, 16 languages).
	ModelMultilingualQ8  = "multilingual-q8"
	ModelMultilingualF16 = "multilingual-f16"
	// ModelNemotronQ8 / ModelNemotronF16 are the nemotron fine-tune.
	ModelNemotronQ8  = "nemotron-q8"
	ModelNemotronF16 = "nemotron-f16"
)

type modelArtifact struct {
	url    string
	sha256 string
}

var modelArtifacts = map[string]modelArtifact{
	ModelQ8: {
		url:    "https://huggingface.co/LocalAI-io/privacy-filter-GGUF/resolve/main/privacy-filter-q8.gguf",
		sha256: "80efc1803eda7c095a79741d2008c07e2e0a57b01bac8825fbeb448fd097998c",
	},
	ModelF16: {
		url:    "https://huggingface.co/LocalAI-io/privacy-filter-GGUF/resolve/main/privacy-filter-f16.gguf",
		sha256: "eb71312b6b9370d0fe582e576b840567bb06603c4de241c6d899205d1b04dc81",
	},
	ModelMultilingualQ8: {
		url:    "https://huggingface.co/LocalAI-io/privacy-filter-multilingual-GGUF/resolve/main/privacy-filter-multilingual-q8.gguf",
		sha256: "968135172ba8202374b4c3bd7d353e100c8fc574035da793fa4d13ca441319b7",
	},
	ModelMultilingualF16: {
		url:    "https://huggingface.co/LocalAI-io/privacy-filter-multilingual-GGUF/resolve/main/privacy-filter-multilingual-f16.gguf",
		sha256: "01b76572f80b7d2ebee80a27cb9c3699c26b04cae1c402eee7664fc17a4b5ce6",
	},
	ModelNemotronQ8: {
		url:    "https://huggingface.co/LocalAI-io/privacy-filter-nemotron-GGUF/resolve/main/privacy-filter-nemotron-q8.gguf",
		sha256: "2ec11c154e572a2686f4d77e861b7f74e6917e09638fe9bd27156d48bd99e21a",
	},
	ModelNemotronF16: {
		url:    "https://huggingface.co/LocalAI-io/privacy-filter-nemotron-GGUF/resolve/main/privacy-filter-nemotron-f16.gguf",
		sha256: "70dfe91ff220ff04594168a83e296dcc2054449cde77f98d0e782edbb6a31f5a",
	},
}

// EnsureModel returns the local path of a privacy-filter GGUF, downloading
// it into the user cache dir (~/.cache/alcatraz/models or the platform
// equivalent) on first use. The pinned sha256 is verified on every call —
// at download time and again on cache hits — so a corrupted cache entry is
// deleted and re-fetched instead of trusted. variant is one of the Model*
// constants. The download is large (1.6–2.8 GB) — pass a cancellable ctx if
// you need to bound it.
func EnsureModel(ctx context.Context, variant string) (string, error) {
	art, ok := modelArtifacts[variant]
	if !ok {
		return "", fmt.Errorf("pfilter: unknown model variant %q (see the pfilter.Model* constants)", variant)
	}
	dir, err := cacheDir("models")
	if err != nil {
		return "", err
	}
	dest := filepath.Join(dir, filepath.Base(art.url))
	if cachedFileValid(dest, art.sha256) {
		return dest, nil
	}
	if err := download(ctx, art.url, dest, art.sha256, 0o644); err != nil {
		return "", fmt.Errorf("pfilter: downloading model %s: %w", variant, err)
	}
	return dest, nil
}

// libraryVersion names the prebuilt-libpf release this module version
// downloads: a "libpf-*" tag on github.com/hoophq/alcatraz whose assets are
// produced by the libpf-release workflow. Bump it together with
// libraryChecksums when a new set is published.
const libraryVersion = "libpf-v1"

// libraryBaseURL is where EnsureLibrary fetches prebuilt binaries from;
// overridable for tests.
var libraryBaseURL = "https://github.com/hoophq/alcatraz/releases/download"

// libraryChecksums pins the sha256 of each published artifact, keyed by
// "GOOS-GOARCH". The values come from the checksums.txt of the
// libpf-release workflow run; pinning them in reviewed source (rather than
// trusting checksums.txt at download time) is what makes the download
// verifiable.
var libraryChecksums = map[string]string{
	// libpf-v1: built from localai-org/privacy-filter.cpp@735a6c2 by the
	// libpf-release workflow (run 28973430209).
	"darwin-arm64": "5b7aedb042244ce0ae6424e2dd9ad5400d251180e316539bd4022125596ad76b",
	"darwin-amd64": "a1cb3be9c0b851d5440ac5ece2a9fa3a61298bdd4a8bf864673feb71987fe097",
	"linux-amd64":  "c1fe627e825f5dc31fdcd0a686aac85f8b9a7238c5846bdea73adec8b189c280",
	"linux-arm64":  "a97f7a2b36edbc3dfbefae68b701b1002a43f2401c9ae3c4f934393f9f55f5a3",
}

// libraryArtifactName returns the release asset name for a platform, e.g.
// "libpf-darwin-arm64.dylib".
func libraryArtifactName(goos, goarch string) string {
	ext := ".so"
	if goos == "darwin" {
		ext = ".dylib"
	}
	return "libpf-" + goos + "-" + goarch + ext
}

// cachedLibraryPath is where EnsureLibrary stores (and loadLibrary looks
// for) the downloaded shared library.
func cachedLibraryPath() (string, error) {
	dir, err := cacheDir("lib", libraryVersion)
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, libraryArtifactName(runtime.GOOS, runtime.GOARCH)), nil
}

// EnsureLibrary returns the local path of a prebuilt privacy-filter.cpp
// shared library for this platform, downloading it from the alcatraz GitHub
// release on first use. The pinned sha256 is verified on every call — at
// download time and again on cache hits — so a corrupted or tampered cache
// entry is deleted and re-fetched instead of dlopen'ed. Once cached, New
// finds it without any configuration (loadLibrary checks the cache path);
// the returned path can also be set explicitly as Config.Library.
//
// Platforms without a published artifact get an error telling them to build
// from source (pfilter/dist) and point $PF_LIBRARY at the result.
func EnsureLibrary(ctx context.Context) (string, error) {
	key := runtime.GOOS + "-" + runtime.GOARCH
	sum, ok := libraryChecksums[key]
	if !ok {
		return "", fmt.Errorf(
			"pfilter: no prebuilt libpf published for %s; build it from source (see pfilter/dist) and set Config.Library or $PF_LIBRARY", key)
	}
	dest, err := cachedLibraryPath()
	if err != nil {
		return "", err
	}
	if cachedFileValid(dest, sum) {
		return dest, nil
	}
	url := libraryBaseURL + "/" + libraryVersion + "/" + libraryArtifactName(runtime.GOOS, runtime.GOARCH)
	if err := download(ctx, url, dest, sum, 0o755); err != nil {
		return "", fmt.Errorf("pfilter: downloading libpf: %w", err)
	}
	return dest, nil
}

// cachedFileValid reports whether path exists and still matches the pinned
// sha256 — cache hits are re-verified on every Ensure call, so an entry
// corrupted after download (disk fault, truncation, tampering) is caught
// even though it was verified when it landed. A mismatched file is deleted,
// which both triggers a fresh download here and keeps loadLibrary's direct
// cache-path probe from dlopen-ing it.
//
// For the GGUF models this hashes 1.6–2.8 GB (~1 s with hardware SHA-256) —
// small next to loading the model, but not free; pass Config.ModelPath
// straight to New to skip EnsureModel entirely if that matters.
func cachedFileValid(path, wantSHA256 string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	hasher := sha256.New()
	if _, err := io.Copy(hasher, f); err != nil {
		return false
	}
	if hex.EncodeToString(hasher.Sum(nil)) == wantSHA256 {
		return true
	}
	os.Remove(path)
	return false
}

// cacheDir returns (creating if needed) a subdirectory of the user cache
// dir, e.g. cacheDir("models") -> ~/.cache/alcatraz/models on Linux.
func cacheDir(parts ...string) (string, error) {
	base, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(append([]string{base, "alcatraz"}, parts...)...)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	return dir, nil
}

// download fetches url into dest atomically: it streams to a temp file in
// the destination directory, verifies the sha256, sets mode, and renames.
// A failed or corrupt download never leaves a file at dest.
func download(ctx context.Context, url, dest, wantSHA256 string, mode os.FileMode) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", url, resp.Status)
	}

	tmp, err := os.CreateTemp(filepath.Dir(dest), filepath.Base(dest)+".partial-*")
	if err != nil {
		return err
	}
	defer func() {
		tmp.Close()
		os.Remove(tmp.Name()) // no-op after successful rename
	}()

	hasher := sha256.New()
	if _, err := io.Copy(io.MultiWriter(tmp, hasher), resp.Body); err != nil {
		return err
	}
	if got := hex.EncodeToString(hasher.Sum(nil)); got != wantSHA256 {
		return fmt.Errorf("sha256 mismatch for %s: got %s, want %s", url, got, wantSHA256)
	}
	if err := tmp.Chmod(mode); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), dest)
}
