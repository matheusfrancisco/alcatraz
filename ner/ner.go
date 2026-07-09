// Package ner provides a statistical NER backend for alcatraz: it detects
// free-text entities — PERSON, LOCATION, NRP, DATE_TIME — that pattern
// recognizers cannot express, using an ONNX token-classification model run
// in-process via hugot.
//
// It lives in a separate module on purpose (mirroring alcatraz/lookaround):
// importing it is the only way to pull in the model runtime, so the alcatraz
// core stays dependency-free. The default hugot backend is pure Go — no cgo,
// no shared libraries. For higher throughput, build with hugot's ORT backend
// ("-tags ORT") and an ONNX Runtime shared library.
//
// The Engine implements analyzer.NlpEngine, so an analyzer.Engine configured
// with SetNlpEngine runs the model once per Analyze call and shares the
// artifacts with every artifact-aware recognizer:
//
//	nlp, err := ner.New(ctx, ner.DefaultConfig())
//	// handle err; model is downloaded on first use
//	defer nlp.Close()
//
//	reg := analyzer.NewRegistry("en")
//	recognizers.LoadDefaults(reg, "en")
//	reg.Add("en", nlp.Recognizer("en"))
//
//	eng := analyzer.NewEngine(reg, []string{"en"})
//	eng.SetNlpEngine(nlp)
package ner

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/hoophq/alcatraz/analyzer"
	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
)

// Engine runs an ONNX token-classification model and implements
// analyzer.NlpEngine. Create it with New and release it with Close. Per
// hugot's guidance a single pipeline may be called from multiple goroutines,
// so Engine is safe for concurrent use after construction.
type Engine struct {
	// runCtx is used for inference calls. It is derived from New's ctx
	// with context.WithoutCancel: the construction ctx bounds the model
	// download and session setup only, and cancelling it afterwards must
	// not poison every future ProcessText. The engine's lifetime is
	// governed by Close, not by a context.
	runCtx   context.Context
	session  *hugot.Session
	pipeline *pipelines.TokenClassificationPipeline
	cfg      Config
}

// New loads (downloading first if needed) the configured model and fails
// fast on any model problem, so a constructed Engine can always run. Zero
// fields of cfg fall back to DefaultConfig values.
//
// ctx bounds construction only (model download, session setup); cancelling
// it after New returns does not affect the engine.
func New(ctx context.Context, cfg Config) (*Engine, error) {
	def := DefaultConfig()
	if cfg.Model == "" && cfg.ModelPath == "" {
		cfg.Model = def.Model
	}
	if cfg.LabelMapping == nil {
		cfg.LabelMapping = def.LabelMapping
		if cfg.LabelsToIgnore == nil {
			cfg.LabelsToIgnore = def.LabelsToIgnore
		}
	}

	modelPath := cfg.ModelPath
	if modelPath == "" {
		var err error
		modelPath, err = ensureModel(ctx, cfg)
		if err != nil {
			return nil, fmt.Errorf("ner: obtaining model %s: %w", cfg.Model, err)
		}
	}

	// hugot's session keeps the ctx it was created with and uses it for
	// inference, so it must not inherit the caller's cancellation either —
	// only the model download above is bound to the construction ctx.
	runCtx := context.WithoutCancel(ctx)
	session, err := hugot.NewGoSession(runCtx)
	if err != nil {
		return nil, fmt.Errorf("ner: creating hugot session: %w", err)
	}

	pipeline, err := hugot.NewPipeline(session, hugot.TokenClassificationConfig{
		ModelPath:    modelPath,
		Name:         "alcatraz-ner",
		OnnxFilename: cfg.OnnxFilename,
		Options: []hugot.TokenClassificationOption{
			pipelines.WithSimpleAggregation(),
		},
	})
	if err != nil {
		// The session owns no other pipelines; release it.
		_ = session.Destroy()
		return nil, fmt.Errorf("ner: creating token classification pipeline: %w", err)
	}

	return &Engine{runCtx: runCtx, session: session, pipeline: pipeline, cfg: cfg}, nil
}

// ensureModel returns the local path of cfg.Model, downloading it from
// Hugging Face into the models directory on first use.
func ensureModel(ctx context.Context, cfg Config) (string, error) {
	dir := cfg.ModelsDir
	if dir == "" {
		cache, err := os.UserCacheDir()
		if err != nil {
			return "", err
		}
		dir = filepath.Join(cache, "alcatraz", "models")
	}
	// DownloadModel stores the model under this derived path; when it is
	// already populated, skip the network entirely.
	modelPath := filepath.Join(dir, strings.ReplaceAll(cfg.Model, "/", "_"))
	if _, err := os.Stat(filepath.Join(modelPath, "tokenizer.json")); err == nil {
		return modelPath, nil
	}
	return hugot.DownloadModel(ctx, cfg.Model, dir, hugot.NewDownloadOptions())
}

// Close releases the model and its runtime resources.
func (e *Engine) Close() error {
	return e.session.Destroy()
}

// Config returns the engine's effective configuration.
func (e *Engine) Config() Config { return e.cfg }

// ProcessText implements analyzer.NlpEngine: it runs the model over text and
// returns the detected entity spans mapped to canonical entity names with
// byte offsets. The language parameter is currently unused (one model per
// engine); it is part of the interface for engines that route per language.
//
// The model sees an ASCII-folded rendering of the text (see foldASCII) and
// the reported spans are mapped back to byte offsets in the original text,
// so the alcatraz invariant text[Start:End] == matched span holds for any
// input.
func (e *Engine) ProcessText(text, language string) (*analyzer.NlpArtifacts, error) {
	all, err := e.ProcessTexts([]string{text}, language)
	if err != nil {
		return nil, err
	}
	return all[0], nil
}

// ProcessTexts implements analyzer.BatchNlpEngine: it runs the model over all
// texts in one inference call and returns one NlpArtifacts per text, in input
// order. Batching amortizes the per-call tokenization and graph overhead, so
// it is substantially faster than calling ProcessText in a loop; the spans of
// each text carry the same byte-offset guarantee as ProcessText.
func (e *Engine) ProcessTexts(texts []string, language string) ([]*analyzer.NlpArtifacts, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	folded := make([]string, len(texts))
	foldOffsets := make([][]int, len(texts))
	for i, text := range texts {
		folded[i], foldOffsets[i] = foldASCII(text)
	}
	out, err := e.pipeline.RunPipeline(e.runCtx, folded)
	if err != nil {
		return nil, err
	}
	artifacts := make([]*analyzer.NlpArtifacts, len(texts))
	for i := range texts {
		artifacts[i] = &analyzer.NlpArtifacts{}
	}
	for i, ents := range out.Entities {
		if i >= len(texts) {
			break
		}
		for _, ent := range ents {
			span, ok := e.toNerSpan(texts[i], foldOffsets[i], ent)
			if !ok {
				continue
			}
			artifacts[i].Ents = append(artifacts[i].Ents, span)
		}
	}
	return artifacts, nil
}

// toNerSpan converts one hugot entity into a NerSpan: label mapping, ignore
// list, score calibration, fold-offset remapping and byte-span validation.
// ok is false when the entity is dropped.
func (e *Engine) toNerSpan(text string, foldOffsets []int, ent pipelines.Entity) (analyzer.NerSpan, bool) {
	label := ent.Entity
	if e.ignored(label) {
		return analyzer.NerSpan{}, false
	}
	entity, ok := e.cfg.LabelMapping[label]
	if !ok {
		// Keep unmapped labels as-is (Presidio's behavior); users silence
		// them via LabelsToIgnore.
		entity = label
	}
	if e.ignored(entity) {
		return analyzer.NerSpan{}, false
	}

	score := float64(ent.Score)
	if e.cfg.LowScoreMultiplier > 0 {
		for _, low := range e.cfg.LowScoreEntities {
			if low == entity {
				score *= e.cfg.LowScoreMultiplier
				break
			}
		}
	}
	if score > analyzer.MaxScore {
		score = analyzer.MaxScore
	}
	if score <= analyzer.MinScore {
		return analyzer.NerSpan{}, false
	}

	start, end := remapSpan(foldOffsets, len(text), int(ent.Start), int(ent.End))
	start, end, ok = byteSpan(text, start, end)
	if !ok {
		return analyzer.NerSpan{}, false
	}
	return analyzer.NerSpan{EntityType: entity, Start: start, End: end, Score: score}, true
}

func (e *Engine) ignored(label string) bool {
	for _, l := range e.cfg.LabelsToIgnore {
		if l == label {
			return true
		}
	}
	return false
}

// Recognizer returns an analyzer.Recognizer emitting this engine's entities
// for the given language. Register it in a Registry alongside the pattern
// recognizers; pair it with Engine.SetNlpEngine so the model runs once per
// Analyze call.
func (e *Engine) Recognizer(language string) *Recognizer {
	return &Recognizer{
		engine:   e,
		language: language,
	}
}

// Recognizer adapts an Engine to the analyzer.Recognizer contract. It is the
// thin consumer of NlpArtifacts: when the analyzer.Engine provides shared
// artifacts the model has already run and AnalyzeWithArtifacts just converts
// spans; the plain Analyze path runs the model itself so the recognizer also
// works in an engine without SetNlpEngine.
type Recognizer struct {
	engine   *Engine
	language string
}

// Name implements analyzer.Recognizer.
func (r *Recognizer) Name() string { return "NERecognizer" }

// SupportedEntities implements analyzer.Recognizer.
func (r *Recognizer) SupportedEntities() []string { return r.engine.cfg.SupportedEntities() }

// SupportedLanguage implements analyzer.Recognizer.
func (r *Recognizer) SupportedLanguage() string { return r.language }

// Analyze implements analyzer.Recognizer by running the model directly. An
// inference error yields no results, matching the engine's degrade-to-
// patterns behavior.
func (r *Recognizer) Analyze(text string, entities []string) []analyzer.Result {
	if entities != nil && !supportsAny(r.SupportedEntities(), entities) {
		return nil
	}
	artifacts, err := r.engine.ProcessText(text, r.language)
	if err != nil {
		return nil
	}
	return r.AnalyzeWithArtifacts(text, entities, artifacts)
}

// AnalyzeWithArtifacts implements analyzer.ArtifactRecognizer.
func (r *Recognizer) AnalyzeWithArtifacts(text string, entities []string, artifacts *analyzer.NlpArtifacts) []analyzer.Result {
	var results []analyzer.Result
	for _, ent := range artifacts.Ents {
		if entities != nil && !contains(entities, ent.EntityType) {
			continue
		}
		results = append(results, analyzer.Result{
			EntityType:     ent.EntityType,
			Start:          ent.Start,
			End:            ent.End,
			Score:          ent.Score,
			RecognizerName: r.Name(),
		})
	}
	return results
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

func supportsAny(have, want []string) bool {
	for _, w := range want {
		if contains(have, w) {
			return true
		}
	}
	return false
}
