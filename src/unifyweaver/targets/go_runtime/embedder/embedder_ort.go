//go:build ort || ORT

package embedder

import (
	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
)

// availableBackend indicates this is the ONNX Runtime backend
var availableBackend = BackendORT

// ortEmbedder uses hugot's ONNX Runtime backend (requires C dependencies)
type ortEmbedder struct {
	session  *hugot.Session
	pipeline *pipelines.FeatureExtractionPipeline
}

// newEmbedder creates an ONNX Runtime embedder
// Requires: onnxruntime.so and libtokenizers.a
func newEmbedder(config EmbedderConfig) (Embedder, error) {
	// Use ONNX Runtime session (requires C dependencies)
	// This requires:
	// 1. onnxruntime.so in /usr/lib/ or LD_LIBRARY_PATH
	// 2. libtokenizers.a linked at build time (from daulet/tokenizers)
	session, err := hugot.NewORTSession()
	if err != nil {
		return nil, err
	}

	// Default to model.onnx if no specific ONNX file is specified
	onnxFile := config.OnnxFilename
	if onnxFile == "" {
		onnxFile = "model.onnx"
	}

	hugotConfig := hugot.FeatureExtractionConfig{
		ModelPath:    config.ModelPath,
		Name:         config.ModelName,
		OnnxFilename: onnxFile,
	}

	pipeline, err := hugot.NewPipeline(session, hugotConfig)
	if err != nil {
		session.Destroy()
		return nil, err
	}

	return &ortEmbedder{session: session, pipeline: pipeline}, nil
}

func (e *ortEmbedder) Embed(text string) ([]float32, error) {
	if text == "" {
		return nil, nil
	}

	output, err := e.pipeline.RunPipeline([]string{text})
	if err != nil {
		return nil, err
	}

	if output == nil || len(output.Embeddings) == 0 {
		return nil, nil
	}

	return output.Embeddings[0], nil
}

func (e *ortEmbedder) Close() {
	if e.session != nil {
		e.session.Destroy()
	}
}
