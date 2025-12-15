//go:build xla || XLA

package embedder

import (
	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
)

// availableBackend indicates this is the XLA/PJRT backend
var availableBackend = BackendXLA

// xlaEmbedder uses hugot's XLA backend (requires C++ PJRT dependencies)
type xlaEmbedder struct {
	session  *hugot.Session
	pipeline *pipelines.FeatureExtractionPipeline
}

// newEmbedder creates an XLA/PJRT embedder
// Requires: PJRT plugin libraries from GoMLX
// Install: curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_linux_amd64.sh | bash
// For GPU: curl -sSf https://raw.githubusercontent.com/gomlx/gopjrt/main/cmd/install_cuda.sh | bash
func newEmbedder(config EmbedderConfig) (Embedder, error) {
	// Build session options
	var sessionOpts []options.WithOption
	if config.UseGPU {
		// Enable CUDA for GPU acceleration
		sessionOpts = append(sessionOpts, options.WithCuda(nil))
	}

	// Use XLA session (requires PJRT C++ libraries)
	session, err := hugot.NewXLASession(sessionOpts...)
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

	return &xlaEmbedder{session: session, pipeline: pipeline}, nil
}

func (e *xlaEmbedder) Embed(text string) ([]float32, error) {
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

func (e *xlaEmbedder) Close() {
	if e.session != nil {
		e.session.Destroy()
	}
}
