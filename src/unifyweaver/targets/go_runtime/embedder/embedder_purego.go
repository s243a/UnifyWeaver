//go:build !candle && !ort && !xla

package embedder

import (
	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
)

// availableBackend indicates this is the pure Go backend
var availableBackend = BackendPureGo

// pureGoEmbedder uses hugot's pure Go backend (no C dependencies)
type pureGoEmbedder struct {
	session  *hugot.Session
	pipeline *pipelines.FeatureExtractionPipeline
}

// newEmbedder creates a pure Go embedder using hugot's GoSession
func newEmbedder(config EmbedderConfig) (Embedder, error) {
	// Use pure Go session (no C dependencies)
	session, err := hugot.NewGoSession()
	if err != nil {
		return nil, err
	}

	hugotConfig := hugot.FeatureExtractionConfig{
		ModelPath: config.ModelPath,
		Name:      config.ModelName,
	}

	pipeline, err := hugot.NewPipeline(session, hugotConfig)
	if err != nil {
		session.Destroy()
		return nil, err
	}

	return &pureGoEmbedder{session: session, pipeline: pipeline}, nil
}

func (e *pureGoEmbedder) Embed(text string) ([]float32, error) {
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

func (e *pureGoEmbedder) Close() {
	if e.session != nil {
		e.session.Destroy()
	}
}
