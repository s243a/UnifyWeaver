package embedder

import (
	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
)

type HugotEmbedder struct {
	session  *hugot.Session
	pipeline *pipelines.FeatureExtractionPipeline
}

func NewHugotEmbedder(modelPath string, name string) (*HugotEmbedder, error) {
	// Use pure Go session (no C dependencies)
	session, err := hugot.NewGoSession()
	if err != nil {
		return nil, err
	}

	config := hugot.FeatureExtractionConfig{
		ModelPath: modelPath,
		Name:      name,
	}

	pipeline, err := hugot.NewPipeline(session, config)
	if err != nil {
		session.Destroy()
		return nil, err
	}

	return &HugotEmbedder{session: session, pipeline: pipeline}, nil
}

func (e *HugotEmbedder) Embed(text string) ([]float32, error) {
	output, err := e.pipeline.RunPipeline([]string{text})
	if err != nil {
		return nil, err
	}

	if output == nil || len(output.Embeddings) == 0 {
		return nil, nil
	}

	return output.Embeddings[0], nil
}

func (e *HugotEmbedder) Close() {
	if e.session != nil {
		e.session.Destroy()
	}
}
