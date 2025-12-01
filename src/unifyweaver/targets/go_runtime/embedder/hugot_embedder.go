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
	session, err := hugot.NewSession()
	if err != nil {
		return nil, err
	}

	config := pipelines.FeatureExtractionConfig{
		ModelPath: modelPath,
		Name:      name,
	}

	pipeline, err := session.NewFeatureExtractionPipeline(config)
	if err != nil {
		session.Destroy()
		return nil, err
	}

	return &HugotEmbedder{session: session, pipeline: pipeline}, nil
}

func (e *HugotEmbedder) Embed(text string) ([]float32, error) {
	batch := []string{text}
	output, err := e.pipeline.Run(batch)
	if err != nil {
		return nil, err
	}
	
	if len(output.Embeddings) == 0 {
		return nil, nil
	}

	return output.Embeddings[0], nil
}

func (e *HugotEmbedder) Close() {
	if e.session != nil {
		e.session.Destroy()
	}
}
