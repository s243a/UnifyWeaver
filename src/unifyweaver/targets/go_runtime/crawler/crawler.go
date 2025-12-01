package crawler

import (
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Interfaces to decouple from specific implementations
type ObjectStore interface {
	UpsertObject(id string, data map[string]interface{}) error
	UpsertEmbedding(id string, vector []float32) error
}

type Embedder interface {
	Embed(text string) ([]float32, error)
}

type Crawler struct {
	store    ObjectStore
	embedder Embedder
	client   *http.Client
	seen     map[string]bool
}

func NewCrawler(store ObjectStore, embedder Embedder) *Crawler {
	return &Crawler{
		store:    store,
		embedder: embedder,
		client:   &http.Client{Timeout: 10 * time.Second},
		seen:     make(map[string]bool),
	}
}

func (c *Crawler) Crawl(seeds []string, maxDepth int) {
	frontier := seeds
	
	for depth := 0; depth < maxDepth; depth++ {
		var nextBatch []string
		
		for _, url := range frontier {
			if c.seen[url] {
				continue
			}
			c.seen[url] = true
			
			fmt.Printf("Crawling %s...\n", url)
			links, err := c.processURL(url)
			if err != nil {
				fmt.Printf("Error crawling %s: %v\n", url, err)
				continue
			}
			
			nextBatch = append(nextBatch, links...)
		}
		
		frontier = nextBatch
		if len(frontier) == 0 {
			break
		}
	}
}

func (c *Crawler) processURL(url string) ([]string, error) {
	resp, err := c.client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	return c.processStream(resp.Body)
}

func (c *Crawler) processStream(r io.Reader) ([]string, error) {
	decoder := xml.NewDecoder(r)
	var links []string
	
	for {
		t, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		
		switch se := t.(type) {
		case xml.StartElement:
			// Heuristic: Flatten any element that looks like a record
			// In a real scenario, we'd filter by tag
			var node XmlNode
			if err := decoder.DecodeElement(&node, &se); err != nil {
				continue
			}
			
			data := FlattenXML(node)
			
			// Extract ID
			id, _ := data["@id"].(string)
			if id == "" {
				id, _ = data["@rdf:about"].(string)
			}
			
			if id != "" {
				c.store.UpsertObject(id, data)
				
				// Embed text
				if c.embedder != nil {
					if text, ok := data["text"].(string); ok && text != "" {
						vec, err := c.embedder.Embed(text)
						if err == nil && vec != nil {
							c.store.UpsertEmbedding(id, vec)
						}
					}
				}
			}
			
			// Collect links (simplified)
			// In real app, we'd inspect data for links
		}
	}
	
	return links, nil
}

// XML Helpers

type XmlNode struct {
	XMLName xml.Name
	Attrs   []xml.Attr `xml:",any,attr"`
	Content string     `xml:",chardata"`
	Nodes   []XmlNode  `xml:",any"`
}

func FlattenXML(n XmlNode) map[string]interface{} {
	m := make(map[string]interface{})
	for _, a := range n.Attrs {
		m["@"+a.Name.Local] = a.Value
	}
	trim := strings.TrimSpace(n.Content)
	if trim != "" {
		m["text"] = trim
	}
	for _, child := range n.Nodes {
		tag := child.XMLName.Local
		flatChild := FlattenXML(child)
		
		if existing, ok := m[tag]; ok {
			if list, isList := existing.([]interface{}); isList {
				m[tag] = append(list, flatChild)
			} else {
				m[tag] = []interface{}{existing, flatChild}
			}
		} else {
			m[tag] = flatChild
		}
	}
	return m
}
