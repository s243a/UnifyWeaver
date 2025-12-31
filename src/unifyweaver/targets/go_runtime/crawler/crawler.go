package crawler

import (
	"bufio"
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"unifyweaver/targets/go_runtime/utils"
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

				// Embed text - try title, about, then text (namespace-agnostic)
				if c.embedder != nil {
					text := utils.GetLocalString(data, "title")
					if text == "" {
						text = utils.GetLocalString(data, "about")
					}
					if text == "" {
						if t, ok := data["text"].(string); ok {
							text = t
						}
					}
					if text != "" {
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

// ProcessFragmentsFromStdin reads null-delimited XML fragments from stdin
// This enables AWK-based ingestion where AWK filters and extracts fragments
// Usage: awk -f extract_fragments.sh input.rdf | go_crawler
func (c *Crawler) ProcessFragmentsFromStdin() error {
	return c.ProcessFragments(os.Stdin)
}

// ProcessFragments reads null-delimited XML fragments from a reader
// Each fragment is a complete XML element separated by null bytes (\0)
func (c *Crawler) ProcessFragments(r io.Reader) error {
	scanner := bufio.NewScanner(r)
	scanner.Split(scanNullDelimited)

	// Set larger buffer for XML fragments (default 64KB may be too small)
	const maxFragmentSize = 1024 * 1024 // 1MB
	buf := make([]byte, maxFragmentSize)
	scanner.Buffer(buf, maxFragmentSize)

	count := 0
	for scanner.Scan() {
		fragment := scanner.Bytes()
		if len(fragment) == 0 {
			continue
		}

		if err := c.processFragment(fragment); err != nil {
			fmt.Fprintf(os.Stderr, "Error processing fragment: %v\n", err)
			continue
		}

		count++
		if count%100 == 0 {
			fmt.Fprintf(os.Stderr, "Processed %d fragments...\n", count)
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("scanner error: %w", err)
	}

	fmt.Fprintf(os.Stderr, "✓ Processed %d total fragments\n", count)
	return nil
}

// processFragment parses and stores a single XML fragment
func (c *Crawler) processFragment(fragment []byte) error {
	decoder := xml.NewDecoder(bytes.NewReader(fragment))

	for {
		t, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("xml decode: %w", err)
		}

		switch se := t.(type) {
		case xml.StartElement:
			var node XmlNode
			if err := decoder.DecodeElement(&node, &se); err != nil {
				return fmt.Errorf("decode element: %w", err)
			}

			data := FlattenXML(node)

			// Extract ID
			id, _ := data["@id"].(string)
			if id == "" {
				id, _ = data["@rdf:about"].(string)
			}
			if id == "" {
				id, _ = data["@about"].(string)
			}

			if id != "" {
				if err := c.store.UpsertObject(id, data); err != nil {
					return fmt.Errorf("upsert object: %w", err)
				}

				// Embed text - try title, about, then text (namespace-agnostic)
				if c.embedder != nil {
					text := utils.GetLocalString(data, "title")
					if text == "" {
						text = utils.GetLocalString(data, "about")
					}
					if text == "" {
						if t, ok := data["text"].(string); ok {
							text = t
						}
					}
					if text != "" {
						vec, err := c.embedder.Embed(text)
						if err == nil && vec != nil {
							c.store.UpsertEmbedding(id, vec)
						}
					}
				}
			}
		}
	}

	return nil
}

// scanNullDelimited is a split function for bufio.Scanner that splits on null bytes
func scanNullDelimited(data []byte, atEOF bool) (advance int, token []byte, err error) {
	if atEOF && len(data) == 0 {
		return 0, nil, nil
	}

	// Look for null delimiter
	if i := bytes.IndexByte(data, 0); i >= 0 {
		// Found null byte - return data up to (but not including) the null
		return i + 1, data[0:i], nil
	}

	// If at EOF, return remaining data
	if atEOF {
		return len(data), data, nil
	}

	// Request more data
	return 0, nil, nil
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
