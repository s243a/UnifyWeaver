use quick_xml::events::Event;
use quick_xml::reader::Reader;
use serde_json::{Map, Value};
use std::fs::File;
use std::io::{self, BufReader, Read};
use crate::importer::PtImporter;
use crate::embedding::EmbeddingProvider;

pub struct PtCrawler {
    importer: PtImporter,
    embedder: EmbeddingProvider,
}

impl PtCrawler {
    pub fn new(importer: PtImporter, embedder: EmbeddingProvider) -> Self {
        Self { importer, embedder }
    }

    pub fn crawl(&self, seeds: &[String], _max_depth: usize) -> Result<(), Box<dyn std::error::Error>> {
        for seed in seeds {
            // Assume seed is a file path for now
            println!("Processing {}", seed);
            self.process_file(seed)?;
        }
        Ok(())
    }

    fn process_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut reader = Reader::from_reader(BufReader::new(file));
        reader.config_mut().trim_text(true);

        let mut buf = Vec::new();
        let mut current_obj: Option<Map<String, Value>> = None;
        let mut current_tag = String::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    // Flatten: Only care about leaf nodes or direct children of root?
                    // Strategy: Every element with an ID is an object.
                    // Attributes are props. Children text is props.
                    
                    let mut obj = Map::new();
                    let mut obj_id = String::new();
                    
                    // Extract attributes
                    for attr in e.attributes() {
                        let attr = attr?;
                        let key = String::from_utf8_lossy(attr.key.as_ref());
                        let val = String::from_utf8_lossy(&attr.value);
                        
                        if key == "rdf:about" || key == "id" || key.ends_with("about") {
                            obj_id = val.to_string();
                        }
                        
                        // Link extraction
                        if key == "rdf:resource" || key.ends_with("resource") {
                            // Store link logic handled later or here?
                            // If this is a child element, the parent ID is needed.
                            // Current simplistic approach: Flat stream.
                        }
                        
                        obj.insert(format!("@{}", key), Value::String(val.to_string()));
                    }
                    
                    // Only track top-level objects that have an ID
                    // This avoids nested elements overwriting the parent
                    if !obj_id.is_empty() && current_obj.is_none() {
                        current_obj = Some(obj);
                        current_tag = tag;
                    }
                }
                Ok(Event::Text(e)) => {
                    if let Some(ref mut obj) = current_obj {
                        let text = std::str::from_utf8(e.as_ref()).unwrap_or("").to_string();
                        if !text.is_empty() {
                            obj.insert("text".to_string(), Value::String(text));
                        }
                    }
                }
                Ok(Event::End(e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    if let Some(obj) = current_obj.take() {
                        if tag == current_tag {
                            // Object complete
                            if let Some(Value::String(id)) = obj.get("@rdf:about").or(obj.get("@id")) {
                                self.importer.upsert_object(id, &tag, &Value::Object(obj.clone()))?;

                                // Generate Embedding
                                if let Some(Value::String(text)) = obj.get("title").or(obj.get("text")) {
                                    if let Ok(vec) = self.embedder.get_embedding(text) {
                                        self.importer.upsert_embedding(id, &vec)?;
                                    }
                                }
                            }
                        } else {
                            // Put it back if it's not the matching end tag (handles nested elements)
                            current_obj = Some(obj);
                        }
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(Box::new(e)),
                _ => (),
            }
            buf.clear();
        }
        Ok(())
    }

    /// Process null-delimited XML fragments from stdin
    /// This enables AWK-based ingestion where AWK filters and extracts fragments
    /// Usage: awk -f extract_fragments.sh input.rdf | rust_crawler
    pub fn process_fragments_from_stdin(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.process_fragments(&mut io::stdin())
    }

    /// Process null-delimited XML fragments from a reader
    /// Each fragment is a complete XML element separated by null bytes (\0)
    pub fn process_fragments<R: Read>(&self, reader: &mut R) -> Result<(), Box<dyn std::error::Error>> {
        let mut buffer = Vec::new();
        let mut fragment = Vec::new();
        let mut count = 0;

        // Read byte by byte, splitting on null delimiter
        for byte_result in reader.bytes() {
            let byte = byte_result?;

            if byte == 0 {
                // Found null delimiter - process accumulated fragment
                if !fragment.is_empty() {
                    if let Err(e) = self.process_fragment(&fragment) {
                        eprintln!("Error processing fragment: {}", e);
                    }
                    count += 1;

                    if count % 100 == 0 {
                        eprintln!("Processed {} fragments...", count);
                    }

                    fragment.clear();
                }
            } else {
                fragment.push(byte);
            }
        }

        // Process final fragment if exists (no trailing null)
        if !fragment.is_empty() {
            if let Err(e) = self.process_fragment(&fragment) {
                eprintln!("Error processing fragment: {}", e);
            }
            count += 1;
        }

        eprintln!("✓ Processed {} total fragments", count);
        Ok(())
    }

    /// Process a single XML fragment
    fn process_fragment(&self, fragment: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        let mut reader = Reader::from_reader(fragment);
        reader.config_mut().trim_text(true);

        let mut buf = Vec::new();
        let mut obj: Option<Map<String, Value>> = None;
        let mut obj_id = String::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    let mut current_obj = Map::new();

                    // Extract attributes
                    for attr in e.attributes() {
                        let attr = attr?;
                        let key = String::from_utf8_lossy(attr.key.as_ref());
                        let val = String::from_utf8_lossy(&attr.value);

                        if key == "rdf:about" || key == "id" || key.ends_with("about") {
                            obj_id = val.to_string();
                        }

                        current_obj.insert(format!("@{}", key), Value::String(val.to_string()));
                    }

                    // Only track top-level object if it has an ID
                    if !obj_id.is_empty() && obj.is_none() {
                        obj = Some(current_obj);
                    }
                }
                Ok(Event::Text(e)) => {
                    if let Some(ref mut current_obj) = obj {
                        let text = std::str::from_utf8(e.as_ref()).unwrap_or("").trim().to_string();
                        if !text.is_empty() {
                            current_obj.insert("text".to_string(), Value::String(text));
                        }
                    }
                }
                Ok(Event::End(_)) => {
                    if let Some(current_obj) = obj.take() {
                        // Extract ID from attributes
                        let id = current_obj.get("@rdf:about")
                            .or(current_obj.get("@id"))
                            .or(current_obj.get("@about"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");

                        if !id.is_empty() {
                            // Get tag for object type (could extract from fragment)
                            let tag = "unknown";
                            self.importer.upsert_object(id, tag, &Value::Object(current_obj.clone()))?;

                            // Generate embedding if we have text
                            if let Some(Value::String(text)) = current_obj.get("title").or(current_obj.get("text")) {
                                if let Ok(vec) = self.embedder.get_embedding(text) {
                                    self.importer.upsert_embedding(id, &vec)?;
                                }
                            }
                        }
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(Box::new(e)),
                _ => (),
            }
            buf.clear();
        }

        Ok(())
    }
}
