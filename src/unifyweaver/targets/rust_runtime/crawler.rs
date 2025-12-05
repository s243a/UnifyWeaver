use quick_xml::events::Event;
use quick_xml::reader::Reader;
use serde_json::{Map, Value};
use std::fs::File;
use std::io::BufReader;
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
                    
                    if !obj_id.is_empty() {
                        current_obj = Some(obj);
                        current_tag = tag;
                    }
                }
                Ok(Event::Text(e)) => {
                    if let Some(ref mut obj) = current_obj {
                        let text = e.unescape()?.to_string();
                        if !text.is_empty() {
                            obj.insert("text".to_string(), Value::String(text));
                        }
                    }
                }
                Ok(Event::End(e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref());
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
