from collections import deque
import sys
try:
    from lxml import etree
except ImportError:
    sys.stderr.write("Error: lxml required for crawler\n")
    sys.exit(1)

class PtCrawler:
    def __init__(self, importer, embedder=None):
        self.importer = importer
        self.embedder = embedder
        self.seen = set()

    def crawl(self, seed_ids, fetch_func, max_depth=5):
        frontier = deque(seed_ids)
        self.seen.update(seed_ids)
        depth = 0
        
        while frontier and depth < max_depth:
            next_batch = []
            print(f"Depth {depth}: Processing {len(frontier)} items", file=sys.stderr)
            
            while frontier:
                obj_id = frontier.popleft()
                
                try:
                    xml_stream = fetch_func(obj_id)
                    if not xml_stream:
                        continue
                        
                    self._process_stream(xml_stream, next_batch)
                except Exception as e:
                    print(f"Error processing {obj_id}: {e}", file=sys.stderr)
            
            for kid in next_batch:
                frontier.append(kid)
            depth += 1

    def _process_stream(self, xml_stream, next_batch):
        context = etree.iterparse(xml_stream, events=('end',), recover=True)
        for event, elem in context:
            # Simple flattening (similar to read_xml_lxml)
            data = {}
            for k, v in elem.attrib.items():
                data['@' + k] = v
            
            tag = elem.tag.split('}')[-1] # Local name
            obj_id = data.get('@id') or data.get('@rdf:about') or data.get('@about')
            
            # If no ID, maybe generate one or skip? C# PtMapper extracts IDs.
            # For now, assume @id exists or skip.
            if not obj_id:
                # Check if it's a link
                pass
            else:
                self.importer.upsert_object(obj_id, tag, data)
                
                # Embeddings
                if self.embedder:
                    text = data.get('title') or data.get('text') or ""
                    if text:
                        vec = self.embedder.get_embedding(text)
                        self.importer.upsert_embedding(obj_id, vec)
                
                # Links (Children)
                # If children are in data? No, children are sub-elements.
                # In lxml iterparse 'end', children are already processed?
                # Flattening logic usually handles children.
                # We need to extract links.
                # Assume specific link logic (e.g. pt:seeAlso) or generic.
                pass
            
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
