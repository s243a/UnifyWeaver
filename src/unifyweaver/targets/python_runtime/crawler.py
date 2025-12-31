from collections import deque
import sys
try:
    from lxml import etree
except ImportError:
    sys.stderr.write("Error: lxml required for crawler\n")
    sys.exit(1)

from .utils import get_local

class PtCrawler:
    def __init__(self, importer, embedder=None):
        self.importer = importer
        self.embedder = embedder
        self.seen = set()

    def crawl(self, seed_ids, fetch_func, max_depth=5, embed_content=True):
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
                        
                    self._process_stream(xml_stream, next_batch, embed_content=embed_content)
                except Exception as e:
                    print(f"Error processing {obj_id}: {e}", file=sys.stderr)
            
            for kid in next_batch:
                frontier.append(kid)
            depth += 1

    def _process_stream(self, xml_stream, next_batch, embed_content=True):
        context = etree.iterparse(xml_stream, events=('end',), recover=True)
        count = 0
        limit = 50 # Safety limit for testing
        
        for event, elem in context:
            count += 1
            if count > limit:
                print(f"DEBUG: Reached limit of {limit} items. Stopping stream processing.", file=sys.stderr)
                break
                
            # Simple flattening (similar to read_xml_lxml)
            data = {}
            # Root element attributes (global keys for backward compatibility)
            for k, v in elem.attrib.items():
                local_k = k.split('}')[-1] if '}' in k else k
                data['@' + local_k] = v

            tag = elem.tag.split('}')[-1] # Local name

            obj_id = data.get('@id') or data.get('@rdf:about') or data.get('@about')

            # Flatten children
            for child in elem:
                child_tag = child.tag.split('}')[-1]
                # Child text content
                if not len(child) and child.text:
                    data[child_tag] = child.text.strip()

                # Child element attributes (element-scoped to prevent conflicts)
                for attr_name, attr_val in child.attrib.items():
                    local_attr = attr_name.split('}')[-1] if '}' in attr_name else attr_name
                    # Element-scoped: e.g., "seeAlso@resource" vs "parentTree@resource"
                    scoped_key = child_tag + '@' + local_attr
                    data[scoped_key] = attr_val
                    # Also store with global key for backward compatibility
                    data['@' + local_attr] = attr_val

                # Link extraction
                # Look for rdf:resource attribute
                resource = None
                for k, v in child.attrib.items():
                    # Check for }resource or just resource
                    if k.endswith('}resource') or k == 'resource':
                        resource = v
                        break

                if resource and obj_id:
                    # Store link: source=obj_id, target=resource
                    # This captures parentTree (Child->Parent) and seeAlso
                    self.importer.upsert_link(obj_id, resource)
            
            # If no ID, maybe generate one or skip? C# PtMapper extracts IDs.
            # For now, assume @id exists or skip.
            if not obj_id:
                # Check if it's a link
                pass
            else:
                self.importer.upsert_object(obj_id, tag, data)
                
                # Embeddings
                if embed_content and self.embedder:
                    text = get_local(data, 'title') or get_local(data, 'about') or data.get('text') or ""
                    if text:
                        # print(f"DEBUG: Embedding {obj_id}: {text[:50]}...", file=sys.stderr)
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
                    parent = elem.getparent()
                    if parent is not None:
                        del parent[0]
                    else:
                        break

    def process_fragments_from_stdin(self):
        """Process null-delimited XML fragments from stdin.

        This enables AWK-based ingestion where AWK filters and extracts fragments.
        Usage: awk -f extract_fragments.awk input.rdf | python_crawler
        """
        return self.process_fragments(sys.stdin.buffer)

    def process_fragments(self, reader):
        """Process null-delimited XML fragments from a binary reader.

        Each fragment is a complete XML element separated by null bytes (\\0).

        Args:
            reader: A binary file-like object (e.g., sys.stdin.buffer, open(file, 'rb'))
        """
        fragment = bytearray()
        count = 0

        # Read byte by byte, splitting on null delimiter
        while True:
            byte = reader.read(1)
            if not byte:
                # EOF - process final fragment if exists
                if fragment:
                    try:
                        self._process_fragment(bytes(fragment))
                        count += 1
                    except Exception as e:
                        print(f"Error processing fragment: {e}", file=sys.stderr)
                break

            if byte == b'\x00':
                # Found null delimiter - process accumulated fragment
                if fragment:
                    try:
                        self._process_fragment(bytes(fragment))
                        count += 1

                        if count % 100 == 0:
                            print(f"Processed {count} fragments...", file=sys.stderr)
                    except Exception as e:
                        print(f"Error processing fragment: {e}", file=sys.stderr)

                    fragment.clear()
            else:
                fragment.extend(byte)

        print(f"✓ Processed {count} total fragments", file=sys.stderr)

    def _process_fragment(self, fragment):
        """Process a single XML fragment.

        Args:
            fragment: Bytes containing a complete XML element
        """
        # Use recover=True to handle fragments without namespace declarations
        parser = etree.XMLParser(recover=True)
        try:
            root = etree.fromstring(fragment, parser=parser)
        except etree.XMLSyntaxError as e:
            raise ValueError(f"Invalid XML fragment: {e}")

        # Extract data similar to _process_stream
        data = {}

        # Root element attributes
        for k, v in root.attrib.items():
            local_k = k.split('}')[-1] if '}' in k else k
            data['@' + local_k] = v

        tag = root.tag.split('}')[-1]  # Local name
        obj_id = data.get('@id') or data.get('@rdf:about') or data.get('@about')

        # Flatten children
        for child in root:
            child_tag = child.tag.split('}')[-1]

            # Child text content
            if not len(child) and child.text:
                data[child_tag] = child.text.strip()

            # Child element attributes (element-scoped to prevent conflicts)
            for attr_name, attr_val in child.attrib.items():
                local_attr = attr_name.split('}')[-1] if '}' in attr_name else attr_name
                # Element-scoped: e.g., "seeAlso@resource" vs "parentTree@resource"
                scoped_key = child_tag + '@' + local_attr
                data[scoped_key] = attr_val
                # Also store with global key for backward compatibility
                data['@' + local_attr] = attr_val

            # Link extraction
            resource = None
            for k, v in child.attrib.items():
                if k.endswith('}resource') or k == 'resource':
                    resource = v
                    break

            if resource and obj_id:
                self.importer.upsert_link(obj_id, resource)

        # Store object if it has an ID
        if obj_id:
            self.importer.upsert_object(obj_id, tag, data)

            # Generate embedding if embedder is available
            if self.embedder:
                text = get_local(data, 'title') or get_local(data, 'about') or data.get('text') or ""
                if text:
                    vec = self.embedder.get_embedding(text)
                    self.importer.upsert_embedding(obj_id, vec)
