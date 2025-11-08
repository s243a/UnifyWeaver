# XML Source Plugin Philosophy

## 1. Introduction

This document outlines the philosophy behind a new proposed data source for UnifyWeaver, designed to extract data from XML and RDF documents. The initial request was for an "rdf source" to handle Pearltrees data, but this document proposes a more generic "XML source" that is more broadly applicable.

## 2. Naming: `xml_source` vs. `rdf_source`

The proposed name for this new feature is `xml_source`.

**Rationale:**

*   **Generality:** While the initial use case is an RDF file from Pearltrees, the core task is extracting elements from an XML-based document. RDF/XML is a specific serialization of RDF, but it is still XML. By naming it `xml_source`, we create a more generic and reusable tool that can be applied to a wider range of XML files, not just RDF.
*   **Future-Proofing:** A generic `xml_source` can be extended in the future to support other XML-related tasks, such as attribute extraction or more complex XPath queries. Naming it `rdf_source` would limit its perceived scope.
*   **Clarity:** The name `xml_source` clearly communicates that the plugin operates on XML data. This is more accurate than `rdf_source`, which might imply a deeper understanding of RDF semantics, which is not the immediate goal.

## 3. Core Principles

The `xml_source` plugin will be designed with the following principles in mind:

*   **True Streaming and Memory Efficiency:** The plugin must process XML files in a true streaming fashion, with constant memory usage, even for very large files. This means that after an element is extracted and emitted, the memory it occupied should be released immediately.
*   **Simplicity and Ease of Use:** The plugin should be easy to configure and use. The primary configuration will be the file path and a list of tags to extract.
*   **Declarative Configuration:** Users will define the XML source declaratively in their Prolog files, specifying the tags they want to extract.
*   **Integration with UnifyWeaver:** The plugin will be fully integrated into the UnifyWeaver ecosystem, including the firewall for security and the template system for code generation.

## 4. The "Filter" Concept: Extracting Multiple Tags

The user suggested the concept of a "filter" to specify which tags to extract. This is a crucial feature for making the plugin flexible.

We will implement this with a `tags` option, which will accept a list of tag names. For example:

```prolog
:- source(xml, my_data, [
    xml_file('data.rdf'),
    tags(['pt:Tree', 'pt:Page'])
]).
```

This will instruct the plugin to extract all `<pt:Tree>` and `<pt:Page>` elements from the `data.rdf` file.

## 5. Tooling

The implementation will rely on a true streaming XML parser to ensure constant memory usage. The proposed solution is to use a small, generated Python script that leverages the `lxml.etree.iterparse` functionality. This approach allows for efficient processing of very large XML files by parsing and emitting elements one by one, and then immediately releasing them from memory. This choice will be detailed further in the `IMPLEMENTATION_STRATEGY.md` document.

## 6. Conclusion

By adopting the name `xml_source` and focusing on a simple, stream-based design, we can create a powerful and flexible tool for extracting data from XML documents within the UnifyWeaver framework. This approach not only satisfies the immediate requirement of processing Pearltrees RDF data but also provides a solid foundation for future enhancements.
