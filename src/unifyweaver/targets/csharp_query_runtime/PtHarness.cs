// SPDX-License-Identifier: MIT OR Apache-2.0
using System;
using System.Collections.Generic;
using System.IO;
using UnifyWeaver.QueryRuntime.Dynamic;

namespace UnifyWeaver.QueryRuntime
{
    public static class PtHarness
    {
        public static void RunIngest(string xmlPath, string dbPath)
        {
            RunIngest(xmlPath, dbPath, embeddingProvider: null, emitEmbeddings: false);
        }

        public static void RunIngest(string xmlPath, string dbPath, bool emitEmbeddings)
        {
            RunIngest(xmlPath, dbPath, embeddingProvider: null, emitEmbeddings: emitEmbeddings);
        }

        public static void RunIngest(string xmlPath, string dbPath, IEmbeddingProvider? embeddingProvider, bool emitEmbeddings)
        {
            var config = new XmlSourceConfig
            {
                InputPath = xmlPath,
                RecordSeparator = RecordSeparatorKind.LineFeed,
                NamespacePrefixes = new Dictionary<string, string>
                {
                    {"http://www.pearltrees.com/rdf/0.1/#", "pt"},
                    {"http://purl.org/dc/elements/1.1/", "dcterms"}
                },
                TreatPearltreesCDataAsText = true
            };

            using var crawler = new PtCrawler(dbPath, config, embeddingProvider);
            crawler.IngestOnce(emitEmbeddings);
        }
    }
}
