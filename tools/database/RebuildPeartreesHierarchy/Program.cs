using System;
using System.Linq;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using LiteDB;

// Rebuild Children lists by extracting parent IDs from About URLs
var dbPath = args.Length > 0 ? args[0] : "../../../pt_ingest_test.db";

Console.WriteLine("=== Rebuilding Tree Children from About URLs ===\n");
Console.WriteLine($"Database: {dbPath}");

using var db = new LiteDatabase(dbPath);
var trees = db.GetCollection("trees");
var pearls = db.GetCollection("pearls");

Console.WriteLine($"Trees: {trees.Count()}");
Console.WriteLine($"Pearls: {pearls.Count()}");

// Extract parent tree ID from About URL
// Example: "https://www.pearltrees.com/s243a/social-media/id10311488?show=item,100408713"
// Parent tree ID: "10311488"
var parentToChildren = new Dictionary<string, List<string>>();
int pearlsProcessed = 0;
int pearlsWithParent = 0;

Console.WriteLine("\nExtracting parent tree IDs from pearl About URLs...");
foreach (var pearl in pearls.FindAll())
{
    pearlsProcessed++;
    var pearlId = pearl["_id"].AsString;
    var about = pearl["About"]?.AsString;

    if (!string.IsNullOrEmpty(about))
    {
        // Extract parent tree ID from URL like: .../id12345?show=item,67890
        // The parent tree ID is "12345"
        var match = Regex.Match(about, @"/id(\d+)\?");
        if (match.Success)
        {
            var parentId = match.Groups[1].Value;
            pearlsWithParent++;

            if (!parentToChildren.ContainsKey(parentId))
            {
                parentToChildren[parentId] = new List<string>();
            }
            parentToChildren[parentId].Add(pearlId);

            if (pearlsWithParent <= 5)
            {
                Console.WriteLine($"  Pearl {pearlId} → Parent tree {parentId}");
            }
        }
    }
}

Console.WriteLine($"\nProcessed: {pearlsProcessed} pearls");
Console.WriteLine($"Pearls with parent: {pearlsWithParent}");
Console.WriteLine($"Unique parent trees: {parentToChildren.Count}");

// Update trees with their children
Console.WriteLine("\nUpdating tree Children fields...");
int treesUpdated = 0;
int treesNotFound = 0;

foreach (var kvp in parentToChildren)
{
    var parentId = kvp.Key;
    var childIds = kvp.Value;

    var tree = trees.FindById(parentId);
    if (tree != null)
    {
        // Update Children field
        var childArray = new BsonArray();
        foreach (var childId in childIds)
        {
            childArray.Add(new BsonValue(childId));
        }
        tree["Children"] = childArray;
        trees.Update(tree);
        treesUpdated++;

        if (treesUpdated <= 5)
        {
            Console.WriteLine($"  Updated tree {parentId}: {childIds.Count} children");
        }
    }
    else
    {
        treesNotFound++;
    }
}

Console.WriteLine($"\n✓ Updated {treesUpdated} trees with Children lists");
if (treesNotFound > 0)
{
    Console.WriteLine($"  Note: {treesNotFound} parent tree IDs not found in database");
}

// Update pearls' ParentTree fields
Console.WriteLine("\nUpdating pearl ParentTree fields...");
int pearlsUpdated = 0;

foreach (var kvp in parentToChildren)
{
    var parentId = kvp.Key;
    var childIds = kvp.Value;

    foreach (var childId in childIds)
    {
        var pearl = pearls.FindById(childId);
        if (pearl != null)
        {
            pearl["ParentTree"] = new BsonValue(parentId);
            pearls.Update(pearl);
            pearlsUpdated++;
        }
    }
}

Console.WriteLine($"✓ Updated {pearlsUpdated} pearls with ParentTree field");

// Extract tree → tree relationships from RefPearl seeAlso
Console.WriteLine("\n=== Extracting Tree-to-Tree Relationships ===");
Console.WriteLine("Scanning RefPearls for seeAlso → child tree links...");

var treeToChildTrees = new Dictionary<string, List<string>>();
int refPearlsProcessed = 0;
int refPearlsWithSeeAlso = 0;
foreach (var pearl in pearls.FindAll())
{
    var pearlType = pearl["Type"]?.AsString ?? "";
    if (pearlType == "pt:RefPearl" || pearlType == "pt:AliasPearl")
    {
        refPearlsProcessed++;
        var pearlId = pearl["_id"].AsString;
        var parentTreeId = pearl["ParentTree"]?.AsString;

        if (!string.IsNullOrEmpty(parentTreeId) && pearl.ContainsKey("Raw"))
        {
            var raw = pearl["Raw"].AsDocument;

            // The seeAlso target is in element-scoped attribute: seeAlso@rdf:resource
            // Try multiple possible key formats (with different namespace representations)
            string? seeAlsoUrl = null;
            var possibleKeys = new[]
            {
                "seeAlso@rdf:resource",
                "seeAlso@{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource",
                "rdfs:seeAlso@rdf:resource",
                "{http://www.w3.org/2000/01/rdf-schema#}seeAlso@rdf:resource",
                "{http://www.w3.org/2000/01/rdf-schema#}seeAlso@{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource",
                // Fallback to old global keys (for backward compatibility with old databases)
                "@rdf:resource",
                "@{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource",
                "@resource"
            };

            foreach (var key in possibleKeys)
            {
                if (raw.ContainsKey(key))
                {
                    seeAlsoUrl = raw[key].ToString();
                    break;
                }
            }

            if (!string.IsNullOrWhiteSpace(seeAlsoUrl))
            {
                // Extract child tree ID from URL like: .../id12345
                var match = Regex.Match(seeAlsoUrl, @"/id(\d+)");
                if (match.Success)
                {
                    var childTreeId = match.Groups[1].Value;

                    // Only count if it's different from parent (not self-reference)
                    if (childTreeId != parentTreeId)
                    {
                        refPearlsWithSeeAlso++;

                        if (!treeToChildTrees.ContainsKey(parentTreeId))
                        {
                            treeToChildTrees[parentTreeId] = new List<string>();
                        }
                        if (!treeToChildTrees[parentTreeId].Contains(childTreeId))
                        {
                            treeToChildTrees[parentTreeId].Add(childTreeId);
                        }

                        if (refPearlsWithSeeAlso <= 5)
                        {
                            Console.WriteLine($"  Tree {parentTreeId} → Tree {childTreeId} (via RefPearl)");
                        }
                    }
                }
            }
        }
    }
}

Console.WriteLine($"\nProcessed: {refPearlsProcessed} RefPearls/AliasPearls");
Console.WriteLine($"RefPearls with seeAlso: {refPearlsWithSeeAlso}");
Console.WriteLine($"Trees with child trees: {treeToChildTrees.Count}");

// Update trees with ChildTrees field
Console.WriteLine("\nUpdating tree ChildTrees fields...");
int treesWithChildTrees = 0;

foreach (var kvp in treeToChildTrees)
{
    var parentId = kvp.Key;
    var childTreeIds = kvp.Value;

    var tree = trees.FindById(parentId);
    if (tree != null)
    {
        var childTreeArray = new BsonArray();
        foreach (var childTreeId in childTreeIds)
        {
            childTreeArray.Add(new BsonValue(childTreeId));
        }
        tree["ChildTrees"] = childTreeArray;
        trees.Update(tree);
        treesWithChildTrees++;

        if (treesWithChildTrees <= 5)
        {
            Console.WriteLine($"  Updated tree {parentId}: {childTreeIds.Count} child trees");
        }
    }
}

Console.WriteLine($"✓ Updated {treesWithChildTrees} trees with ChildTrees lists");

// Show sample tree with both children and child trees
Console.WriteLine("\n=== Sample Tree with Children & Child Trees ===");
var sampleTree = trees.FindAll().FirstOrDefault(t =>
    t.ContainsKey("ChildTrees") &&
    t["ChildTrees"].AsArray.Count > 0);

if (sampleTree != null)
{
    Console.WriteLine($"Tree: {sampleTree["Title"]?.AsString ?? sampleTree["_id"].AsString}");
    Console.WriteLine($"  ID: {sampleTree["_id"].AsString}");

    // Show child trees (tree → tree relationships)
    if (sampleTree.ContainsKey("ChildTrees") && sampleTree["ChildTrees"].AsArray.Count > 0)
    {
        Console.WriteLine($"  Child Trees: {sampleTree["ChildTrees"].AsArray.Count}");
        foreach (var childTreeId in sampleTree["ChildTrees"].AsArray.Take(5))
        {
            var childTree = trees.FindById(childTreeId.AsString);
            if (childTree != null)
            {
                var childTitle = childTree["Title"]?.AsString ?? childTreeId.AsString;
                if (childTitle.Length > 50) childTitle = childTitle.Substring(0, 50) + "...";
                Console.WriteLine($"    → {childTitle}");
            }
        }
        if (sampleTree["ChildTrees"].AsArray.Count > 5)
        {
            Console.WriteLine($"    ... and {sampleTree["ChildTrees"].AsArray.Count - 5} more child trees");
        }
    }

    // Show children (pearls)
    if (sampleTree.ContainsKey("Children") && sampleTree["Children"].AsArray.Count > 0)
    {
        Console.WriteLine($"  Children (pearls): {sampleTree["Children"].AsArray.Count}");
        foreach (var childId in sampleTree["Children"].AsArray.Take(5))
        {
            var childPearl = pearls.FindById(childId.AsString);
            if (childPearl != null)
            {
                var childTitle = childPearl["Title"]?.AsString ?? childId.AsString;
                var childType = childPearl["Type"]?.AsString ?? "";
                if (childTitle.Length > 50) childTitle = childTitle.Substring(0, 50) + "...";
                Console.WriteLine($"    • [{childType}] {childTitle}");
            }
        }
        if (sampleTree["Children"].AsArray.Count > 5)
        {
            Console.WriteLine($"    ... and {sampleTree["Children"].AsArray.Count - 5} more pearls");
        }
    }
}

// Check if any trees have ParentTree (they shouldn't - only pearls should)
Console.WriteLine("\n=== Checking Tree ParentTree Fields ===");
var treesWithParent = trees.FindAll().Count(t =>
    t.ContainsKey("ParentTree") && t["ParentTree"] != null && !t["ParentTree"].IsNull && t["ParentTree"].AsString != "");
Console.WriteLine($"Trees with ParentTree: {treesWithParent} (should be 0 - only pearls have parents)");

Console.WriteLine("\n✓ Done!");
