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

// Show sample tree with children
Console.WriteLine("\n=== Sample Tree with Children ===");
var sampleTree = trees.FindAll().FirstOrDefault(t =>
    t.ContainsKey("Children") &&
    t["Children"].AsArray.Count > 0);

if (sampleTree != null)
{
    Console.WriteLine($"Tree: {sampleTree["Title"]?.AsString ?? sampleTree["_id"].AsString}");
    Console.WriteLine($"  ID: {sampleTree["_id"].AsString}");
    Console.WriteLine($"  Children count: {sampleTree["Children"].AsArray.Count}");
    foreach (var childId in sampleTree["Children"].AsArray.Take(10))
    {
        var childPearl = pearls.FindById(childId.AsString);
        if (childPearl != null)
        {
            var childTitle = childPearl["Title"]?.AsString ?? childId.AsString;
            if (childTitle.Length > 60) childTitle = childTitle.Substring(0, 60) + "...";
            Console.WriteLine($"    - {childTitle}");
        }
    }
    if (sampleTree["Children"].AsArray.Count > 10)
    {
        Console.WriteLine($"    ... and {sampleTree["Children"].AsArray.Count - 10} more");
    }
}

// Check if any trees have ParentTree (they shouldn't - only pearls should)
Console.WriteLine("\n=== Checking Tree ParentTree Fields ===");
var treesWithParent = trees.FindAll().Count(t =>
    t.ContainsKey("ParentTree") && t["ParentTree"] != null && !t["ParentTree"].IsNull && t["ParentTree"].AsString != "");
Console.WriteLine($"Trees with ParentTree: {treesWithParent} (should be 0 - only pearls have parents)");

Console.WriteLine("\n✓ Done!");
