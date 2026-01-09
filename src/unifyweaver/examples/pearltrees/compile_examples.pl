%% pearltrees/compile_examples.pl - Cross-target compilation examples
%%
%% Phase 4: Demonstrates compiling Pearltrees queries to multiple targets.
%% Each target generates idiomatic code for its runtime.

:- module(pearltrees_compile_examples, [
    demo_python_generation/0,
    demo_csharp_generation/0,
    demo_go_generation/0,
    show_target_comparison/0,
    % Format selection demos
    demo_format_selection/0,
    demo_all_formats/0,
    generate_for_format/5
]).

%% --------------------------------------------------------------------
%% Example: What Generated Python Would Look Like
%%
%% This shows the OUTPUT that UnifyWeaver's Python target would generate
%% from the queries defined in queries.pl
%% --------------------------------------------------------------------

python_tree_with_children_example(Code) :-
    Code = '
# Generated from: tree_with_children/3
# UnifyWeaver Python Target

from dataclasses import dataclass
from typing import List, Optional
import sqlite3

@dataclass
class Child:
    type: str
    title: str
    url: Optional[str]
    order: int

def tree_with_children(db_path: str, tree_id: str) -> tuple[str, List[Child]]:
    """Get tree title and children from SQLite index."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get tree title
    cursor.execute(
        "SELECT title FROM trees WHERE tree_id = ?",
        (tree_id,)
    )
    row = cursor.fetchone()
    title = row[0] if row else ""

    # Get children (aggregate_all -> list comprehension)
    cursor.execute(
        """SELECT pearl_type, title, external_url, pos_order
           FROM children WHERE parent_tree_id = ?
           ORDER BY pos_order""",
        (tree_id,)
    )
    children = [
        Child(type=r[0], title=r[1], url=r[2], order=r[3])
        for r in cursor.fetchall()
    ]

    conn.close()
    return title, children

def incomplete_trees(db_path: str) -> List[tuple[str, str]]:
    """Find trees with only root pearl (count <= 1)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT t.tree_id, t.title
        FROM trees t
        LEFT JOIN children c ON t.tree_id = c.parent_tree_id
        GROUP BY t.tree_id
        HAVING COUNT(c.rowid) <= 1
    """)
    results = cursor.fetchall()
    conn.close()
    return results
'.

%% --------------------------------------------------------------------
%% Example: What Generated C# Would Look Like
%% --------------------------------------------------------------------

csharp_tree_with_children_example(Code) :-
    Code = '
// Generated from: tree_with_children/3
// UnifyWeaver C# Target

using Microsoft.Data.Sqlite;
using System.Collections.Generic;
using System.Threading.Tasks;

public record Child(string Type, string Title, string? Url, int Order);

public class PearltreesQueries
{
    private readonly string _dbPath;

    public PearltreesQueries(string dbPath) => _dbPath = dbPath;

    public async Task<(string Title, List<Child> Children)> TreeWithChildrenAsync(string treeId)
    {
        await using var conn = new SqliteConnection($"Data Source={_dbPath}");
        await conn.OpenAsync();

        // Get title
        await using var titleCmd = conn.CreateCommand();
        titleCmd.CommandText = "SELECT title FROM trees WHERE tree_id = @id";
        titleCmd.Parameters.AddWithValue("@id", treeId);
        var title = (string?)await titleCmd.ExecuteScalarAsync() ?? "";

        // Get children (aggregate_all -> LINQ-style)
        await using var childCmd = conn.CreateCommand();
        childCmd.CommandText = @"
            SELECT pearl_type, title, external_url, pos_order
            FROM children WHERE parent_tree_id = @id
            ORDER BY pos_order";
        childCmd.Parameters.AddWithValue("@id", treeId);

        var children = new List<Child>();
        await using var reader = await childCmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            children.Add(new Child(
                reader.GetString(0),
                reader.GetString(1),
                reader.IsDBNull(2) ? null : reader.GetString(2),
                reader.GetInt32(3)
            ));
        }

        return (title, children);
    }

    public async IAsyncEnumerable<(string TreeId, string Title)> IncompleteTreesAsync()
    {
        await using var conn = new SqliteConnection($"Data Source={_dbPath}");
        await conn.OpenAsync();

        await using var cmd = conn.CreateCommand();
        cmd.CommandText = @"
            SELECT t.tree_id, t.title
            FROM trees t
            LEFT JOIN children c ON t.tree_id = c.parent_tree_id
            GROUP BY t.tree_id
            HAVING COUNT(c.rowid) <= 1";

        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            yield return (reader.GetString(0), reader.GetString(1));
        }
    }
}
'.

%% --------------------------------------------------------------------
%% Example: What Generated Go Would Look Like
%% --------------------------------------------------------------------

go_tree_with_children_example(Code) :-
    Code = '
// Generated from: tree_with_children/3
// UnifyWeaver Go Target

package pearltrees

import (
    "database/sql"
    _ "github.com/mattn/go-sqlite3"
)

type Child struct {
    Type  string
    Title string
    URL   *string
    Order int
}

type TreeWithChildren struct {
    TreeID   string
    Title    string
    Children []Child
}

func GetTreeWithChildren(dbPath, treeID string) (*TreeWithChildren, error) {
    db, err := sql.Open("sqlite3", dbPath)
    if err != nil {
        return nil, err
    }
    defer db.Close()

    // Get title
    var title string
    err = db.QueryRow("SELECT title FROM trees WHERE tree_id = ?", treeID).Scan(&title)
    if err != nil && err != sql.ErrNoRows {
        return nil, err
    }

    // Get children (aggregate_all -> slice append)
    rows, err := db.Query(`
        SELECT pearl_type, title, external_url, pos_order
        FROM children WHERE parent_tree_id = ?
        ORDER BY pos_order`, treeID)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    var children []Child
    for rows.Next() {
        var c Child
        var url sql.NullString
        if err := rows.Scan(&c.Type, &c.Title, &url, &c.Order); err != nil {
            return nil, err
        }
        if url.Valid {
            c.URL = &url.String
        }
        children = append(children, c)
    }

    return &TreeWithChildren{
        TreeID:   treeID,
        Title:    title,
        Children: children,
    }, nil
}

func IncompleteTreesChan(dbPath string) (<-chan struct{ TreeID, Title string }, <-chan error) {
    results := make(chan struct{ TreeID, Title string })
    errs := make(chan error, 1)

    go func() {
        defer close(results)
        defer close(errs)

        db, err := sql.Open("sqlite3", dbPath)
        if err != nil {
            errs <- err
            return
        }
        defer db.Close()

        rows, err := db.Query(`
            SELECT t.tree_id, t.title
            FROM trees t
            LEFT JOIN children c ON t.tree_id = c.parent_tree_id
            GROUP BY t.tree_id
            HAVING COUNT(c.rowid) <= 1`)
        if err != nil {
            errs <- err
            return
        }
        defer rows.Close()

        for rows.Next() {
            var treeID, title string
            if err := rows.Scan(&treeID, &title); err != nil {
                errs <- err
                return
            }
            results <- struct{ TreeID, Title string }{treeID, title}
        }
    }()

    return results, errs
}
'.

%% --------------------------------------------------------------------
%% Demo predicates
%% --------------------------------------------------------------------

demo_python_generation :-
    format('~n=== Python Target Example ===~n'),
    python_tree_with_children_example(Code),
    format('~w~n', [Code]).

demo_csharp_generation :-
    format('~n=== C# Target Example ===~n'),
    csharp_tree_with_children_example(Code),
    format('~w~n', [Code]).

demo_go_generation :-
    format('~n=== Go Target Example ===~n'),
    go_tree_with_children_example(Code),
    format('~w~n', [Code]).

show_target_comparison :-
    format('~n========================================~n'),
    format('Cross-Target Code Generation Comparison~n'),
    format('========================================~n'),
    format('~nSource predicate: tree_with_children/3~n'),
    format('~nThis single Prolog predicate generates idiomatic code for:~n'),
    format('  - Python: sqlite3 + dataclasses + list comprehension~n'),
    format('  - C#: Microsoft.Data.Sqlite + async/await + records~n'),
    format('  - Go: database/sql + structs + channels~n'),
    format('~nEach target respects its runtime conventions:~n'),
    format('  - Python: duck typing, generators~n'),
    format('  - C#: strong typing, IAsyncEnumerable~n'),
    format('  - Go: explicit error handling, goroutines~n'),
    format('~nRun demo_python_generation/0, demo_csharp_generation/0,~n'),
    format('or demo_go_generation/0 to see full examples.~n').

%% ====================================================================
%% Format Selection Demos
%% ====================================================================
%%
%% Demonstrates generating the same tree data to multiple output formats.
%% Uses templates.pl predicates for actual generation.

:- use_module(templates).

%% Sample data for demos
sample_tree_data(TreeId, Title, Children) :-
    TreeId = '12345',
    Title = 'Science Topics',
    Children = [
        child(pagepearl, 'Wikipedia - Physics', 'https://en.wikipedia.org/wiki/Physics', 1),
        child(pagepearl, 'Khan Academy', 'https://khanacademy.org', 2),
        child(tree, 'Chemistry Notes', null, 3),
        child(section, 'Resources', null, 4),
        child(alias, 'Link to Math', null, 5)
    ].

%% Available output formats
output_format(smmx, 'SimpleMind (.smmx)', generate_mindmap/4).
output_format(freemind, 'FreeMind (.mm)', generate_freemind/4).
output_format(opml, 'OPML (.opml)', generate_opml/4).
output_format(graphml, 'GraphML (.graphml)', generate_graphml/4).
output_format(vue, 'VUE (.vue)', generate_vue/4).
output_format(mermaid, 'Mermaid (.md)', generate_mermaid/4).

%% generate_for_format(+Format, +TreeId, +Title, +Children, -Output) is semidet.
%%   Generate output for a specific format.
generate_for_format(smmx, TreeId, Title, Children, Output) :-
    pearltrees_templates:generate_mindmap(TreeId, Title, Children, Output).
generate_for_format(freemind, TreeId, Title, Children, Output) :-
    pearltrees_templates:generate_freemind(TreeId, Title, Children, Output).
generate_for_format(opml, TreeId, Title, Children, Output) :-
    pearltrees_templates:generate_opml(TreeId, Title, Children, Output).
generate_for_format(graphml, TreeId, Title, Children, Output) :-
    pearltrees_templates:generate_graphml(TreeId, Title, Children, Output).
generate_for_format(vue, TreeId, Title, Children, Output) :-
    pearltrees_templates:generate_vue(TreeId, Title, Children, Output).
generate_for_format(mermaid, TreeId, Title, Children, Output) :-
    pearltrees_templates:generate_mermaid(TreeId, Title, Children, Output).

%% demo_format_selection/0 is det.
%%   Show how to select and generate different output formats.
demo_format_selection :-
    format('~n========================================~n'),
    format('Multi-Format Output Selection Demo~n'),
    format('========================================~n'),
    format('~nSame tree data can be output to 6 different formats:~n~n'),
    forall(
        output_format(Format, Description, _Pred),
        format('  ~w: ~w~n', [Format, Description])
    ),
    format('~nExample - Generate FreeMind format:~n'),
    format('~n?- sample_tree_data(Id, Title, Children),~n'),
    format('   generate_for_format(freemind, Id, Title, Children, Output).~n'),
    format('~nExample - Generate Mermaid for documentation:~n'),
    sample_tree_data(TreeId, Title, Children),
    generate_for_format(mermaid, TreeId, Title, Children, MermaidOutput),
    format('~n~w~n', [MermaidOutput]).

%% demo_all_formats/0 is det.
%%   Generate sample data in all available formats.
demo_all_formats :-
    format('~n========================================~n'),
    format('All Formats Demo~n'),
    format('========================================~n'),
    sample_tree_data(TreeId, Title, Children),
    format('~nTree: ~w (ID: ~w)~n', [Title, TreeId]),
    format('Children: ~w items~n', [5]),
    format('~nGenerating all formats...~n'),
    forall(
        output_format(Format, Description, _),
        (
            format('~n--- ~w ---~n', [Description]),
            generate_for_format(Format, TreeId, Title, Children, Output),
            % Show first 500 chars of each format
            atom_length(Output, Len),
            (   Len > 500
            ->  sub_atom(Output, 0, 500, _, Preview),
                format('~w...~n[truncated, ~w total chars]~n', [Preview, Len])
            ;   format('~w~n', [Output])
            )
        )
    ).
