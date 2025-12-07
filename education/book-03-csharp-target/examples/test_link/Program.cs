using System;
using System.Linq;
using UnifyWeaver.QueryRuntime;

var result = UnifyWeaver.Generated.TestLinkQueryModule.Build();
var executor = new QueryExecutor(result.Provider);
foreach (var row in executor.Execute(result.Plan))
{
    Console.WriteLine(string.Join(",", row.Select(v => v?.ToString() ?? string.Empty)));
}
