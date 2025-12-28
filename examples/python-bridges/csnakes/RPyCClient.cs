/**
 * CSnakes + RPyC Example - C# Client
 *
 * NOTE: This is a conceptual example showing the intended usage pattern.
 * CSnakes uses source generators to create wrapper classes at compile time.
 *
 * For a working example, use the CSnakes templates:
 *   dotnet new install CSnakes.Templates
 *   dotnet new pyapp
 *
 * Then add rpyc_wrapper.py to your project.
 */

// When CSnakes processes rpyc_wrapper.py, it generates:
// - IRpycWrapper interface
// - RpycWrapper implementation
// - Extension methods for IPythonEnvironment

namespace UnifyWeaver.Bridges.CSnakes;

/// <summary>
/// Example showing how CSnakes-generated code would be used.
/// The actual generated code comes from rpyc_wrapper.py.
/// </summary>
public class RPyCClientExample
{
    public static void Main(string[] args)
    {
        Console.WriteLine("CSnakes + RPyC Example");
        Console.WriteLine("======================");
        Console.WriteLine();
        Console.WriteLine("CSnakes uses source generators to create typed wrappers");
        Console.WriteLine("from Python files at compile time.");
        Console.WriteLine();
        Console.WriteLine("To use this example:");
        Console.WriteLine("1. Install CSnakes templates: dotnet new install CSnakes.Templates");
        Console.WriteLine("2. Create new project: dotnet new pyapp");
        Console.WriteLine("3. Add rpyc_wrapper.py to the project");
        Console.WriteLine("4. CSnakes generates IRpycWrapper at compile time");
        Console.WriteLine();
        Console.WriteLine("Generated usage would look like:");
        Console.WriteLine();
        Console.WriteLine("  var wrapper = env.RpycWrapper();");
        Console.WriteLine("  bool ok = wrapper.ConnectRpyc(\"localhost\", 18812);");
        Console.WriteLine("  double mean = wrapper.GetNumpyMean(\"localhost\", 18812, [1,2,3,4,5]);");
        Console.WriteLine("  string version = wrapper.GetServerPythonVersion(\"localhost\", 18812);");
        Console.WriteLine();
        Console.WriteLine("For dynamic Python execution without source generators,");
        Console.WriteLine("see the Python.NET example instead.");
    }
}
