/**
 * CSnakes + RPyC Example - Working Implementation
 *
 * This example demonstrates CSnakes with RPyC using:
 * 1. System Python via PYTHON_HOME environment variable
 * 2. Virtual environment with rpyc installed
 * 3. CSnakes source-generated wrappers from rpyc_wrapper.py
 */

using CSnakes.Runtime;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace UnifyWeaver.Bridges.CSnakes;

public class Program
{
    public static void Main(string[] args)
    {
        Console.WriteLine("CSnakes + RPyC Integration");
        Console.WriteLine("==========================");
        Console.WriteLine();

        // Check for Python
        var pythonHome = Environment.GetEnvironmentVariable("PYTHON_HOME");
        if (string.IsNullOrEmpty(pythonHome))
        {
            // Try common locations
            var possiblePaths = new[]
            {
                "/usr/bin/python3",
                "/usr/local/bin/python3",
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData) + "/Programs/Python"
            };

            Console.WriteLine("PYTHON_HOME not set. Trying common locations...");
            Console.WriteLine();
            Console.WriteLine("To run this example:");
            Console.WriteLine("  1. Set PYTHON_HOME to your Python installation");
            Console.WriteLine("  2. Ensure rpyc is installed: pip install rpyc");
            Console.WriteLine("  3. Start RPyC server: python examples/rpyc-integration/rpyc_server.py");
            Console.WriteLine("  4. Run: dotnet run");
            Console.WriteLine();
            Console.WriteLine("Example:");
            Console.WriteLine("  export PYTHON_HOME=/usr");
            Console.WriteLine("  dotnet run");
            return;
        }

        try
        {
            // Build host with CSnakes
            var builder = Host.CreateApplicationBuilder(args);

            // Configure CSnakes Python
            builder.Services
                .WithPython()
                .WithHome(AppContext.BaseDirectory)
                // Use system Python from PYTHON_HOME
                .FromEnvironmentVariable("PYTHON_HOME", "3.8")
                // Or explicitly: .FromFolder("/usr", "3.8")
                // Install dependencies from requirements.txt
                .WithPipInstaller();

            var app = builder.Build();

            // Get Python environment
            var env = app.Services.GetRequiredService<IPythonEnvironment>();

            Console.WriteLine($"Python initialized");
            Console.WriteLine();

            // Get the generated wrapper (from rpyc_wrapper.py)
            // CSnakes generates IRpycWrapper interface and RpycWrapper class
            var wrapper = env.RpycWrapper();

            // Test RPyC connection
            Console.WriteLine("Testing RPyC connection...");
            try
            {
                bool connected = wrapper.ConnectRpyc("localhost", 18812);
                Console.WriteLine($"  connect_rpyc: {(connected ? "OK" : "Failed")}");

                double mean = wrapper.GetNumpyMean("localhost", 18812, new List<double> { 1, 2, 3, 4, 5 });
                Console.WriteLine($"  numpy.mean([1,2,3,4,5]) = {mean}");

                string version = wrapper.GetServerPythonVersion("localhost", 18812);
                Console.WriteLine($"  Server Python: {version}");

                Console.WriteLine();
                Console.WriteLine("All tests passed!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  Error: {ex.Message}");
                Console.WriteLine();
                Console.WriteLine("Make sure RPyC server is running:");
                Console.WriteLine("  python examples/rpyc-integration/rpyc_server.py");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Initialization error: {ex.Message}");
            Console.WriteLine();
            Console.WriteLine("This may indicate:");
            Console.WriteLine("  - Python not found at PYTHON_HOME");
            Console.WriteLine("  - Missing Python packages (rpyc, plumbum)");
            Console.WriteLine("  - CSnakes version mismatch");
            Console.WriteLine();
            Console.WriteLine("Try installing rpyc in your Python environment:");
            Console.WriteLine("  pip install rpyc plumbum");
        }
    }
}
