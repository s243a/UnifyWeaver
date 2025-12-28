/**
 * CSnakes + RPyC Example - Working Implementation
 *
 * This example demonstrates CSnakes with RPyC using:
 * 1. Redistributable Python (auto-downloaded) - RECOMMENDED
 * 2. Or system Python via PYTHON_HOME environment variable
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

        // Check for mode selection
        var useRedistributable = Environment.GetEnvironmentVariable("CSNAKES_USE_REDIST") != "0";
        var pythonHome = Environment.GetEnvironmentVariable("PYTHON_HOME");

        try
        {
            // Build host with CSnakes
            var builder = Host.CreateApplicationBuilder(args);

            // Configure CSnakes Python
            var pythonBuilder = builder.Services
                .WithPython()
                .WithHome(AppContext.BaseDirectory);

            // Set up virtual environment path
            var venvPath = Path.Combine(AppContext.BaseDirectory, ".venv");

            if (useRedistributable)
            {
                // Use redistributable Python - downloads automatically (50-80MB first time)
                Console.WriteLine("Using redistributable Python 3.12 (auto-download)...");
                Console.WriteLine("(Set CSNAKES_USE_REDIST=0 and PYTHON_HOME to use system Python)");
                Console.WriteLine();
                pythonBuilder
                    .FromRedistributable("3.12")
                    .WithVirtualEnvironment(venvPath)
                    .WithPipInstaller();
            }
            else if (!string.IsNullOrEmpty(pythonHome))
            {
                // Use system Python from PYTHON_HOME
                Console.WriteLine($"Using system Python from PYTHON_HOME={pythonHome}");
                Console.WriteLine();
                pythonBuilder
                    .FromEnvironmentVariable("PYTHON_HOME", "3.8")
                    .WithVirtualEnvironment(venvPath)
                    .WithPipInstaller();
            }
            else
            {
                Console.WriteLine("Error: CSNAKES_USE_REDIST=0 but PYTHON_HOME not set");
                Console.WriteLine();
                Console.WriteLine("Options:");
                Console.WriteLine("  1. Use redistributable (default): just run 'dotnet run'");
                Console.WriteLine("  2. Use system Python: export PYTHON_HOME=/usr && dotnet run");
                return;
            }

            var app = builder.Build();

            // Get Python environment
            var env = app.Services.GetRequiredService<IPythonEnvironment>();

            Console.WriteLine($"Python initialized successfully!");
            Console.WriteLine();

            // Get the generated wrapper (from rpyc_wrapper.py)
            // CSnakes generates IRpycWrapper interface and RpycWrapper class
            var wrapper = env.RpycWrapper();

            // Test RPyC connection
            Console.WriteLine("Testing RPyC connection...");
            Console.WriteLine("(Make sure RPyC server is running first)");
            Console.WriteLine();
            try
            {
                bool connected = wrapper.ConnectRpyc("localhost", 18812);
                Console.WriteLine($"  connect_rpyc: {(connected ? "OK" : "Failed")}");

                double mean = wrapper.GetNumpyMean("localhost", 18812, new List<double> { 1, 2, 3, 4, 5 });
                Console.WriteLine($"  numpy.mean([1,2,3,4,5]) = {mean}");

                string version = wrapper.GetServerPythonVersion("localhost", 18812);
                Console.WriteLine($"  Server Python: {version}");

                Console.WriteLine();
                Console.WriteLine("==========================");
                Console.WriteLine("All tests passed!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  RPyC Error: {ex.Message}");
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
            Console.WriteLine("  - Network issue downloading redistributable Python");
            Console.WriteLine("  - Python not found at PYTHON_HOME (if using system Python)");
            Console.WriteLine("  - Missing write permissions for cache directory");
            Console.WriteLine();
            Console.WriteLine("Set CSNAKES_REDIST_CACHE to a writable directory, or try:");
            Console.WriteLine("  export PYTHON_HOME=/usr");
            Console.WriteLine("  export CSNAKES_USE_REDIST=0");
            Console.WriteLine("  dotnet run");
        }
    }
}
