/**
 * Python.NET + RPyC Example - C# Client
 *
 * Demonstrates calling RPyC from C# via Python.NET embedding.
 *
 * Build and run:
 *   cd examples/python-bridges/pythonnet
 *   dotnet run
 *
 * Prerequisites:
 *   - .NET SDK 6.0+ (8.0 recommended)
 *   - Python 3.8+ with rpyc installed
 */
using System;
using Python.Runtime;

namespace UnifyWeaver.Bridges.PythonNet
{
    class RPyCClient
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Python.NET + RPyC C# Client");
            Console.WriteLine("===========================");

            // Initialize Python runtime
            Runtime.PythonDLL = GetPythonDLL();
            PythonEngine.Initialize();

            try
            {
                using (Py.GIL())
                {
                    // Import RPyC
                    dynamic rpyc = Py.Import("rpyc");

                    // Connect to server
                    Console.WriteLine("\nConnecting to RPyC server...");
                    dynamic conn = rpyc.classic.connect("localhost", 18812);

                    try
                    {
                        // Use remote math module
                        dynamic math = conn.modules.math;
                        double result = math.sqrt(16);
                        Console.WriteLine($"math.sqrt(16) = {result}");

                        // Try NumPy
                        try
                        {
                            dynamic np = conn.modules.numpy;
                            dynamic arr = np.array(new[] { 1, 2, 3, 4, 5 });
                            double mean = np.mean(arr);
                            Console.WriteLine($"numpy.mean([1,2,3,4,5]) = {mean}");
                        }
                        catch (Exception e)
                        {
                            Console.WriteLine($"NumPy not available: {e.Message}");
                        }

                        // Get server info
                        dynamic info = conn.root.get_info();
                        Console.WriteLine($"Server Python: {info["python_version"]}");

                        Console.WriteLine("\nAll tests passed!");
                    }
                    finally
                    {
                        conn.close();
                        Console.WriteLine("Connection closed");
                    }
                }
            }
            finally
            {
                PythonEngine.Shutdown();
            }
        }

        static string GetPythonDLL()
        {
            // Find Python shared library
            // Adjust path based on your Python installation
            if (OperatingSystem.IsLinux())
            {
                return "libpython3.8.so";
            }
            else if (OperatingSystem.IsWindows())
            {
                return "python38.dll";
            }
            else if (OperatingSystem.IsMacOS())
            {
                return "/usr/local/Frameworks/Python.framework/Versions/3.8/lib/libpython3.8.dylib";
            }
            throw new PlatformNotSupportedException();
        }
    }
}
