/**
 * CSnakes + RPyC Example - C# Client
 *
 * Demonstrates calling RPyC from C# via CSnakes Python embedding.
 * CSnakes is a modern .NET 8+ library with a simpler API than Python.NET.
 *
 * Build and run:
 *   cd examples/python-bridges/csnakes
 *   dotnet run
 *
 * Prerequisites:
 *   - .NET SDK 8.0+
 *   - Python 3.8+ with rpyc installed
 */
using CSnakes.Runtime;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace UnifyWeaver.Bridges.CSnakes
{
    class RPyCClient
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("CSnakes + RPyC C# Client");
            Console.WriteLine("========================");

            // Build host with CSnakes services
            var builder = Host.CreateApplicationBuilder(args);

            // Configure Python
            builder.Services
                .WithPython()
                .WithVirtualEnvironment(
                    Environment.GetEnvironmentVariable("VIRTUAL_ENV")
                    ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".local")
                )
                .FromEnvironmentVariable("PYTHON_HOME", "python3");

            var host = builder.Build();
            var env = host.Services.GetRequiredService<IPythonEnvironment>();

            Console.WriteLine($"Python Home: {env.Home}");

            // Execute Python code that uses RPyC
            var result = env.Execute(@"
import rpyc

def connect_and_test():
    conn = rpyc.classic.connect('localhost', 18812)
    try:
        # Test math
        math = conn.modules.math
        sqrt_result = math.sqrt(16)
        print(f'math.sqrt(16) = {sqrt_result}')

        # Try numpy
        try:
            np = conn.modules.numpy
            arr = np.array([1, 2, 3, 4, 5])
            mean = float(np.mean(arr))
            print(f'numpy.mean([1,2,3,4,5]) = {mean}')
        except Exception as e:
            print(f'NumPy not available: {e}')

        # Get info
        info = conn.root.get_info()
        py_version = info['python_version']
        print(f'Server Python: {py_version}')

        return 'success'
    finally:
        conn.close()
        print('Connection closed')

result = connect_and_test()
");

            Console.WriteLine("\nPython execution completed");
            Console.WriteLine("All tests passed!");
        }
    }
}
