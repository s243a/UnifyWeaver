"""RPyC wrapper functions with type annotations for CSnakes.

CSnakes generates C# wrapper classes from Python files at compile time.
These functions provide a typed interface for RPyC operations.
"""


def connect_rpyc(host: str, port: int) -> bool:
    """Connect to RPyC server and verify connection works.

    Args:
        host: RPyC server hostname
        port: RPyC server port

    Returns:
        True if connection succeeded and math.sqrt(16) == 4.0
    """
    import rpyc
    conn = rpyc.classic.connect(host, port)
    try:
        math = conn.modules.math
        result = math.sqrt(16)
        return result == 4.0
    finally:
        conn.close()


def get_numpy_mean(host: str, port: int, values: list[float]) -> float:
    """Calculate mean of values using remote NumPy.

    Args:
        host: RPyC server hostname
        port: RPyC server port
        values: List of numbers to average

    Returns:
        Mean of the values
    """
    import rpyc
    conn = rpyc.classic.connect(host, port)
    try:
        np = conn.modules.numpy
        arr = np.array(values)
        return float(np.mean(arr))
    finally:
        conn.close()


def get_server_python_version(host: str, port: int) -> str:
    """Get Python version from remote server.

    Args:
        host: RPyC server hostname
        port: RPyC server port

    Returns:
        Python version string (e.g., "3.8.10")
    """
    import rpyc
    conn = rpyc.classic.connect(host, port)
    try:
        sys = conn.modules.sys
        return sys.version.split()[0]
    finally:
        conn.close()
