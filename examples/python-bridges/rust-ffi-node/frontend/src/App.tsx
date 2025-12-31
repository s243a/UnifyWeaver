import React, { useState, useEffect } from 'react';
import './App.css';

interface ApiResponse<T = unknown> {
  success: boolean;
  result?: T;
  error?: string;
}

interface HealthStatus {
  status: string;
  connected: boolean;
  rpycServer: string;
}

function App() {
  const [connected, setConnected] = useState(false);
  const [rpycServer, setRpycServer] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Data input
  const [dataInput, setDataInput] = useState('1, 2, 3, 4, 5');
  const [sqrtInput, setSqrtInput] = useState('16');

  // Results
  const [meanResult, setMeanResult] = useState<number | null>(null);
  const [stdResult, setStdResult] = useState<number | null>(null);
  const [sqrtResult, setSqrtResult] = useState<number | null>(null);
  const [piResult, setPiResult] = useState<number | null>(null);

  // Check health on mount
  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch('/health');
      const data: ApiResponse<HealthStatus> = await res.json();
      if (data.success !== false) {
        const health = data as unknown as HealthStatus;
        setConnected(health.connected);
        setRpycServer(health.rpycServer);
      }
    } catch (err) {
      setError('Cannot reach API server');
    }
  };

  const parseData = (): number[] => {
    return dataInput
      .split(',')
      .map(s => s.trim())
      .filter(s => s !== '')
      .map(s => parseFloat(s))
      .filter(n => !isNaN(n));
  };

  const calculateMean = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = parseData();
      if (data.length === 0) {
        setError('Please enter valid numbers');
        return;
      }

      const res = await fetch('/numpy/mean', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data }),
      });

      const result: ApiResponse<number> = await res.json();
      if (result.success) {
        setMeanResult(result.result!);
        setConnected(true);
      } else {
        setError(result.error || 'Unknown error');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed');
    } finally {
      setLoading(false);
    }
  };

  const calculateStd = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = parseData();
      if (data.length === 0) {
        setError('Please enter valid numbers');
        return;
      }

      const res = await fetch('/numpy/std', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data }),
      });

      const result: ApiResponse<number> = await res.json();
      if (result.success) {
        setStdResult(result.result!);
        setConnected(true);
      } else {
        setError(result.error || 'Unknown error');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed');
    } finally {
      setLoading(false);
    }
  };

  const calculateSqrt = async () => {
    setLoading(true);
    setError(null);
    try {
      const value = parseFloat(sqrtInput);
      if (isNaN(value) || value < 0) {
        setError('Please enter a valid non-negative number');
        return;
      }

      const res = await fetch('/math/sqrt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value }),
      });

      const result: ApiResponse<number> = await res.json();
      if (result.success) {
        setSqrtResult(result.result!);
        setConnected(true);
      } else {
        setError(result.error || 'Unknown error');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed');
    } finally {
      setLoading(false);
    }
  };

  const getPi = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/math/pi');
      const result: ApiResponse<number> = await res.json();
      if (result.success) {
        setPiResult(result.result!);
        setConnected(true);
      } else {
        setError(result.error || 'Unknown error');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Python Bridge Demo</h1>
        <p className="subtitle">Node.js + React + Rust FFI + RPyC</p>
      </header>

      <main className="App-main">
        {/* Connection Status */}
        <section className="status-section">
          <div className={`status-indicator ${connected ? 'connected' : 'disconnected'}`}>
            {connected ? 'Connected' : 'Disconnected'}
          </div>
          {rpycServer && <p className="server-info">RPyC Server: {rpycServer}</p>}
        </section>

        {/* Error Display */}
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {/* NumPy Operations */}
        <section className="operation-section">
          <h2>NumPy Operations</h2>

          <div className="input-group">
            <label>Data (comma-separated numbers):</label>
            <input
              type="text"
              value={dataInput}
              onChange={e => setDataInput(e.target.value)}
              placeholder="1, 2, 3, 4, 5"
            />
          </div>

          <div className="button-group">
            <button onClick={calculateMean} disabled={loading}>
              Calculate Mean
            </button>
            <button onClick={calculateStd} disabled={loading}>
              Calculate Std Dev
            </button>
          </div>

          <div className="results">
            {meanResult !== null && (
              <div className="result">
                <span className="label">Mean:</span>
                <span className="value">{meanResult}</span>
              </div>
            )}
            {stdResult !== null && (
              <div className="result">
                <span className="label">Std Dev:</span>
                <span className="value">{stdResult.toFixed(4)}</span>
              </div>
            )}
          </div>
        </section>

        {/* Math Operations */}
        <section className="operation-section">
          <h2>Math Operations</h2>

          <div className="input-group">
            <label>Number:</label>
            <input
              type="text"
              value={sqrtInput}
              onChange={e => setSqrtInput(e.target.value)}
              placeholder="16"
            />
          </div>

          <div className="button-group">
            <button onClick={calculateSqrt} disabled={loading}>
              Calculate sqrt
            </button>
            <button onClick={getPi} disabled={loading}>
              Get pi
            </button>
          </div>

          <div className="results">
            {sqrtResult !== null && (
              <div className="result">
                <span className="label">sqrt({sqrtInput}):</span>
                <span className="value">{sqrtResult}</span>
              </div>
            )}
            {piResult !== null && (
              <div className="result">
                <span className="label">pi:</span>
                <span className="value">{piResult}</span>
              </div>
            )}
          </div>
        </section>

        {/* Architecture Diagram */}
        <section className="architecture-section">
          <h2>How It Works</h2>
          <pre className="architecture-diagram">
{`Browser (React)
     |
     | HTTP (fetch)
     v
Node.js (Express)
     |
     | FFI (ffi-napi)
     v
Rust (PyO3)
     |
     | TCP
     v
Python (RPyC + NumPy)`}
          </pre>
        </section>
      </main>

      <footer className="App-footer">
        <p>Part of the UnifyWeaver Python Bridges Examples</p>
      </footer>
    </div>
  );
}

export default App;
