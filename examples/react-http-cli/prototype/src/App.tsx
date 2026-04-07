import React, { useState, useEffect, useRef } from 'react'

// Configuration - update to match your server
const API_BASE = 'https://localhost:3001'

// Types
interface FileEntry {
  name: string
  type: 'file' | 'directory'
  size: number
}

interface User {
  email: string
  roles: string[]
}

// Helper function for formatting file sizes
const formatSize = (bytes: number): string => {
  if (!bytes || bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

// Strip ANSI escape codes for clean terminal display
const stripAnsi = (str: string): string => {
  // eslint-disable-next-line no-control-regex
  return str.replace(/\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b\[\?[0-9;]*[a-zA-Z]/g, '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '')
}

// API helper with auth
const api = {
  token: '',

  async fetch(endpoint: string, options: RequestInit = {}) {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string> || {})
    }
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`
    }

    const res = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers
    })

    if (!res.ok) {
      const text = await res.text()
      throw new Error(text || res.statusText)
    }

    return res.json()
  },

  async login(email: string, password: string) {
    const res = await this.fetch('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password })
    })
    // Server returns {success: true, data: {token, user}}
    if (res.success && res.data) {
      this.token = res.data.token
      return res.data
    }
    throw new Error(res.error || 'Login failed')
  },

  logout() {
    this.token = ''
  }
}

// Tab definitions
const TABS = [
  { id: 'browse', label: 'Browse', icon: '📁' },
  { id: 'upload', label: 'Upload', icon: '📤' },
  { id: 'cat', label: 'Cat', icon: '📄' },
  { id: 'shell', label: 'Shell', icon: '🔐', highlight: true }
]

function App() {
  // Auth state
  const [user, setUser] = useState<User | null>(null)
  const [loginEmail, setLoginEmail] = useState('shell@local')
  const [loginPassword, setLoginPassword] = useState('shell')
  const [loginError, setLoginError] = useState('')

  // UI state
  const [activeTab, setActiveTab] = useState('browse')
  const [loading, setLoading] = useState(false)
  const [workingDir, setWorkingDir] = useState('.')

  // Browse state
  const [browsePath, setBrowsePath] = useState('.')
  const [browseEntries, setBrowseEntries] = useState<FileEntry[]>([])
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const [browseRoot, setBrowseRoot] = useState('sandbox') // sandbox | project | home

  // Upload state
  const [uploadFiles, setUploadFiles] = useState<File[]>([])
  const [uploadResult, setUploadResult] = useState('')

  // Cat state
  const [catPath, setCatPath] = useState('')
  const [catContent, setCatContent] = useState('')

  // Shell state
  const [shellOutput, setShellOutput] = useState('')
  const [shellInput, setShellInput] = useState('')
  const [shellConnected, setShellConnected] = useState(false)
  const shellWs = useRef<WebSocket | null>(null)

  // Login handler
  const handleLogin = async () => {
    setLoading(true)
    setLoginError('')
    try {
      const res = await api.login(loginEmail, loginPassword)
      setUser({ email: res.user.email, roles: res.user.roles })
      loadBrowse('.')
    } catch (err: any) {
      setLoginError(err.message || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  // Logout handler
  const handleLogout = () => {
    api.logout()
    setUser(null)
    setShellConnected(false)
    if (shellWs.current) {
      shellWs.current.close()
    }
  }

  // Browse handlers
  const loadBrowse = async (path: string, root?: string) => {
    setLoading(true)
    const useRoot = root || browseRoot
    try {
      const res = await api.fetch('/browse', {
        method: 'POST',
        body: JSON.stringify({ path, root: useRoot })
      })
      // Server returns {success: true, data: {path, entries}}
      const data = res.data || res
      setBrowsePath(data.path || path)
      setBrowseEntries(data.entries || [])
      setSelectedFile(null)
    } catch (err: any) {
      console.error('Browse failed:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleRootChange = (newRoot: string) => {
    setBrowseRoot(newRoot)
    setBrowsePath('.')
    loadBrowse('.', newRoot)
  }

  const handleEntryClick = (entry: FileEntry) => {
    if (entry.type === 'directory') {
      const newPath = browsePath === '.' ? entry.name : `${browsePath}/${entry.name}`
      loadBrowse(newPath)
    } else {
      setSelectedFile(entry.name)
    }
  }

  const navigateUp = () => {
    if (browsePath !== '.' && browsePath !== '/') {
      const parts = browsePath.split('/')
      parts.pop()
      loadBrowse(parts.join('/') || '.')
    }
  }

  const viewFile = async () => {
    if (!selectedFile) return
    const path = browsePath === '.' ? selectedFile : `${browsePath}/${selectedFile}`
    setCatPath(path)
    setActiveTab('cat')
    handleCat(path)
  }

  const downloadFile = async () => {
    if (!selectedFile) return
    const path = browsePath === '.' ? selectedFile : `${browsePath}/${selectedFile}`
    try {
      const res = await fetch(`${API_BASE}/download?path=${encodeURIComponent(path)}&root=${browseRoot}`, {
        headers: { 'Authorization': `Bearer ${api.token}` }
      })
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = selectedFile
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Download failed:', err)
    }
  }

  // Upload handlers
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setUploadFiles(Array.from(e.target.files))
    }
  }

  // File System Access API - better file picker on Android
  const openFilePicker = async () => {
    // @ts-ignore - File System Access API
    if (!window.showOpenFilePicker) {
      setUploadResult('File System Access API not available - use standard file input')
      return
    }
    try {
      // @ts-ignore
      const handles = await window.showOpenFilePicker({ multiple: true })
      const files: File[] = []
      for (const handle of handles) {
        const file = await handle.getFile()
        files.push(file)
      }
      if (files.length === 0) return

      // Upload immediately
      setLoading(true)
      setUploadResult(`Uploading ${files.length} file(s)...`)

      const formData = new FormData()
      formData.append('destination', workingDir)
      formData.append('root', browseRoot)
      files.forEach(file => formData.append('files', file))

      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${api.token}` },
        body: formData
      })
      const json = await res.json()
      if (json.success) {
        setUploadResult(`Uploaded ${json.data.count} file(s) to ${json.data.destination}`)
      } else {
        setUploadResult(`Error: ${json.error || 'Upload failed'}`)
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setUploadResult(`Error: ${err.message}`)
      }
    } finally {
      setLoading(false)
    }
  }

  const handleUpload = async () => {
    if (uploadFiles.length === 0) return
    setLoading(true)
    setUploadResult('')

    const formData = new FormData()
    uploadFiles.forEach(file => formData.append('files', file))
    formData.append('destination', workingDir)
    formData.append('root', browseRoot)

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${api.token}` },
        body: formData
      })
      const json = await res.json()
      // Server returns {success: true, data: {count, destination, uploaded}}
      if (json.success) {
        setUploadResult(`Uploaded ${json.data.count} file(s) to ${json.data.destination}`)
      } else {
        setUploadResult(`Error: ${json.error || 'Upload failed'}`)
      }
      setUploadFiles([])
    } catch (err: any) {
      setUploadResult(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  // Cat handler
  const handleCat = async (path?: string) => {
    const targetPath = path || catPath
    if (!targetPath) return
    setLoading(true)
    setCatContent('')
    try {
      const res = await api.fetch('/cat', {
        method: 'POST',
        body: JSON.stringify({ path: targetPath, root: browseRoot })
      })
      // Server returns {success: true, data: {content}}
      const data = res.data || res
      setCatContent(data.content || '')
    } catch (err: any) {
      setCatContent(`Error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  // Shell handlers
  const connectShell = () => {
    const wsUrl = API_BASE.replace('https:', 'wss:').replace('http:', 'ws:')
    shellWs.current = new WebSocket(`${wsUrl}/shell?token=${api.token}`)

    shellWs.current.onopen = () => {
      setShellConnected(true)
      setShellOutput(prev => prev + '--- Connected ---\n')
    }

    shellWs.current.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'output' && msg.data) {
          setShellOutput(prev => prev + msg.data)
        }
      } catch {
        // If not JSON, append as-is
        setShellOutput(prev => prev + e.data)
      }
    }

    shellWs.current.onclose = () => {
      setShellConnected(false)
      setShellOutput(prev => prev + '\n--- Disconnected ---\n')
    }

    shellWs.current.onerror = () => {
      setShellOutput(prev => prev + '\n--- Connection error ---\n')
    }
  }

  const disconnectShell = () => {
    if (shellWs.current) {
      shellWs.current.close()
    }
  }

  const sendShellCommand = () => {
    if (shellWs.current && shellInput) {
      shellWs.current.send(shellInput + '\n')
      setShellInput('')
    }
  }

  // Render login form
  if (!user) {
    return (
      <div className="app-container">
        <h1 style={{ textAlign: 'center', marginBottom: 30 }}>🔍 UnifyWeaver CLI</h1>
        <div style={{ maxWidth: 400, margin: '0 auto', background: '#16213e', padding: 30, borderRadius: 10 }}>
          <h2 style={{ marginBottom: 20 }}>Login Required</h2>
          <div style={{ marginBottom: 15 }}>
            <label style={{ display: 'block', marginBottom: 5, color: '#94a3b8' }}>Email</label>
            <input
              type="email"
              value={loginEmail}
              onChange={e => setLoginEmail(e.target.value)}
              onKeyPress={e => e.key === 'Enter' && handleLogin()}
              style={{ width: '100%', padding: 10, background: '#1a1a2e', border: '1px solid #0f3460', borderRadius: 5, color: '#fff' }}
            />
          </div>
          <div style={{ marginBottom: 20 }}>
            <label style={{ display: 'block', marginBottom: 5, color: '#94a3b8' }}>Password</label>
            <input
              type="password"
              value={loginPassword}
              onChange={e => setLoginPassword(e.target.value)}
              onKeyPress={e => e.key === 'Enter' && handleLogin()}
              style={{ width: '100%', padding: 10, background: '#1a1a2e', border: '1px solid #0f3460', borderRadius: 5, color: '#fff' }}
            />
          </div>
          <button
            onClick={handleLogin}
            disabled={loading}
            style={{ width: '100%', padding: 12, background: '#e94560', border: 'none', borderRadius: 5, color: '#fff', cursor: 'pointer', fontWeight: 'bold' }}
          >
            {loading ? 'Logging in...' : 'Login'}
          </button>
          {loginError && (
            <p style={{ color: '#ff6b6b', marginTop: 15, textAlign: 'center' }}>{loginError}</p>
          )}
          <p style={{ color: '#94a3b8', fontSize: 12, marginTop: 20, textAlign: 'center' }}>
            Default: shell@local / shell
          </p>
        </div>
      </div>
    )
  }

  // Render main app
  return (
    <div className="app-container">
      <h1 style={{ marginBottom: 20 }}>🔍 UnifyWeaver CLI</h1>

      {/* User header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20, padding: '10px 15px', background: '#16213e', borderRadius: 5 }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <span>{user.email}</span>
          {user.roles.map(role => (
            <span key={role} style={{ background: '#0f3460', padding: '2px 8px', borderRadius: 3, fontSize: 12 }}>{role}</span>
          ))}
        </div>
        <button onClick={handleLogout} style={{ padding: '8px 16px', background: '#0f3460', border: 'none', borderRadius: 5, color: '#fff', cursor: 'pointer' }}>
          Logout
        </button>
      </div>

      {/* Working directory bar */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 20, padding: '8px 12px', background: '#16213e', borderRadius: 5, flexWrap: 'wrap' }}>
        <select
          value={browseRoot}
          onChange={e => handleRootChange(e.target.value)}
          style={{ padding: '6px 10px', background: '#1a1a2e', border: '1px solid #0f3460', borderRadius: 5, color: '#fff', fontSize: 12 }}
        >
          <option value="sandbox">Sandbox</option>
          <option value="project">Project</option>
          <option value="home">Home</option>
        </select>
        <span style={{ color: '#94a3b8', fontSize: 12 }}>Working Dir:</span>
        <code style={{ background: '#1a1a2e', padding: '4px 8px', borderRadius: 3, color: '#4ade80' }}>{workingDir}</code>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 5, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '10px 20px',
              background: activeTab === tab.id ? '#e94560' : (tab.highlight ? '#a855f7' : '#16213e'),
              border: 'none',
              borderRadius: 5,
              color: '#fff',
              cursor: 'pointer'
            }}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {/* Browse panel */}
      {activeTab === 'browse' && (
        <div style={{ background: '#16213e', padding: 20, borderRadius: 5 }}>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 15, flexWrap: 'wrap' }}>
            {browsePath !== '.' && (
              <button onClick={navigateUp} style={{ padding: '8px 16px', background: '#0f3460', border: 'none', borderRadius: 5, color: '#fff', cursor: 'pointer' }}>⬆️ Up</button>
            )}
            <span>📁</span>
            <code style={{ background: '#1a1a2e', padding: '4px 8px', borderRadius: 3 }}>{browsePath}</code>
            <button
              onClick={() => setWorkingDir(browsePath)}
              disabled={workingDir === browsePath}
              style={{ padding: '8px 16px', background: workingDir === browsePath ? '#555' : '#4ade80', border: 'none', borderRadius: 5, color: '#fff', cursor: 'pointer' }}
            >
              📌 Set as Working Dir
            </button>
          </div>

          <div style={{ maxHeight: 300, overflowY: 'auto' }}>
            {browseEntries.map((entry, i) => (
              <div
                key={i}
                onClick={() => handleEntryClick(entry)}
                style={{
                  padding: '12px 16px',
                  background: selectedFile === entry.name ? '#0f3460' : '#1a1a2e',
                  marginBottom: 4,
                  borderRadius: 5,
                  cursor: 'pointer',
                  borderLeft: `3px solid ${entry.type === 'directory' ? '#e94560' : '#3b82f6'}`
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>{entry.type === 'directory' ? '📁' : '📄'} {entry.name}</span>
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{formatSize(entry.size)}</span>
                </div>
              </div>
            ))}
            {browseEntries.length === 0 && !loading && (
              <p style={{ color: '#94a3b8', textAlign: 'center' }}>Empty directory</p>
            )}
          </div>

          {selectedFile && (
            <div style={{ marginTop: 15, padding: 15, background: '#0f3460', borderRadius: 5 }}>
              <p style={{ color: '#94a3b8', fontSize: 12, marginBottom: 10 }}>Selected: <code>{selectedFile}</code></p>
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                <button onClick={viewFile} style={{ padding: '10px 20px', background: '#e94560', border: 'none', borderRadius: 5, color: '#fff', cursor: 'pointer' }}>View Contents</button>
                <button onClick={downloadFile} style={{ padding: '10px 20px', background: '#e94560', border: 'none', borderRadius: 5, color: '#fff', cursor: 'pointer' }}>📥 Download</button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Upload panel */}
      {activeTab === 'upload' && (
        <div style={{ background: '#16213e', padding: 20, borderRadius: 5 }}>
          <div style={{ marginBottom: 15, padding: 10, background: '#1a1a2e', borderRadius: 5 }}>
            <p style={{ color: '#94a3b8', fontSize: 12, marginBottom: 5 }}>Destination: <code>{workingDir}</code> ({browseRoot})</p>
          </div>

          {/* File System Access API - better for Android */}
          <div style={{ border: '2px dashed #4ade80', padding: 30, borderRadius: 10, textAlign: 'center', marginBottom: 15, cursor: 'pointer', background: '#0a2e1a' }} onClick={openFilePicker}>
            <p style={{ fontSize: 18, marginBottom: 5 }}>📂 Open File Picker</p>
            <p style={{ color: '#94a3b8', fontSize: 12 }}>Recommended for Android - picks and uploads immediately</p>
          </div>

          {/* Fallback standard file input */}
          <div style={{ border: '2px dashed #0f3460', padding: 20, borderRadius: 10, textAlign: 'center', marginBottom: 20 }}>
            <p style={{ fontSize: 14, marginBottom: 10, color: '#94a3b8' }}>Or use standard file input:</p>
            <input
              type="file"
              multiple
              accept="*/*"
              onChange={handleFileSelect}
              style={{ padding: 10 }}
            />
          </div>

          {uploadFiles.length > 0 && (
            <div style={{ marginBottom: 20 }}>
              <p style={{ color: '#94a3b8', marginBottom: 10 }}>Selected files:</p>
              {uploadFiles.map((file, i) => (
                <div key={i} style={{ padding: '8px 12px', background: '#1a1a2e', marginBottom: 4, borderRadius: 5, display: 'flex', justifyContent: 'space-between' }}>
                  <span>{file.name}</span>
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{formatSize(file.size)}</span>
                </div>
              ))}
            </div>
          )}

          {uploadFiles.length > 0 && (
            <button
              onClick={handleUpload}
              disabled={loading}
              style={{ width: '100%', padding: 12, background: '#e94560', border: 'none', borderRadius: 5, color: '#fff', cursor: 'pointer' }}
            >
              {loading ? 'Uploading...' : '📤 Upload Files'}
            </button>
          )}

          {uploadResult && (
            <p style={{ marginTop: 15, padding: 10, background: uploadResult.startsWith('Error') ? '#7f1d1d' : '#065f46', borderRadius: 5 }}>{uploadResult}</p>
          )}
        </div>
      )}

      {/* Cat panel */}
      {activeTab === 'cat' && (
        <div style={{ background: '#16213e', padding: 20, borderRadius: 5 }}>
          <div style={{ display: 'flex', gap: 10, marginBottom: 15 }}>
            <input
              type="text"
              value={catPath}
              onChange={e => setCatPath(e.target.value)}
              onKeyPress={e => e.key === 'Enter' && handleCat()}
              placeholder="File path..."
              style={{ flex: 1, padding: 10, background: '#1a1a2e', border: '1px solid #0f3460', borderRadius: 5, color: '#fff' }}
            />
            <button
              onClick={() => handleCat()}
              disabled={loading}
              style={{ padding: '10px 20px', background: '#e94560', border: 'none', borderRadius: 5, color: '#fff', cursor: 'pointer' }}
            >
              {loading ? 'Loading...' : 'Read File'}
            </button>
          </div>

          {catContent && (
            <>
              <div style={{ marginBottom: 10 }}>
                <button
                  onClick={() => { setActiveTab('browse'); setCatContent(''); }}
                  style={{ padding: '8px 16px', background: '#0f3460', border: 'none', borderRadius: 5, color: '#fff', cursor: 'pointer' }}
                >
                  ← Back to Browse
                </button>
              </div>
              <div style={{ background: '#0a0a0a', padding: 15, borderRadius: 5, maxHeight: 400, overflowY: 'auto' }}>
                <pre style={{ margin: 0, fontFamily: 'monospace', fontSize: 13, whiteSpace: 'pre-wrap', color: '#cdd6f4' }}>{catContent}</pre>
              </div>
            </>
          )}
        </div>
      )}

      {/* Shell panel */}
      {activeTab === 'shell' && (
        <div style={{ background: '#16213e', padding: 0, borderRadius: 5 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', background: '#0f3460' }}>
            <span style={{ color: '#a855f7', fontWeight: 'bold' }}>🔐 Shell</span>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              <span style={{ color: shellConnected ? '#4ade80' : '#ff6b6b', fontSize: 12 }}>
                ● {shellConnected ? 'Connected' : 'Disconnected'}
              </span>
              {!shellConnected ? (
                <button onClick={connectShell} style={{ padding: '6px 12px', background: '#4ade80', border: 'none', borderRadius: 3, color: '#000', cursor: 'pointer', fontSize: 12 }}>Connect</button>
              ) : (
                <button onClick={disconnectShell} style={{ padding: '6px 12px', background: '#ff6b6b', border: 'none', borderRadius: 3, color: '#fff', cursor: 'pointer', fontSize: 12 }}>Disconnect</button>
              )}
              <button onClick={() => setShellOutput('')} style={{ padding: '6px 12px', background: '#0f3460', border: '1px solid #16213e', borderRadius: 3, color: '#fff', cursor: 'pointer', fontSize: 12 }}>Clear</button>
            </div>
          </div>

          <div style={{ background: '#0a0a0a', padding: 10, height: 300, overflowY: 'auto', fontFamily: 'monospace', fontSize: 13, whiteSpace: 'pre-wrap' }}>
            {shellOutput ? stripAnsi(shellOutput) : 'Click "Connect" to start a shell session.'}
          </div>

          <div style={{ display: 'flex', gap: 10, padding: '8px 12px', background: '#0f3460' }}>
            <span style={{ color: '#4ade80' }}>$</span>
            <input
              type="text"
              value={shellInput}
              onChange={e => setShellInput(e.target.value)}
              onKeyPress={e => e.key === 'Enter' && sendShellCommand()}
              placeholder="Enter command..."
              disabled={!shellConnected}
              style={{ flex: 1, background: '#0a0a0a', border: 'none', color: '#fff', fontFamily: 'monospace', padding: 5 }}
            />
            <button
              onClick={sendShellCommand}
              disabled={!shellConnected}
              style={{ padding: '6px 12px', background: '#e94560', border: 'none', borderRadius: 3, color: '#fff', cursor: 'pointer' }}
            >
              Send
            </button>
          </div>
        </div>
      )}

      {loading && (
        <div style={{ position: 'fixed', top: 10, right: 10, padding: '8px 16px', background: '#e94560', borderRadius: 5 }}>
          Loading...
        </div>
      )}
    </div>
  )
}

export default App
