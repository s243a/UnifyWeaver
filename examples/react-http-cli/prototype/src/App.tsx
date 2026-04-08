import { useState, useRef } from 'react'
import './index.css'

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
        <h1 className="text-center" style={{ marginBottom: 30 }}>🔍 UnifyWeaver CLI</h1>
        <div className="panel login-container">
          <h2 style={{ marginBottom: 20 }}>Login Required</h2>
          <div style={{ marginBottom: 15 }}>
            <label className="text-muted" style={{ display: 'block', marginBottom: 5 }}>Email</label>
            <input
              type="email"
              value={loginEmail}
              onChange={e => setLoginEmail(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleLogin()}
              className="input-field"
            />
          </div>
          <div style={{ marginBottom: 20 }}>
            <label className="text-muted" style={{ display: 'block', marginBottom: 5 }}>Password</label>
            <input
              type="password"
              value={loginPassword}
              onChange={e => setLoginPassword(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleLogin()}
              className="input-field"
            />
          </div>
          <button
            onClick={handleLogin}
            disabled={loading}
            className="btn btn-primary"
            style={{ width: '100%' }}
          >
            {loading ? 'Logging in...' : 'Login'}
          </button>
          {loginError && (
            <p className="text-error text-center" style={{ marginTop: 15 }}>{loginError}</p>
          )}
          <p className="text-muted text-center" style={{ marginTop: 20 }}>
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
      <div className="header-panel flex-between">
        <div className="flex-row">
          <span>{user.email}</span>
          {user.roles.map(role => (
            <span key={role} className="role-badge">{role}</span>
          ))}
        </div>
        <button onClick={handleLogout} className="btn btn-small btn-secondary">
          Logout
        </button>
      </div>

      {/* Working directory bar */}
      <div className="header-panel flex-row">
        <select
          value={browseRoot}
          onChange={e => handleRootChange(e.target.value)}
          className="select-field"
        >
          <option value="sandbox">Sandbox</option>
          <option value="project">Project</option>
          <option value="home">Home</option>
        </select>
        <span className="text-muted">Working Dir:</span>
        <code className="path-code path-code-success">{workingDir}</code>
      </div>

      {/* Tabs */}
      <div className="flex-row" style={{ marginBottom: 20 }}>
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`btn ${activeTab === tab.id ? 'btn-primary' : (tab.highlight ? 'btn-accent' : 'btn-panel')}`}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {/* Browse panel */}
      {activeTab === 'browse' && (
        <div className="panel">
          <div className="flex-row" style={{ marginBottom: 15 }}>
            {browsePath !== '.' && (
              <button onClick={navigateUp} className="btn btn-small btn-secondary">⬆️ Up</button>
            )}
            <span>📁</span>
            <code className="path-code">{browsePath}</code>
            <button
              onClick={() => setWorkingDir(browsePath)}
              disabled={workingDir === browsePath}
              className={`btn btn-small ${workingDir === browsePath ? 'btn-outline' : 'btn-success'}`}
            >
              📌 Set as Working Dir
            </button>
          </div>

          <div className="file-list">
            {browseEntries.map((entry, i) => (
              <div
                key={i}
                onClick={() => handleEntryClick(entry)}
                className={`file-item ${selectedFile === entry.name ? 'file-item-selected' : 'file-item-normal'} ${entry.type === 'directory' ? 'file-item-dir' : 'file-item-file'}`}
              >
                <div className="flex-between">
                  <span>{entry.type === 'directory' ? '📁' : '📄'} {entry.name}</span>
                  <span className="text-muted">{formatSize(entry.size)}</span>
                </div>
              </div>
            ))}
            {browseEntries.length === 0 && !loading && (
              <p className="text-muted text-center">Empty directory</p>
            )}
          </div>

          {selectedFile && (
            <div className="selected-panel">
              <p className="text-muted" style={{ marginBottom: 10 }}>Selected: <code className="path-code">{selectedFile}</code></p>
              <div className="flex-row">
                <button onClick={viewFile} className="btn btn-primary">View Contents</button>
                <button onClick={downloadFile} className="btn btn-primary">📥 Download</button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Upload panel */}
      {activeTab === 'upload' && (
        <div className="panel">
          <div className="header-panel" style={{ marginBottom: 15, padding: 10 }}>
            <p className="text-muted" style={{ margin: 0 }}>Destination: <code className="path-code">{workingDir}</code> ({browseRoot})</p>
          </div>

          {/* File System Access API - better for Android */}
          <div className="drop-zone" onClick={openFilePicker}>
            <p style={{ fontSize: 18, margin: '0 0 5px 0' }}>📂 Open File Picker</p>
            <p className="text-muted" style={{ margin: 0 }}>Recommended for Android - picks and uploads immediately</p>
          </div>

          {/* Fallback standard file input */}
          <div className="drop-zone-fallback">
            <p className="text-muted" style={{ fontSize: 14, margin: '0 0 10px 0' }}>Or use standard file input:</p>
            <input
              type="file"
              multiple
              onChange={handleFileSelect}
              style={{ padding: 10 }}
            />
          </div>

          {uploadFiles.length > 0 && (
            <div style={{ marginBottom: 20 }}>
              <p className="text-muted" style={{ marginBottom: 10 }}>Selected files:</p>
              {uploadFiles.map((file, i) => (
                <div key={i} className="flex-between" style={{ padding: '8px 12px', background: 'var(--color-bg-item)', marginBottom: 4, borderRadius: 5 }}>
                  <span>{file.name}</span>
                  <span className="text-muted">{formatSize(file.size)}</span>
                </div>
              ))}
            </div>
          )}

          {uploadFiles.length > 0 && (
            <button
              onClick={handleUpload}
              disabled={loading}
              className="btn btn-primary"
              style={{ width: '100%' }}
            >
              {loading ? 'Uploading...' : '📤 Upload Files'}
            </button>
          )}

          {uploadResult && (
            <div className={`result-box ${uploadResult.startsWith('Error') ? 'result-error' : 'result-success'}`}>
              {uploadResult}
            </div>
          )}
        </div>
      )}

      {/* Cat panel */}
      {activeTab === 'cat' && (
        <div className="panel">
          <div className="flex-row" style={{ marginBottom: 15 }}>
            <input
              type="text"
              value={catPath}
              onChange={e => setCatPath(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleCat()}
              placeholder="File path..."
              className="input-field"
              style={{ flex: 1 }}
            />
            <button
              onClick={() => handleCat()}
              disabled={loading}
              className="btn btn-primary"
            >
              {loading ? 'Loading...' : 'Read File'}
            </button>
          </div>

          {catContent && (
            <>
              <div style={{ marginBottom: 10 }}>
                <button
                  onClick={() => { setActiveTab('browse'); setCatContent(''); }}
                  className="btn btn-small btn-secondary"
                >
                  ← Back to Browse
                </button>
              </div>
              <div className="content-viewer">
                <pre className="content-pre">{catContent}</pre>
              </div>
            </>
          )}
        </div>
      )}

      {/* Shell panel */}
      {activeTab === 'shell' && (
        <div style={{ borderRadius: 5, overflow: 'hidden' }}>
          <div className="shell-header flex-between">
            <span style={{ color: 'var(--color-accent)', fontWeight: 'bold' }}>🔐 Shell</span>
            <div className="flex-row">
              <span style={{ color: shellConnected ? 'var(--color-success)' : 'var(--color-error)', fontSize: 12 }}>
                ● {shellConnected ? 'Connected' : 'Disconnected'}
              </span>
              {!shellConnected ? (
                <button onClick={connectShell} className="btn btn-tiny btn-success">Connect</button>
              ) : (
                <button onClick={disconnectShell} className="btn btn-tiny btn-error">Disconnect</button>
              )}
              <button onClick={() => setShellOutput('')} className="btn btn-tiny btn-outline">Clear</button>
            </div>
          </div>

          <div className="shell-body">
            {shellOutput ? stripAnsi(shellOutput) : 'Click "Connect" to start a shell session.'}
          </div>

          <div className="shell-footer flex-row">
            <span style={{ color: 'var(--color-success)' }}>$</span>
            <input
              type="text"
              value={shellInput}
              onChange={e => setShellInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && sendShellCommand()}
              placeholder="Enter command..."
              disabled={!shellConnected}
              className="input-shell"
            />
            <button
              onClick={sendShellCommand}
              disabled={!shellConnected}
              className="btn btn-tiny btn-primary"
            >
              Send
            </button>
          </div>
        </div>
      )}

      {loading && (
        <div className="loading-toast">
          Loading...
        </div>
      )}
    </div>
  )
}

export default App
