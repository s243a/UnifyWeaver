import { useState } from 'react'
import './index.css'

// Helper function for formatting file sizes
const formatSize = (bytes: number): string => {
  if (!bytes || bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

// Mock file system for demo
const mockFS: Record<string, Array<{name: string, type: string, size: number}>> = {
  '/home/user': [
    { name: 'projects', type: 'directory', size: 4096 },
    { name: 'documents', type: 'directory', size: 4096 },
    { name: '.bashrc', type: 'file', size: 220 },
  ],
  '/home/user/projects': [
    { name: 'src', type: 'directory', size: 4096 },
    { name: 'docs', type: 'directory', size: 4096 },
    { name: 'package.json', type: 'file', size: 1240 },
    { name: 'README.md', type: 'file', size: 3500 },
    { name: 'tsconfig.json', type: 'file', size: 562 },
    { name: 'vite.config.ts', type: 'file', size: 180 },
  ],
  '/home/user/projects/src': [
    { name: 'App.tsx', type: 'file', size: 2400 },
    { name: 'main.tsx', type: 'file', size: 150 },
    { name: 'index.css', type: 'file', size: 800 },
  ],
  '/home/user/projects/docs': [
    { name: 'GUIDE.md', type: 'file', size: 5200 },
    { name: 'API.md', type: 'file', size: 8900 },
  ],
  '/home/user/documents': [
    { name: 'notes.txt', type: 'file', size: 450 },
    { name: 'todo.md', type: 'file', size: 120 },
  ],
}

function App() {
  // State
  const [loading, setLoading] = useState(false)
  const [fileContent, setFileContent] = useState<string | null>(null)
  const [notification, setNotification] = useState<string | null>(null)
  
  // Search state
  const [isSearching, setIsSearching] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')

  // Browse state
  const [browse, setBrowse] = useState<{
    path: string,
    parent: string | null,
    entries: Array<{name: string, type: string, size: number}>,
    selected: string | null
  }>({
    path: '/home/user/projects',
    parent: '/home/user',
    entries: mockFS['/home/user/projects'] || [],
    selected: null
  })

  const [workingDir, setWorkingDir] = useState('.')

  // Get parent path
  const getParent = (path: string): string | null => {
    if (path === '/' || path === '/home/user') return null
    const parts = path.split('/')
    parts.pop()
    return parts.join('/') || '/'
  }

  // Event handlers
  const navigateUp = () => {
    if (browse.parent) {
      const newPath = browse.parent
      setBrowse({
        path: newPath,
        parent: getParent(newPath),
        entries: mockFS[newPath] || [],
        selected: null
      })
      setFileContent(null)
      setNotification(null)
      setIsSearching(false)
    }
  }

  const handleEntryClick = (entry: {name: string, type: string, size: number}) => {
    if (entry.type === 'directory') {
      const newPath = `${browse.path}/${entry.name}`
      setBrowse({
        path: newPath,
        parent: browse.path,
        entries: mockFS[newPath] || [],
        selected: null
      })
      setFileContent(null)
      setNotification(null)
      setIsSearching(false)
    } else {
      setBrowse(prev => ({ ...prev, selected: entry.name }))
      setFileContent(null)
      setNotification(null)
      setIsSearching(false)
    }
  }

  const setWorkingDirTo = (path: string) => {
    setWorkingDir(path)
    setNotification(`Working directory set to: ${path}`)
  }

  const viewFile = () => {
    if (browse.selected) {
      setLoading(true)
      // NOTE: demo simulation
      setTimeout(() => {
        setFileContent(`// Contents of ${browse.selected}\n\nexport default function Example() {\n  return <div>Hello World</div>\n}`)
        setLoading(false)
      }, 300)
    }
  }

  const downloadFile = () => {
    if (browse.selected) {
      setNotification(`Downloading: ${browse.path}/${browse.selected}`)
    }
  }

  const handleSearchSubmit = () => {
    if (searchQuery) {
      setNotification(`Searching for "${searchQuery}" in ${browse.path}...`)
      setIsSearching(false)
      setSearchQuery('')
    }
  }

  return (
    <div className="app-container">
      <div className="panel">
        <div className="flex-col">
          {/* Navigation bar */}
          <div className="flex-row">
            {browse.parent && (
              <button onClick={navigateUp} className="btn btn-secondary">⬆️ Up</button>
            )}
            <span style={{ fontSize: 18 }}>📁 </span>
            <code className="path-code">{browse.path}</code>
            <button
              onClick={() => setWorkingDirTo(browse.path)}
              disabled={workingDir === browse.path}
              className={`btn ${workingDir === browse.path ? 'btn-panel' : 'btn-primary'}`}
            >📌 Set as Working Dir</button>
          </div>

          {/* Entry count */}
          <span className="text-muted">{browse.entries.length} items</span>

          {/* File list */}
          <div className="file-list">
            {browse.entries.map((entry, index) => (
              <div
                key={index}
                onClick={() => handleEntryClick(entry)}
                className={`file-item ${browse.selected === entry.name ? 'file-item-selected' : 'file-item-normal'} ${entry.type === 'directory' ? 'file-item-dir' : 'file-item-file'}`}
              >
                <div className="file-item-content">
                  <div className="file-item-left">
                    <span>{entry.type === 'directory' ? '📁' : '📄'}</span>
                    <span>{entry.name}</span>
                  </div>
                  <span className="text-muted">{formatSize(entry.size)}</span>
                </div>
              </div>
            ))}
          </div>

          {/* Empty state */}
          {browse.entries.length === 0 && !loading && (
            <span className="text-muted text-center">Empty directory</span>
          )}

          {/* Selected file actions */}
          {browse.selected && (
            <div className="selected-panel">
              <div className="selected-panel-col">
                <span className="text-muted">Selected file:</span>
                <code className="path-code">{browse.selected}</code>
                <div className="flex-row">
                  <button onClick={viewFile} disabled={loading} className="btn btn-primary">
                    {loading ? "Loading..." : "View Contents"}
                  </button>
                  <button onClick={downloadFile} className="btn btn-primary">📥 Download</button>
                  <button onClick={() => setIsSearching(true)} className="btn btn-panel">Search Here</button>
                </div>
                
                {isSearching && (
                  <div className="flex-row" style={{ marginTop: '10px' }}>
                    <input 
                      type="text" 
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                      placeholder="Enter search pattern..."
                      className="search-input"
                      autoFocus
                      onKeyDown={e => e.key === 'Enter' && handleSearchSubmit()}
                    />
                    <button onClick={handleSearchSubmit} className="btn btn-secondary">Go</button>
                    <button onClick={() => setIsSearching(false)} className="btn btn-text text-muted">Cancel</button>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Notification */}
          {notification && (
            <div className="notification">
              <span className="text-success">{notification}</span>
              <button onClick={() => setNotification(null)} className="btn-text text-muted">✕</button>
            </div>
          )}

          {/* File content viewer */}
          {fileContent && (
            <div className="content-viewer">
              <div className="content-header">
                <span className="text-success">File contents:</span>
                <button onClick={() => setFileContent(null)} className="btn-text btn-text-error">✕ Close</button>
              </div>
              <pre className="content-pre">{fileContent}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
