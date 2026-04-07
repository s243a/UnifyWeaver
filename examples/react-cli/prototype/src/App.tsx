import React, { useState } from 'react'

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

  // Browse state
  const [browse, setBrowse] = useState({
    path: '/home/user/projects',
    parent: '/home/user',
    entries: mockFS['/home/user/projects'] || [],
    selected: null as string | null
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
    } else {
      setBrowse(prev => ({ ...prev, selected: entry.name }))
      setFileContent(null)
    }
  }

  const setWorkingDirTo = (path: string) => {
    setWorkingDir(path)
    alert(`Working directory set to: ${path}`)
  }

  const viewFile = () => {
    if (browse.selected) {
      setLoading(true)
      // Simulate loading
      setTimeout(() => {
        setFileContent(`// Contents of ${browse.selected}\n\nexport default function Example() {\n  return <div>Hello World</div>\n}`)
        setLoading(false)
      }, 300)
    }
  }

  const downloadFile = () => {
    if (browse.selected) {
      alert(`Downloading: ${browse.path}/${browse.selected}`)
    }
  }

  const searchHere = () => {
    const pattern = prompt('Enter search pattern:')
    if (pattern) {
      alert(`Searching for "${pattern}" in ${browse.path}...`)
    }
  }

  return (
    <div className="app-container">
      <div style={{ background: "#16213e", padding: 20, borderRadius: 5 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 15 }}>
          {/* Navigation bar */}
          <div style={{ display: "flex", flexDirection: "row", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            {browse.parent && (
              <button onClick={navigateUp} style={{ padding: "10px 20px", background: "#0f3460", border: "none", color: "#fff", cursor: "pointer", borderRadius: 5 }}>⬆️ Up</button>
            )}
            <span style={{ fontSize: 18 }}>📁 </span>
            <code style={{ background: "#1a1a2e", padding: "4px 8px", borderRadius: 3, fontFamily: "monospace" }}>{browse.path}</code>
            <button
              onClick={() => setWorkingDirTo(browse.path)}
              disabled={workingDir === browse.path}
              style={{ padding: "10px 20px", background: workingDir === browse.path ? "#555" : "#e94560", border: "none", color: "#fff", cursor: workingDir === browse.path ? "not-allowed" : "pointer", borderRadius: 5 }}
            >📌 Set as Working Dir</button>
          </div>

          {/* Entry count */}
          <span style={{ color: "#94a3b8", fontSize: 12 }}>{browse.entries.length} items</span>

          {/* File list */}
          <div style={{ overflowY: "auto", maxHeight: 300 }}>
            {browse.entries.map((entry, index) => (
              <div
                key={index}
                onClick={() => handleEntryClick(entry)}
                style={{
                  background: browse.selected === entry.name ? "#0f3460" : "#1a1a2e",
                  padding: "12px 16px",
                  borderRadius: 5,
                  marginBottom: 4,
                  cursor: "pointer",
                  borderLeft: entry.type === 'directory' ? "3px solid #e94560" : "3px solid #3b82f6",
                  transition: "background 0.2s"
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <span>{entry.type === 'directory' ? '📁' : '📄'}</span>
                    <span>{entry.name}</span>
                  </div>
                  <span style={{ color: "#94a3b8", fontSize: 12 }}>{formatSize(entry.size)}</span>
                </div>
              </div>
            ))}
          </div>

          {/* Empty state */}
          {browse.entries.length === 0 && !loading && (
            <span style={{ color: "#94a3b8", fontSize: 12, textAlign: "center" }}>Empty directory</span>
          )}

          {/* Selected file actions */}
          {browse.selected && (
            <div style={{ background: "#0f3460", padding: 16, borderRadius: 5 }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <span style={{ color: "#94a3b8", fontSize: 12 }}>Selected file:</span>
                <code style={{ background: "#1a1a2e", padding: "4px 8px", borderRadius: 3, fontFamily: "monospace" }}>{browse.selected}</code>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  <button onClick={viewFile} disabled={loading} style={{ padding: "10px 20px", background: "#e94560", border: "none", color: "#fff", cursor: "pointer", borderRadius: 5, fontWeight: "bold" }}>
                    {loading ? "Loading..." : "View Contents"}
                  </button>
                  <button onClick={downloadFile} style={{ padding: "10px 20px", background: "#e94560", border: "none", color: "#fff", cursor: "pointer", borderRadius: 5, fontWeight: "bold" }}>📥 Download</button>
                  <button onClick={searchHere} style={{ padding: "10px 20px", background: "#16213e", border: "none", color: "#fff", cursor: "pointer", borderRadius: 5 }}>Search Here</button>
                </div>
              </div>
            </div>
          )}

          {/* File content viewer */}
          {fileContent && (
            <div style={{ background: "#0a0a0a", padding: 16, borderRadius: 5, maxHeight: 200, overflowY: "auto" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                <span style={{ color: "#4ade80", fontSize: 12 }}>File contents:</span>
                <button onClick={() => setFileContent(null)} style={{ background: "none", border: "none", color: "#ff6b6b", cursor: "pointer" }}>✕ Close</button>
              </div>
              <pre style={{ margin: 0, fontFamily: "monospace", fontSize: 13, color: "#cdd6f4", whiteSpace: "pre-wrap" }}>{fileContent}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
