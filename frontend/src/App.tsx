import { useEffect, useState } from 'react'
import { api, type Health } from './api'
import Home from './pages/Home'
import Processing from './pages/Processing'
import Result from './pages/Result'

// Tiny hash router: #/ (home), #/job/<id> (processing or result, decided by status).
function useHashRoute(): [string | null, (id: string | null) => void] {
  const read = () => {
    const m = window.location.hash.match(/^#\/job\/([\w-]+)/)
    return m ? m[1] : null
  }
  const [jobId, setJobId] = useState<string | null>(read)
  useEffect(() => {
    const on = () => setJobId(read())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  const go = (id: string | null) => {
    window.location.hash = id ? `/job/${id}` : '/'
  }
  return [jobId, go]
}

export default function App() {
  const [jobId, go] = useHashRoute()
  const [health, setHealth] = useState<Health | null>(null)
  const [backendDown, setBackendDown] = useState(false)
  const [done, setDone] = useState<Record<string, boolean>>({})

  useEffect(() => {
    api.health().then(setHealth).catch(() => setBackendDown(true))
  }, [])

  return (
    <div className="app">
      <header className="topbar">
        <button className="wordmark" onClick={() => go(null)} aria-label="Keylight home">
          <span className="wordmark-dot" aria-hidden />
          Keylight
        </button>
        <span className="locality" title="Video processing runs on this computer">
          {health ? `Processing on this computer (${health.device.toUpperCase()})` : backendDown ? '' : ' '}
        </span>
      </header>

      {backendDown && (
        <div className="banner banner-error" role="alert">
          Keylight can’t reach its local engine. Start it with <code>python backend/main.py</code>, then reload this page.
        </div>
      )}

      <main>
        {!jobId && <Home health={health} onStarted={(id) => go(id)} onOpen={(id) => { setDone((d) => ({ ...d, [id]: true })); go(id) }} />}
        {jobId && !done[jobId] && (
          <Processing jobId={jobId} onComplete={() => setDone((d) => ({ ...d, [jobId]: true }))} onBack={() => go(null)} />
        )}
        {jobId && done[jobId] && <Result jobId={jobId} health={health} onBack={() => go(null)} />}
      </main>
    </div>
  )
}
