// Thin client for the local FastAPI backend. Everything stays on this machine.

export type Settings = {
  strength: number
  background: number
  glow: number
  smoothing: number
  focus: string
  use_claude: boolean
}

export type Job = {
  id: string
  name: string
  status: 'queued' | 'processing' | 'complete' | 'error'
  stage: string
  label: string
  progress: number
  error: string | null
  settings: Settings
  version: number
  created: number
}

export type Segment = {
  start: number
  end: number
  t0: number
  t1: number
  target: number | null
  cls: string | null
  source: 'semantic' | 'heuristic' | 'none'
  reason: string
}

export type Metadata = {
  video: { width: number; height: number; fps: number; duration: number; frame_count: number; has_audio: boolean }
  device: string
  shots: number
  cuts_s: number[]
  focus_timeline: Segment[]
  semantic: { enabled: boolean; ok?: number; calls?: number; error?: string | null; model?: string; cached?: boolean }
  timings_s: Record<string, number>
  realtime_factor: number
}

export type Result = Job & {
  original: string; enhanced: string; attention: string; metadata: Metadata
  original_hdr: string | null; enhanced_hdr: string | null  // BT.2020/PQ pair, when rendered
}

export type Health = { ok: boolean; device: string; claude_available: boolean; claude_model: string; defaults: Settings }

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try {
      const body = await r.json()
      if (body?.detail) msg = String(body.detail)
    } catch { /* not JSON */ }
    throw new Error(msg)
  }
  return r.json() as Promise<T>
}

export const api = {
  health: () => fetch('/api/health').then(json<Health>),
  samples: () => fetch('/api/samples').then(json<{ name: string; size: number }[]>),
  jobs: () => fetch('/api/jobs').then(json<Job[]>),
  status: (id: string) => fetch(`/api/status/${id}`).then(json<Job>),
  result: (id: string) => fetch(`/api/result/${id}`).then(json<Result>),

  processFile(file: File, settings: Settings, onUpload?: (f: number) => void): Promise<Job> {
    // XHR rather than fetch so we can show progress while the file streams to localhost.
    return new Promise((resolve, reject) => {
      const form = new FormData()
      form.append('file', file)
      form.append('settings', JSON.stringify(settings))
      const xhr = new XMLHttpRequest()
      xhr.open('POST', '/api/process')
      xhr.upload.onprogress = (e) => e.lengthComputable && onUpload?.(e.loaded / e.total)
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText))
        else {
          let msg = `Upload failed (${xhr.status})`
          try { msg = JSON.parse(xhr.responseText).detail ?? msg } catch { /* ignore */ }
          reject(new Error(msg))
        }
      }
      xhr.onerror = () => reject(new Error('Could not reach the Keylight backend on this computer. Is it running?'))
      xhr.send(form)
    })
  },

  processSample(name: string, settings: Settings): Promise<Job> {
    const form = new FormData()
    form.append('sample', name)
    form.append('settings', JSON.stringify(settings))
    return fetch('/api/process', { method: 'POST', body: form }).then(json<Job>)
  },

  deleteJob: (id: string) => fetch(`/api/jobs/${id}`, { method: 'DELETE' }).then(json<{ ok: boolean }>),

  rerender: (id: string, settings: Partial<Settings>) =>
    fetch(`/api/rerender/${id}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ settings }),
    }).then(json<Job>),
}
