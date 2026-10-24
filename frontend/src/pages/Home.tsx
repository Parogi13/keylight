import { useEffect, useRef, useState } from 'react'
import { api, type Health, type Job, type Settings } from '../api'
import Slider from '../components/Slider'

const FALLBACK: Settings = { strength: 0.6, background: 0.15, glow: 0.35, smoothing: 0.6, focus: '', use_claude: true }

type Props = { health: Health | null; onStarted: (id: string) => void; onOpen: (id: string) => void }

export default function Home({ health, onStarted, onOpen }: Props) {
  const [settings, setSettings] = useState<Settings>(FALLBACK)
  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const [upload, setUpload] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [samples, setSamples] = useState<{ name: string; size: number }[]>([])
  const [recent, setRecent] = useState<Job[]>([])
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (health?.defaults) setSettings({ ...FALLBACK, ...health.defaults })
  }, [health])
  useEffect(() => {
    api.samples().then(setSamples).catch(() => {})
    api.jobs().then((j) => setRecent(j.filter((x) => x.status === 'complete').slice(0, 6))).catch(() => {})
  }, [])

  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => setSettings((s) => ({ ...s, [k]: v }))
  const claude = health?.claude_available ?? false

  function pick(f: File | undefined | null) {
    setError(null)
    if (!f) return
    if (!f.type.startsWith('video/') && !/\.(mp4|mov|m4v|mkv|webm|avi)$/i.test(f.name)) {
      setError(`${f.name} isn’t a video file. Choose an MP4 or MOV.`)
      return
    }
    setFile(f)
  }

  async function start() {
    if (!file) return
    setError(null)
    setUpload(0)
    try {
      const job = await api.processFile(file, { ...settings, use_claude: claude && settings.use_claude }, setUpload)
      onStarted(job.id)
    } catch (e) {
      setError((e as Error).message)
      setUpload(null)
    }
  }

  async function startSample(name: string) {
    setError(null)
    try {
      const job = await api.processSample(name, { ...settings, use_claude: claude && settings.use_claude })
      onStarted(job.id)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div className="home">
      <section className="hero">
        <h1 className="hero-title">
          Make what matters easier to see.
        </h1>
        <p className="hero-copy">
          Keylight finds the subject of every scene and lights it the way a cinematographer would: the subject
          lifts, the background settles back. Built for people with low vision who lose track of the action in
          dark or busy video. Nothing is uploaded; your video stays on this computer.
        </p>
      </section>

      <section className="workbench">
        <div
          className={`drop ${dragging ? 'is-dragging' : ''} ${file ? 'has-file' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files?.[0]) }}
        >
          <div className="drop-light" aria-hidden />
          {file ? (
            <div className="drop-file">
              <p className="drop-name">{file.name}</p>
              <p className="drop-meta">{(file.size / 1e6).toFixed(1)} MB</p>
              <button className="link" onClick={() => setFile(null)}>Choose a different video</button>
            </div>
          ) : (
            <div className="drop-empty">
              <p className="drop-title">Drop a video here</p>
              <p className="drop-meta">MP4 or MOV, up to a few minutes long</p>
              <button className="btn btn-quiet" onClick={() => input.current?.click()}>Browse files</button>
            </div>
          )}
          <input ref={input} type="file" accept="video/*" hidden onChange={(e) => pick(e.target.files?.[0])} />
        </div>

        <div className="controls">
          <Slider label="Subject lift" format={pct} hint="How much the subject brightens and sharpens."
            value={settings.strength} onChange={(v) => set('strength', v)} />
          <Slider label="Background dimming" hint="How far everything else settles back."
            value={settings.background} max={0.8} format={pct} onChange={(v) => set('background', v)} />
          <Slider label="Smoothness" format={pct} hint="How gently focus moves between subjects."
            value={settings.smoothing} onChange={(v) => set('smoothing', v)} />

          <div className="field">
            <label htmlFor="focus">What should Keylight follow? <span className="optional">Optional</span></label>
            <input
              id="focus"
              type="text"
              placeholder="Follow the woman in the red jacket"
              value={settings.focus}
              disabled={!claude}
              onChange={(e) => set('focus', e.target.value)}
            />
            <p className="field-hint">
              {claude
                ? 'Uses Claude to pick the subject, sending only a few still frames. Leave empty to choose automatically.'
                : 'Needs an Anthropic API key. Without one, Keylight picks the subject on this computer.'}
            </p>
          </div>

          <button className="btn btn-key" disabled={!file || upload !== null} onClick={start}>
            {upload === null ? 'Enhance video' : upload < 1 ? `Reading video… ${Math.round(upload * 100)}%` : 'Starting…'}
          </button>
          {error && <p className="error" role="alert">{error}</p>}
        </div>
      </section>

      {(samples.length > 0 || recent.length > 0) && (
        <section className="shelf">
          {samples.length > 0 && (
            <div>
              <h2>Try a sample clip</h2>
              <ul className="chips">
                {samples.map((s) => (
                  <li key={s.name}>
                    <button className="chip" onClick={() => startSample(s.name)}>{prettify(s.name)}</button>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {recent.length > 0 && (
            <div>
              <h2>Open a finished video</h2>
              <ul className="chips">
                {recent.map((j) => (
                  <li key={j.id}>
                    <button className="chip" onClick={() => onOpen(j.id)}>{prettify(j.name)}</button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}
    </div>
  )
}

function prettify(name: string) {
  const base = name.replace(/\.[^.]+$/, '').replace(/[-_]+/g, ' ')
  return base.charAt(0).toUpperCase() + base.slice(1)
}

const pct = (v: number) => `${Math.round(v * 100)}%`
