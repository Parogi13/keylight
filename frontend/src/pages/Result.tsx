import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type Health, type Result as R, type Settings } from '../api'
import Slider from '../components/Slider'
import WipePlayer, { type CompareMode } from '../components/WipePlayer'

type Props = { jobId: string; health: Health | null; onBack: () => void }

export default function Result({ jobId, health, onBack }: Props) {
  const [res, setRes] = useState<R | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<CompareMode>('wipe')
  const [showAttention, setShowAttention] = useState(false)
  const [draft, setDraft] = useState<Settings | null>(null)
  const [busy, setBusy] = useState<{ label: string; progress: number } | null>(null)
  const poll = useRef<number>(0)
  const [confirmDelete, setConfirmDelete] = useState(false)

  async function remove() {
    try {
      await api.deleteJob(jobId)
      onBack()
    } catch (e) {
      setError((e as Error).message)
      setConfirmDelete(false)
    }
  }

  const load = useCallback(() => {
    api.result(jobId).then((r) => { setRes(r); setDraft(r.settings) }).catch((e) => setError(e.message))
  }, [jobId])
  useEffect(() => { load(); return () => clearTimeout(poll.current) }, [load])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return
      if (e.key === 'a' || e.key === 'A') setShowAttention((v) => !v)
      if (e.key === 'f' || e.key === 'F') setMode((m) => (m === 'enhanced' ? 'original' : 'enhanced'))
      if (e.key === 'w' || e.key === 'W') setMode('wipe')
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  async function apply() {
    if (!draft) return
    setError(null)
    try {
      await api.rerender(jobId, draft)
      setBusy({ label: 'Waiting', progress: 0 })
      const tick = async () => {
        const j = await api.status(jobId)
        if (j.status === 'complete') { setBusy(null); load(); return }
        if (j.status === 'error') { setBusy(null); setError(j.error); return }
        setBusy({ label: j.label, progress: j.progress })
        poll.current = window.setTimeout(tick, 700)
      }
      tick()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  if (error && !res) return <div className="processing"><p className="error">{error}</p><button className="btn btn-quiet" onClick={onBack}>Back</button></div>
  if (!res || !draft) return <div className="processing"><p className="proc-file">Loading…</p></div>

  const m = res.metadata
  // Auto HDR: play the BT.2020/PQ pair whenever it exists and this browser can decode
  // 10-bit HEVC (Safari/Chrome on Apple Silicon). Both sides of the wipe switch
  // together, so the comparison stays fair. Add ?sdr to the URL to force SDR.
  const hdr = !!(res.original_hdr && res.enhanced_hdr) && canPlayHdr() && !location.search.includes('sdr')
  const claude = health?.claude_available ?? false
  const dirty = (['strength', 'background', 'smoothing', 'focus'] as const).some((k) => draft[k] !== res.settings[k])
  const total = Object.values(m.timings_s).reduce((a, b) => a + b, 0)
  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => setDraft((d) => (d ? { ...d, [k]: v } : d))
  const following = new Set(m.focus_timeline.filter((s) => s.target != null).map((s) => s.cls)).size

  return (
    <div className="result">
      <div className="result-head">
        <h1 className="result-title">{res.name.replace(/\.[^.]+$/, '')}</h1>
        <div className="result-actions">
          {confirmDelete ? (
            <span className="library-confirm">
              <span>Delete this video?</span>
              <button className="btn-danger" onClick={remove}>Delete</button>
              <button className="link" onClick={() => setConfirmDelete(false)}>Keep</button>
            </span>
          ) : (
            <button className="link" disabled={!!busy} onClick={() => setConfirmDelete(true)}>Delete video</button>
          )}
          <button className="link" onClick={onBack}>Enhance another video</button>
        </div>
      </div>

      <WipePlayer
        original={hdr ? res.original_hdr! : res.original}
        enhanced={hdr ? res.enhanced_hdr! : res.enhanced}
        attention={res.attention}
        showAttention={showAttention}
        mode={mode}
        duration={m.video.duration}
        aspect={m.video.width / m.video.height}
        segments={m.focus_timeline}
        cuts={m.cuts_s}
      />

      <div className="result-grid">
        <section className="panel">
          <h2>Compare</h2>
          <div className="segmented" role="radiogroup" aria-label="Comparison mode">
            {([['wipe', 'Wipe'], ['original', 'Original'], ['enhanced', 'Keylight']] as const).map(([k, label]) => (
              <button key={k} role="radio" aria-checked={mode === k} className={mode === k ? 'on' : ''} onClick={() => setMode(k)}>{label}</button>
            ))}
          </div>
          <label className="switch">
            <input type="checkbox" checked={showAttention} onChange={(e) => setShowAttention(e.target.checked)} />
            <span className="switch-track" aria-hidden><span className="switch-thumb" /></span>
            Show what Keylight is following
          </label>
          <p className="keys">Keyboard: Space plays, W wipe, F flips between versions, A shows attention.</p>
        </section>

        <section className="panel">
          <h2>Adjust</h2>
          <Slider label="Subject lift" format={pct} value={draft.strength} onChange={(v) => set('strength', v)} />
          <Slider label="Background dimming" value={draft.background} max={0.8} format={pct} onChange={(v) => set('background', v)} />
          <Slider label="Smoothness" format={pct} value={draft.smoothing} onChange={(v) => set('smoothing', v)} />
          <div className="field">
            <label htmlFor="focus2">What should Keylight follow?</label>
            <input id="focus2" type="text" disabled={!claude} placeholder={claude ? 'Follow the person on the left' : 'Needs an Anthropic API key'}
              value={draft.focus} onChange={(e) => set('focus', e.target.value)} />
          </div>
          <button className="btn btn-key" disabled={!dirty || !!busy} onClick={apply}>
            {busy ? `${busy.label}… ${Math.round(busy.progress * 100)}%` : 'Apply changes'}
          </button>
          {error && <p className="error" role="alert">{error}</p>}
        </section>

        <section className="panel panel-facts">
          <h2>How this was made</h2>
          <dl>
            <div><dt>Processed in</dt><dd>{total.toFixed(0)} s on {m.device.toUpperCase()}</dd></div>
            <div><dt>Scenes</dt><dd>{m.shots}</dd></div>
            <div><dt>Subjects followed</dt><dd>{following}</dd></div>
            <div><dt>Subject choice</dt><dd>{m.semantic?.enabled && m.semantic.ok ? `Claude (${m.semantic.ok} still frames sent)` : 'On this computer'}</dd></div>
            <div><dt>Playing</dt><dd>{hdr ? 'HDR (subject above SDR white)' : 'SDR'}</dd></div>
            <div><dt>Video</dt><dd>{m.video.width}×{m.video.height}, {m.video.fps.toFixed(2).replace(/\.00$/, '')} fps{m.video.has_audio ? ', with sound' : ''}</dd></div>
          </dl>
          {m.semantic?.enabled && m.semantic.error && (
            <p className="facts-note">{m.semantic.ok ? 'Some frames fell back to the local choice: ' : 'Claude wasn’t used: '}{m.semantic.error}</p>
          )}
        </section>
      </div>
    </div>
  )
}

const pct = (v: number) => `${Math.round(v * 100)}%`

function canPlayHdr(): boolean {
  try {
    return document.createElement('video').canPlayType('video/mp4; codecs="hvc1.2.4.L123.B0"') !== ''
  } catch {
    return false
  }
}
