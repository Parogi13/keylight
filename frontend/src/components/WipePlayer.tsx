import { useCallback, useEffect, useRef, useState } from 'react'
import type { Segment } from '../api'

export type CompareMode = 'wipe' | 'original' | 'enhanced'

type Props = {
  original: string
  enhanced: string
  attention: string
  showAttention: boolean
  mode: CompareMode
  duration: number
  aspect: number
  segments: Segment[]
  cuts: number[]
}

/**
 * Original and enhanced videos stacked in exactly the same box. The original is
 * the timing master (and the only one with sound); the others follow it. A
 * draggable divider reveals the enhanced video to its right via clip-path.
 */
export default function WipePlayer({ original, enhanced, attention, showAttention, mode, duration, aspect, segments, cuts }: Props) {
  const master = useRef<HTMLVideoElement>(null)
  const enh = useRef<HTMLVideoElement>(null)
  const att = useRef<HTMLVideoElement>(null)
  const stage = useRef<HTMLDivElement>(null)
  const [split, setSplit] = useState(0.5)
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [dragging, setDragging] = useState(false)

  const followers = useCallback(() => [enh.current, att.current].filter(Boolean) as HTMLVideoElement[], [])
  const showAtt = useRef(showAttention)
  showAtt.current = showAttention

  // Keep the *visible* follower locked to the master clock. Small drift is corrected
  // by nudging playbackRate; large drift (seeks, stalls) by setting currentTime. The
  // hidden overlay is paused so the browser decodes two videos, not three; it is
  // re-synced the moment it becomes visible.
  useEffect(() => {
    let raf = 0
    let lastUi = -1
    const loop = () => {
      const m = master.current
      if (m) {
        // Re-render the transport at ~15 Hz, not every animation frame.
        if (Math.abs(m.currentTime - lastUi) > 1 / 15 || m.paused) {
          lastUi = m.currentTime
          setTime(m.currentTime)
        }
        const visible = showAtt.current ? att.current : enh.current
        const hidden = showAtt.current ? enh.current : att.current
        if (hidden && !hidden.paused) hidden.pause()
        for (const f of visible ? [visible] : []) {
          const d = f.currentTime - m.currentTime
          if (Math.abs(d) > 0.25 || m.paused) {
            if (Math.abs(d) > 0.02) f.currentTime = m.currentTime
            f.playbackRate = m.playbackRate
          } else {
            f.playbackRate = m.playbackRate * (1 - Math.max(-0.08, Math.min(0.08, d * 2)))
          }
          if (m.paused !== f.paused) (m.paused ? f.pause() : f.play().catch(() => {}))
        }
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [followers])

  // New render (cache-busted URLs) → keep the playhead where it was.
  useEffect(() => {
    const m = master.current
    if (!m) return
    const t = m.currentTime
    const restore = () => { m.currentTime = t; for (const f of followers()) f.currentTime = t }
    m.addEventListener('loadedmetadata', restore, { once: true })
    return () => m.removeEventListener('loadedmetadata', restore)
  }, [original, enhanced, attention, followers])

  const toggle = useCallback(() => {
    const m = master.current
    if (!m) return
    if (m.paused) m.play().catch(() => {})
    else m.pause()
  }, [])

  const seek = (t: number) => {
    const m = master.current
    if (!m) return
    m.currentTime = Math.max(0, Math.min(duration, t))
    for (const f of followers()) f.currentTime = m.currentTime
  }

  const moveSplit = (clientX: number) => {
    const r = stage.current?.getBoundingClientRect()
    if (!r) return
    setSplit(Math.max(0, Math.min(1, (clientX - r.left) / r.width)))
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT' || (e.target as HTMLElement)?.tagName === 'TEXTAREA') return
      if (e.code === 'Space') { e.preventDefault(); toggle() }
      if (e.key === ',') seek((master.current?.currentTime ?? 0) - 1 / 30)
      if (e.key === '.') seek((master.current?.currentTime ?? 0) + 1 / 30)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const overlay = showAttention ? 'attention' : 'enhanced'
  const clip =
    mode === 'wipe' ? `inset(0 0 0 ${split * 100}%)` : mode === 'enhanced' ? 'inset(0 0 0 0)' : 'inset(0 0 0 100%)'
  const current = segments.find((s) => time >= s.t0 && time < s.t1)

  return (
    <div className="player">
      <div
        ref={stage}
        className={`stage ${dragging ? 'is-dragging' : ''}`}
        style={{ aspectRatio: String(aspect) }}
        onPointerDown={(e) => {
          if (mode !== 'wipe') return toggle()
          ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
          setDragging(true)
          moveSplit(e.clientX)
        }}
        onPointerMove={(e) => dragging && moveSplit(e.clientX)}
        onPointerUp={() => setDragging(false)}
        onPointerCancel={() => setDragging(false)}
      >
        <video ref={master} src={original} playsInline preload="auto" loop
          onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} />
        <video ref={enh} src={enhanced} muted playsInline preload="auto" loop
          className="layer" style={{ clipPath: clip, visibility: overlay === 'enhanced' ? 'visible' : 'hidden' }} />
        <video ref={att} src={attention} muted playsInline preload="auto" loop
          className="layer" style={{ clipPath: clip, visibility: overlay === 'attention' ? 'visible' : 'hidden' }} />

        {mode === 'wipe' && (
          <div className="divider" style={{ left: `${split * 100}%` }}>
            <div
              className="divider-handle"
              role="slider"
              tabIndex={0}
              aria-label="Comparison divider"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(split * 100)}
              onKeyDown={(e) => {
                if (e.key === 'ArrowLeft') { e.preventDefault(); setSplit((s) => Math.max(0, s - 0.05)) }
                if (e.key === 'ArrowRight') { e.preventDefault(); setSplit((s) => Math.min(1, s + 0.05)) }
              }}
            />
          </div>
        )}
        <div className="stage-tags" aria-hidden>
          {mode !== 'enhanced' && <span className="tag tag-left">Original</span>}
          {mode !== 'original' && <span className="tag tag-right">{showAttention ? 'What Keylight is following' : 'Keylight'}</span>}
        </div>
      </div>

      <div className="transport">
        <button className="play" onClick={toggle} aria-label={playing ? 'Pause' : 'Play'}>
          {playing ? (
            <svg viewBox="0 0 24 24" aria-hidden><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
          ) : (
            <svg viewBox="0 0 24 24" aria-hidden><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z" /></svg>
          )}
        </button>
        <Timeline duration={duration} time={time} segments={segments} cuts={cuts} onSeek={seek} />
        <span className="clock">{fmt(time)} / {fmt(duration)}</span>
      </div>
      <p className="now-following" aria-live="polite">
        {current?.target != null
          ? <>Following <strong>{current.cls ?? 'subject'}</strong>{current.source === 'semantic' ? ', chosen by Claude' : ''}{current.reason ? `: ${current.reason}` : ''}</>
          : 'No subject in view'}
      </p>
    </div>
  )
}

function Timeline({ duration, time, segments, cuts, onSeek }: {
  duration: number; time: number; segments: Segment[]; cuts: number[]; onSeek: (t: number) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [drag, setDrag] = useState(false)
  const at = (x: number) => {
    const r = ref.current!.getBoundingClientRect()
    return Math.max(0, Math.min(1, (x - r.left) / r.width)) * duration
  }
  // Alternate shading per distinct target so focus changes are visible on the bar.
  let shade = 0
  let prev: string | null = null
  return (
    <div
      ref={ref}
      className="timeline"
      role="slider"
      tabIndex={0}
      aria-label="Seek"
      aria-valuemin={0}
      aria-valuemax={Math.round(duration)}
      aria-valuenow={Math.round(time)}
      onPointerDown={(e) => { (e.target as HTMLElement).setPointerCapture?.(e.pointerId); setDrag(true); onSeek(at(e.clientX)) }}
      onPointerMove={(e) => drag && onSeek(at(e.clientX))}
      onPointerUp={() => setDrag(false)}
      onKeyDown={(e) => {
        if (e.key === 'ArrowLeft') onSeek(time - 2)
        if (e.key === 'ArrowRight') onSeek(time + 2)
      }}
    >
      {segments.map((s, i) => {
        const key = `${s.target}-${s.cls}`
        if (key !== prev) { shade = (shade + 1) % 2; prev = key }
        return (
          <div key={i}
            className={`seg ${s.target == null ? 'seg-none' : shade ? 'seg-a' : 'seg-b'} ${s.source === 'semantic' ? 'seg-ai' : ''}`}
            style={{ left: `${(s.t0 / duration) * 100}%`, width: `${((s.t1 - s.t0) / duration) * 100}%` }}
            title={s.target == null ? 'No subject' : `${s.cls}${s.source === 'semantic' ? ' (Claude)' : ''}`} />
        )
      })}
      {cuts.slice(1).map((c) => <div key={c} className="cut" style={{ left: `${(c / duration) * 100}%` }} />)}
      <div className="playhead" style={{ left: `${(time / Math.max(duration, 1e-6)) * 100}%` }} />
    </div>
  )
}

function fmt(t: number) {
  const m = Math.floor(t / 60)
  const s = Math.floor(t % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}
