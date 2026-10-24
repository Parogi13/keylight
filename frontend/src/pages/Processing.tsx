import { useEffect, useRef, useState } from 'react'
import { api, type Job } from '../api'

const STEPS: [string, string][] = [
  ['prepare', 'Preparing video'],
  ['perceive', 'Detecting subjects'],
  ['understand', 'Understanding the scene'],
  ['attention', 'Tracking attention'],
  ['render', 'Enhancing frames'],
  ['finalize', 'Finalizing'],
]

type Props = { jobId: string; onComplete: () => void; onBack: () => void }

export default function Processing({ jobId, onComplete, onBack }: Props) {
  const [job, setJob] = useState<Job | null>(null)
  const [error, setError] = useState<string | null>(null)
  const done = useRef(onComplete)
  done.current = onComplete

  useEffect(() => {
    let alive = true
    let timer: number
    const tick = async () => {
      try {
        const j = await api.status(jobId)
        if (!alive) return
        setJob(j)
        if (j.status === 'complete') return done.current()
        if (j.status === 'error') return
      } catch (e) {
        if (!alive) return
        const msg = (e as Error).message
        setError(msg === 'job not found'
          ? 'This job is no longer running. Keylight may have restarted while it was processing. Start it again from the home screen.'
          : msg)
        return
      }
      timer = window.setTimeout(tick, 700)
    }
    tick()
    return () => { alive = false; clearTimeout(timer) }
  }, [jobId])

  const current = job ? STEPS.findIndex(([k]) => k === job.stage) : -1
  const pct = Math.round((job?.progress ?? 0) * 100)

  if (error || job?.status === 'error') {
    return (
      <div className="processing">
        <h1 className="proc-title">This video couldn’t be enhanced</h1>
        <p className="error" role="alert">{job?.error ?? error}</p>
        <button className="btn btn-quiet" onClick={onBack}>Choose another video</button>
      </div>
    )
  }

  return (
    <div className="processing" aria-live="polite">
      <p className="proc-file">{job?.name ?? ' '}</p>
      <h1 className="proc-title">{job?.status === 'queued' ? 'Waiting to start' : job?.label ?? 'Starting'}</h1>
      <div className="meter" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
        <div className="meter-fill" style={{ width: `${pct}%` }} />
      </div>
      <p className="proc-pct">{pct}%</p>
      <ol className="steps">
        {STEPS.map(([key, label], i) => (
          <li key={key} className={i < current ? 'done' : i === current ? 'now' : ''}>{label}</li>
        ))}
      </ol>
      <p className="proc-note">Everything runs on this computer. You can leave this page open; it updates on its own.</p>
    </div>
  )
}
