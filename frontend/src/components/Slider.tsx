import { useId } from 'react'

type Props = {
  label: string
  hint?: string
  value: number
  min?: number
  max?: number
  step?: number
  onChange: (v: number) => void
  format?: (v: number) => string
}

export default function Slider({ label, hint, value, min = 0, max = 1, step = 0.05, onChange, format }: Props) {
  const id = useId()
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div className="slider">
      <div className="slider-head">
        <label htmlFor={id}>{label}</label>
        <output htmlFor={id}>{format ? format(value) : `${Math.round(pct)}%`}</output>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ '--fill': `${pct}%` } as React.CSSProperties}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      {hint && <p className="slider-hint">{hint}</p>}
    </div>
  )
}
