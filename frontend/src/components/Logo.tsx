// Keylight mark: a gooseneck key light whose panel is lit in Tuscan Sun.
// Drawn as a vector (no product photography), so it scales from favicon to poster.
export function LampMark({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="40 40 260 430" aria-hidden focusable="false">
      <defs>
        <radialGradient id="km-panel" cx="45%" cy="40%" r="70%">
          <stop offset="0" stopColor="#FFF6D6" />
          <stop offset=".55" stopColor="#F8DB86" />
          <stop offset="1" stopColor="#F5CB5C" />
        </radialGradient>
      </defs>
      <ellipse cx="88" cy="452" rx="62" ry="11" fill="#E8EDDF" />
      <rect x="84" y="208" width="8" height="244" rx="3" fill="#E8EDDF" />
      <rect x="80" y="232" width="16" height="22" rx="4" fill="#E8EDDF" />
      <path d="M88 210 C88 168 108 142 150 128" stroke="#E8EDDF" strokeWidth="8" fill="none" strokeLinecap="round" />
      <g transform="rotate(-28 225 112)">
        <rect x="160" y="52" width="130" height="120" rx="38" fill="url(#km-panel)" stroke="#E8EDDF" strokeWidth="9" />
      </g>
    </svg>
  )
}

export default function Wordmark() {
  return (
    <>
      <LampMark className="wordmark-mark" />
      <span className="wordmark-text">keylight</span>
    </>
  )
}
