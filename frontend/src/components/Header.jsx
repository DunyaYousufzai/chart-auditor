function Logo() {
  return (
    <svg width="30" height="30" viewBox="0 0 28 28" aria-hidden="true">
      <rect x="3" y="12" width="5" height="13" rx="1" fill="currentColor" />
      <rect x="11.5" y="6" width="5" height="19" rx="1" fill="currentColor" />
      <rect x="20" y="15" width="5" height="10" rx="1" fill="currentColor" />
      <line
        x1="18"
        y1="9"
        x2="27"
        y2="9"
        stroke="#c2410c"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  )
}

function Header() {
  return (
    <header className="sticky top-0 z-10 border-b border-rule bg-paper/90 backdrop-blur">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
        <a href="#top" className="flex items-center gap-3 text-ink">
          <Logo />
          <span className="font-display text-xl font-semibold">Chart Auditor</span>
        </a>

        <div className="flex items-center gap-6 text-sm">
          <nav className="hidden gap-6 text-ink/70 sm:flex">
            <a href="#tool" className="hover:text-ink">Try it</a>
            <a href="#how" className="hover:text-ink">How it works</a>
            <a
              href="https://github.com/dunyaYousufzai/chart-auditor"
              target="_blank"
              rel="noreferrer"
              className="hover:text-ink"
            >
              GitHub
            </a>
          </nav>
          <span className="rounded-full border border-ink/20 px-3 py-1 text-xs text-ink/70">
            Prototype
          </span>
        </div>
      </div>
    </header>
  )
}

export default Header