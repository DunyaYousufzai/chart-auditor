const MEANINGS = [
  { dot: "bg-green-400", name: "Match", text: "The chart and the table agree." },
  { dot: "bg-red-400", name: "Possible mismatch", text: "Please check this by hand." },
  { dot: "bg-amber-400", name: "Not sure", text: "The chart was too hard to read." },
]

function Footer() {
  return (
    <footer className="mt-20 bg-ink text-white/75">
      <div className="mx-auto grid max-w-5xl gap-10 px-6 py-12 md:grid-cols-3">
        <div>
          <p className="font-display text-xl font-semibold text-white">Chart Auditor</p>
          <p className="mt-3 text-sm leading-relaxed">
            A small open-model tool that reads the numbers in a chart and checks
            them against the table in the same paper.
          </p>
        </div>

        <div>
          <p className="font-semibold text-white">What the answers mean</p>
          <ul className="mt-3 space-y-3 text-sm">
            {MEANINGS.map((m) => (
              <li key={m.name} className="flex gap-3">
                <span className={`mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full ${m.dot}`} />
                <span>
                  <span className="text-white">{m.name}.</span> {m.text}
                </span>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <p className="font-semibold text-white">Project</p>
          <ul className="mt-3 space-y-2 text-sm">
            <li>University of Neuchatel</li>
            <li>Generative AI course, Autumn 2026</li>
            <li>
              <a
                href="https://github.com/dunyaYousufzai/chart-auditor"
                target="_blank"
                rel="noreferrer"
                className="underline underline-offset-4 hover:text-white"
              >
                Source code on GitHub
              </a>
            </li>
          </ul>
        </div>
      </div>

      <div className="border-t border-white/15">
        <p className="mx-auto max-w-5xl px-6 py-5 text-sm text-white/60">
          This tool never says a paper is wrong. It asks a person to look.
        </p>
      </div>
    </footer>
  )
}

export default Footer