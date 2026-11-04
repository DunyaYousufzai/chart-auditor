import { useState } from "react"
import Header from "./components/Header"
import Footer from "./components/Footer"
import UploadBox from "./components/UploadBox"
import ResultCard from "./components/ResultCard"

const STEPS = [
  {
    title: "Read the chart",
    text: "A small vision model writes the chart as a table. It reads the chart several times and checks how much its readings differ.",
  },
  {
    title: "Compare with your table",
    text: "A short Python program matches the labels and compares every pair of numbers. Exact arithmetic, no guessing.",
  },
  {
    title: "Give a verdict",
    text: "Match, possible mismatch, or not sure. When the chart is hard to read, we allow a bigger gap before we raise a flag.",
  },
]

function App() {
  const [chartFile, setChartFile] = useState(null)
  const [chartUrl, setChartUrl] = useState(null)
  const [tableFile, setTableFile] = useState(null)
  const [tableRows, setTableRows] = useState([])
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  function handleChart(file) {
    if (!file) return
    setChartFile(file)
    setChartUrl(URL.createObjectURL(file))
    setResult(null)
  }

  async function handleTable(file) {
    if (!file) return
    setTableFile(file)
    const text = await file.text()
    const rows = text.trim().split("\n").map((line) => line.split(","))
    setTableRows(rows)
    setResult(null)
  }

  function handleCheck() {
    setLoading(true)
    // FAKE result for now. Later we call the real backend here.
    setTimeout(() => {
      setResult({
        verdict: "mismatch",
        note: "This is a fake answer to test the page.",
        differences: [{ label: "B", chart: 60, table: 72, error: 0.167 }],
      })
      setLoading(false)
    }, 1000)
  }

  const ready = chartFile && tableFile

  return (
    <div id="top" className="flex min-h-screen flex-col bg-paper font-sans text-ink">
      <Header />

      <main className="flex-1">
        <section className="mx-auto max-w-5xl px-6 pb-10 pt-14">
          <h1 className="max-w-2xl font-display text-4xl font-semibold leading-tight sm:text-5xl">
            Do your chart and your table tell the same story?
          </h1>
          <p className="mt-5 max-w-xl text-lg text-ink/70">
            Upload a bar or line chart and the table it comes from. We read the
            numbers off the picture and compare them with the table.
          </p>
        </section>

        <section id="tool" className="mx-auto max-w-5xl px-6">
          <div className="grid gap-4 md:grid-cols-2">
            <UploadBox
              title="1. Chart image"
              hint="PNG or JPG"
              accept="image/*"
              fileName={chartFile?.name}
              onFile={handleChart}
            />
            <UploadBox
              title="2. Table"
              hint="CSV file"
              accept=".csv"
              fileName={tableFile?.name}
              onFile={handleTable}
            />
          </div>

          {(chartUrl || tableRows.length > 0) && (
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <div className="rounded-lg border border-rule bg-white p-4">
                {chartUrl && (
                  <img src={chartUrl} alt="Your chart" className="mx-auto max-h-80" />
                )}
              </div>
              <div className="overflow-auto rounded-lg border border-rule bg-white p-4">
                <table className="w-full text-left text-sm tabular-nums">
                  <tbody>
                    {tableRows.map((row, i) => (
                      <tr
                        key={i}
                        className={i === 0 ? "border-b border-rule font-semibold" : ""}
                      >
                        {row.map((cell, j) => (
                          <td key={j} className="py-1 pr-4">
                            {cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="mt-6 flex items-center gap-4">
            <button
              onClick={handleCheck}
              disabled={!ready || loading}
              className="rounded-lg bg-ink px-6 py-2.5 font-semibold text-white hover:bg-cobalt focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cobalt disabled:cursor-not-allowed disabled:bg-ink/25"
            >
              {loading ? "Checking..." : "3. Check"}
            </button>
            {!ready && (
              <p className="text-sm text-ink/60">Choose both files to continue.</p>
            )}
          </div>

          {result && (
            <div className="mt-6">
              <ResultCard result={result} />
            </div>
          )}
        </section>

        <section id="how" className="mx-auto mt-20 max-w-5xl px-6">
          <h2 className="font-display text-2xl font-semibold">How it works</h2>
          <ol className="mt-6 grid gap-8 border-t border-rule pt-8 md:grid-cols-3">
            {STEPS.map((s, i) => (
              <li key={s.title}>
                <p className="font-display text-3xl text-cobalt">{i + 1}</p>
                <p className="mt-2 font-semibold">{s.title}</p>
                <p className="mt-1 text-sm leading-relaxed text-ink/70">{s.text}</p>
              </li>
            ))}
          </ol>
        </section>
      </main>

      <Footer />
    </div>
  )
}

export default App