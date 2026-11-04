const STYLES = {
  match: {
    label: "Match",
    badge: "bg-green-100 text-green-800",
    stripe: "border-l-green-600",
  },
  mismatch: {
    label: "Possible mismatch",
    badge: "bg-red-100 text-red-800",
    stripe: "border-l-red-600",
  },
  not_sure: {
    label: "Not sure",
    badge: "bg-amber-100 text-amber-800",
    stripe: "border-l-amber-500",
  },
}

function ResultCard({ result }) {
  const style = STYLES[result.verdict]

  return (
    <div className={`rounded-lg border border-l-4 border-rule bg-white p-6 ${style.stripe}`}>
      <span className={`inline-block rounded-full px-4 py-1 text-sm font-semibold ${style.badge}`}>
        {style.label}
      </span>

      <p className="mt-3 text-ink/70">{result.note}</p>

      {result.differences.length > 0 && (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-sm tabular-nums">
            <thead>
              <tr className="border-b border-rule text-ink/60">
                <th className="py-2 font-medium">Label</th>
                <th className="font-medium">Chart says</th>
                <th className="font-medium">Table says</th>
                <th className="font-medium">Difference</th>
              </tr>
            </thead>
            <tbody>
              {result.differences.map((d) => (
                <tr key={d.label} className="border-b border-rule last:border-0">
                  <td className="py-2 font-medium">{d.label}</td>
                  <td>{d.chart}</td>
                  <td>{d.table}</td>
                  <td className="font-medium text-red-700">
                    {(d.error * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export default ResultCard