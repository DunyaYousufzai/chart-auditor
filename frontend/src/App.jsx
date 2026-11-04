function App() {
  return (
    <div className="min-h-screen bg-slate-100 flex items-center justify-center">
      <div className="bg-white rounded-xl shadow p-8 max-w-md text-center">
        <h1 className="text-2xl font-bold text-slate-800">
          Chart Auditor
        </h1>
        <p className="mt-2 text-slate-600">
          Upload a chart and its table. We check if they match.
        </p>
      </div>
    </div>
  )
}

export default App