function UploadBox({ title, hint, accept, fileName, onFile }) {
  const done = Boolean(fileName)

  return (
    <label
      className={`block cursor-pointer rounded-lg border p-6 transition-colors focus-within:ring-2 focus-within:ring-cobalt ${
        done
          ? "border-cobalt bg-white"
          : "border-dashed border-ink/30 bg-white/60 hover:border-cobalt"
      }`}
    >
      <p className="font-display text-lg font-semibold">{title}</p>
      <p className="mt-1 text-sm text-ink/60">{hint}</p>
      <p className={`mt-4 text-sm ${done ? "font-medium text-cobalt" : "text-ink/50"}`}>
        {done ? `✓ ${fileName}` : "Click to choose a file"}
      </p>
      <input
        type="file"
        accept={accept}
        className="sr-only"
        onChange={(e) => onFile(e.target.files[0])}
      />
    </label>
  )
}

export default UploadBox