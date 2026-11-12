import sys
from pathlib import Path

# The checker and table code live in src/ (shared with the experiments).
# In Docker they are copied to /app/src; when run from the repo they are in ../../src.
for p in (Path(__file__).resolve().parents[1] / "src", Path(__file__).resolve().parents[2] / "src"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from app.reader import read_chart  # noqa: E402
from checker import compare  # noqa: E402
from tables import from_csv  # noqa: E402

app = FastAPI(title="Chart Auditor API")

# Allow our frontend page (door 5174) to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.get("/health") means: “when someone asks the door /health, run the function below.”
@app.get("/health")
def health():
    return {"status": "ok"}


# @app.post("/check") is the same for /check, but with POST, which means “I am sending you something” (our two files).
@app.post("/check")
async def check(chart: UploadFile = File(...), table: UploadFile = File(...)):
    # 1. Read the table: labels in the first column, one column per series
    raw = await table.read()
    try:
        paper = from_csv(raw.decode("utf-8-sig"))
        if not paper.cells():
            raise ValueError("no numbers")
    except Exception:
        # 400 means “your request has a problem”. The message tells the user how to fix it.
        raise HTTPException(
            status_code=400,
            detail="Could not read the table. Use a CSV with labels in the first column and numbers in the other columns.",
        )

    # 2. Read the chart (fake for now)
    image_bytes = await chart.read()
    chart_table, spread = read_chart(chart.filename, image_bytes, paper)

    # 3. Compare and answer (same keys as before: verdict, note, differences, plus details)
    return compare(chart_table, paper, spread)
