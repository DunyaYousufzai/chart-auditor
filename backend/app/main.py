import io

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.checker import compare
from app.reader import read_chart

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
    # 1. Read the table
    raw = await table.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
        # df.iloc[:, 0] is the first column (labels) and df.iloc[:, 1] is the second column (numbers).
        labels = df.iloc[:, 0].astype(str)
        values = pd.to_numeric(df.iloc[:, 1])
        table_values = {l: float(v) for l, v in zip(labels, values)}
    except Exception:
        # HTTPException(status_code=400, ...) sends a clear error if the CSV is bad. 400 means “your request has a problem”. 
        # The message tells the user how to fix it.
        raise HTTPException(
            status_code=400,
            detail="Could not read the table. Use a CSV with labels in the first column and numbers in the second.",
        )

    # 2. Read the chart (fake for now)
    image_bytes = await chart.read()
    chart_values = read_chart(chart.filename, image_bytes, table_values)

    # 3. Compare and answer
    return compare(chart_values, table_values)