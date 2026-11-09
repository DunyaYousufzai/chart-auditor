# Build from the repository root, so the shared code in src/ can be copied:
#   docker build -f backend/Dockerfile -t chart-auditor-api .
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY backend/app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
