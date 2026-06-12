# AskGloucester API image — FastAPI served by uvicorn on Azure Container Apps.
#
# Only the `api/` package is copied: it is fully self-contained. `api/query.py`
# imports stdlib + azure-* / openai / dotenv only, and the meeting_category
# retrieval filter uses the literal string 'full_committee' — it does NOT import
# ingestion/utils.classify_meeting_category. So `ingestion/` is intentionally
# left out of the image (verified: no `api/` -> `ingestion/` import).

FROM python:3.12-slim

# Don't write .pyc files; stream stdout/stderr straight to the container logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so the layer is cached across code-only changes.
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy the API package. `from . import query` resolves because `api/` is an
# importable package directory under /app.
COPY api/ ./api/

EXPOSE 8000

# uvicorn serves the FastAPI app defined in api/main.py as `app`.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
