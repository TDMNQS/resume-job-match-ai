"""Vercel-compatible API for the Resume Job-Match analyzer."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from analyzer import AnalyzerError, analyze


app = FastAPI(title="Resume Job-Match AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AnalyzeRequest(BaseModel):
    resume: str = Field(min_length=1, max_length=8000)
    job: str = Field(min_length=1, max_length=8000)


@app.get("/", response_class=FileResponse)
def homepage() -> FileResponse:
    """Serve the existing Vercel-compatible frontend at the deployment root."""
    return FileResponse(Path(__file__).resolve().parents[1] / "index.html")


@app.get("/api/health")
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "geminiConfigured": bool(
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        ),
    }


@app.post("/api/analyze")
@app.post("/analyze")
def analyze_match(payload: AnalyzeRequest) -> dict:
    try:
        result = analyze(payload.resume, payload.job)
        return result.model_dump()
    except AnalyzerError as exc:
        message = str(exc)
        status = 503 if "API_KEY" in message else 400
        raise HTTPException(status_code=status, detail=message) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The analysis service encountered an unexpected error.",
        ) from exc
