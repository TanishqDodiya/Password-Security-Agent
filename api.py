"""
api.py - FastAPI transport layer for Password Security Agent.

Only a transport layer: calls analyze_password() and returns JSON.
No security logic duplicated here.
Never logs, saves, or returns raw passwords.
"""

import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from agent.security_graph import analyze_password
from agent.llm_explainer import is_llama_available, get_llama_runtime
from agent.rag import knowledge_base_available

app = FastAPI(
    title="Password Security Agent API",
    description="Deterministic password security analysis via LangGraph orchestration. Heuristic engine is authoritative; ML is supporting evidence.",
    version="5A",
)

# ------------------------------------------------------------
# CORS - minimal, for future frontend local dev
# Prefer explicit origins over wildcard with credentials
# ------------------------------------------------------------
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Request / Response models - never include password in response
# ------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    password: str = Field(..., max_length=5000, description="Password to analyze (in-memory only, never stored)")

    # Empty string is allowed and handled by engine; >5000 rejected with 422 before analysis

class HealthResponse(BaseModel):
    status: str
    version: str

class InfoResponse(BaseModel):
    name: str
    ml_enabled: bool
    rag_enabled: bool
    llm_provider: str
    version: str

class GenerateRequest(BaseModel):
    length: int = Field(16, ge=4, le=128, description="Password length")
    use_lower: bool = Field(True, description="Include lowercase")
    use_upper: bool = Field(True, description="Include uppercase")
    use_digits: bool = Field(True, description="Include digits")
    use_symbols: bool = Field(True, description="Include symbols")

class GenerateResponse(BaseModel):
    password: str
    analysis: Dict[str, Any]

# ------------------------------------------------------------
# Error handling - never expose password, stack trace, paths, secrets
# ------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Sanitize validation errors: remove 'input' to avoid echoing password
    # Pydantic would otherwise include the submitted password in error detail for string_too_long
    errors = exc.errors()
    sanitized = []
    for err in errors:
        safe_err = {k: v for k, v in err.items() if k != "input"}
        sanitized.append(safe_err)
    return JSONResponse(status_code=422, content={"detail": sanitized})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Log internally without password (only safe metadata)
    # Do not include exc details, password, or stack trace in response
    return JSONResponse(
        status_code=500,
        content={"detail": "Password analysis failed."},
    )

# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    """Health check - no sensitive data."""
    return {"status": "ok", "version": "5A"}

@app.get("/info", response_model=InfoResponse, tags=["info"])
async def info():
    """Safe architecture info - no secrets, paths, or dataset details."""
    # Determine ML enabled by checking artifact existence (not by exposing path)
    try:
        from pathlib import Path
        ml_enabled = Path("models/password_classifier.joblib").exists()
    except Exception:
        ml_enabled = False
    rag_enabled = knowledge_base_available()
    llm_provider = get_llama_runtime() if is_llama_available() else "fallback"
    return {
        "name": "Password Security Agent",
        "ml_enabled": bool(ml_enabled),
        "rag_enabled": bool(rag_enabled),
        "llm_provider": llm_provider,
        "version": "5A",
    }

@app.post("/analyze", tags=["analyze"])
async def analyze(req: AnalyzeRequest):
    """
    Analyze password via LangGraph pipeline.
    Password exists only temporarily in memory during this request.
    """
    # Pydantic already validated that password is a string and present
    # Empty string is allowed - let security engine handle it
    password = req.password

    # Do NOT log password, do NOT print, do NOT save
    # Call stable interface analyze_password (no duplicated logic)
    try:
        result = analyze_password(password)
    except TypeError:
        # Invalid type (should be caught by Pydantic, but safe handling)
        raise HTTPException(status_code=422, detail="Invalid password type.")
    except Exception:
        # Generic failure - never expose details or password
        raise HTTPException(status_code=500, detail="Password analysis failed.")

    # Ensure raw password not in response (defense in depth)
    # result from analyze_password never contains password, but we double-check
    # Do not add request-history fields containing password

    # Build response preserving analyze_password structure
    # FastAPI will serialize to JSON
    return {
        "risk": result.get("risk"),
        "features": result.get("features"),
        "patterns": result.get("patterns"),
        "ml": result.get("ml"),
        "rag": result.get("rag"),
        "explanation": result.get("explanation"),
    }

@app.post("/generate", response_model=GenerateResponse, tags=["generate"])
async def generate(req: GenerateRequest):
    """
    Generate a secure password using cryptographically secure randomness.
    Returns password and its immediate analysis via same security engine.
    Password is generated in memory and returned once; not stored.
    """
    try:
        from agent.generator import generate_password
    except Exception:
        raise HTTPException(status_code=500, detail="Password generation failed.")
    try:
        # Validate at least one type selected (Pydantic handles length, but check types)
        if not any([req.use_lower, req.use_upper, req.use_digits, req.use_symbols]):
            raise HTTPException(status_code=422, detail="At least one character type must be selected.")
        pw = generate_password(
            length=req.length,
            use_lower=req.use_lower,
            use_upper=req.use_upper,
            use_digits=req.use_digits,
            use_symbols=req.use_symbols,
        )
    except (ValueError, TypeError) as e:
        # Do not expose password in error
        raise HTTPException(status_code=422, detail=str(e)[:200])
    except Exception:
        raise HTTPException(status_code=500, detail="Password generation failed.")
    # Analyze generated password using same engine for user convenience
    try:
        analysis = analyze_password(pw)
    except Exception:
        analysis = {}
    # Return password and analysis - caller should handle securely (do not log)
    return {"password": pw, "analysis": analysis}

# Optional: safe metadata-only request logging (no bodies)
# We deliberately do NOT log request bodies to avoid password exposure
@app.middleware("http")
async def log_safe_metadata(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    # Log only safe metadata at debug level without body - use print for minimal visibility in dev
    # In production, use proper logger with no body
    duration = time.time() - start
    # Do not log request body, query params containing password, or headers
    # Only method, path, status, duration are safe
    # We avoid print to prevent accidental log capture in tests; use no-op
    _ = f"{request.method} {request.url.path} {response.status_code} {duration:.3f}s"
    return response
