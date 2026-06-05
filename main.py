import os
import sys
import asyncio
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from utils.rate_limiter import limiter

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
APP_ENV = os.getenv("APP_ENV", "development")

# ── Structured logging ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO" if APP_ENV == "production" else "DEBUG")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger("face-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from utils.face_helper import preload_models
    await asyncio.to_thread(preload_models)
    yield


app = FastAPI(title="Face Recognition Service", version="2.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if APP_ENV == "production":
    origins_str = os.getenv("ALLOWED_ORIGINS", "")
    origins = [o.strip() for o in origins_str.split(",") if o.strip()]
    allow_creds = bool(origins)
else:
    origins = ["*"]
    allow_creds = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_creds,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)
    if API_KEY:
        public_paths = {"/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}
        if request.url.path in public_paths:
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[len("Bearer "):] != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return await call_next(request)


os.makedirs(os.getenv("TEMP_FOLDER", "./temp"), exist_ok=True)

from routes import face, health

app.include_router(health.router)
app.include_router(face.router)

if __name__ == "__main__":
    import uvicorn
    is_dev = APP_ENV == "development"
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8001)),
        reload=is_dev,
        log_level=LOG_LEVEL.lower(),
    )
