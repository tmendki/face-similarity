import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading face embedding model…")
    model.load_model()
    logger.info("Model ready.")
    yield


app = FastAPI(title="Face Similarity API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error(code: str, message: str, status: int = 400):
    return JSONResponse(status_code=status, content={"error": code, "message": message})


async def _read_upload(upload: UploadFile, label: str) -> bytes:
    if upload.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail={"error": "invalid_file_type", "message": f"{label}: only JPEG, PNG, and WebP are accepted."},
        )
    data = await upload.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail={"error": "file_too_large", "message": f"{label} exceeds the 10MB limit."},
        )
    return data


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "request_error", "message": str(exc.detail)},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "An unexpected error occurred."},
    )


@app.post("/compare")
async def compare(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    data1 = await _read_upload(image1, "Image 1")
    data2 = await _read_upload(image2, "Image 2")

    embeddings = []
    geometries = []
    faces_detected = [False, False]

    for idx, (data, label) in enumerate([(data1, "image 1"), (data2, "image 2")]):
        try:
            emb, geo, detected = model.get_face_data(data)
        except ValueError as e:
            return _error("processing_error", str(e))

        if not detected:
            return _error("no_face_detected", f"No face could be detected in {label}.")

        faces_detected[idx] = True
        embeddings.append(emb)
        geometries.append(geo)

    similarity = model.compute_similarity(embeddings[0], embeddings[1])
    breakdown = model.compute_breakdown(geometries[0], geometries[1])

    return {**similarity, "faces_detected": faces_detected, "breakdown": breakdown}


@app.get("/health")
async def health():
    return {"status": "ok", "backend": model._backend}
