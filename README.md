---
title: Face Similarity API
emoji: 🔍
colorFrom: blue
colorTo: cyan
sdk: docker
pinned: false
app_port: 7860
---

# Face Similarity

Upload two face photos and get a 0–100 similarity score computed from ArcFace embeddings, plus a geometric breakdown of per-region landmark similarity.

## How it works

1. **Detection** — InsightFace (RetinaFace) finds faces in each image. Highest-confidence face is used.
2. **Alignment** — 5 facial landmarks are used to apply a similarity transform to a canonical 112×112 pose.
3. **Embedding** — `buffalo_l` (ArcFace R100) produces a 512-d L2-normalized vector per face.
4. **Similarity** — Cosine similarity mapped to 0–100: `score = (cosine + 1) / 2 × 100`.
5. **Breakdown** — Scale-invariant geometry ratios (eye spacing, face shape, nose position, mouth width, vertical proportions) are computed from raw landmarks and compared independently of the embedding.

Fallback: if InsightFace is unavailable, the pipeline uses FaceNet (InceptionResnetV1, VGGFace2) + MTCNN.

---

## Run locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# First run downloads buffalo_l (~500 MB) — takes a few minutes
uvicorn main:app --reload --port 8000
```

Keep this terminal running while you use the site. Interactive API docs: `http://localhost:8000/docs`.

**Frontend** — open a second terminal:

```bash
cd frontend
python -m http.server 5500
# open http://localhost:5500
```

---

## Deploy publicly

### Backend → Hugging Face Spaces (free)

1. Push this repo to a HF Space (Docker SDK). The `Dockerfile` at the root handles everything.
2. The `buffalo_l` model is pre-downloaded during the Docker build (~500 MB) so startup is instant.
3. Your API will be at `https://YOUR_HF_USERNAME-face-similarity-api.hf.space`.

### Frontend → Vercel

1. Import your GitHub repo on [vercel.com](https://vercel.com). `vercel.json` points it at `frontend/`.
2. Update `API_BASE` in `frontend/main.js` to your HF Space URL before deploying.

---

## API

`POST /compare` — `multipart/form-data` with `image1` and `image2` (JPEG/PNG/WebP, max 10 MB).

```json
{
  "score": 78.4,
  "raw_cosine": 0.5680,
  "label": "Strong resemblance",
  "faces_detected": [true, true],
  "breakdown": {
    "Eye Spacing":          83.2,
    "Face Shape":           72.1,
    "Nose Position":        90.3,
    "Mouth Width":          68.4,
    "Vertical Proportions": 78.9
  }
}
```

`GET /health` — returns backend status and active model backend.

---

## Known limitations

- **Single face per image** — multiple faces present → highest-confidence one is used.
- **CPU only by default** — no GPU required. Cold start: 5–15 s. Per-request: 0.5–3 s.
- **Front-facing photos work best** — extreme profiles degrade detection/alignment.
- **Geometric breakdown is heuristic** — ratios are compared with fixed tolerance scales.
- **No persistence** — images processed in memory, never written to disk or logged.
- **HF free Spaces sleep** after 15 min of inactivity; first request after sleep takes ~30 s to wake up.
