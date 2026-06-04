import time
import logging
import numpy as np
import cv2
from PIL import Image, ExifTags
import io

logger = logging.getLogger(__name__)

_app = None
_facenet = None
_backend = None

ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def load_model():
    global _app, _facenet, _backend
    t0 = time.time()

    try:
        from insightface.app import FaceAnalysis
        _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=-1, det_size=(640, 640))
        _backend = "insightface"
        logger.info(f"InsightFace buffalo_l loaded in {time.time() - t0:.2f}s")
        return
    except Exception as e:
        logger.warning(f"InsightFace unavailable ({e}), trying FaceNet fallback")

    try:
        from facenet_pytorch import InceptionResnetV1, MTCNN
        _facenet = {
            "mtcnn": MTCNN(keep_all=True, device="cpu"),
            "resnet": InceptionResnetV1(pretrained="vggface2").eval(),
        }
        _backend = "facenet"
        logger.info(f"FaceNet+MTCNN loaded in {time.time() - t0:.2f}s")
        return
    except Exception as e:
        logger.error(f"FaceNet also unavailable ({e})")
        raise RuntimeError("No face embedding backend could be loaded.")


def _fix_exif_rotation(pil_img: Image.Image) -> Image.Image:
    try:
        exif = pil_img._getexif()
        if exif is None:
            return pil_img
        orientation_key = next(
            (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
        )
        if orientation_key is None or orientation_key not in exif:
            return pil_img
        rotations = {3: 180, 6: 270, 8: 90}
        orientation = exif[orientation_key]
        if orientation in rotations:
            pil_img = pil_img.rotate(rotations[orientation], expand=True)
    except Exception:
        pass
    return pil_img


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _align_face(img_bgr: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    try:
        src = landmarks.astype(np.float32)
        dst = ARCFACE_DST * (size / 112.0)
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
        if M is None:
            raise ValueError("estimateAffinePartial2D returned None")
        return cv2.warpAffine(img_bgr, M, (size, size))
    except Exception as e:
        logger.warning(f"Alignment failed ({e}), using resize fallback")
        return cv2.resize(img_bgr, (size, size))


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def _face_geometry(kps: np.ndarray, bbox: np.ndarray) -> dict:
    """Compute scale-invariant facial geometry ratios from 5 landmarks."""
    x1, y1, x2, y2 = bbox[:4]
    w = max(float(x2 - x1), 1.0)
    h = max(float(y2 - y1), 1.0)

    le, re, nt, lm, rm = kps.astype(float)

    eye_dist = float(np.linalg.norm(re - le)) / w
    face_ar = h / w
    nose_h = float(nt[1] - y1) / h
    mouth_w = float(np.linalg.norm(rm - lm)) / w
    eye_mid_y = (le[1] + re[1]) / 2.0
    nose_to_mouth = max((lm[1] + rm[1]) / 2.0 - nt[1], 1.0)
    vert_prop = float(nt[1] - eye_mid_y) / nose_to_mouth

    return {
        "eye_spacing": eye_dist,
        "face_shape": face_ar,
        "nose_position": nose_h,
        "mouth_width": mouth_w,
        "vertical_proportions": vert_prop,
    }


def _ratio_sim(r1: float, r2: float, scale: float) -> float:
    return max(0.0, min(100.0, (1.0 - abs(r1 - r2) / scale) * 100.0))


def _detect_insightface(img_bgr: np.ndarray):
    faces = _app.get(img_bgr)
    if not faces:
        return None, None, False

    def _score(f):
        b = f.bbox
        return (float(f.det_score), float((b[2] - b[0]) * (b[3] - b[1])))

    best = max(faces, key=_score)
    emb = _l2_normalize(best.embedding.astype(np.float64))
    geo = _face_geometry(best.kps, best.bbox)
    return emb, geo, True


def _detect_facenet(img_bgr: np.ndarray):
    import torch

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    mtcnn = _facenet["mtcnn"]
    resnet = _facenet["resnet"]

    boxes, probs, landmarks = mtcnn.detect(pil, landmarks=True)
    if boxes is None or len(boxes) == 0:
        return None, None, False

    best_idx = int(np.argmax(probs))
    box = boxes[best_idx]
    lm = landmarks[best_idx]

    geo = _face_geometry(lm, box)

    # Crop with padding for embedding
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * 0.2), int(bh * 0.2)
    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(w, x2 + px)
    cy2 = min(h, y2 + py)
    face_crop = img_bgr[cy1:cy2, cx1:cx2]

    lm_crop = lm - np.array([cx1, cy1], dtype=np.float32)
    aligned = _align_face(face_crop, lm_crop)
    aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

    face_pil = Image.fromarray(aligned_rgb).resize((160, 160))
    face_tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float()
    face_tensor = (face_tensor - 127.5) / 128.0
    face_tensor = face_tensor.unsqueeze(0)

    with torch.no_grad():
        emb = resnet(face_tensor).squeeze().numpy()

    emb = _l2_normalize(emb.astype(np.float64))
    return emb, geo, True


def get_face_data(image_bytes: bytes):
    """Returns (embedding, geometry, face_detected). Raises ValueError on bad input."""
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = _fix_exif_rotation(pil_img)
        img_bgr = _pil_to_bgr(pil_img)
    except Exception as e:
        raise ValueError(f"Cannot decode image: {e}")

    if _backend == "insightface":
        return _detect_insightface(img_bgr)
    elif _backend == "facenet":
        return _detect_facenet(img_bgr)
    else:
        raise RuntimeError("No backend loaded")


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> dict:
    raw_cosine = float(np.dot(emb1, emb2))
    score = (raw_cosine + 1) / 2 * 100

    if score >= 85:
        label = "Very likely the same person"
    elif score >= 65:
        label = "Strong resemblance"
    elif score >= 45:
        label = "Moderate resemblance"
    elif score >= 25:
        label = "Weak resemblance"
    else:
        label = "Little to no resemblance"

    return {"score": round(score, 1), "raw_cosine": round(raw_cosine, 4), "label": label}


def compute_breakdown(geo1: dict, geo2: dict) -> dict:
    """
    Compare normalized facial geometry ratios between two faces.
    Scores are 0-100: higher = more geometrically similar in that region.
    These are independent of the embedding similarity.
    """
    specs = [
        ("Eye Spacing",          "eye_spacing",         0.20),
        ("Face Shape",           "face_shape",          0.40),
        ("Nose Position",        "nose_position",       0.20),
        ("Mouth Width",          "mouth_width",         0.20),
        ("Vertical Proportions", "vertical_proportions",0.50),
    ]
    return {
        label: round(_ratio_sim(geo1[key], geo2[key], scale), 1)
        for label, key, scale in specs
    }
