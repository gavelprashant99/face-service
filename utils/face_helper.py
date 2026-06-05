import os
import logging
import cv2
import numpy as np
from deepface import DeepFace

logger = logging.getLogger("face-service.face_helper")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_PROCESSING_DIMENSION = 960
BLUR_THRESHOLD = 100.0
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 215
ARC_COSINE_THRESHOLD = 0.40


def preload_models():
    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    dummy_path = os.path.join(os.getenv("TEMP_FOLDER", "./temp"), "_model_warmup.jpg")
    os.makedirs(os.path.dirname(dummy_path), exist_ok=True)
    cv2.imwrite(dummy_path, dummy)
    try:
        DeepFace.represent(
            img_path=dummy_path,
            model_name="ArcFace",
            enforce_detection=False,
        )
        logger.info("ArcFace model loaded successfully")
    except Exception as e:
        logger.warning("ArcFace model preload warning", exc_info=True)

    # Preload anti-spoofing model for liveness detection
    try:
        DeepFace.extract_faces(
            img_path=dummy_path,
            detector_backend="ssd",
            enforce_detection=False,
            anti_spoofing=True,
        )
        logger.info("Anti-spoofing model loaded successfully")
    except Exception as e:
        logger.warning("Anti-spoofing model preload warning (will use fallback)", exc_info=True)
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)


def validate_file(filename: str, file_size: int) -> tuple[bool, str]:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    if file_size is not None and file_size > MAX_FILE_SIZE:
        return False, f"File too large. Max size: {MAX_FILE_SIZE // (1024 * 1024)}MB"
    return True, ""


def check_image_quality(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {"success": False, "error": "Failed to read image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < BLUR_THRESHOLD:
        return {
            "success": False,
            "error": f"Image too blurry (sharpness: {lap_var:.1f}, min: {BLUR_THRESHOLD})",
        }

    mean_brightness = float(np.mean(gray))
    if mean_brightness < MIN_BRIGHTNESS:
        return {
            "success": False,
            "error": f"Image too dark (brightness: {mean_brightness:.1f}, min: {MIN_BRIGHTNESS})",
        }
    if mean_brightness > MAX_BRIGHTNESS:
        return {
            "success": False,
            "error": f"Image too bright (brightness: {mean_brightness:.1f}, max: {MAX_BRIGHTNESS})",
        }

    return {
        "success": True,
        "blur_score": round(float(lap_var), 1),
        "brightness": round(mean_brightness, 1),
    }


def resize_for_processing(image_path: str, inplace: bool = False) -> None:
    if not inplace:
        return
    img = cv2.imread(image_path)
    if img is None:
        return
    h, w = img.shape[:2]
    largest = max(w, h)
    if largest <= MAX_PROCESSING_DIMENSION:
        return
    scale = MAX_PROCESSING_DIMENSION / largest
    resized = cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )
    cv2.imwrite(image_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])


def detect_single_face(image_path: str) -> dict:
    try:
        resize_for_processing(image_path, inplace=True)

        img = cv2.imread(image_path)
        if img is None:
            return {"success": False, "error": "Failed to read image"}

        opencv_result = detect_single_face_with_opencv(image_path)
        if opencv_result["success"]:
            return opencv_result

        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="ssd",
            enforce_detection=True,
        )

        if len(faces) > 1:
            return {
                "success": False,
                "error": "Multiple faces detected. Please ensure only one face is visible.",
            }

        return {"success": True, "face_count": 1}
    except Exception as e:
        return {"success": False, "error": str(e)}


def extract_embedding(image_path: str) -> dict:
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            detector_backend="ssd",
            enforce_detection=True,
        )
        if not result:
            return {"success": False, "error": "No face found in image"}
        embedding = result[0]["embedding"]
        return {"success": True, "embedding": embedding}
    except Exception as e:
        return {"success": False, "error": str(e)}


def cosine_distance(a: list, b: list) -> float:
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - np.dot(arr_a, arr_b) / (norm_a * norm_b))


def compare_embeddings(emb1: list, emb2: list) -> dict:
    distance = cosine_distance(emb1, emb2)
    verified = distance <= ARC_COSINE_THRESHOLD
    confidence = max(0.0, 1.0 - distance)
    return {
        "verified": verified,
        "distance": round(distance, 4),
        "confidence": round(confidence, 4),
    }


def check_liveness(image_path: str) -> dict:
    try:
        resize_for_processing(image_path, inplace=True)

        # Try anti-spoofing with multiple backends
        anti_spoofing_errors = []
        faces = []

        for backend in ("ssd", "opencv"):
            try:
                faces = DeepFace.extract_faces(
                    img_path=image_path,
                    detector_backend=backend,
                    enforce_detection=True,
                    anti_spoofing=True,
                )
                if faces:
                    break
            except Exception as e:
                anti_spoofing_errors.append(f"{backend}: {str(e)}")

        if faces:
            # Anti-spoofing model worked — use its result
            face = get_largest_face(faces)
            is_real = face.get("is_real", False)
            real_score = face.get("antispoof_score", face.get("real_score", 0.0))

            return {
                "success": True,
                "is_real": bool(is_real),
                "confidence": round(float(real_score), 4),
            }

        # Anti-spoofing model unavailable — fall back to face detection only
        logger.warning("Anti-spoofing unavailable, falling back to face-only check. Errors: %s", anti_spoofing_errors)

        # Verify a face can still be detected (ensures it's not a blank image)
        try:
            detect_result = detect_single_face(image_path)
            if not detect_result["success"]:
                return {"success": False, "error": detect_result["error"]}
        except Exception as e:
            return {"success": False, "error": f"Face detection failed: {e}"}

        return {
            "success": True,
            "is_real": True,
            "confidence": 0.5,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def verify_faces(image_path_1: str, image_path_2: str, stored_embedding_path: str = None) -> dict:
    if stored_embedding_path and os.path.exists(stored_embedding_path):
        emb_result = extract_embedding(image_path_1)
        if not emb_result["success"]:
            return {"success": False, "error": emb_result["error"]}

        try:
            stored_emb = np.load(stored_embedding_path).tolist()
        except Exception as e:
            return {"success": False, "error": f"Failed to load stored embedding: {e}"}

        result = compare_embeddings(emb_result["embedding"], stored_emb)
        result["success"] = True
        result["is_real"] = True
        result["detector"] = "embedding"
        return result

    resize_for_processing(image_path_1, inplace=True)

    last_error = None
    for backend in ("opencv", "ssd", "retinaface"):
        try:
            result = DeepFace.verify(
                img1_path=image_path_1,
                img2_path=image_path_2,
                model_name="ArcFace",
                detector_backend=backend,
                anti_spoofing=False,
                enforce_detection=True,
            )

            verified = result["verified"]
            distance = result.get("distance", 0)
            is_real = result.get("is_real", result.get("anti_spoofing", {}).get("is_real", True))
            confidence = max(0.0, 1.0 - float(distance))

            return {
                "success": True,
                "verified": verified,
                "is_real": bool(is_real),
                "distance": round(float(distance), 4),
                "confidence": round(confidence, 4),
                "detector": backend,
            }
        except Exception as e:
            last_error = e
            continue

    detail = f"{type(last_error).__name__}: {last_error}" if last_error else "Unknown error"
    return {"success": False, "error": detail}


def get_largest_face(faces: list) -> dict:
    def area(face: dict) -> int:
        fa = face.get("facial_area") or {}
        return int(fa.get("w", 0)) * int(fa.get("h", 0))
    return max(faces, key=area)


def detect_single_face_with_opencv(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {"success": False, "error": "Failed to read image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(
        cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
    )
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        return {"success": False, "error": "OpenCV face detector is unavailable."}

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    if len(faces) == 0:
        return {"success": False, "error": "No face detected in the image."}

    return {"success": True, "face_count": len(faces)}
