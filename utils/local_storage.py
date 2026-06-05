import os
import re
import shutil
import uuid
import numpy as np

STORAGE_DIR = os.getenv(
    "LOCAL_STORAGE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage/employee_faces"),
)


def ensure_storage_dir():
    os.makedirs(STORAGE_DIR, exist_ok=True)


def _embedding_path(image_path: str) -> str:
    base, _ = os.path.splitext(image_path)
    return base + ".npy"


def sanitize_filename(filename: str) -> str:
    safe = os.path.basename(filename)
    if not re.match(r"^emp_\d+_[a-f0-9]+\.(jpg|jpeg|png)$", safe, re.IGNORECASE):
        raise ValueError(f"Invalid filename: {safe}")
    return safe


def store_file_locally(source_path: str, employee_id: str) -> dict:
    ensure_storage_dir()
    ext = os.path.splitext(source_path)[1].lower()
    if not ext:
        ext = ".jpg"
    filename = f"emp_{employee_id}_{uuid.uuid4().hex[:8]}{ext}"
    dest_path = os.path.join(STORAGE_DIR, filename)
    shutil.copy2(source_path, dest_path)
    return {"success": True, "filename": filename, "path": dest_path}


def store_embedding(image_path: str, embedding: list) -> dict:
    emb_path = _embedding_path(image_path)
    np.save(emb_path, np.array(embedding, dtype=np.float32))
    return {"success": True, "path": emb_path}


def load_embedding(image_path: str) -> dict:
    emb_path = _embedding_path(image_path)
    if not os.path.exists(emb_path):
        return {"success": False, "error": "Embedding not found"}
    try:
        emb = np.load(emb_path)
        return {"success": True, "embedding": emb.tolist()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_file_locally(filename: str) -> dict:
    filename = os.path.basename(filename)
    file_path = os.path.join(STORAGE_DIR, filename)
    deleted = False
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            deleted = True
        except OSError as e:
            return {"success": False, "error": str(e)}

    emb_path = _embedding_path(file_path)
    if os.path.exists(emb_path):
        try:
            os.remove(emb_path)
        except OSError:
            pass

    if not deleted:
        return {"success": False, "error": "File not found"}
    return {"success": True, "message": "File and embedding deleted successfully"}


def get_local_path(filename: str) -> str:
    return os.path.join(STORAGE_DIR, filename)


def find_employee_face(employee_id: str) -> dict:
    """
    Find the stored face image for an employee by employee ID.
    Looks for files matching emp_{employee_id}_*.jpg|jpeg|png.

    Returns:
        success: True/False
        path: full path to the image file (if found)
        filename: filename only (if found)
        embedding_path: path to the .npy embedding file (if exists)
        error: error message (if not found)
    """
    ensure_storage_dir()
    for fname in os.listdir(STORAGE_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parts = fname.split("_")
        if len(parts) >= 3 and parts[1] == str(employee_id):
            full_path = os.path.join(STORAGE_DIR, fname)
            emb_path = _embedding_path(full_path)
            return {
                "success": True,
                "path": full_path,
                "filename": fname,
                "embedding_path": emb_path if os.path.exists(emb_path) else None,
            }
    return {"success": False, "error": f"Stored face not found for employee {employee_id}"}
