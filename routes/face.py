import os
import uuid
import asyncio
import shutil
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from dotenv import load_dotenv

from utils.face_helper import (
    validate_file,
    detect_single_face,
    verify_faces,
    check_liveness,
    check_image_quality,
    extract_embedding,
)
from utils.local_storage import (
    store_file_locally,
    store_embedding,
    load_embedding,
    delete_file_locally,
    get_local_path,
    sanitize_filename,
    find_employee_face,
)
from utils.rate_limiter import limiter

load_dotenv()

router = APIRouter(prefix="/face", tags=["face"])

TEMP_FOLDER = os.getenv("TEMP_FOLDER", "./temp")


def save_temp_file(file: UploadFile) -> str:
    file_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename)[1].lower()
    temp_path = os.path.join(TEMP_FOLDER, f"{file_id}{ext}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_path


def cleanup_files(*paths: str):
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def resolve_stored_path(employee_id: str, stored_face_url: str = "") -> str:
    if not stored_face_url:
        return "", ""

    raw = stored_face_url.rstrip("/").split("/")[-1]
    try:
        safe_name = sanitize_filename(raw)
    except ValueError:
        return "", ""

    stored_path = get_local_path(safe_name)
    if not os.path.exists(stored_path):
        return "", ""

    return stored_path, safe_name


@router.post("/register")
@limiter.limit("10/minute")
async def register_face(
    request: Request,
    employeeId: str = Form(...),
    faceImage: UploadFile = File(...),
):
    temp_path = None
    try:
        is_valid, error_msg = validate_file(faceImage.filename, faceImage.size)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        temp_path = save_temp_file(faceImage)

        face_result = await asyncio.to_thread(detect_single_face, temp_path)
        if not face_result["success"]:
            raise HTTPException(status_code=400, detail=face_result["error"])

        quality = await asyncio.to_thread(
            check_image_quality, temp_path, face_result.get("facial_area")
        )
        if not quality["success"]:
            raise HTTPException(status_code=400, detail=quality["error"])

        embedding = await asyncio.to_thread(
            lambda: extract_embedding(temp_path)
        )
        if not embedding["success"]:
            raise HTTPException(
                status_code=500, detail=f"Failed to extract face embedding: {embedding['error']}"
            )

        store_result = store_file_locally(temp_path, employeeId)
        if not store_result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store face image: {store_result.get('error', 'unknown')}",
            )

        store_embedding(store_result["path"], embedding["embedding"])

        return {
            "success": True,
            "faceUrl": store_result["filename"],
            "message": "Face registered successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_files(temp_path)


@router.post("/verify")
@limiter.limit("30/minute")
async def verify_face(
    request: Request,
    employeeId: str = Form(...),
    faceImage: UploadFile = File(...),
    storedFaceUrl: str = Form(""),
    storedFaceImage: Optional[UploadFile] = File(None),
):
    temp_input_path = None
    temp_stored_path = None
    cleanup_stored = False
    try:
        is_valid, error_msg = validate_file(faceImage.filename, faceImage.size)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        temp_input_path = save_temp_file(faceImage)

        stored_embedding_path = None
        if storedFaceImage is not None and storedFaceImage.filename:
            is_valid, error_msg = validate_file(
                storedFaceImage.filename, storedFaceImage.size
            )
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Stored face: {error_msg}")
            temp_stored_path = save_temp_file(storedFaceImage)
            cleanup_stored = True
        elif storedFaceUrl:
            stored_path, safe_name = resolve_stored_path(employeeId, storedFaceUrl)
            if not stored_path:
                raise HTTPException(
                    status_code=404, detail="Stored face image not found"
                )
            temp_stored_path = stored_path

            emb = load_embedding(stored_path)
            if emb["success"]:
                stored_embedding_path = get_local_path(
                    os.path.splitext(safe_name)[0] + ".npy"
                )
        else:
            # No storedFaceImage or storedFaceUrl provided — look up by employeeId
            stored_face = find_employee_face(employeeId)
            if not stored_face["success"]:
                raise HTTPException(
                    status_code=404,
                    detail=stored_face["error"],
                )
            temp_stored_path = stored_face["path"]
            stored_embedding_path = stored_face.get("embedding_path")

        face_result = await asyncio.to_thread(detect_single_face, temp_input_path)
        if not face_result["success"]:
            raise HTTPException(status_code=400, detail=face_result["error"])

        liveness_result = await asyncio.to_thread(check_liveness, temp_input_path)
        if not liveness_result["success"]:
            return {
                "verified": False,
                "is_real": False,
                "distance": 1.0,
                "confidence": 0.0,
                "liveness_error": liveness_result["error"],
                "message": "Liveness check unavailable. Real face required.",
            }

        if not liveness_result["is_real"]:
            return {
                "verified": False,
                "is_real": False,
                "distance": 1.0,
                "confidence": 0.0,
                "message": "Liveness check failed. Real face required.",
            }

        verify_result = await asyncio.to_thread(
            verify_faces,
            temp_input_path,
            temp_stored_path,
            stored_embedding_path,
        )

        if not verify_result["success"]:
            raise HTTPException(status_code=500, detail=verify_result["error"])

        if not verify_result["verified"]:
            return {
                "verified": False,
                "is_real": True,
                "distance": verify_result["distance"],
                "confidence": verify_result["confidence"],
                "message": "Face does not match registered employee.",
            }

        return {
            "verified": True,
            "is_real": True,
            "distance": verify_result["distance"],
            "confidence": verify_result["confidence"],
            "message": "Face verified successfully",
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_files(temp_input_path)
        if cleanup_stored:
            cleanup_files(temp_stored_path)


@router.post("/clear-cache/{employee_id}")
@limiter.limit("20/minute")
async def clear_employee_cache(
    request: Request,
    employee_id: int,
):
    """
    Clear the cached face image and embedding for a specific employee.
    Called by Laravel whenever an employee's face photo is updated or deleted,
    ensuring the Python service always uses the latest image on next verification.
    """
    try:
        stored_face = find_employee_face(str(employee_id))
        if not stored_face["success"]:
            return {
                "success": True,
                "message": f"No cached face found for employee {employee_id} — nothing to clear.",
            }

        result = delete_file_locally(stored_face["filename"])
        return {
            "success": result["success"],
            "message": f"Cache cleared for employee {employee_id}"
            if result["success"]
            else f"Failed to clear cache: {result.get('error', '')}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete")
@limiter.limit("5/minute")
async def delete_face(
    request: Request,
    faceUrl: str = Form(...),
):
    try:
        safe_name = sanitize_filename(faceUrl)
        result = delete_file_locally(safe_name)
        return {
            "success": result["success"],
            "message": "Face deleted successfully"
            if result["success"]
            else f"Delete failed: {result.get('error', '')}",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/liveness")
@limiter.limit("30/minute")
async def check_face_liveness(
    request: Request,
    faceImage: UploadFile = File(...),
):
    temp_path = None
    try:
        is_valid, error_msg = validate_file(faceImage.filename, faceImage.size)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        temp_path = save_temp_file(faceImage)

        liveness_result = await asyncio.to_thread(check_liveness, temp_path)

        if not liveness_result["success"]:
            # Return a graceful response instead of a 400 error so the mobile
            # app can display a user-friendly message through its normal flow.
            return {
                "is_real": False,
                "confidence": 0.0,
                "message": liveness_result["error"],
            }

        if not liveness_result["is_real"]:
            return {
                "is_real": False,
                "confidence": liveness_result["confidence"],
                "message": "Liveness check failed. Real face required.",
            }

        return {
            "is_real": True,
            "confidence": liveness_result["confidence"],
            "message": "Human verified successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_files(temp_path)
