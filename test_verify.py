import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
BASE_URL = "http://127.0.0.1:8001"

headers = {}
if API_KEY:
    headers["Authorization"] = f"Bearer {API_KEY}"

stored_file = os.path.join(
    os.path.dirname(__file__),
    "storage/employee_faces/emp_1_cedea063.jpeg",
)
print(f"Stored file exists: {os.path.exists(stored_file)}")
print(f"Stored file size: {os.path.getsize(stored_file) if os.path.exists(stored_file) else 0}")

url = f"{BASE_URL}/face/verify"

# Test 1: Verify with itself (should match)
with open(stored_file, "rb") as f:
    files = {"faceImage": ("test_face.jpg", f, "image/jpeg")}
    data = {"employeeId": "1", "storedFaceUrl": "emp_1_cedea063.jpeg"}
    print("\n--- Test 1: Self-verification ---")
    try:
        resp = requests.post(url, files=files, data=data, headers=headers, timeout=60)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")
    except Exception as e:
        print(f"Error: {e}")

# Test 2: Verify with a full URL as storedFaceUrl
print("\n--- Test 2: Full URL as storedFaceUrl ---")
with open(stored_file, "rb") as f:
    files = {"faceImage": ("test_face.jpg", f, "image/jpeg")}
    data = {
        "employeeId": "1",
        "storedFaceUrl": "http://localhost:8000/storage/employee-faces/emp_1_cedea063.jpeg",
    }
    try:
        resp = requests.post(url, files=files, data=data, headers=headers, timeout=60)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")
    except Exception as e:
        print(f"Error: {e}")
