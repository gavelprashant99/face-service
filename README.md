# Face Recognition Service

FastAPI-based face verification service using DeepFace (ArcFace), anti-spoofing liveness detection, and embedding-based matching.

## API Endpoints

| Method | Path                | Rate Limit  | Description                        |
|--------|---------------------|-------------|------------------------------------|
| GET    | `/health`           | —           | Health check                       |
| POST   | `/face/register`    | 10/min      | Register a face for an employee    |
| POST   | `/face/verify`      | 30/min      | Verify a face against stored face  |
| POST   | `/face/liveness`    | 30/min      | Check liveness (is it a real face?) |
| POST   | `/face/delete`      | 5/min       | Delete a stored face               |

All endpoints except `/health` require `Authorization: Bearer <API_KEY>` header.

## Quick Start (Local Development)

```bash
# 1. Setup
cp .env.example .env
# Edit .env — set API_KEY (or leave blank to disable auth in dev)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
# Server starts at http://localhost:8001
# Swagger docs at http://localhost:8001/docs
```

## Deploy with Docker

```bash
# 1. Setup
cp .env.example .env
# Edit .env — set API_KEY, APP_ENV=production, ALLOWED_ORIGINS

# 2. Build & start
docker compose up -d --build

# 3. Check logs
docker compose logs -f

# 4. Check health
curl http://<vps-ip>:8001/health
```

## Deploy manually on a VPS

```bash
# Install system deps
sudo apt update
sudo apt install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0

# Clone & setup
git clone <repo> face-service
cd face-service
cp .env.example .env
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run with systemd (recommended — see below)
```

### systemd service unit

Create `/etc/systemd/system/face-service.service`:

```ini
[Unit]
Description=Face Recognition Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/face-service
EnvironmentFile=/opt/face-service/.env
ExecStart=/opt/face-service/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001 --log-level info
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable face-service
sudo systemctl start face-service
```

## Environment Variables

| Variable             | Default              | Description                               |
|----------------------|----------------------|-------------------------------------------|
| `APP_ENV`            | `development`        | `development` or `production`             |
| `LOG_LEVEL`          | `INFO` (prod)        | Log level                                 |
| `API_KEY`            | `""`                 | Bearer token (empty = auth disabled)      |
| `HOST`               | `0.0.0.0`            | Bind address                              |
| `PORT`               | `8001`               | Port                                      |
| `ALLOWED_ORIGINS`    | `""`                 | Comma-separated CORS origins (prod only)  |
| `TEMP_FOLDER`        | `./temp`             | Directory for temp uploaded files         |
| `LOCAL_STORAGE_DIR`  | `./storage/...`      | Directory for registered face images      |

## Architecture

```
Mobile App ──HTTPS──> nginx (SSL) ──> face-service:8001
                              │
                        ┌─────┴─────┐
                        │  storage/  │
                        │ employee_faces │
                        └───────────┘
```

- Face images are stored locally as `emp_{id}_{uuid}.jpg`
- Face embeddings are cached as `emp_{id}_{uuid}.npy` for fast comparison
- Anti-spoofing uses DeepFace's built-in liveness detection
- Falls back to OpenCV Haar cascade if DeepFace backends fail
