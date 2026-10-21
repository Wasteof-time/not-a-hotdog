import base64
import os

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import infer

MAX_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="SeeFood", description="Hotdog / Not Hotdog classifier")

infer.load()  # load weights (or the stub head) once at startup


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        return JSONResponse({"error": "Only JPEG, PNG or WebP images are allowed"}, status_code=415)
    data = await file.read()
    if len(data) > MAX_BYTES:
        return JSONResponse({"error": "File is larger than 5 MB"}, status_code=413)
    try:
        res = infer.predict(data)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    try:
        heat = infer.gradcam(data)
    except Exception:
        heat = b""  # keep the prediction even if the heatmap fails
    res["heatmap_b64"] = base64.b64encode(heat).decode()
    return res
