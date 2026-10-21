# Agent C: API + Frontend + Repo + README + Delivery

Read `work.md` first (rules, contract). You own: `app.py`, `static/index.html`, `smoke_test.py`, `requirements.txt`, `.gitignore`, `README.md`, `Dockerfile`, `comms/C_report.md`.

## Tasks

**1. app.py (FastAPI)**
- Import `infer`; call `infer.load()` once at startup.
- `GET /` serves `static/index.html`. `GET /health`.
- `POST /predict`: validate content type (jpeg/png/webp) -> 415; size > 5 MB -> 413; `ValueError` from infer -> 400 with `{"error": "..."}`. Success returns the JSON in the contract, `heatmap_b64` = base64 of `infer.gradcam()`.
- Build against stub mode. Do not wait for weights.

**2. static/index.html (single file, vanilla JS, no framework, no build step)**
- Drag-and-drop + click-to-upload, image preview.
- States: idle, loading (spinner + disabled button), result, error (shows server message).
- Result: label big, confidence %, "Not sure" notice if `uncertain`, original and heatmap side by side.
- Client-side check for type/size before upload (server still validates).

**3. Repo hygiene**
- `requirements.txt` with versions: torch, torchvision, fastapi, uvicorn, python-multipart, pillow, numpy, scikit-learn, matplotlib, onnx, onnxruntime, kagglehub.
- `.gitignore`: `data/`, `__pycache__/`, `.venv/`, `*.zip`, `.ipynb_checkpoints/`. Weights (~9 MB) are fine to commit.
- `Dockerfile` `[Y2]`: python slim, install requirements, expose 7860, `uvicorn app:app --host 0.0.0.0 --port 7860` (Hugging Face Docker Space).

**4. smoke_test.py**
- With server running: POST a valid sample image -> 200 and expected keys; POST a `.txt` file -> 415; POST random bytes labeled `image/png` -> 400. Print PASS/FAIL. Nothing more.

**5. README.md**
Sections: what it is, setup (venv, `pip install -r requirements.txt`), dataset download (`python data.py`), train (`python train.py`), run (`uvicorn app:app`), local inference one-liner, architecture, hyperparameters with reasons, results table (from A's `metrics.json`), calibration + quantization numbers (from B), the one technical challenge (from A's report, in your words, keep it true), demo/weights link. Fill numbers only from the other agents' reports and `outputs/`; never invent them.

**6. Deliver (after `M3`)**
- Run `smoke_test.py` with real weights; run the app and test one hotdog and one non-hotdog image by hand.
- Push to a public GitHub repo; confirm `data/` is absent from the repo.
- Deploy to a Hugging Face Docker Space (or put weights on GitHub release / HF Hub). Post links, then `M4 SUBMITTED`.

## Acceptance
- Fresh clone + README steps works on a clean venv.
- Wrong file type and huge file show a readable error in the UI, never a crash.

## Report must contain
Endpoint test results, deployment link, README completeness check against the `work.md` checklist, and `Explain-me` bullets: how an uploaded file's raw bytes become a tensor (multipart -> bytes -> PIL -> RGB -> resize -> normalize -> batch dim), why validation on both client and server, what the loading state protects against.
