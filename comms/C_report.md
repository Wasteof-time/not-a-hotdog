# Agent C Report

**Status:** IN PROGRESS
**Last updated:** 2026-10-21 23:30

## Milestones
- M4 SUBMITTED - **not posted yet**: it needs the human's deploy (GitHub repo + Hugging Face Space). The exact commands are in the C7 section below; as soon as the human pastes the links back I record them in the README and post the signal.

## Done
- C1 DONE. `requirements.txt` (13 pinned packages) + `.gitignore` written.
  Gate: `.venv\Scripts\python.exe -m pip install -r requirements.txt` -> exit code 0, every pin "already satisfied" in `.venv` (torch 2.13.0, torchvision 0.28.0, fastapi 0.135.2, uvicorn 0.42.0, python-multipart 0.0.29, pillow 12.2.0, numpy 2.4.3, scikit-learn 1.8.0, matplotlib 3.10.8, onnx 1.23.2, onnxruntime 1.31.0, onnxconverter-common 1.16.0, kagglehub 1.0.2).
- Note: pins match the versions that are actually installed in the shared `.venv` (Python 3.14.3, `--system-site-packages`) so a clean install is reproducible and needs no re-download of torch.
- `.gitignore`: `data/`, `__pycache__/`, `*.pyc`, `.venv/`, `venv/`, `*.zip`, `.ipynb_checkpoints/`, `.DS_Store`. Weights are intentionally NOT ignored (~9 MB, they must ship).
- C2 DONE. `app.py` + `static/index.html` built against B's stub `infer.py`.
  Gate (server on 127.0.0.1:8000): `curl /health` -> HTTP 200 `{"status":"ok"}`; `curl -F "file=@sample.jpg;type=image/jpeg" /predict` -> HTTP 200 with keys `confidence, heatmap_b64, label, prob_hotdog, uncertain`; `curl -F "file=@notes.txt;type=text/plain"` -> HTTP 415 `{"error":"Only JPEG, PNG or WebP images are allowed"}`; `curl -F "file=@fake.png;type=image/png"` (64 random bytes) -> HTTP 400 `{"error":"invalid image"}`; `curl -F "file=@big.png;type=image/png"` (6 MB) -> HTTP 413 `{"error":"File is larger than 5 MB"}`. `GET /` returns the HTML page.
- C3 DONE (real browser, headless Chrome 138 + a local same-origin test proxy that forwards `/predict` to the app).
  Gate: wrong type -> readable message "Only JPEG, PNG or WebP images are allowed." shown, result area stays hidden; after `Predict` is clicked the spinner has class `on`, the button is `disabled` and the hint reads "Classifying..."; when the answer arrives the result block is visible with label "Hotdog!", "Confidence: 50.2%" and a `data:image/png;base64,` heatmap; with the server answering HTTP 500 the UI shows the server message "Internal test failure" and hides the result. No crash in any case.
- C4 DONE. `python smoke_test.py` -> `PASS valid image returns 200 with all keys (status=200)` / `PASS text file returns 415 (status=415)` / `PASS random bytes as image/png return 400 (status=400)`, exit code 0.
- C5 IN PROGRESS: `Dockerfile` (python:3.11-slim, CPU torch, port 7860) and `README.md` skeleton with the Hugging Face front-matter written; `docker build` running (Docker Desktop had to be started first, the daemon was not running).
- C6 DONE (real weights, server restarted so `infer.load()` picked up `weights/seefood.pt` + `weights/calibration.json`).
  Gate: `python smoke_test.py` -> the same three PASS lines, exit code 0.
  By hand: `data/test/hot_dog/133012.jpg` -> `{"label": "Hotdog!", "confidence": 61.9, "prob_hotdog": 0.619, "uncertain": false}`; `data/test/not_hot_dog/13023.jpg` -> `{"label": "Not Hotdog!", "confidence": 84.7, "prob_hotdog": 0.153, "uncertain": false}`.
  Browser re-check with the real weights: result "Not Hotdog!" 64.5% + heatmap PNG, and an ambiguous validation image (`data/train/hot_dog/1195698.jpg`, prob 0.474) shows the notice "Not sure - the model is torn on this one. Try a clearer photo."
  README numbers are copied from `outputs/metrics.json` (accuracy 0.906, precision 0.928, recall 0.880, F1 0.903, best val loss 0.276) and `weights/calibration.json` (threshold 0.370, entropy_max 0.666).
- C6 COMPLETE (after B's `M3 CALIBRATION + ONNX READY`, 2026-10-21 23:20): README now carries the real benchmark table from `outputs/benchmark.json` (torch 9.037 MB / 20.70 ms / acc 0.906, ONNX FP32 8.873 MB / 3.41 ms / acc 0.906, ONNX INT8 2.409 MB / 77.87 ms / acc 0.906), the technical challenge from A's report (stage 2 fine-tuning made validation worse, so the best-val-loss checkpoint from stage 1 is the one that ships) and an honest "what the model gets wrong" section.
- README number check (command I ran): `python creadme.py` -> threshold 0.37 on `outputs/val_probs.npy` gives acc 0.9300 / F1 0.9293 / prec 0.9388 / rec 0.9200, threshold 0.50 gives acc 0.9100 / F1 0.9053 / prec 0.9556 / rec 0.8600. The false positives quoted in the README re-measured with `infer.predict`: `not_hot_dog/68684.jpg` p=0.830, `44810.jpg` p=0.802, `61822.jpg` p=0.740, `19143.jpg` p=0.682, `26489.jpg` p=0.665; false negatives `hot_dog/607289.jpg` p=0.074, `225367.jpg` p=0.477.
- C7 DONE. The work.md section 6 checklist is ticked in the C7 section below and the exact `git` / Hugging Face commands are written there. Gate: `git ls-files` -> 14 tracked files, **0** of them under `data/`; `git check-ignore -v data/` -> `.gitignore:1:data/ data/`; `git ls-files --others --exclude-standard` lists exactly the 24 files that would be committed and `data/` is not among them.
- C8 DONE. Explain-me written (8 bullets, including the three the spec asks for: bytes -> tensor, client+server validation, what the loading state protects against).
- Re-verification after the machine restarted (final files, real weights, real calibration): `python smoke_test.py` -> the same three PASS lines, exit 0; `data/test/hot_dog/133012.jpg` -> Hotdog! 61.9% (p=0.619); `data/test/not_hot_dog/13023.jpg` -> Not Hotdog! 84.7% (p=0.153); headless Chrome run -> spinner + disabled button while loading, result "Not Hotdog!" 64.5% with a PNG heatmap, and the "Not sure - the model is torn on this one." notice for the ambiguous image (prob 0.474).

## Artifacts produced
| File | Notes |
|---|---|
| requirements.txt | 13 pinned deps (adds `onnxconverter-common` for B's FP16 fallback, request B-R2) |
| .gitignore | dataset + venv + caches ignored, weights kept |
| app.py | FastAPI: `GET /`, `GET /health`, `POST /predict` |
| static/index.html | single-file vanilla JS UI |
| smoke_test.py | three checks against a running server |
| Dockerfile | python:3.11-slim, CPU torch, port 7860 |
| README.md | full README with the HF front-matter, results, calibration, the technical challenge |

## Requests (To: X | Need | Needed by)
| # | To | Need | Needed by |
|---|----|------|-----------|
| C-R1 | B | `weights/seefood.pt` + `outputs/val_probs.npy` / `val_labels.npy` (A is training) so B can post `M3 CALIBRATION + ONNX READY`; C6 needs real weights | C6 |
| C-R2 | B | Please also confirm the final `infer.predict` JSON keys stay `label, confidence, prob_hotdog, uncertain` and that `infer.gradcam` returns PNG bytes (app.py depends on both) | now |
| C-R3 | A | A6 (the technical challenge) is not in `comms/A_report.md` yet; the README needs it. Please post it when A6 is done. | C6 |
| C-R4 | Human | Please run the three command blocks in the C7 section below (GitHub push + Hugging Face Docker Space) and paste the two links back. I record them in the README and post `M4 SUBMITTED`. | M4 |

## Replies (answer requests addressed to me, cite the number, e.g. "A-R1")
- A-R1: `requirements.txt` now pins `torch==2.13.0`, `torchvision==0.28.0`, `kagglehub==1.0.2`, `scikit-learn==1.8.0`, `matplotlib==3.10.8`, `pillow==12.2.0`, `numpy==2.4.3` - exactly what the shared `.venv` has.
- B-R2: `onnxconverter-common==1.16.0` added to `requirements.txt`.
- B-R6: done - `requirements.txt` now pins `torch==2.13.0` / `torchvision==0.28.0`, i.e. exactly the environment that trains and runs here. `pip install -r requirements.txt` in `.venv` exits 0 with everything already satisfied.
- B-R7: the README now shows `infer.gradcam(...)` returning PNG bytes and links the overlay files. I rendered the same overlays through the API (`heatmap_b64` in the `/predict` answer) and put them next to the original in the UI.
- C-R1 (resolved): B posted `M3 CALIBRATION + ONNX READY` at 23:20 with the final `outputs/benchmark.json`; the README benchmark table is filled from it.
- C-R2 (resolved): B's B1/B2 gates confirm `predict()` returns `label, confidence, prob_hotdog, uncertain` as plain Python types and `gradcam()` returns PNG bytes - exactly what `app.py` consumes.
- C-R3 (resolved): A finished at 22:54 and wrote the technical challenge in `comms/A_report.md`; it is now in the README in my own words (stage 2 fine-tuning made validation worse, so the best-val-loss stage-1 snapshot ships).

## Blockers
- 

## Contract change requests (human approval needed)
- 

## Results / numbers
- `.venv`: Python 3.14.3, created with `--system-site-packages`; `pip install -r requirements.txt` exit code 0, 0 packages downloaded.
- `.gitignore` size: 7 lines, dataset (`data/`) ignored, weights tracked.
- Endpoint tests (real weights): `GET /health` 200; valid JPEG 200 (keys `label, confidence, prob_hotdog, uncertain, heatmap_b64`); `text/plain` 415; random bytes as `image/png` 400; 6 MB file 413.
- Test-set metrics from `outputs/metrics.json`: accuracy 0.906, precision 0.928, recall 0.880, F1 0.903, n_test 500, best val loss 0.276.
- Calibration from `weights/calibration.json`: threshold 0.370, entropy_max 0.666.
- Hand test: `data/test/hot_dog/133012.jpg` -> Hotdog! 61.9% (p=0.619); `data/test/not_hot_dog/13023.jpg` -> Not Hotdog! 84.7% (p=0.153).
- `git ls-files` -> 14 tracked files, 0 of them under `data/`. `git check-ignore -v data/` -> `.gitignore:1:data/ data/`.

## C7 - checklist (work.md section 6) and the commands for the human

Checklist:

- [x] Base: preprocessing (RGB -> Resize 224 -> ToTensor -> ImageNet Normalize), leak-free split (val carved out of the train folder, stratified, seed 42, index-overlap assert in `data.py`), trained classifier (`weights/seefood.pt`, test metrics in `outputs/metrics.json`)
- [x] Accuracy, Precision, Recall, F1 + confusion matrix on the test set (`outputs/metrics.json`, `outputs/confusion_matrix.png`, `outputs/curves.png`)
- [x] `predict()` returns Hotdog!/Not Hotdog! + confidence for a brand-new image (`infer.predict(bytes)`; checked on two test images through the API and on a validation image through the UI)
- [x] Augmentation + training vs validation curves (`data.py` TRAIN_TF, `outputs/curves.png`)
- [x] `[Y2]` Transfer learning with frozen then fine-tuned backbone (`train.py` stage 1 frozen, stage 2 last 4 blocks)
- [x] `[Y2]` FastAPI + frontend with loading state and invalid-file handling (verified in headless Chrome: spinner, disabled button, readable messages for wrong type and for server errors)
- [x] `[Y2]` Grad-CAM overlay, ONNX + quantization benchmark, threshold/entropy calibration (B's files: `weights/seefood.onnx`, `weights/seefood_int8.onnx`, `outputs/benchmark.json`, `weights/calibration.json`)
- [x] Public GitHub repo, no dataset committed, `requirements.txt`, `.gitignore` (the repo push itself is the human's step, commands below; `git ls-files` shows 0 files under `data/`)
- [ ] Demo link or weights link (the human creates the repo / Space and pastes the links back, then I post `M4 SUBMITTED`)
- [x] README: setup, dataset download, local inference, architecture + hyperparameters + why, one technical challenge (all sections present; the challenge text waits for A6, request C-R3)
- [x] Human can answer: why BCE-with-logits, why Adam, how a conv works, how backprop flows, why these false positives, how bytes become a tensor (all explained in the README and in the Explain-me below)

Exact commands for the human (run from the repo root; I did NOT run any of these):

```bash
# 1) GitHub - create the empty repo "not-a-hotdog" (public) on github.com first, then:
cd /path/to/not-a-hotdog
git add -A
git status --short                 # data/ must NOT appear in this list
git ls-files | grep "^data/" ; echo "exit=$?"   # must print nothing (exit 1)
git commit -m "SeeFood: hotdog classifier with FastAPI, web UI, Grad-CAM, ONNX and calibration"
git remote add origin https://github.com/<YOUR-USERNAME>/not-a-hotdog.git
git push -u origin main
```

```bash
# 2) Hugging Face Docker Space
pip install "huggingface_hub[cli]"
hf auth login                                        # paste the HF token
hf repo create seefood --repo-type space --sdk docker --space-sdk docker
git remote add space https://huggingface.co/spaces/<YOUR-USERNAME>/seefood
git push space main
# README.md already starts with the required front-matter (title / sdk: docker / app_port: 7860)
```

```bash
# 3) optional - weights on the Hugging Face Hub as a backup link
hf repo create seefood-weights --repo-type model
hf upload seefood-weights weights/ --repo-type model
```

Note for the human: `.agent/` (the internal plan) and `comms/` (the agent reports) are currently tracked in git. If you want a clean public repo, remove them with
`git rm -r --cached .agent comms && git commit -m "Remove internal agent notes"`. Everything else in the repo is part of the deliverable.

## Explain-me
- **How an uploaded file's raw bytes become a tensor:** the browser sends `multipart/form-data`; FastAPI's `UploadFile` gives me the part as a stream and `await file.read()` returns the raw `bytes` of the file - nothing decoded yet. `PIL.Image.open(io.BytesIO(data))` wraps those bytes in a file-like object, PIL sniffs the format and decodes the pixels, `.convert("RGB")` drops any alpha channel and puts every image on three channels, `.load()` forces the decode so a truncated file fails here and not later. `transforms.Resize((224, 224))` makes the image square (aspect ratio is not preserved, exactly like training), `ToTensor()` turns the PIL image into a CHW float tensor scaled to `[0, 1]`, `Normalize(MEAN, STD)` subtracts the ImageNet mean and divides by the ImageNet std, and `.unsqueeze(0)` adds the batch dimension so the network gets `[1, 3, 224, 224]`. The model returns `[1, 1]` logits, `sigmoid` turns that into `P(hotdog)`.
- **Why validate on both the client and the server:** the client check is a courtesy - it saves the user a 10 MB upload and gives an instant message instead of a round trip. But the browser is not trusted: anyone can `curl` the endpoint, so the server repeats every rule itself (content type -> 415, size -> 413, decodability -> 400) and returns a JSON `{"error": "..."}` that the UI can show verbatim. If only the client checked, a `curl -F "file=@huge.bin"` would be decoded by PIL and could exhaust the server's memory. The server is the only place where the rules actually have to hold.
- **What the loading state protects against:** it tells the user that the request is alive (the model has to load a 9 MB checkpoint, resize and run a forward pass and a backward pass for Grad-CAM, which is not instant) and, more importantly, it stops a double submit - the button is disabled and the drag-and-drop area is highlighted, so the same photo cannot be uploaded twice and the user cannot change the file while a prediction is in flight. Without it a user who is impatient would click again, waste CPU on a second forward pass and race two responses into the same result area.
- **Why `infer.load()` is called once at startup:** loading the checkpoint and building the model costs about a second, so doing it per request would make every prediction slow. The module-level call plus the module-level cache in `infer` means the first request pays the cost and every later one reuses the same model. It also means a missing `weights/seefood.pt` shows up at startup (the app still starts on the untrained head in stub mode) instead of as a 500 on the first upload.
- **Why the error codes are what they are:** 415 means "the media type is not something I accept" (the browser sent `text/plain` for a `.txt`), 413 means "the body is too large" (over 5 MB), 400 means "the content type was allowed but the bytes are not a decodable image". They are ordered type -> size -> content so the most specific problem is reported first, and each one comes back as `{"error": "..."}` so the frontend can show the server's own words instead of a generic failure.
- **Why the requirements are pinned to the versions that actually run:** the shared `.venv` is Python 3.14 with `torch 2.13.0+cpu` and `torchvision 0.28.0+cpu`, and the network here could not reliably fetch the 2.14.1 wheels. Pinning the versions that trained and were tested means `pip install -r requirements.txt` reproduces the exact setup that produced the reported numbers - a lock file in everything but name. The one extra package I added (`onnxconverter-common`) was requested by Agent B for its FP16 quantization fallback.
- **Why `data/` is in `.gitignore` but the weights are not:** the dataset is 48 MB of third-party Kaggle images that can be re-downloaded with `python data.py`, and shipping it would put someone else's data in a public repo. The weights are 9 MB, they *are* the deliverable (without them the API only answers with an untrained head), and they cannot be regenerated without retraining, so they stay in the repo. `git ls-files` confirms: 0 tracked files under `data/`.
- **Why every number in the README is traceable:** a README with plausible-looking numbers is worthless in a demo. Each figure comes from a file in the repo (`outputs/metrics.json`, `outputs/benchmark.json`, `weights/calibration.json`) or from a command I ran myself (the validation threshold sweep, the false-positive probabilities). Where a result is not flattering - INT8 quantization being 23x slower than FP32 on this CPU, stage 2 fine-tuning making validation worse - the README says so instead of hiding it.
