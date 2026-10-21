# SeeFood: Master Work Plan

Goal: hotdog / not-hotdog classifier with REST API + web UI, Grad-CAM, ONNX quantization, threshold calibration. Submission = public GitHub repo + weights/demo + README.

**Assumption:** Second-year scope (superset of first-year). If you are first-year, skip everything tagged `[Y2]` (Agent B's ONNX work, API-only parts, Dockerfile). Everything else still applies.

**Stack (fixed, do not debate):** PyTorch, torchvision MobileNetV2 (ImageNet pretrained), FastAPI, single-file vanilla HTML frontend, ONNX Runtime.
**Dataset:** Kaggle `dansbecker/hot-dog-not-hot-dog` (train/ and test/ folders, each with `hot_dog/` and `not_hot_dog/`).

---

## 1. Rules

All rules are in `rules.md`. Per-agent jobs and gates are in `jobs.md`, code recipes in `skills.md`, entry point in `AGENTS.md`.

## 2. Files and owners

```
seefood/
  AGENTS.md, rules.md,
  jobs.md, skills.md   HUMAN (done, frozen)
  agents/agent_*.md    HUMAN (done, frozen)
  config.py            HUMAN (done, frozen)
  data.py              A
  model.py             A
  train.py             A
  infer.py             B
  calibrate.py         B
  export_onnx.py       B  [Y2]
  app.py               C
  static/index.html    C
  smoke_test.py        C
  requirements.txt     C
  .gitignore           C
  README.md            C   (fills in numbers from A and B reports)
  Dockerfile           C   [Y2, optional if only weights link is submitted]
  comms/A_report.md    A
  comms/B_report.md    B
  comms/C_report.md    C
  data/                gitignored, created by data.py
  weights/             seefood.pt (A), calibration.json (B), seefood.onnx / seefood_int8.onnx (B)
  outputs/             metrics.json, confusion_matrix.png, curves.png, val_probs.npy, val_labels.npy (A); benchmark.json (B)
  edge_cases/          5-6 ambiguous images (human supplies; B uses)
```

## 3. Contract (frozen)

**config.py** holds: `SIZE=224`, `MEAN`, `STD` (ImageNet), `SEED=42`, `CLASSES=["hot_dog","not_hot_dog"]`, paths `DATA_DIR`, `WEIGHTS`, `CALIB`, `OUT_DIR`.

**Labels:** torchvision `ImageFolder` assigns `hot_dog=0, not_hot_dog=1` (alphabetical). The model has **one logit**; `sigmoid(logit) = P(hot_dog)`. Therefore training target = `1 - folder_label`. Saved `val_labels.npy` uses **hotdog = 1**. (This is a classic bug source: everyone must follow it.)

**model.py (A)**
- `build_model(pretrained=True) -> nn.Module`: MobileNetV2, classifier replaced with `Dropout(0.2) -> Linear(1280, 1)`. Input `(N,3,224,224)`, output `(N,1)` logits.
- `set_trainable(model, backbone: bool, last_n_blocks: int = 0)`: freeze or unfreeze for transfer learning.
- Grad-CAM target layer: `model.features[-1]`.

**Preprocessing (identical in training eval and inference):** RGB -> `Resize((224,224))` -> ToTensor -> Normalize(MEAN, STD).

**weights/seefood.pt:** `state_dict` of `build_model(pretrained=False)`.

**weights/calibration.json (B):** `{"threshold": float, "entropy_max": float}`. If missing, infer.py uses `0.5` and no entropy check.

**infer.py (B)**
- `load()`: loads weights once (module-level cache). If `seefood.pt` does not exist yet, load the untrained head so C can build against it (STUB MODE).
- `predict(image_bytes: bytes) -> dict` with keys: `label` ("Hotdog!" or "Not Hotdog!"), `confidence` (float 0-100, confidence in the shown label), `prob_hotdog` (0-1), `uncertain` (bool, true when entropy > entropy_max).
- `gradcam(image_bytes: bytes) -> bytes`: PNG of heatmap overlaid on the image.
- Invalid or unreadable image: raise `ValueError("invalid image")`.

**API (C)**
- `GET /health` -> `{"status":"ok"}`
- `POST /predict` (multipart field `file`) -> `{"label","confidence","prob_hotdog","uncertain","heatmap_b64"}`
- Allowed types: jpeg, png, webp. Max 5 MB. Wrong type -> 415. Too large -> 413. Undecodable -> 400.

## 4. Phases and sync points

| Phase | A | B | C | Sync gate |
|---|---|---|---|---|
| 0 (human, 10 min) | | | | Create repo, drop these files in, Kaggle API key configured, create venv |
| 1 parallel | `model.py` + `data.py` first (30 min), then `train.py` | `infer.py` in stub mode, Grad-CAM, `calibrate.py` draft | `app.py`, `index.html`, requirements, .gitignore | **M1**: A posts "model.py + data.py ready" |
| 2 parallel | Train stage 1 + fine-tune, metrics, plots | Build `export_onnx.py` against stub weights | Frontend states (loading, error), Dockerfile, README skeleton | **M2**: A posts "WEIGHTS READY" |
| 3 integrate | Fix train issues, write challenge notes | Run `calibrate.py` on real val probs, run edge cases, ONNX export + benchmark | Run `smoke_test.py` with real weights, fill README numbers | **M3**: B posts "CALIBRATION + ONNX READY" |
| 4 ship (human + C) | | | Deploy to HF Space (Docker) or upload weights, final checklist | **M4**: C posts "SUBMITTED" |

Rough effort: A 3 to 4 h, B 2 to 3 h, C 2 to 3 h, wall clock about 4 to 5 h because of overlap.

## 5. Communication protocol

Each agent edits **only its own report** in `comms/` using the template in that file.

- **Read first:** before starting any task and whenever blocked, read the other two reports.
- **Write on:** every milestone, every blocker, every request.
- **Status values:** `NOT STARTED`, `IN PROGRESS`, `BLOCKED`, `DONE`.
- **Requests:** put in your `Requests` table as `To: X | Need | Needed by`. The recipient answers in its own report under `Replies`, citing the request number.
- **Signals other agents wait on** (exact phrases, in the Milestones section):
  - A: `M1 READY`, `M2 WEIGHTS READY`
  - B: `M3 CALIBRATION + ONNX READY`
  - C: `M4 SUBMITTED`
- If waiting would block you, work on a stub or the next independent task. Do not idle.

## 6. Final definition of done

- [ ] Base: preprocessing, leak-free split (val carved from train, test untouched until the end), trained classifier
- [ ] Accuracy, Precision, Recall, F1 + confusion matrix on test set (`outputs/`)
- [ ] `predict()` returns Hotdog!/Not Hotdog! + confidence for a brand-new image
- [ ] Augmentation + training vs validation curves
- [ ] `[Y2]` Transfer learning with frozen then fine-tuned backbone
- [ ] `[Y2]` FastAPI + frontend with loading state and invalid-file handling
- [ ] `[Y2]` Grad-CAM overlay, ONNX + quantization benchmark, threshold/entropy calibration
- [ ] Public GitHub repo, no dataset committed, `requirements.txt`, `.gitignore`
- [ ] Demo link or weights link
- [ ] README: setup, dataset download, local inference, architecture + hyperparameters + why, one technical challenge
- [ ] Human can answer: why BCE-with-logits, why Adam, how a conv works, how backprop flows, why these false positives, how bytes become a tensor
