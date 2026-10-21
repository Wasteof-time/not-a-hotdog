# Jobs

Each job has a **Gate**: the check you must run and record before marking it DONE. Detailed specs are in `agents/agent_<letter>.md`; code recipes and pitfalls are in `skills.md`.

## Who runs what

| Agent | Role | Suggested model | Why |
|---|---|---|---|
| A | Data, model, training (critical path, silent bugs) | Grok 4.6 (medium) | Everything downstream waits on A's weights; label and leakage mistakes are quiet |
| B | Inference, Grad-CAM, calibration, ONNX | Free OpenCode model #1 at max/high effort (pick the stronger coder of the two) | Most intricate code, but it has exact recipes in `skills.md` and strict gates |
| C | API, frontend, repo, README, deploy | Free OpenCode model #2 at max/high effort | Most standard work, least ambiguity |

Swap freely: any agent works with any model because everything is in these files. Start each with: `You are Agent <letter>. Follow AGENTS.md.`

Human tasks (not for agents): configure Kaggle API key, create `.venv`, put 5 to 6 ambiguous images in `edge_cases/`, create the GitHub repo and run `git push` / Hugging Face upload when C hands you the commands.

---

## Agent A

| Job | What | Gate |
|---|---|---|
| A1 | `model.py`: `build_model`, `set_trainable` | `python -c "import torch; from model import build_model as b; print(b(False)(torch.randn(2,3,224,224)).shape)"` prints `torch.Size([2, 1])`; after `set_trainable(m, False)` only classifier params have `requires_grad` |
| A2 | `data.py`: `download()`, `get_loaders()` | Class counts printed for train/val/test; `assert` no index overlap between train and val; one train batch has shape `[32,3,224,224]` |
| | Post **`M1 READY`** | |
| A3 | `train.py` stage 1 (frozen backbone) | Val accuracy after stage 1 is above 0.70; if not, check label flip (skills A) before anything else |
| A4 | Stage 2 fine-tune, save best-val-loss weights | `weights/seefood.pt` loads into `build_model(False)` with no key errors |
| A5 | Test metrics, `confusion_matrix.png`, `curves.png`, `val_probs.npy`, `val_labels.npy` | Files exist; confusion matrix entries sum to the test set size; metrics.json has accuracy, precision, recall, f1 |
| | Post **`M2 WEIGHTS READY`** | |
| A6 | Look at 5 to 10 test errors, note causes; write the technical challenge (before/after numbers) | Written in report |
| A7 | Answer requests from B and C; write Explain-me; set status DONE | Report complete |

## Agent B

| Job | What | Gate |
|---|---|---|
| B1 | `infer.py`: `load`, `preprocess`, `predict` (stub mode if no weights) | A generated JPEG's bytes return a dict with `label, confidence, prob_hotdog, uncertain` as plain Python types; random bytes raise `ValueError("invalid image")` |
| B2 | `gradcam(bytes) -> PNG bytes` | Output opens with PIL; heatmap array is not constant; hooks removed afterwards (second call works) |
| | Post progress: "predict + gradcam ready (stub)" so C can drop mocks | |
| B3 | `export_onnx.py`: export, check match, quantize, benchmark (build against stub weights) | ONNX vs torch outputs `allclose` (atol 1e-4); quantized model loads in ONNX Runtime and runs |
| | Poll `comms/A_report.md` for `M2 WEIGHTS READY`; keep doing B3 meanwhile | |
| B4 | `calibrate.py` on real validation probs | `weights/calibration.json` exists; `infer.predict` now reads it (print threshold) |
| B5 | Edge cases from `edge_cases/` | Table in report: file, label, confidence, uncertain |
| B6 | Re-run `export_onnx.py` with real weights; benchmark; quantized test accuracy | `outputs/benchmark.json` with `size_mb` and `ms_per_img` for torch/ONNX/quantized; quantized accuracy within a few points of FP32, or reported honestly |
| | Post **`M3 CALIBRATION + ONNX READY`** | |
| B7 | Explain-me; set status DONE | Report complete |

B may import `data.get_loaders()` (A's file) for the test set in B6.

## Agent C

| Job | What | Gate |
|---|---|---|
| C1 | `requirements.txt` (pinned), `.gitignore` | `pip install -r requirements.txt` succeeds in `.venv` |
| C2 | `app.py` | With server up: `curl /health` ok; image upload returns 200 with all keys; `.txt` upload returns 415; random bytes as `image/png` return 400; >5 MB returns 413 |
| C3 | `static/index.html` | In a browser: spinner shows while loading; wrong type and server errors show a readable message; result shows label, confidence, heatmap |
| C4 | `smoke_test.py` | Prints PASS for the three checks |
| C5 | `Dockerfile`, README skeleton (with Hugging Face front-matter) | `docker build` works if Docker is available, otherwise say it was not tested |
| | Wait for `M3 CALIBRATION + ONNX READY` (poll `comms/B_report.md`); keep C3/C5 polishing meanwhile | |
| C6 | Real weights end to end; fill README numbers from reports and `outputs/` | Smoke test PASS with real weights; one hotdog and one non-hotdog image tested by hand; every README number traceable to a file |
| C7 | Checklist from `work.md` section 6; write exact `git` / Hugging Face commands for the human | Checklist ticked in report; `git ls-files` shows no `data/` |
| | After the human deploys, record links, post **`M4 SUBMITTED`** | |
| C8 | Explain-me; set status DONE | Report complete |
