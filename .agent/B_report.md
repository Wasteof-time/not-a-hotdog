# Agent B Report

**Status:** DONE (B1, B2, B3, B4, B6, B7 done; B5 blocked on the human's `edge_cases/` images - see Blockers)
**Last updated:** 2026-10-21 23:55

## Milestones
- predict + gradcam ready (stub) - 2026-10-21 22:45 (C can drop the mock now: `infer.predict(bytes)` and `infer.gradcam(bytes)` both work)
- M3 CALIBRATION + ONNX READY - 2026-10-21 23:20

## Done
- B1 DONE: `infer.py` -> `load()` (stub mode while `weights/seefood.pt` is missing), `preprocess()`, `predict()` (plain Python types), `ValueError("invalid image")` on garbage bytes.
- B2 DONE: `gradcam(bytes) -> PNG bytes`, hand written (~25 lines), forward hook + tensor gradient hook on `model.features[-1]`, jet colormap blended 50/50 with the original, hooks removed after every call.
- B3 DONE (against stub weights, then again with the real weights in B6): `export_onnx.py` exports `weights/seefood.onnx`, checks ONNX vs torch with `np.allclose(atol=1e-4)`, quantizes to `weights/seefood_int8.onnx`, benchmarks on CPU and writes `outputs/benchmark.json`.
- B4 DONE: `calibrate.py` on the real validation probabilities -> `weights/calibration.json` (threshold 0.37, entropy_max 0.6661). `infer.predict` reads it (it re-reads the file when it changes, so the API always uses the calibrated threshold).
- B5 PARTIAL / BLOCKED: `edge_cases/` does not exist yet (human task, see B-R3). I ran the 7 hardest test images instead (4 ambiguous hotdogs + 3 worst false positives) as a stand-in and the table is below; the real edge-case table will follow as soon as the human drops the images in.
- B6 DONE: re-ran `export_onnx.py` with the real weights, benchmarked torch / ONNX FP32 / ONNX INT8 and measured the test accuracy of all three -> `outputs/benchmark.json`.
- Grad-CAM sanity check DONE: on a clear hotdog photo the hot region sits on the sausage inside the bun, not on the background (checked visually on 3 test images).

## Artifacts produced
| File | Notes |
|---|---|
| `infer.py` | `load()` caches the model, `predict()` returns plain types, `gradcam()` returns PNG bytes, `ValueError("invalid image")` for unreadable input |
| `calibrate.py` | `python calibrate.py` -> prints threshold + entropy_max, writes `weights/calibration.json` |
| `export_onnx.py` | `python export_onnx.py` -> `weights/seefood.onnx`, `weights/seefood_int8.onnx`, `outputs/benchmark.json` |
| `weights/calibration.json` | `{"threshold": 0.37, "entropy_max": 0.6661}` |
| `weights/seefood.onnx` | FP32 export of the trained model, 8.87 MB |
| `weights/seefood_int8.onnx` | INT8 dynamic quantization, 2.41 MB |
| `outputs/benchmark.json` | final numbers, real weights |

## Requests (To: X | Need | Needed by)
| # | To | Need | Needed by |
|---|----|------|-----------|
| B-R1 | A | `outputs/val_probs.npy` + `val_labels.npy` were there before you posted the signal, so I calibrated already - please still post `M2 WEIGHTS READY` so C knows the weights are final. | now |
| B-R2 | C | `requirements.txt` is missing `onnxconverter-common` (needed for the FP16 fallback of quantization). Please add it. | before C6 |
| B-R3 | Human | `edge_cases/` does not exist yet. Please drop 5-6 ambiguous images there (bratwurst, corndog, dog in a hotdog costume, ...) so I can finish B5 with the real edge-case table. | B5 |
| B-R4 | A, C | Environment resolved: `.venv` is Python 3.14.3 with `--system-site-packages` and works (`torch 2.13.0+cpu`, `torchvision 0.28.0+cpu`, `onnxruntime 1.31.0`). I cancelled my own `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` because the 2.14.1+cpu wheel (124 MB) kept timing out on this network and the system packages already cover it. Please do not start a second pip install into `.venv` while other agents are running. | resolved |
| B-R5 | A | Your `data.get_loaders()` works from the repo root and the dataset is already in `data/`. My `export_onnx.py` uses `data.get_loaders()[-1]` as the test loader, as allowed by `jobs.md` ("B may import data.get_loaders()"). | done |
| B-R6 | C | `requirements.txt` pins `torch==2.14.1` / `torchvision==0.29.1`, but the environment that actually trains and runs here is `torch==2.13.0+cpu` / `torchvision==0.28.0+cpu` (the 2.14.1+cpu wheel download times out on this network). Please pin the versions that work so a fresh `pip install -r requirements.txt` reproduces the tested setup. | before C6 |
| B-R7 | C | For the README demo: `infer.gradcam(open("some.jpg","rb").read())` returns PNG bytes (base64 it for the web UI). Example overlays I generated for the Grad-CAM check are in `C:\Users\Asus\AppData\Local\Temp\opencode\cam_hotdog_*.png` - the heat sits on the sausage. | C6 |

## Replies (answer requests addressed to me, cite the number, e.g. "A-R1")
- A confirmed `outputs/val_labels.npy` uses hotdog = 1 (training target `1 - folder_label`), so my `calibrate.py` compares `probs >= thr` against that array directly.
- A posted `M1 READY` (2026-10-21 22:30). `model.py` and `data.py` work, so B1/B2/B3 could be finished in stub mode.

## Blockers
- **B5 only:** `edge_cases/` does not exist (checked at 23:50). The human has to put the 5-6 ambiguous images there (B-R3). The command that will produce the final table is `.venv\Scripts\python.exe C:\Users\Asus\AppData\Local\Temp\opencode\b_edge.py` (or the same three lines inline: `infer.predict(open(f,"rb").read())` for every file in `edge_cases/`). Until then the stand-in table above documents the behaviour on the hardest test images.
- Nothing else: all my other jobs are done and verified with their gate commands.

## Contract change requests (human approval needed)
- None.

## Results / numbers

**B1 gate** - `py -3.14 C:\Users\Asus\AppData\Local\Temp\opencode\btest1.py` ->
`{'label': 'Not Hotdog!', 'confidence': 50.33285617828369, 'prob_hotdog': 0.4966714382171631, 'uncertain': False}`, types `{'label': 'str', 'confidence': 'float', 'prob_hotdog': 'float', 'uncertain': 'bool'}`, garbage bytes -> `ValueError: invalid image`.

**B2 gate** - `py -3.14 C:\Users\Asus\AppData\Local\Temp\opencode\btest2.py` -> `png bytes: 11748 size: (300, 200) pixel std: 25.364`, `second call ok, bytes: 11748`, `hooks left: 0`.

**B3 gate (stub weights)** - `py -3.14 export_onnx.py` -> `exported weights\seefood.onnx (8.82 MB)`, `onnx vs torch allclose(atol=1e-4): True (max abs diff 9.31e-10)`, `quantized weights\seefood_int8.onnx (2.36 MB)`, the quantized model loads in ONNX Runtime and runs.

**B4 gate (real validation probs)** - `.venv\Scripts\python.exe calibrate.py` -> `threshold: 0.37 (val f1 0.9293)`, `entropy_max: 0.6661`, `wrote weights\calibration.json`.
`infer` now reads it: `{'threshold': 0.37, 'entropy_max': 0.6661}`.
Validation sweep (100 val images, 50 hotdog): thr 0.37 -> acc 0.930 / F1 0.9293 / prec 0.9388 / rec 0.9200; thr 0.50 -> acc 0.910 / F1 0.9053. Entropy of correctly classified val samples: median 0.4037, p95 0.6661; entropy of the wrongly classified ones: median 0.6007.

**B5 gate (stand-in table - the real `edge_cases/` folder is still missing, see B-R3)**

The 4 test hotdogs the model is least sure about + the 3 not-hotdogs it is most sure about (worst false positives):

| file | true | label | confidence | prob_hotdog | uncertain | correct @0.37 |
|---|---|---|---|---|---|---|
| 741637.jpg (sub in a bun with fries) | hotdog | Hotdog! | 37.54 | 0.3754 | False | yes |
| 267629.jpg | hotdog | Not Hotdog! | 64.04 | 0.3596 | False | no |
| 498141.jpg | hotdog | Not Hotdog! | 64.39 | 0.3561 | False | no |
| 315747.jpg | hotdog | Hotdog! | 38.39 | 0.3839 | False | yes |
| 68684.jpg (a pizza slice) | not | Hotdog! | 83.00 | 0.8300 | False | no |
| 44810.jpg | not | Hotdog! | 80.20 | 0.8020 | False | no |
| 61822.jpg | not | Hotdog! | 74.04 | 0.7404 | False | no |

Observations, honestly: the ambiguous hotdogs are real hotdogs that look like long sandwiches (`741637.jpg` is a hot dog in a long bun next to fries) and the model gives them p ~ 0.36-0.38 - right at the calibrated threshold, so 2 of 4 flip to the correct class with threshold 0.37 while all 4 would be wrong with 0.5. The worst false positives are a pizza slice and two similar food photos: the model is *confidently* wrong (p 0.74-0.83), so the `uncertain` flag does **not** catch them - entropy measures how undecided the model is, not whether it is right.

**B6 gate (real weights)** - `.venv\Scripts\python.exe export_onnx.py` -> `exported weights\seefood.onnx (8.87 MB)`, `onnx vs torch allclose(atol=1e-4): True (max abs diff 1.25e-06)`, `quantized weights\seefood_int8.onnx (2.41 MB)`, `accuracy torch 0.9060 / onnx 0.9060 / int8 0.9060`.

| model | size_mb | ms_per_img | test accuracy |
|---|---|---|---|
| torch (PyTorch, CPU) | 9.037 | 20.70 | 0.906 |
| ONNX FP32 | 8.873 | 3.41 | 0.906 |
| ONNX INT8 (dynamic) | 2.409 | 77.87 | 0.906 |

Honest note: quantization did **not** hurt accuracy (identical 0.906 for all three, so no collapse), but on this CPU the INT8 model is ~23x slower than the FP32 ONNX model while being 3.7x smaller. Dynamic quantization of the conv layers of MobileNetV2 does not pay off on this machine (no VNNI fast path). The alternative variants I measured on the stub model: INT8 with MatMul/Gemm only = 8.835 MB (no size gain) and 40.5 ms/img; FP16 = 4.447 MB and 58.4 ms/img. Conclusion: FP32 ONNX is the best runtime here, INT8 is the best for size / download; I shipped the INT8 file because the gate asks for a quantized model, and the numbers above are the honest ones. Timings were taken while the other agents were using the same CPU (a second run gave torch 14.10 / ONNX 2.46 / INT8 71.53 ms/img), so treat them as relative, not absolute.

**Grad-CAM check** - on `data/test/hot_dog/133245.jpg` (p = 0.9863) the red/yellow heat sits exactly on the two sausages in the buns; on `135628.jpg` it sits on the sausage in the bun. On a not-hotdog picture (a steak) the heat is diffuse over the meat. No background-only heatmaps so far.

## Explain-me
- **What Grad-CAM computes:** for the chosen target layer I keep the activations `A` (1280 x 7 x 7) and the gradient of the hotdog logit with respect to them, `dL/dA`. The channel weights are the spatial mean of the gradient, `w_c = mean(dL/dA_c)`, then `cam = ReLU(sum_c w_c * A_c)`. The idea: a channel that increases the hotdog logit gets a positive weight, so the map lights up where the evidence for "hotdog" is. The map is normalised to 0-1, resized to the original image, coloured with the jet colormap and blended 50/50 with the photo.
- **Why the last conv layer:** `model.features[-1]` is the deepest convolutional block before the global pooling and the classifier, so its 7x7 map still has spatial structure (a real "where"), while still being close enough to the output that the gradient is meaningful. Earlier layers are more generic (edges, textures) and the classifier is 1x1, so it has no spatial information at all.
- **Why the threshold is tuned on validation and never on test:** the threshold is a hyperparameter of the decision rule, not of the model. If I picked it on the test set I would silently fit to the test set and my reported accuracy would be optimistic and dishonest. The validation set is carved out of the training folder (80/20, stratified, seed 42) and was never used for the test numbers, so tuning the threshold there keeps the test set a clean, one-shot measurement. The test set is used exactly once, at the end, to report.
- **What entropy means:** binary entropy `-(p ln p + (1-p) ln(1-p))` measures how undecided the model is, not how correct it is. It is 0 at p = 0 or 1 and maximal at ln 2 = 0.693 when p = 0.5. I flag a prediction as `uncertain` when its entropy is above the 95th percentile of the entropy of the correctly classified validation samples (0.6661), so only the most torn predictions are marked. The consequence, which I report honestly: this rule is conservative, because the model is often unsure on hard hotdogs, so `uncertain` fires rarely.
- **What quantization trades:** INT8 stores the weights in 8 bits instead of 32, so the model file shrinks 3.7x (8.87 MB -> 2.41 MB) and memory traffic drops, but the convolution is no longer computed in floating point. On this CPU the accuracy is unchanged (0.906) but the runtime is much slower (77.9 vs 3.4 ms/img), because ONNX Runtime emulates the integer convolutions here. So quantization buys download size, not speed, in this setup - a classic trade-off I measured instead of assuming.
