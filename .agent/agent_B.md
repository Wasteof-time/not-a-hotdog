# Agent B: Inference + Explainability + Calibration + Optimization

Read `work.md` first (rules, contract). You own: `infer.py`, `calibrate.py`, `export_onnx.py`, `comms/B_report.md`.

## Tasks

**1. infer.py (start immediately, stub mode until `seefood.pt` exists)**
- `load()`: build model via `model.build_model(pretrained=False)`, load `WEIGHTS` if present, `eval()`, cache. Read `CALIB` if present.
- `preprocess(bytes)`: `PIL.Image.open(io.BytesIO(...)).convert("RGB")`, then the contract transform. Any failure -> `ValueError("invalid image")`.
- `predict(bytes)`: `p = sigmoid(logit)` under `torch.no_grad()`.
  - label = "Hotdog!" if `p >= threshold` else "Not Hotdog!"
  - confidence = `max(p, 1-p) * 100`
  - entropy = `-(p*ln p + (1-p)*ln(1-p))`; `uncertain = entropy > entropy_max`
- `gradcam(bytes)`: forward hook (activations) + backward hook (gradients) on `model.features[-1]`; weights = mean of gradients over H,W; cam = ReLU(sum(weights * activations)); normalize 0-1; resize to image size; apply a colormap, blend 50/50 with the original, return PNG bytes. Implement by hand (about 25 lines), no extra library. The target is the hotdog logit.

Post progress after `predict` works so C can stop using a mock.

**2. calibrate.py (after `M2 WEIGHTS READY`)**
- Load `outputs/val_probs.npy` and `val_labels.npy`.
- `threshold`: scan 0.05 to 0.95, pick the one with the best F1 on validation.
- `entropy_max`: 95th percentile of entropy among correctly classified validation samples.
- Write `weights/calibration.json`.
- Run every image in `edge_cases/` (bratwurst, corndog, dog in hotdog costume, etc.; human supplies 5 to 6) and report label, confidence, uncertain flag in your report as a small table. If `edge_cases/` is empty, post a request to the human.

**3. export_onnx.py `[Y2]`**
- Export with `torch.onnx.export` (dynamic batch axis), check ONNX output matches torch (`allclose`, atol 1e-4).
- Quantize: INT8 via `onnxruntime.quantization.quantize_dynamic`; if INT8 hurts or fails on conv layers, use FP16 and say so.
- Benchmark on CPU, 100 runs after warmup: torch vs ONNX vs quantized -> `outputs/benchmark.json` with `size_mb` and `ms_per_img`. Also report accuracy on the test set for the quantized model (it must not collapse).

## Acceptance
- `predict()` works on a brand-new image in both stub and real mode.
- Grad-CAM highlights the sausage/bun on a clear hotdog image; if it highlights background, say so honestly in the report.
- `calibration.json` exists; benchmark file exists.

## Report must contain
Signature confirmations, calibration numbers (threshold, entropy_max), edge case table, benchmark table, and `Explain-me` bullets: what Grad-CAM computes, why last conv layer, why threshold tuned on validation not test, what entropy means, what quantization trades.
