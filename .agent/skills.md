# Skills: recipes and pitfalls

Copy these patterns; adjust names. They are known-good shapes, not extra code to add.

## Shared (everyone)

**Label convention.** `ImageFolder` gives `hot_dog=0, not_hot_dog=1`. Model output is one logit, `sigmoid(logit) = P(hot_dog)`. Training target is `1 - folder_label`. Saved `val_labels.npy` uses hotdog = 1. Everything else (precision, recall, F1) treats hotdog as the positive class.

**Bytes to tensor (inference path).**
```python
img = Image.open(io.BytesIO(data)).convert("RGB")      # raises on non-images -> catch, raise ValueError("invalid image")
x = tf(img).unsqueeze(0)                                # tf = Resize((224,224)), ToTensor, Normalize(MEAN, STD)
```

**Report update.** Open your own `comms/<letter>_report.md`, set Status and time, tick what you did, put gate commands and one-line results under Done, put numbers under Results. Do not touch any other file in `comms/`.

**Checking a signal.** Read the other agent's report and look for the exact phrase under Milestones.

---

## A: training

**Leak-free split with different transforms.**
```python
base = ImageFolder(train_dir)
tr, va = train_test_split(list(range(len(base))), test_size=0.2, stratify=base.targets, random_state=SEED)
train_ds = Subset(ImageFolder(train_dir, TRAIN_TF), tr)
val_ds   = Subset(ImageFolder(train_dir, EVAL_TF), va)
test_ds  = ImageFolder(test_dir, EVAL_TF)
```
After `download()`, find the real `train`/`test` folders with `glob`; the archive may nest them (e.g. `.../seefood/train`).

**set_trainable semantics.** `backbone=False`: freeze all of `model.features`. `backbone=True, last_n_blocks=0`: unfreeze all features. `backbone=True, last_n_blocks=n`: freeze all, then unfreeze `model.features[-n:]`. Classifier is always trainable.

**Pitfalls.**
- Build the optimizer **after** calling `set_trainable`: `Adam([p for p in model.parameters() if p.requires_grad], lr=...)`. Stage 2 needs a new optimizer.
- Stage 1: after `model.train()`, also call `model.features.eval()` so frozen BatchNorm statistics do not drift.
- Target shape must match logits: `y = (1 - labels).float().unsqueeze(1)`.
- Evaluate with `model.eval()` and `torch.no_grad()`.
- Best weights: `best = copy.deepcopy(model.state_dict())` when val loss improves; `load_state_dict(best)` before the test evaluation and before saving.
- Metrics: `probs = sigmoid(logits)`, `pred = probs >= 0.5`, `precision_recall_fscore_support(y_true, pred, average="binary")` with `y_true` hotdog=1.
- Val accuracy stuck near 0.5 almost always means flipped labels or missing normalization.
- Plot: 2 subplots (loss, accuracy), train and val lines, vertical line where stage 2 starts. Use `plt.savefig`, not `plt.show()`.

---

## B: inference, Grad-CAM, calibration, ONNX

**Return plain types.** `float(p)`, `bool(unc)`. numpy/torch scalars break JSON in the API.

**Grad-CAM (hand-written).** Use a tensor hook, not a module backward hook (ReLU6 is in-place in MobileNetV2 and breaks module backward hooks).
```python
store = {}
def fwd(m, i, o):                      # must return None
    store["a"] = o
    o.register_hook(lambda g: store.__setitem__("g", g))
h = model.features[-1].register_forward_hook(fwd)
for p in model.parameters(): p.requires_grad_(True)
model.zero_grad()
out = model(x)                         # NOT inside torch.no_grad()
out[0, 0].backward()                   # hotdog logit
h.remove()
w = store["g"].mean(dim=(2, 3), keepdim=True)
cam = torch.relu((w * store["a"]).sum(1)).squeeze().detach().numpy()   # 7x7
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
heat = Image.fromarray((cam * 255).astype("uint8")).resize(img.size)    # then matplotlib cm.jet, blend 0.5/0.5, save PNG to BytesIO
```
Sanity: on a clear hotdog photo, hot regions should sit on the food, not the corners.

**Entropy and threshold.**
```python
ent = lambda p: -(p * np.log(p + 1e-9) + (1 - p) * np.log(1 - p + 1e-9))
thr = max(np.arange(0.05, 0.96, 0.01), key=lambda t: f1_score(y, probs >= t))
ok = (probs >= thr) == y
entropy_max = float(np.percentile(ent(probs[ok]), 95))
json.dump({"threshold": float(thr), "entropy_max": entropy_max}, open(CALIB, "w"))
```
Binary entropy maxes at ln 2 (0.693). Entropy near that means the model is torn.

**ONNX export, check, quantize.**
```python
torch.onnx.export(model, torch.randn(1,3,224,224), "weights/seefood.onnx", input_names=["x"], output_names=["logit"],
                  dynamic_axes={"x": {0: "n"}, "logit": {0: "n"}}, opset_version=17)   # if it errors, add dynamo=False
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("weights/seefood.onnx", "weights/seefood_int8.onnx", weight_type=QuantType.QInt8)
```
If the INT8 model fails to load or run on CPU (conv ops), retry once with `op_types_to_quantize=["MatMul","Gemm"]` (small size gain), or convert to FP16 with `onnxconverter_common.float16.convert_float_to_float16`. State which one you used and why. Feed the same preprocessed batch to torch and ONNX and `np.allclose(..., atol=1e-4)` for the FP32 export.

**Benchmark (CPU, ms per image).**
```python
s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
x = np.random.randn(1,3,224,224).astype(np.float32)
for _ in range(10): s.run(None, {"x": x})                 # warmup
t = time.perf_counter()
for _ in range(100): s.run(None, {"x": x})
ms = (time.perf_counter() - t) * 10
size_mb = os.path.getsize(path) / 1e6
```
Time torch the same way (`model.eval()`, `no_grad`). Use identical settings for all three.

---

## C: API, frontend, delivery

**Upload endpoint.**
```python
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        return JSONResponse({"error": "Only JPEG, PNG or WebP images are allowed"}, status_code=415)
    data = await file.read()
    if len(data) > 5 * 1024 * 1024:
        return JSONResponse({"error": "File is larger than 5 MB"}, status_code=413)
    try:
        res = infer.predict(data); heat = infer.gradcam(data)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    res["heatmap_b64"] = base64.b64encode(heat).decode()
    return res
```
Call `infer.load()` once at module level. `python-multipart` must be installed or uploads fail.

**Frontend fetch.**
```js
const fd = new FormData(); fd.append("file", file);
const r = await fetch("/predict", { method: "POST", body: fd });
const j = await r.json();
if (!r.ok) throw new Error(j.error || "Request failed");
img.src = "data:image/png;base64," + j.heatmap_b64;
```
Set loading true before `fetch`, false in `finally`.

**Hugging Face Docker Space.** `README.md` must start with this front-matter, and the app must listen on 7860:
```
---
title: SeeFood
sdk: docker
app_port: 7860
---
```
In the Dockerfile install CPU torch to keep the image small: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`, then the rest of `requirements.txt`. Weights must be inside the repo/image.

**Deploy boundary.** You prepare exact commands for `git init/add/commit/remote/push` and the Hugging Face upload. The human runs them (credentials). Do not push yourself.
