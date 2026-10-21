---
title: SeeFood
sdk: docker
app_port: 7860
---

# SeeFood - Hotdog or Not Hotdog?

A binary image classifier that tells you whether a photo shows a hotdog, with a REST API,
a single-page web UI, a Grad-CAM heatmap that shows what the model looked at, a calibrated
decision threshold, and an ONNX / INT8 export with a CPU benchmark.

Trained from scratch on the [Kaggle hotdog / not-hotdog dataset](https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog).

---

## What it is

| Piece | What it does |
|---|---|
| `model.py` | MobileNetV2 (ImageNet pretrained), classifier replaced with `Dropout(0.2) -> Linear(1280, 1)`, one logit |
| `data.py` | downloads the dataset, leak-free 80/20 stratified split of the train folder, augmentations |
| `train.py` | two-stage transfer learning (frozen backbone, then fine-tune), metrics and plots |
| `infer.py` | `predict(bytes)` and `gradcam(bytes)` - the inference API used by the server |
| `calibrate.py` | picks the decision threshold and the entropy limit on the validation set |
| `export_onnx.py` | ONNX export, FP32-vs-torch check, INT8 dynamic quantization, CPU benchmark |
| `app.py` | FastAPI: `GET /`, `GET /health`, `POST /predict` |
| `static/index.html` | the web UI (one file, vanilla JS, no build step) |
| `smoke_test.py` | three end-to-end checks against a running server |

---

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ (developed on 3.14). Everything is pinned in `requirements.txt`.

## Dataset

```bash
python data.py
```

Downloads `dansbecker/hot-dog-not-hot-dog` through `kagglehub` into `data/` (git-ignored).
If `kagglehub` cannot reach Kaggle, set up the Kaggle CLI credentials (`~/.kaggle/kaggle.json`)
and `data.py` falls back to `kaggle datasets download`. The script prints the train/val/test
class counts and asserts that no training image leaks into the validation split.

## Train

```bash
python train.py
```

Stage 1 trains the classifier head on a frozen backbone, stage 2 fine-tunes the last blocks.
It writes `weights/seefood.pt`, `outputs/metrics.json`, `outputs/confusion_matrix.png`,
`outputs/curves.png`, `outputs/val_probs.npy` and `outputs/val_labels.npy`.

## Run the app

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open <http://127.0.0.1:8000/>. The API:

| Method | Path | Answer |
|---|---|---|
| GET | `/` | the web UI |
| GET | `/health` | `{"status":"ok"}` |
| POST | `/predict` | `{"label","confidence","prob_hotdog","uncertain","heatmap_b64"}` |

`POST /predict` takes a multipart field named `file`. JPEG, PNG and WebP up to 5 MB.
Wrong type -> `415`, too large -> `413`, undecodable bytes -> `400` with `{"error": "..."}`.

```bash
curl -F "file=@photo.jpg" http://127.0.0.1:8000/predict
```

## Local inference, one line

```bash
python -c "import infer; print(infer.predict(open('photo.jpg','rb').read()))"
```

`infer.predict` returns plain Python types: `label` (`"Hotdog!"` / `"Not Hotdog!"`),
`confidence` (0-100, in the shown label), `prob_hotdog` (0-1) and `uncertain` (bool).
`infer.gradcam(bytes)` returns the overlay PNG bytes.

## Smoke test

```bash
uvicorn app:app --port 8000
python smoke_test.py
```

Prints one `PASS` line per check: valid image -> 200 with all keys, `.txt` -> 415,
random bytes labelled `image/png` -> 400.

---

## Architecture

```
input image (RGB, any size)
  -> Resize((224, 224))
  -> ToTensor()            # [0,1], CHW
  -> Normalize(mean, std)  # ImageNet statistics
  -> [1, 3, 224, 224] float32
  -> MobileNetV2 features  # 1x1 depthwise-separable convs + inverted residuals
  -> Dropout(0.2) + Linear(1280, 1)
  -> [1, 1] logit
  -> sigmoid(logit) = P(hotdog)
```

The model has a single output logit, so the training target is `1 - folder_label`
(`ImageFolder` numbers `hot_dog=0`, `not_hot_dog=1`) and the loss is
`BCEWithLogitsLoss`, which is the numerically stable version of BCE for a one-logit model.

**Grad-CAM** hooks the last convolutional block (`model.features[-1]`, 7x7x1280),
backpropagates the hotdog logit, weights the activations by the mean gradient, upsamples
the resulting map to the original image size and blends it with a jet colormap at 50%.

**Calibration**: `weights/calibration.json` holds the decision `threshold` (the one that
maximises F1 on the validation probabilities) and `entropy_max` (the 95th percentile of the
binary entropy over the correctly classified validation samples). A prediction whose
entropy is above `entropy_max` is returned with `uncertain: true`.

---

## Hyperparameters and why

| Hyperparameter | Value | Why |
|---|---|---|
| Backbone | MobileNetV2, ImageNet pretrained | Small and fast on CPU, and the ImageNet features already know what food looks like |
| Input size | 224 x 224 | The size MobileNetV2 was pretrained on, so the pretrained weights still fit |
| Loss | `BCEWithLogitsLoss` | One logit per image; BCE-with-logits is the stable version (sigmoid applied inside) |
| Optimizer | Adam | Adaptive per-parameter step size, works well without a hand-tuned schedule |
| Stage 1 lr | 1e-3 | The backbone is frozen, so only the new head is learning - it can afford a large step |
| Stage 1 epochs | 5 | Enough to fit the head; the backbone cannot overfit |
| Stage 2 lr | 1e-5 | Fine-tuning a pretrained backbone needs a small step, otherwise the pretrained features are destroyed |
| Stage 2 epochs | 10 | Short fine-tune of the last blocks only |
| Unfrozen blocks | the last 4 modules of `model.features` | Keeps the early generic features intact and limits the number of trainable parameters |
| Batch size | 32 | Fits comfortably in RAM/CPU time and keeps the BN statistics stable |
| Augmentation | `RandomResizedCrop(0.7-1.0)`, horizontal flip, 15 deg rotation, colour jitter 0.2 | The dataset is small (498 train images), so augmentation is what prevents the classifier from memorising backgrounds |
| Split | 80/20 stratified from the train folder, seed 42 | The test folder is only touched once, at the end, for reporting |
| Normalisation | ImageNet mean/std | The pretrained backbone expects ImageNet statistics |
| Dropout | 0.2 before the classifier | Regularises the only randomly initialised layer |
| Threshold | from `weights/calibration.json` | F1-optimal on validation, not the default 0.5 |

---

## Results

All numbers below are copied from files in this repo - `outputs/metrics.json` (Agent A),
`weights/calibration.json` and `outputs/benchmark.json` (Agent B). Nothing is estimated.

Test set metrics - source `outputs/metrics.json`, 500 test images (250 hotdog / 250 not hotdog):

| Metric | Value |
|---|---|
| Accuracy | 0.906 |
| Precision | 0.928 |
| Recall | 0.880 |
| F1 | 0.903 |
| Best validation loss | 0.276 |

Calibration - source `weights/calibration.json`:

| Item | Value |
|---|---|
| Decision threshold | 0.370 |
| Entropy limit | 0.666 |

CPU benchmark (single CPU core, batch of 1, 10 warm-up + 100 timed runs) and test accuracy -
source `outputs/benchmark.json`:

| Model | Size (MB) | ms/image | Test accuracy |
|---|---|---|---|
| PyTorch (CPU) | 9.037 | 20.70 | 0.906 |
| ONNX FP32 | 8.873 | 3.41 | 0.906 |
| ONNX INT8 (dynamic quantization) | 2.409 | 77.87 | 0.906 |

Honest note: quantization did **not** hurt accuracy (all three are 0.906), but on this CPU the
INT8 model is about 23x slower than the FP32 ONNX model while being 3.7x smaller - ONNX Runtime
emulates the integer convolutions here and there is no VNNI fast path. So the INT8 file is the
one to ship if download size matters, and FP32 ONNX is the one to run if speed matters.
(The timings were measured while the other agents were using the same machine, so read them as
relative, not absolute.)

Training curves and the confusion matrix are in `outputs/curves.png` and
`outputs/confusion_matrix.png`.

![Training and validation curves](outputs/curves.png)

![Confusion matrix on the test set](outputs/confusion_matrix.png)

---

## The one technical challenge

**Stage 2 fine-tuning made the model worse, and the honest fix was to throw it away.**

The plan was classic two-stage transfer learning: freeze the whole MobileNetV2 backbone, train the
new classification head for 5 epochs (Adam, lr 1e-3), then unfreeze the last 4 feature blocks and
fine-tune for 10 more epochs with a small learning rate (Adam, lr 1e-5).

Stage 1 worked: validation accuracy 0.91, validation loss 0.276.

Stage 2 made things worse immediately. On the first fine-tuned epoch the validation loss jumped from
0.276 to 0.479 and the validation accuracy fell from 0.91 to 0.74 (bottoming out at 0.69). After 10
epochs it had only recovered to val_loss 0.356 / val_acc 0.79 - still clearly worse than the
frozen-backbone snapshot.

The likely causes:

- The validation set is small (100 images), so a couple of bad epochs dominate the curve.
- Unfreezing the last blocks also switches BatchNorm back from `features.eval()` (frozen ImageNet
  running statistics) to `train()` mode, so the normalisation statistics of those blocks start
  drifting on a 398-image training set.
- The head was already calibrated to the frozen features; moving the last blocks under a new
  optimizer knocks that alignment off.

What was done about it: **nothing was retuned on the test set.** The training script keeps the
checkpoint with the lowest validation loss, so the stage-2 weights were discarded and
`weights/seefood.pt` is the stage-1 (frozen backbone) snapshot. The shipped weights score 0.906 on
the test set, measured once, at the end. "Fixing" stage 2 by looking at the test set would have
made the reported number a lie.

## What the model gets wrong

From the inspected test errors (47 misclassified out of 500):

- **False negatives** are mostly atypical or occluded hotdogs: a sausage with no bun in a takeout
  box, bahn-mi style sandwiches, chili dogs buried under toppings, and context shots where the food
  is tiny or missing entirely ("Good Dog" food truck with a queue). A few are label noise - a
  cheeseburger filed under `hot_dog`.
- **False positives** are other elongated or bun-like foods: pizza slices and whole pizzas
  (p = 0.83, 0.80, 0.74), prime rib (0.68) and tacos (0.67). The model has learned "elongated
  brown food in a bread-like shape" more than the specific notion of a hotdog.

That is also what the calibrated threshold is for: moving the decision boundary from 0.5 to 0.37
fixes several of the borderline hotdogs (on validation: F1 0.905 -> 0.929) at the cost of a few
more false positives.

---

## Demo and weights

- Live demo: TBD (the human creates the GitHub repo and the Hugging Face Space, see the commands
  in `comms/C_report.md`)
- Weights in this repo: `weights/seefood.pt` (PyTorch, 9.1 MB), `weights/seefood.onnx`
  (ONNX FP32, 8.9 MB), `weights/seefood_int8.onnx` (INT8, 2.4 MB)
- Docker image: `docker build -t seefood .` then `docker run -p 7860:7860 seefood`

## How the bytes become a tensor

`multipart/form-data` upload -> FastAPI `UploadFile` -> `await file.read()` gives raw `bytes` ->
`PIL.Image.open(io.BytesIO(bytes)).convert("RGB")` -> `transforms.Resize((224, 224))` ->
`ToTensor()` (pixels to `[0, 1]`, CHW) -> `Normalize(ImageNet mean/std)` -> `unsqueeze(0)` adds the
batch dimension -> `[1, 3, 224, 224]` float32 -> MobileNetV2 -> one logit -> `sigmoid` =
`P(hotdog)`.
