# Agent A Report

**Status:** DONE
**Last updated:** 2026-10-21 22:54

## Milestones
- M1 READY (2026-10-21 22:30)
- M2 WEIGHTS READY (2026-10-21 22:54)

## Done
- A1 DONE. Gate: `python -c "import torch; from model import build_model as b; print(b(False)(torch.randn(2,3,224,224)).shape)"` -> `torch.Size([2, 1])`. After `set_trainable(m, False)` only `classifier.1.weight` and `classifier.1.bias` have `requires_grad`.
- A2 DONE. Gate: `python data.py` -> class counts train `hot_dog=199, not_hot_dog=199`; val `50/50`; test `250/250`; one train batch `(32, 3, 224, 224)`; train/val index overlap assert passed.
- A3 DONE. Gate: `python train.py` (exit 0, ~608s CPU). Stage 1 val_acc by epoch: 0.86, 0.91, 0.89, 0.85, 0.91. Last stage-1 val_acc=0.9100, val_loss=0.2761 (all > 0.70).
- A4 DONE. Gate: `python -c "import torch; from model import build_model; m=build_model(False); sd=torch.load('weights/seefood.pt', map_location='cpu'); m.load_state_dict(sd); print('keys', len(sd)); print('load ok', m(torch.randn(1,3,224,224)).shape)"` -> `keys 314` / `load ok torch.Size([1, 1])` (no key errors). Stage 2 ran 10 epochs (unfreeze last 4 blocks, Adam 1e-5). Best-val-loss snapshot is stage-1 epoch 5 (`best_val_loss=0.276062670648098`); saved `weights/seefood.pt` (9,140,363 bytes).
- A5 DONE. Gate: files exist; `metrics.json` has accuracy/precision/recall/f1; confusion matrix 233+17+30+220=500. `val_probs.npy` shape (100,) range ~0.004–0.998; `val_labels.npy` unique {0,1}, n=100, mean 0.5 (hotdog=1).
- A6 DONE. Inspected 10 test errors (see Results). Technical challenge written.
- A7 DONE. Replied to B-R1/B-R4/B-R5. Explain-me written. Status DONE.

## Artifacts produced
| File | Notes |
|---|---|
| model.py | `build_model`, `set_trainable` |
| data.py | `download()`, `get_loaders()`; dataset copied into `data/train` and `data/test` |
| train.py | two-stage training, metrics, plots, val_probs |
| weights/seefood.pt | best-val-loss state_dict, 314 keys, 9,140,363 bytes |
| outputs/metrics.json | test acc/prec/rec/f1 + history + best_val_loss |
| outputs/confusion_matrix.png | [[233, 17], [30, 220]] (tn/fp/fn/tp), sums to 500 |
| outputs/curves.png | train/val loss+acc, vertical line at stage 2 |
| outputs/val_probs.npy | (100,) sigmoid P(hot_dog) on val |
| outputs/val_labels.npy | (100,) hotdog=1 |

## Requests (To: X | Need | Needed by)
| # | To | Need | Needed by |
|---|----|------|-----------|
| A-R1 | C | requirements already has torch/torchvision/kagglehub/sklearn/matplotlib. Shared `.venv` uses system Python 3.14 torch 2.13.0+cpu (CDN for 2.14.1+cpu was unstable). | noted |
| A-R2 | Human | Kaggle key was not needed; kagglehub downloaded `dansbecker/hot-dog-not-hot-dog` without `kaggle.json`. | resolved |

## Replies (answer requests addressed to me, cite the number, e.g. "A-R1")
- B-R1: `M1 READY` posted 2026-10-21 22:30. `M2 WEIGHTS READY` posted 2026-10-21 22:54. `outputs/val_probs.npy` and `outputs/val_labels.npy` are written. `val_labels.npy` uses hotdog = 1 (training target `1 - folder_label`; ImageFolder `hot_dog=0`).
- B-R4: No extra pip during training. Shared `.venv` stayed on system torch 2.13.0+cpu / torchvision 0.28.0+cpu.
- B-R5: `data.get_loaders()` is usable from the repo root; dataset is in `data/`. B may import it for the test loader as jobs.md allows.

## Blockers
- 

## Contract change requests (human approval needed)
- 

## Results / numbers
- Dataset: train folder 249 hot_dog + 249 not_hot_dog; test folder 250 + 250.
- Split seed 42, 80/20 stratified from train folder: train 199/199, val 50/50, test 250/250. Roughly balanced.
- Stage 1 (frozen backbone, Adam 1e-3, 5 epochs, `features.eval()` for BN): val_acc 0.86 / 0.91 / 0.89 / 0.85 / 0.91; best val_loss 0.2761 at epoch 5, val_acc 0.91.
- Stage 2 (unfreeze last 4 feature blocks, Adam 1e-5, 10 epochs): val_loss jumped 0.2761 -> 0.4791 on epoch 6; val_acc 0.91 -> 0.74 then 0.69. Recovered to val_loss 0.3559 / val_acc 0.79 by epoch 15. Never beat stage-1 best val_loss, so saved weights are the stage-1 snapshot.
- Test (once, on best-val-loss weights): accuracy=0.906, precision=0.9282700421940928, recall=0.88, f1=0.9034907597535934, n_test=500. CM [[233,17],[30,220]] (tn/fp/fn/tp). 47 misclassified (30 FN, 17 FP).

### Test errors inspected (10)
False negatives (labeled hot_dog, p < 0.5):
- `data/test/hot_dog/607289.jpg` p=0.074: sausage only, no bun, in a black takeout box. Looks like a plain hot dog link, missing the bun cue the model learned.
- `data/test/hot_dog/607287.jpg` p=0.102: same scene, person holding the open box; food is a tiny fraction of the frame.
- `data/test/hot_dog/154409.jpg` p=0.088: bahn-mi style sandwich, heavy cilantro, pickled radish, dark grilled meat. Looks like a sandwich, not a classic hotdog.
- `data/test/hot_dog/160523.jpg` p=0.133: cheeseburger with broccoli and onion. Label noise; this is not a hotdog.
- `data/test/hot_dog/502977.jpg` p=0.102: "Good Dog" food truck, people in line, no food in frame. Context label, not a food photo.
- `data/test/hot_dog/493727.jpg` p=0.075: plated restaurant dish with wine; pastry/sausage-like roll on salad, not a bun-and-frank.
- `data/test/hot_dog/259058.jpg` p=0.285: loaded corn/cheese dogs buried in foil and toppings; bun+sausage shape is hidden.
- `data/test/hot_dog/225367.jpg` p=0.477: classic bun+frank, close miss. Split bun + fries + condiment cups; frank is offset, probability just under 0.5.
- `data/test/hot_dog/505584.jpg` p=0.158: Varsity chili dogs on a tray with onion rings. Wide shot, chili covers the frank; model likely keys on bun+exposed sausage.

False positives (labeled not_hot_dog, p >= 0.5):
- `data/test/not_hot_dog/68684.jpg` p=0.830: pizza in a box. Elongated slices + brown toppings; high-confidence miss.
- `data/test/not_hot_dog/44810.jpg` p=0.802: two pizza slices on a pan. Same elongated-food cue.
- `data/test/not_hot_dog/61822.jpg` p=0.740: pepperoni pizza, close crop.
- `data/test/not_hot_dog/19143.jpg` p=0.682: prime rib + steak knife. Cylindrical meat, reddish, looks sausage-like in close-up.
- `data/test/not_hot_dog/26489.jpg` p=0.665: tacos with salsa. Folded bread + filling, similar to a loaded bun.

Pattern: FNs are occluded/atypical hotdogs, label noise (burger), or no-food context shots. FPs are other elongated or bun-like foods (pizza, tacos, roast).

### Technical challenge
Stage 2 hurt validation. Unfreezing the last 4 MobileNetV2 blocks, switching BN from `features.eval()` to `train()`, and building a new Adam at 1e-5 jumped val_loss from 0.276 to 0.479 and val_acc from 0.91 to 0.69–0.79. Likely cause: small val set (n=100) plus BN running-stats / last-block weights moving under a new optimizer, so the head that was already calibrated to frozen features was knocked off. What I changed: nothing in the contract. I kept best-by-val-loss, so the stage-2 snapshot was discarded and `weights/seefood.pt` is stage-1 epoch 5. Before: stage-1 val_acc 0.91 / val_loss 0.276. After stage 2 (last epoch): val_acc 0.79 / val_loss 0.356. Test accuracy on the kept weights: 0.906. No test-set retuning.

## Explain-me
- BCE-with-logits vs softmax+CE: one logit, two classes. `BCEWithLogitsLoss` folds sigmoid into the loss (stable). Softmax+CE on two logits is equivalent in theory but needs a 2-wide head; the contract is one logit with `sigmoid = P(hot_dog)`.
- Adam + 100x lr drop: Adam adapts per-parameter steps, which helps a new linear head on frozen ImageNet features. Stage 2 touches pretrained conv weights; lr 1e-5 (100x smaller than 1e-3) keeps those features from being washed out in a few batches.
- Freeze first: the ImageNet backbone already extracts edges/textures. Training only the new `Linear(1280,1)` for 5 epochs gets a usable decision boundary without destroying those filters. Then (intended) unfreeze the last 4 blocks at a tiny lr to specialize.
- Augmentation (`RandomResizedCrop`, flip, rotation 15, ColorJitter): forces the head to use shape/texture rather than exact crop, lighting, or left/right placement. Val/test use only Resize+Normalize so metrics measure the real distribution.
- Val is carved from the train folder (80/20, seed 42, stratified), test folder untouched until the end. That keeps model selection (best val loss) leak-free. Two `ImageFolder` instances share indices so val is not augmented.
- Label flip: ImageFolder is alphabetical (`hot_dog=0`). Target is `1 - folder_label` so the positive class is hotdog. Precision/recall/F1 and `val_labels.npy` all use hotdog=1.
- Stage-1 `model.features.eval()` while the classifier trains: frozen BatchNorm must keep ImageNet running mean/var. If BN stays in train mode it updates stats on a 398-image set and val accuracy collapses toward chance.
- Best checkpoint is min val loss, not last epoch and not test acc. That is why the saved file is the stage-1 snapshot even though stage 2 ran to completion.
