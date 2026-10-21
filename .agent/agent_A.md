# Agent A: Data + Model + Training

Read `work.md` first (rules, contract). You own: `data.py`, `model.py`, `train.py`, `comms/A_report.md`.

## Tasks (in order)

**1. model.py (do first, post `M1 READY` after step 2)**
- `build_model(pretrained=True)` and `set_trainable(...)` exactly as in the contract.

**2. data.py**
- `download()`: fetch the Kaggle dataset into `data/` (kagglehub or kaggle CLI). Verify the real folder layout after download and set paths accordingly.
- `get_loaders()` returns `train_loader, val_loader, test_loader`.
  - Split: 80/20 stratified from the **train** folder, seed 42. Test folder is never used for tuning.
  - Two `ImageFolder` instances over the same indices: train transform (augmented) for train subset, eval transform for val subset. Otherwise augmentation leaks into validation.
  - Train transform: `RandomResizedCrop(224, scale=(0.7,1))`, `RandomHorizontalFlip`, `RandomRotation(15)`, `ColorJitter(0.2,0.2,0.2)`, ToTensor, Normalize.
  - Eval transform: `Resize((224,224))`, ToTensor, Normalize (must match contract).
  - Batch 32, `num_workers=2`.
- Print class counts for train/val/test (check imbalance; dataset should be roughly balanced).

**3. train.py**
- Loss: `BCEWithLogitsLoss`. Target = `1 - folder_label` (see contract).
- Stage 1 (head only): backbone frozen, Adam lr 1e-3, 5 epochs.
- Stage 2 (fine-tune): unfreeze last ~4 feature blocks, Adam lr 1e-5, 10 epochs.
- Keep the best weights by validation loss; save `weights/seefood.pt`.
- Record per-epoch train/val loss and accuracy across both stages; plot to `outputs/curves.png` (mark where stage 2 starts).
- After training, on the **test set**: Accuracy, Precision, Recall, F1 (sklearn, hotdog = positive class) -> `outputs/metrics.json`; confusion matrix -> `outputs/confusion_matrix.png`.
- Save validation sigmoid probabilities and labels (hotdog=1) -> `outputs/val_probs.npy`, `outputs/val_labels.npy`. B needs these. Then post `M2 WEIGHTS READY`.
- Look at 5 to 10 misclassified test images and note why they failed (needed for the interview).

## Acceptance
- `python data.py` downloads; `python train.py` runs end to end on CPU or Colab GPU and writes every file above.
- Test accuracy is reported honestly. Do not chase 99%; if below ~85%, check label flip, normalization, and frozen layers first.

## Report must contain
Class counts, final metrics, the one technical challenge (what you saw, what you changed, before/after numbers), and `Explain-me` bullets: BCE-with-logits vs softmax+CE, why Adam and why lr drops 100x for fine-tuning, why freeze first, what augmentation does, why val is carved from train.
