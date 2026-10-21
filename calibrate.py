import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".agent"))
import config

PROBS = os.path.join(config.OUT_DIR, "val_probs.npy")
LABELS = os.path.join(config.OUT_DIR, "val_labels.npy")
GRID = np.arange(0.05, 0.96, 0.01)


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(p + 1e-9) + (1 - p) * np.log(1 - p + 1e-9))


def main():
    if not (os.path.exists(PROBS) and os.path.exists(LABELS)):
        print("missing %s / %s - wait for A (M2 WEIGHTS READY)" % (PROBS, LABELS))
        return
    probs = np.load(PROBS).reshape(-1).astype(np.float64)
    y = np.load(LABELS).reshape(-1).astype(np.int64)
    thr = float(max(GRID, key=lambda t: f1_score(y, probs >= t)))
    ent = entropy(probs)
    ok = (probs >= thr) == (y == 1)
    entropy_max = float(np.percentile(ent[ok], 95))
    os.makedirs(os.path.dirname(config.CALIB) or ".", exist_ok=True)
    with open(config.CALIB, "w") as f:
        json.dump({"threshold": thr, "entropy_max": entropy_max}, f, indent=2)
    f1 = float(f1_score(y, probs >= thr))
    print("threshold: %.2f (val f1 %.4f)" % (thr, f1))
    print("entropy_max: %.4f" % entropy_max)
    print("wrote %s" % config.CALIB)


if __name__ == "__main__":
    main()
