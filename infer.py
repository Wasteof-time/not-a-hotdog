import io
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".agent"))  # fallback if config.py/model.py live in .agent
import config
from model import build_model

TF = transforms.Compose(
    [
        transforms.Resize((config.SIZE, config.SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ]
)

DEFAULT_THRESHOLD = 0.5
DEFAULT_ENTROPY_MAX = None  # no entropy check until calibration.json exists

_model = None
_calib = None


def _load_calib():
    global _calib
    if os.path.exists(config.CALIB):
        mtime = os.path.getmtime(config.CALIB)
        if _calib is None or _calib.get("mtime") != mtime:
            with open(config.CALIB, "r") as f:
                c = json.load(f)
            c["mtime"] = mtime
            _calib = c
    elif _calib is None:
        _calib = {"threshold": DEFAULT_THRESHOLD, "entropy_max": DEFAULT_ENTROPY_MAX}
    return _calib


def _load_weights(m):
    try:
        sd = torch.load(config.WEIGHTS, map_location="cpu")
    except Exception:
        sd = torch.load(config.WEIGHTS, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd)


def load():
    global _model
    if _model is None:
        m = build_model(pretrained=False)
        if os.path.exists(config.WEIGHTS):
            _load_weights(m)
        m.eval()
        _model = m
    _load_calib()
    return _model


def _open(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except Exception:
        raise ValueError("invalid image")


def preprocess(data: bytes) -> torch.Tensor:
    img = _open(data)
    try:
        x = TF(img)
    except Exception:
        raise ValueError("invalid image")
    return x.unsqueeze(0)


def _entropy(p: float) -> float:
    return -(p * math.log(p + 1e-9) + (1 - p) * math.log(1 - p + 1e-9))


def predict(data: bytes) -> dict:
    m = load()
    x = preprocess(data)
    with torch.no_grad():
        logit = m(x)
    p = float(torch.sigmoid(logit).reshape(-1)[0].item())
    c = _load_calib()
    thr = float(c["threshold"])
    hotdog = p >= thr
    label = "Hotdog!" if hotdog else "Not Hotdog!"
    confidence = (p if hotdog else 1.0 - p) * 100.0  # confidence in the label we show
    emax = c["entropy_max"]
    uncertain = bool(_entropy(p) > float(emax)) if emax is not None else False
    return {
        "label": label,
        "confidence": float(confidence),
        "prob_hotdog": p,
        "uncertain": uncertain,
    }


def gradcam(data: bytes) -> bytes:
    m = load()
    img = _open(data)
    x = preprocess(data)
    store = {}

    def fwd(mod, inp, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = m.features[-1].register_forward_hook(fwd)
    try:
        for prm in m.parameters():
            prm.requires_grad_(True)
        m.zero_grad()
        out = m(x)
        out[0, 0].backward()
    finally:
        h.remove()

    grads = store["g"].detach()
    acts = store["a"].detach()
    w = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts).sum(1)).squeeze(0).numpy()
    cam = cam.astype(np.float32)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    small = Image.fromarray((cam * 255).astype(np.uint8)).resize(img.size)
    heat = colormaps["jet"](np.asarray(small).astype(np.float32) / 255.0)[:, :, :3]
    heat_img = Image.fromarray((heat * 255).astype(np.uint8))
    blended = Image.blend(img.convert("RGB"), heat_img, 0.5)

    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    return buf.getvalue()
