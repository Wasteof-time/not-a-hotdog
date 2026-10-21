import json
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".agent"))
import config
from model import build_model

ONNX_PATH = os.path.join("weights", "seefood.onnx")
INT8_PATH = os.path.join("weights", "seefood_int8.onnx")
FP16_PATH = os.path.join("weights", "seefood_fp16.onnx")
BENCH_PATH = os.path.join(config.OUT_DIR, "benchmark.json")
WARMUP = 10
RUNS = 100


def build():
    m = build_model(pretrained=False)
    if os.path.exists(config.WEIGHTS):
        try:
            sd = torch.load(config.WEIGHTS, map_location="cpu")
        except Exception:
            sd = torch.load(config.WEIGHTS, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        m.load_state_dict(sd)
    m.eval()
    return m


def export(model):
    os.makedirs("weights", exist_ok=True)
    x = torch.randn(1, 3, config.SIZE, config.SIZE)
    try:
        torch.onnx.export(
            model,
            x,
            ONNX_PATH,
            input_names=["x"],
            output_names=["logit"],
            dynamic_axes={"x": {0: "n"}, "logit": {0: "n"}},
            opset_version=17,
            dynamo=False,
        )
    except Exception as e:
        print("legacy exporter failed (%s), trying dynamo" % e)
        torch.onnx.export(
            model,
            x,
            ONNX_PATH,
            input_names=["x"],
            output_names=["logit"],
            dynamic_axes={"x": {0: "n"}, "logit": {0: "n"}},
            opset_version=17,
            dynamo=True,
        )
    print("exported %s (%.2f MB)" % (ONNX_PATH, os.path.getsize(ONNX_PATH) / 1e6))


def ort_session(path):
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def check_match(model, session, n=5):
    x = torch.randn(n, 3, config.SIZE, config.SIZE)
    with torch.no_grad():
        ref = model(x).cpu().numpy()
    got = session.run(None, {"x": x.numpy()})[0]
    ok = bool(np.allclose(ref, got, atol=1e-4))
    print("onnx vs torch allclose(atol=1e-4): %s (max abs diff %.2e)" % (ok, float(np.abs(ref - got).max())))
    return ok


def quantize():
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(ONNX_PATH, INT8_PATH, weight_type=QuantType.QInt8)
    print("quantized %s (%.2f MB)" % (INT8_PATH, os.path.getsize(INT8_PATH) / 1e6))


def quantize_matmul_only():
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(ONNX_PATH, INT8_PATH, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    print("quantized (MatMul/Gemm only) %s (%.2f MB)" % (INT8_PATH, os.path.getsize(INT8_PATH) / 1e6))


def quantize_fp16():
    from onnxconverter_common.float16 import convert_float_to_float16

    convert_float_to_float16(ONNX_PATH, FP16_PATH)
    print("fp16 %s (%.2f MB)" % (FP16_PATH, os.path.getsize(FP16_PATH) / 1e6))


def bench_ort(path, x):
    s = ort_session(path)
    for _ in range(WARMUP):
        s.run(None, {"x": x})
    t = time.perf_counter()
    for _ in range(RUNS):
        s.run(None, {"x": x})
    ms = (time.perf_counter() - t) * 1000.0 / RUNS
    size = os.path.getsize(path) / 1e6
    return ms, size, s


def bench_torch(model, x):
    x = torch.from_numpy(x)
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        t = time.perf_counter()
        for _ in range(RUNS):
            model(x)
    ms = (time.perf_counter() - t) * 1000.0 / RUNS
    size = sum(v.numel() * v.element_size() for v in model.state_dict().values()) / 1e6
    return ms, size


def test_loader():
    try:
        import data
    except Exception as e:
        print("no data.py yet (%s), skipping test accuracy" % e)
        return None
    loaders = data.get_loaders()
    if isinstance(loaders, (tuple, list)):
        return loaders[-1]
    return loaders


def hotdog_index(loader):
    ds = getattr(loader, "dataset", None)
    classes = getattr(ds, "classes", None)
    if classes:
        for i, c in enumerate(classes):
            if "hot_dog" in c:
                return i
    return 0


def accuracy(session, loader, thr=0.5):
    idx = hotdog_index(loader)
    correct = 0
    total = 0
    for x, y in loader:
        logits = session.run(None, {"x": x.numpy().astype(np.float32)})[0]
        p = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
        yh = (y.numpy() == idx).astype(int)
        correct += int(((p >= thr).astype(int) == yh).sum())
        total += len(y)
    return correct / max(total, 1)


def torch_accuracy(model, loader, thr=0.5):
    idx = hotdog_index(loader)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x).numpy().reshape(-1)
            p = 1.0 / (1.0 + np.exp(-logits))
            yh = (y.numpy() == idx).astype(int)
            correct += int(((p >= thr).astype(int) == yh).sum())
            total += len(y)
    return correct / max(total, 1)


def main():
    model = build()
    export(model)
    s32 = ort_session(ONNX_PATH)
    check_match(model, s32)

    quant_mode = "int8_dynamic"
    try:
        quantize()
        s8 = ort_session(INT8_PATH)
        x = np.random.randn(1, 3, config.SIZE, config.SIZE).astype(np.float32)
        s8.run(None, {"x": x})
    except Exception as e:
        print("int8 quantization failed (%s), retrying with MatMul/Gemm only" % e)
        quantize_matmul_only()
        s8 = ort_session(INT8_PATH)
        quant_mode = "int8_dynamic_matmul_gemm_only"

    x = np.random.randn(1, 3, config.SIZE, config.SIZE).astype(np.float32)
    t_ms, t_mb = bench_torch(model, x)
    o_ms, o_mb, _ = bench_ort(ONNX_PATH, x)
    q_ms, q_mb, _ = bench_ort(INT8_PATH, x)

    out = {
        "torch": {"size_mb": round(t_mb, 3), "ms_per_img": round(t_ms, 3)},
        "onnx_fp32": {"size_mb": round(o_mb, 3), "ms_per_img": round(o_ms, 3)},
        "onnx_quantized": {"size_mb": round(q_mb, 3), "ms_per_img": round(q_ms, 3)},
        "quantization": quant_mode,
        "accuracy": {},
    }

    loader = test_loader()
    if loader is not None:
        out["accuracy"]["torch"] = round(torch_accuracy(model, loader), 4)
        out["accuracy"]["onnx_fp32"] = round(accuracy(s32, loader), 4)
        out["accuracy"]["onnx_quantized"] = round(accuracy(s8, loader), 4)
        print("accuracy torch %.4f / onnx %.4f / int8 %.4f" % (out["accuracy"]["torch"], out["accuracy"]["onnx_fp32"], out["accuracy"]["onnx_quantized"]))
    else:
        print("no test loader, accuracy not measured")

    os.makedirs(config.OUT_DIR, exist_ok=True)
    with open(BENCH_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
