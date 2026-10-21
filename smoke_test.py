"""Smoke test for the SeeFood API. Start the server first: uvicorn app:app --port 8000"""
import io
import os
import sys
import urllib.error
import urllib.request

from PIL import Image

URL = os.environ.get("SEEFOOD_URL", "http://127.0.0.1:8000")
BOUNDARY = "----seefoodsmoke"
EXPECTED_KEYS = {"label", "confidence", "prob_hotdog", "uncertain", "heatmap_b64"}


def post(path, filename, content_type, data):
    body = b"".join([
        ("--%s\r\n" % BOUNDARY).encode(),
        ('Content-Disposition: form-data; name="file"; filename="%s"\r\n' % filename).encode(),
        ("Content-Type: %s\r\n\r\n" % content_type).encode(),
        data,
        ("\r\n--%s--\r\n" % BOUNDARY).encode(),
    ])
    req = urllib.request.Request(URL + path, data=body, method="POST")
    req.add_header("Content-Type", "multipart/form-data; boundary=%s" % BOUNDARY)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def main():
    img = Image.new("RGB", (64, 64), (200, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    jpeg = buf.getvalue()

    results = []

    status, body = post("/predict", "sample.jpg", "image/jpeg", jpeg)
    ok = status == 200 and all(k in body.decode("utf-8", "ignore") for k in EXPECTED_KEYS)
    results.append(("valid image returns 200 with all keys", ok, "status=%d" % status))

    status, _ = post("/predict", "notes.txt", "text/plain", b"this is not an image")
    results.append(("text file returns 415", status == 415, "status=%d" % status))

    status, _ = post("/predict", "fake.png", "image/png", b"\x00\x01\x02not-a-real-png")
    results.append(("random bytes as image/png return 400", status == 400, "status=%d" % status))

    failed = 0
    for name, ok, info in results:
        print("%s %s (%s)" % ("PASS" if ok else "FAIL", name, info))
        failed += 0 if ok else 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
