FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch keeps the image small, then the pinned set fills in the rest
# long timeouts + retries and a pip cache mount: the CPU wheel is ~190 MB and the
# network can stall, a cached wheel must not be downloaded again on every build
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --cache-dir /root/.cache/pip --timeout 300 --retries 20 \
      torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --cache-dir /root/.cache/pip --timeout 300 --retries 20 -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
