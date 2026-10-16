# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=False \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
WORKDIR /app

# ---------- test：只装单测所需依赖，不拉 torch ----------
FROM base AS test
ENV MEDLLM_HOME=/tmp/medllm
COPY requirements-test.txt ./
RUN pip install --no-cache-dir -r requirements-test.txt
COPY . .
CMD ["python", "-m", "unittest", "test_rag", "-v"]

# ---------- serve：完整应用依赖，工作目录以卷挂载 ----------
# 生成侧不在这里：默认后端连的是 llama-server（见 README 第四节），
# 模型权重、向量库与语料都从 MEDLLM_HOME 挂载进来，不打进镜像。
FROM base AS serve
COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MEDLLM_HOME=/data \
    RAG_GENERATOR_BACKEND=llamacpp \
    RAG_LLAMA_BASE_URL=http://127.0.0.1:8081 \
    RAG_LLAMA_SFT_URL=http://127.0.0.1:8080
EXPOSE 7860
CMD ["python", "rag_api.py", "--port", "7860"]
