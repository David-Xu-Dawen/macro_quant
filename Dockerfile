# 宏观量化工作台 — Render / Docker 部署
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8765

WORKDIR /app

# 先装依赖，便于利用 Docker 层缓存
COPY web/requirements.txt /app/web/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/web/requirements.txt

# 复制项目（体积由 .dockerignore 控制）
COPY . /app

WORKDIR /app/web

EXPOSE 8765

# Render 会覆盖 PORT；本地 docker run -p 8765:8765 也可直接用
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8765}"]
