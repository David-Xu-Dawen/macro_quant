"""Streamlit 入口：全屏嵌入原有 FastAPI + HTML 工作台，保留原界面与功能。

本地:
  streamlit run streamlit_app.py
  （会自动拉起 web/app.py / uvicorn）

Streamlit Community Cloud:
  Main file path 填: streamlit_app.py
  Secrets 里配置:
    BACKEND_URL = "https://你的-render-服务.onrender.com"
  （Cloud 无法对外暴露第二端口，所以 API/页面本体需部署在 Render）
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
DEFAULT_PORT = int(os.environ.get("MACRO_QUANT_PORT", "8765"))


def _is_streamlit_cloud() -> bool:
    return Path("/mount/src").exists() or bool(os.environ.get("STREAMLIT_APP_BASE_URL"))


def _secret_backend() -> str:
    try:
        return str(st.secrets.get("BACKEND_URL", "")).strip().rstrip("/")
    except Exception:
        return ""


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def _wait_healthy(base: str, timeout: float = 45.0) -> bool:
    import urllib.request

    deadline = time.time() + timeout
    url = f"{base.rstrip('/')}/healthz"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _ensure_local_backend() -> str:
    """本地自动启动 FastAPI；返回可访问的 base URL。"""
    host = "127.0.0.1"
    port = DEFAULT_PORT
    base = f"http://{host}:{port}"
    if _port_open(host, port):
        return base

    env = os.environ.copy()
    env["HOST"] = host
    env["PORT"] = str(port)
    log_path = ROOT / ".streamlit_backend.log"
    log_f = open(log_path, "a", encoding="utf-8")
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=str(WEB_DIR),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    if not _wait_healthy(base):
        raise RuntimeError(
            f"本地后端启动失败，请查看 {log_path}，或手动执行: cd web && python3 app.py"
        )
    return base


def _resolve_backend() -> tuple[str, str]:
    """返回 (backend_url, mode_note)。"""
    env_url = os.environ.get("BACKEND_URL", "").strip().rstrip("/")
    secret_url = _secret_backend()
    configured = secret_url or env_url

    if configured:
        return configured, "使用已配置的 BACKEND_URL（推荐用于 Streamlit Cloud）"

    if _is_streamlit_cloud():
        return "", (
            "Streamlit Cloud 无法单独暴露 FastAPI 端口。"
            "请先把本项目部署到 Render，然后在 App settings → Secrets 填写：\n\n"
            'BACKEND_URL = "https://你的服务.onrender.com"'
        )

    return _ensure_local_backend(), "本地模式：已自动启动 FastAPI 后端"


st.set_page_config(
    page_title="宏观量化工作台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      [data-testid="stHeader"],
      [data-testid="stToolbar"],
      [data-testid="stDecoration"],
      #MainMenu,
      footer { visibility: hidden; height: 0; }
      .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
      }
      iframe { border: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    backend, note = _resolve_backend()
except Exception as exc:
    st.error(f"无法启动工作台：{exc}")
    st.stop()

if not backend:
    st.title("宏观量化工作台 · Streamlit 入口")
    st.warning(note)
    st.markdown(
        """
### 部署步骤（保留原界面）

1. **Render**：用仓库里的 `render.yaml` / `Dockerfile` 部署 FastAPI（完整原站）
2. 拿到公开地址，例如 `https://macro-quant-xxxx.onrender.com`
3. **Streamlit Cloud**
   - Main file path: `streamlit_app.py`
   - Secrets:
     ```toml
     BACKEND_URL = "https://macro-quant-xxxx.onrender.com"
     ```
4. 之后本地改代码 → `git push` → GitHub → Render / Streamlit 自动更新

本地预览（无需 Secrets）:
```bash
streamlit run streamlit_app.py
```
"""
    )
    st.stop()

# 全屏嵌入原站：界面与功能与直接打开 FastAPI 一致
components.iframe(backend, height=1400, scrolling=True)
