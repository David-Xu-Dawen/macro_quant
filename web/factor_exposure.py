"""因子暴露：按结束周计算滚动窗口 Bootstrap + Lasso 矩阵。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FE_DIR = ROOT / "factor exposure"
OUTPUT_JSON = FE_DIR / "factor_exposure_latest.json"

_fe_module = None


def _load_fe_module():
    global _fe_module
    if _fe_module is not None:
        return _fe_module
    spec = importlib.util.spec_from_file_location(
        "compute_factor_exposure",
        FE_DIR / "compute_factor_exposure.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("无法加载 factor exposure/compute_factor_exposure.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["compute_factor_exposure"] = module
    spec.loader.exec_module(module)
    _fe_module = module
    return module


def default_params() -> dict:
    if OUTPUT_JSON.exists():
        data = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        return {
            "rolling_window_weeks": int(data.get("rolling_window_weeks", 260)),
            "sample_length_weeks": int(data.get("sample_length_weeks", 104)),
            "bootstrap_samples": int(data.get("bootstrap_samples", 3000)),
            "alpha_scale": float(data.get("alpha_scale", 0.5)),
            "default_end": data.get("window_end"),
        }
    return {
        "rolling_window_weeks": 260,
        "sample_length_weeks": 104,
        "bootstrap_samples": 3000,
        "alpha_scale": 0.5,
        "default_end": None,
    }


def load_mom_panel():
    """与因子暴露一致的周度环比/变化面板（含流动性）。"""
    fe = _load_fe_module()
    return fe.load_macro_weekly_mom()


def load_latest_json() -> dict:
    if not OUTPUT_JSON.exists():
        raise FileNotFoundError(
            "factor_exposure_latest.json not found，请先运行 python 'factor exposure/compute_factor_exposure.py'"
        )
    return json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))


def available_weeks(rolling_window_weeks: int | None = None) -> dict:
    fe = _load_fe_module()
    defaults = default_params()
    rolling = rolling_window_weeks or defaults["rolling_window_weeks"]
    all_weeks = fe.list_available_weeks(min_window=1)
    weeks = fe.list_available_weeks(min_window=rolling)
    return {
        "weeks": weeks,
        "total_weeks": len(all_weeks),
        "rolling_window_weeks": rolling,
        "default_end": defaults["default_end"] or (weeks[-1] if weeks else None),
    }


def compute_exposure(
    end: str,
    rolling_window_weeks: int | None = None,
    sample_length_weeks: int | None = None,
    bootstrap: int | None = None,
    alpha_scale: float | None = None,
    seed: int = 42,
) -> dict:
    fe = _load_fe_module()
    defaults = default_params()
    exposure, r_squared, meta = fe.compute_latest_exposure(
        n_bootstrap=bootstrap or defaults["bootstrap_samples"],
        rolling_window=rolling_window_weeks or defaults["rolling_window_weeks"],
        sample_length=sample_length_weeks or defaults["sample_length_weeks"],
        seed=seed,
        alpha_scale=alpha_scale if alpha_scale is not None else defaults["alpha_scale"],
        end_date=end,
        write_panel=False,
    )
    return fe.build_exposure_payload(exposure, r_squared, meta)
