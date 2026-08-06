#!/usr/bin/env python3
"""
中债数据更新脚本

包含：
  - 3 年期国开债 / AA 中短期票据到期收益率
  - 企业债 AA / 国开债总 财富指数（3-5 年）

数据源：中国债券信息网 https://yield.chinabond.com.cn
默认读写：同目录下 中债_3Y_国开债_AA中票_收益率_2020至今.csv
日期按北京时间（Asia/Shanghai）解析，避免 UTC 导致交易日少 1 天。

用法：
  python update_bond_yields.py              # 增量更新（从 CSV 最后日期往后补）
  python update_bond_yields.py --full       # 从 2020-01-01 全量重建
  python update_bond_yields.py --days 30    # 强制回刷最近 N 天再合并
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# 中债返回的时间戳对应北京时间交易日，必须用 Asia/Shanghai，不能用 UTC（否则日期少 1 天）
TZ_CN = ZoneInfo("Asia/Shanghai")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "中债_3Y_国开债_AA中票_收益率_2020至今.csv"
FULL_START = date(2020, 1, 1)

# 收益率曲线（到期）
YIELD_CURVES = {
    "国开债_3Y": "8a8b2ca037a7ca910137bfaa94fa5057",
    "中短期票据AA_3Y": "2c9081e50a2f9606010a30acdae40176",
}

# 财富指数：indexid + 待偿期分段 qxlxt（03 = 3-5年）
WEALTH_INDICES = {
    "企业债AA财富_3-5年": {
        "indexid": "8a8b2ca0408e9fa70140949f012d0002",  # 中债-企业债AA指数
        "qxlxt": "03",
    },
    "国开债总财富_3-5年": {
        "indexid": "2c908188111fac07011125068f91044d",  # 中债-国开行债券总指数
        "qxlxt": "03",
    },
}

VALUE_COLS = [
    "国开债_3Y",
    "中短期票据AA_3Y",
    "企业债AA财富_3-5年",
    "国开债总财富_3-5年",
]
OUTPUT_COLS = ["日期", *VALUE_COLS, "利差_AA中票减国开_bp"]

DCQ_3Y = "3.0,3y;"  # 关键期限：3 年
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://yield.chinabond.com.cn/cbweb-mn/yield_main?locale=zh_CN",
}
YIELD_URL = "https://yield.chinabond.com.cn/cbweb-mn/yc/queryYz"
INDEX_URL = "https://yield.chinabond.com.cn/cbweb-mn/indices/singleIndexQuery"
REQUEST_PAUSE_SEC = 0.3


def _year_windows(start: date, end: date) -> list[tuple[date, date]]:
    """按自然年切分查询窗口，避免接口一次区间过大。"""
    windows: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        year_end = date(cur.year, 12, 31)
        chunk_end = min(year_end, end)
        windows.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return windows


def _ts_to_date(ts_ms: float | int) -> str:
    return datetime.fromtimestamp(float(ts_ms) / 1000, tz=TZ_CN).date().isoformat()


def fetch_yield_curve(
    curve_id: str, start: date, end: date
) -> list[tuple[str, float]]:
    """拉取单条收益率曲线 [start, end]。"""
    points: list[tuple[str, float]] = []
    for w_start, w_end in _year_windows(start, end):
        qs = (
            f"bjlx=no&&dcq={DCQ_3Y}"
            f"&&startTime={w_start.isoformat()}&&endTime={w_end.isoformat()}"
            f"&&qxlx=0,&&yqqxN=N&&yqqxK=K&&par="
            f"&&ycDefIds={curve_id},&&locale=zh_CN"
        )
        resp = requests.post(f"{YIELD_URL}?{qs}", headers=HEADERS, timeout=120)
        resp.raise_for_status()
        payload = resp.json()
        series = (payload[0].get("seriesData") or []) if payload else []
        for ts_ms, yld in series:
            points.append((_ts_to_date(ts_ms), float(yld)))
        time.sleep(REQUEST_PAUSE_SEC)
    return points


def fetch_wealth_index(
    indexid: str,
    qxlxt: str,
    start: date,
    end: date,
) -> list[tuple[str, float]]:
    """拉取财富指数全历史后按 [start, end] 截取（接口一次返回全序列）。"""
    params = {
        "indexid": indexid,
        "qxlxt": qxlxt,
        "ltcslx": "",
        "zslxt": "CFZS",
        "lx": "1",
        "locale": "zh_CN",
    }
    resp = requests.post(INDEX_URL, params=params, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    series = data.get(f"CFZS_{qxlxt}") or {}
    start_s, end_s = start.isoformat(), end.isoformat()
    points: list[tuple[str, float]] = []
    for ts_ms, value in series.items():
        dt = _ts_to_date(ts_ms)
        if start_s <= dt <= end_s:
            points.append((dt, float(value)))
    time.sleep(REQUEST_PAUSE_SEC)
    return points


def fetch_panel(start: date, end: date) -> pd.DataFrame:
    """拉取收益率 + 财富指数，拼成宽表。"""
    if start > end:
        return pd.DataFrame(columns=["日期", *VALUE_COLS])

    rows: dict[str, dict[str, float]] = {}

    for col, curve_id in YIELD_CURVES.items():
        print(f"  拉取 {col}: {start} ~ {end}")
        for dt, val in fetch_yield_curve(curve_id, start, end):
            rows.setdefault(dt, {})[col] = val

    for col, meta in WEALTH_INDICES.items():
        print(f"  拉取 {col}: {start} ~ {end}")
        for dt, val in fetch_wealth_index(
            meta["indexid"], meta["qxlxt"], start, end
        ):
            rows.setdefault(dt, {})[col] = val

    if not rows:
        return pd.DataFrame(columns=["日期", *VALUE_COLS])

    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "日期"
    df = df.reset_index()
    for col in VALUE_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[["日期", *VALUE_COLS]]


def load_existing(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=OUTPUT_COLS)
    return pd.read_csv(csv_path, dtype={"日期": str})


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in VALUE_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    out["利差_AA中票减国开_bp"] = (
        out["中短期票据AA_3Y"] - out["国开债_3Y"]
    ) * 100
    return out[OUTPUT_COLS]


def merge_update(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if new.empty:
        return finalize(old) if not old.empty else finalize(new)

    old_core = (
        old[["日期", *[c for c in VALUE_COLS if c in old.columns]]].copy()
        if not old.empty
        else pd.DataFrame(columns=["日期", *VALUE_COLS])
    )
    new_core = new[["日期", *VALUE_COLS]].copy()

    # 新数据覆盖同日旧值；缺失列用对方补齐
    combined = old_core.merge(new_core, on="日期", how="outer", suffixes=("_old", ""))
    for col in VALUE_COLS:
        old_col = f"{col}_old"
        if old_col in combined.columns:
            combined[col] = combined[col].combine_first(combined[old_col])
            combined = combined.drop(columns=[old_col])
        elif col not in combined.columns:
            combined[col] = pd.NA

    combined = combined.sort_values("日期").reset_index(drop=True)
    return finalize(combined)


def resolve_range(
    old: pd.DataFrame,
    *,
    full: bool,
    days: int | None,
) -> tuple[date, date]:
    today = date.today()
    if full or old.empty:
        return FULL_START, today

    # 若缺少财富指数列，自动从 2020 起重拉并入
    missing_wealth = any(c not in old.columns for c in WEALTH_INDICES)
    if missing_wealth:
        print("检测到缺少财富指数列，将从 2020-01-01 补全财富指数并做增量合并。")
        return FULL_START, today

    last = datetime.strptime(str(old["日期"].iloc[-1]), "%Y-%m-%d").date()
    if days is not None:
        start = today - timedelta(days=days)
    else:
        start = last - timedelta(days=5)
    return start, today


def save_csv(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.4f")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="更新中债收益率与财富指数 CSV"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"CSV 路径（默认: {DEFAULT_CSV.name}）",
    )
    parser.add_argument("--full", action="store_true", help="从 2020-01-01 全量重建")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="强制回刷最近 N 个自然日（与 --full 互斥时优先 --full）",
    )
    args = parser.parse_args(argv)

    csv_path: Path = args.csv
    old = load_existing(csv_path)
    start, end = resolve_range(old, full=args.full, days=args.days)

    print(f"目标文件: {csv_path}")
    print(f"更新区间: {start} ~ {end}")
    if not args.full and not old.empty:
        print(
            f"现有数据: {old['日期'].iloc[0]} ~ {old['日期'].iloc[-1]}"
            f"（{len(old)} 行）"
        )

    try:
        new = fetch_panel(start, end)
    except requests.RequestException as exc:
        print(f"请求失败: {exc}", file=sys.stderr)
        return 1

    if new.empty:
        print("未获取到新数据（可能是非交易日或接口暂无）。")
        return 0

    if args.full or old.empty:
        result = finalize(new)
    else:
        result = merge_update(old, new)

    # 全量场景下只保留至少有收益率或财富指数的行
    result = result.dropna(how="all", subset=VALUE_COLS).reset_index(drop=True)

    save_csv(result, csv_path)
    print(
        f"已更新: {result['日期'].iloc[0]} ~ {result['日期'].iloc[-1]}，"
        f"共 {len(result)} 行"
    )
    print(result.tail(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
