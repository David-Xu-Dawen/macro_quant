#!/usr/bin/env python3
"""从公开网络源拉取流动性因子所需数据。

低频:
  - M2 同比: 东方财富 / akshare.macro_china_money_supply
  - 社融存量同比: 免费源不稳定，优先沿用本地 Wind 导出 `中国_M2_同比.csv`

高频代理（替代 Wind 申万大盘/小盘市盈率）:
  - 沪深300 滚动市盈率 → 申万大盘市盈率
  - 中证1000 滚动市盈率 → 申万小盘市盈率
  来源: 乐咕乐股 / akshare.stock_index_pe_lg
  与原 Wind 序列日环比相关约 0.88–0.90，可自动更新至最近交易日。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).parent
WIND_CSV = ROOT / "中国_M2_同比.csv"

PE_LARGE_SYMBOL = "沪深300"
PE_SMALL_SYMBOL = "中证1000"
PE_LARGE_COL = "申万大盘市盈率"
PE_SMALL_COL = "申万小盘市盈率"


def _parse_cn_month(series: pd.Series) -> pd.DatetimeIndex:
    text = (
        series.astype(str)
        .str.replace("月份", "", regex=False)
        .str.replace("年", "-", regex=False)
        .str.replace("月", "", regex=False)
    )
    return pd.to_datetime(text, format="%Y-%m", errors="coerce")


def fetch_m2_yoy() -> pd.Series:
    """东方财富货币供应量：M2 同比（%）。"""
    df = ak.macro_china_money_supply()
    df = df.copy()
    df["date"] = _parse_cn_month(df["月份"])
    df = df.dropna(subset=["date"]).sort_values("date")
    out = pd.to_numeric(df["货币和准货币(M2)-同比增长"], errors="coerce")
    out.index = pd.DatetimeIndex(df["date"]).to_period("M").to_timestamp("M")
    out = out.groupby(level=0).last().dropna().sort_index()
    out.name = "m2_yoy"
    if out.empty:
        raise RuntimeError("akshare.macro_china_money_supply 未返回有效 M2 同比")
    return out


def load_sf_yoy_from_wind(path: Path = WIND_CSV) -> pd.Series:
    """从本地 Wind 导出读取社融存量同比。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少 Wind 导出: {path}")
    raw = pd.read_csv(path, encoding="gbk").iloc[4:].copy()
    raw = raw[raw["指标名称"] != "数据来源：Wind"]
    raw["date"] = pd.to_datetime(raw["指标名称"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date")
    col = "中国:社会融资规模存量:同比"
    if col not in raw.columns:
        raise KeyError(f"Wind 文件缺少列: {col}")
    s = pd.to_numeric(raw[col], errors="coerce")
    s.index = pd.DatetimeIndex(raw["date"])
    s = s.dropna().resample("ME").last().dropna()
    s.index = s.index.to_period("M").to_timestamp("M")
    s.name = "sf_yoy"
    if s.empty:
        raise RuntimeError("Wind 文件中社融存量同比为空")
    return s


def try_fetch_sf_increment_mofcom() -> pd.DataFrame | None:
    """商务部社融增量（非存量同比）。仅作连通性探测，不进入因子。"""
    url = "https://data.mofcom.gov.cn/datamofcom/front/gnmy/shrzgmQuery"
    try:
        out = subprocess.check_output(
            [
                "curl",
                "-sS",
                "-k",
                "-X",
                "POST",
                url,
                "-H",
                "User-Agent: Mozilla/5.0",
                "--max-time",
                "30",
            ],
            stderr=subprocess.DEVNULL,
            timeout=35,
        )
        if not out.strip():
            return None
        df = pd.read_json(out.decode("utf-8"))
        return df
    except Exception:
        return None


def fetch_pe_daily() -> pd.DataFrame:
    """乐咕乐股指数滚动市盈率，映射为原管线列名。"""
    frames = []
    for symbol, col in ((PE_LARGE_SYMBOL, PE_LARGE_COL), (PE_SMALL_SYMBOL, PE_SMALL_COL)):
        df = ak.stock_index_pe_lg(symbol=symbol)
        pe = df[["日期", "滚动市盈率"]].copy()
        pe["日期"] = pd.to_datetime(pe["日期"], errors="coerce")
        pe[col] = pd.to_numeric(pe["滚动市盈率"], errors="coerce")
        pe = pe.dropna(subset=["日期", col]).set_index("日期")[[col]].sort_index()
        frames.append(pe)
    out = pd.concat(frames, axis=1).sort_index()
    out = out.ffill().dropna(how="any")
    if out.empty:
        raise RuntimeError("乐咕乐股 PE 拉取结果为空")
    out.index.name = "date"
    return out


def build_mobility_monthly(
    m2: pd.Series | None = None,
    sf: pd.Series | None = None,
) -> pd.DataFrame:
    m2 = fetch_m2_yoy() if m2 is None else m2
    sf = load_sf_yoy_from_wind() if sf is None else sf
    panel = pd.concat([m2.rename("m2_yoy"), sf.rename("sf_yoy")], axis=1).dropna(how="any")
    panel["mobility_factor"] = panel["m2_yoy"] - panel["sf_yoy"]
    panel = panel.sort_index()
    panel.index.name = "date"
    return panel.reset_index()


def fetch_all() -> dict[str, pd.DataFrame | pd.Series]:
    """拉取并返回各序列摘要，供管线写入。"""
    m2 = fetch_m2_yoy()
    sf = load_sf_yoy_from_wind()
    pe = fetch_pe_daily()
    monthly = build_mobility_monthly(m2=m2, sf=sf)
    return {
        "m2_yoy": m2,
        "sf_yoy": sf,
        "pe_daily": pe,
        "mobility_monthly": monthly,
    }


if __name__ == "__main__":
    data = fetch_all()
    m2, sf, pe, monthly = data["m2_yoy"], data["sf_yoy"], data["pe_daily"], data["mobility_monthly"]
    print(f"M2 同比: {m2.index.min().date()} ~ {m2.index.max().date()} ({len(m2)} 月)")
    print(f"社融存量同比(Wind): {sf.index.min().date()} ~ {sf.index.max().date()} ({len(sf)} 月)")
    print(
        f"PE 日频代理({PE_LARGE_SYMBOL}/{PE_SMALL_SYMBOL}): "
        f"{pe.index.min().date()} ~ {pe.index.max().date()} ({len(pe)} 日)"
    )
    print(
        f"流动性月频因子: {monthly['date'].min().date()} ~ {monthly['date'].max().date()} "
        f"({len(monthly)} 月)"
    )
