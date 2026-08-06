"""拉取中债-国债总净价指数日频数据。

优先走本地 chinabond 直连（兼容新版 akshare 移除的接口），
若失败再尝试旧版 akshare API。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cbond_index import fetch_cbond_index

START_DATE = "20150101"
OUTPUT_CSV = "cn_gov_bond_index_daily.csv"


def fetch_gov_bond_index_daily(start_date: str = START_DATE) -> pd.DataFrame:
    try:
        df = fetch_cbond_index(
            index_category="国债总指数",
            indicator="净价",
            period="总值",
        )
    except Exception:
        import akshare as ak

        if not hasattr(ak, "bond_index_general_cbond"):
            raise
        df = ak.bond_index_general_cbond(
            index_category="国债总指数",
            indicator="净价",
            period="总值",
        )

    df = df.rename(columns={"date": "日期", "value": "index_net"})
    df["日期"] = pd.to_datetime(df["日期"])
    df = df[df["日期"] >= pd.Timestamp(start_date)].copy()
    df = df.dropna(subset=["index_net"]).sort_values("日期").reset_index(drop=True)
    return df


if __name__ == "__main__":
    data = fetch_gov_bond_index_daily()
    data.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved {len(data)} rows to {OUTPUT_CSV}")
    print(f"Range: {data['日期'].min().date()} ~ {data['日期'].max().date()}")
    print(data.tail())
