"""拉取中国十年期国债收益率日频数据（AKShare）"""

import akshare as ak
import pandas as pd

START_DATE = "20150101"
OUTPUT_CSV = "cn10y_yield_daily.csv"


def fetch_cn10y_daily(start_date: str = START_DATE) -> pd.DataFrame:
    df = ak.bond_zh_us_rate(start_date=start_date)
    df["日期"] = pd.to_datetime(df["日期"])
    df = df[df["日期"] >= pd.Timestamp(start_date)].copy()
    result = (
        df[["日期", "中国国债收益率10年"]]
        .dropna(subset=["中国国债收益率10年"])
        .rename(columns={"中国国债收益率10年": "yield_10y"})
        .sort_values("日期")
        .reset_index(drop=True)
    )
    return result


if __name__ == "__main__":
    data = fetch_cn10y_daily()
    data.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved {len(data)} rows to {OUTPUT_CSV}")
    print(f"Range: {data['日期'].min().date()} ~ {data['日期'].max().date()}")
    print(data.tail())
