"""中债指数直连拉取（兼容新版 akshare 移除的 cbond 接口）。"""

from __future__ import annotations

import pandas as pd
import requests

INDICATOR_MAPPING = {
    "全价": "QJZS",
    "净价": "JJZS",
    "财富": "CFZS",
    "平均市值法久期": "PJSZFJQ",
    "平均现金流法久期": "PJXJLFJQ",
    "平均市值法凸性": "PJSZFTX",
    "平均现金流法凸性": "PJXJLFTX",
    "平均现金流法到期收益率": "PJDQSYL",
    "平均市值法到期收益率": "PJSZFDQSYL",
    "平均基点价值": "PJJDJZ",
    "平均待偿期": "PJDCQ",
    "平均派息率": "PJPXL",
    "指数上日总市值": "ZSZSZ",
    "财富指数涨跌幅": "CFZSZDF",
    "全价指数涨跌幅": "QJZSZDF",
    "净价指数涨跌幅": "JJZSZDF",
    "现券结算量": "XQJSL",
}

PERIOD_MAPPING = {
    "总值": "00",
    "1年以下": "01",
    "1-3年": "02",
    "3-5年": "03",
    "5-7年": "04",
    "7-10年": "05",
    "10年以上": "06",
    "0-3个月": "07",
    "3-6个月": "08",
    "6-9个月": "09",
    "9-12个月": "10",
    "0-6个月": "11",
    "6-12个月": "12",
}

# 常用指数 ID（来自中国债券信息网 / 旧版 akshare INDEX_MAPPING）
INDEX_MAPPING = {
    "国债总指数": "2c9081e50e8767dc010e879acb220021",
    "企业债总指数": "2c90818811d3f4fa01123837e6b30d4a",
    "企业债AAA指数": "8a8b2ca0408e9fa7014094a20f21001f",
    "新综合指数": "8a8b2ca0332abed20134ea76d8885831",
    "综合指数": "2c90818811afed8d0111c0c672b31578",
}

TREASURY_PERIOD_MAPPING = {
    "0-1Y": "8a8b2cef70bc61380170be069828032b",
    "0-3Y": "61f69682dc3ec18fe9664ff59308314a",
    "0-5Y": "0beafb51867009998c2f4932bf22ede3",
    "0-10Y": "8a8b2cef7832f8920178350801470014",
    "1-3Y": "cc1cfe89b0cbd0800420a0e037026407",
    "1-5Y": "7c3110e5305f9301482517066427a554",
    "1-10Y": "a5d90802e3259978a027267de651106d",
    "3-5Y": "8a8b2ca04bf69582014c10b60f376c77",
    "5Y": "8a8b2ca03a3feea1013a44b98fc533f5",
    "7Y": "2c9081e50e8767dc010e87b6e26c0080",
    "7-10Y": "8a8b2c8f5a492a01015a4ac986480043",
    "10Y": "8a8b2ca04b666362014b723482bc4f49",
    "30Y": "8a8b2cef77b239980177b485d20a6379",
}


def _parse_cbond_response(raw_json: dict, indicator: str) -> pd.DataFrame:
    indicator_code = INDICATOR_MAPPING[indicator]
    key_col_map = {
        f"{indicator_code}_{p_code}": freq_col
        for p_code, freq_col in raw_json["dqcName"].items()
    }
    data_json = {key: raw_json[key] for key in key_col_map}
    temp_df = pd.DataFrame.from_dict(data_json, orient="columns")
    temp_df.index = pd.to_datetime(
        pd.to_numeric(temp_df.index), unit="ms", utc=True
    ).tz_convert("Asia/Shanghai")
    temp_df = temp_df.reset_index()
    temp_df.columns = ["date", "value"]
    temp_df["date"] = pd.to_datetime(temp_df["date"], errors="coerce").dt.tz_localize(None)
    temp_df["value"] = pd.to_numeric(temp_df["value"], errors="coerce")
    return temp_df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)


def fetch_cbond_index(
    index_category: str = "国债总指数",
    indicator: str = "净价",
    period: str = "总值",
) -> pd.DataFrame:
    """通用中债指数查询，返回列 date / value。"""
    if index_category not in INDEX_MAPPING:
        raise KeyError(f"未知指数类别: {index_category}")
    if indicator not in INDICATOR_MAPPING:
        raise KeyError(f"未知指标: {indicator}")
    if period not in PERIOD_MAPPING:
        raise KeyError(f"未知期限: {period}")

    url = "https://yield.chinabond.com.cn/cbweb-mn/indices/singleIndexQueryResult"
    params = {
        "indexid": INDEX_MAPPING[index_category],
        "qxlxt": PERIOD_MAPPING[period],
        "ltcslx": "",
        "zslxt": INDICATOR_MAPPING[indicator],
        "zslxt1": INDICATOR_MAPPING[indicator],
        "lx": "1",
        "locale": "zh_CN",
    }
    response = requests.post(url, params=params, timeout=60)
    response.raise_for_status()
    return _parse_cbond_response(response.json(), indicator)


def fetch_treasury_index(
    indicator: str = "财富",
    period: str = "5Y",
) -> pd.DataFrame:
    """中债国债指数（按期限分段）。"""
    if period not in TREASURY_PERIOD_MAPPING:
        raise KeyError(f"未知国债期限: {period}")
    if indicator not in INDICATOR_MAPPING:
        raise KeyError(f"未知指标: {indicator}")

    url = "https://yield.chinabond.com.cn/cbweb-mn/indices/singleIndexQueryResult"
    params = {
        "indexid": TREASURY_PERIOD_MAPPING[period],
        "qxlxt": "00",
        "ltcslx": "",
        "zslxt": INDICATOR_MAPPING[indicator],
        "zslxt1": INDICATOR_MAPPING[indicator],
        "lx": "1",
        "locale": "zh_CN",
    }
    response = requests.post(url, params=params, timeout=60)
    response.raise_for_status()
    return _parse_cbond_response(response.json(), indicator)
