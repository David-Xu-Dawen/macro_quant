"""简易检索：项目文档 + JSON 摘要，关键词打分取 Top-K。"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOC_SOURCES = [
    ROOT / "README.md",
    ROOT / "macro_factor_corr.json",
    ROOT / "macro_hf_factor_corr.json",
    ROOT / "factor exposure" / "factor_exposure_latest.json",
    ROOT / "politics" / "hf_regression_results.json",
    ROOT / "mobility" / "hf_regression_results.json",
    ROOT / "growth" / "hf_regression_results.json",
    ROOT / "inflasion" / "hf_regression_results.json",
    ROOT / "credit" / "hf_regression_results.json",
    ROOT / "interest rate" / "hf_regression_results.json",
]


def _tokenize(text: str) -> list[str]:
    """中文用字级 bigram/trigram，英文按词；避免整句中文变成单一超长 token。"""
    text = text.lower()
    tokens: list[str] = []
    for eng in re.findall(r"[a-z0-9_\-]{2,}", text):
        tokens.append(eng)
    # 抽出连续中文段，再切成 2/3-gram
    for zh in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(zh) == 1:
            tokens.append(zh)
            continue
        for n in (2, 3):
            if len(zh) < n:
                continue
            for i in range(len(zh) - n + 1):
                tokens.append(zh[i : i + n])
    return tokens


def _chunk_markdown(text: str, source: str, max_chars: int = 700) -> list[dict]:
    blocks = re.split(r"\n(?=#{1,3}\s)", text)
    chunks = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            chunks.append({"source": source, "text": block})
            continue
        for i in range(0, len(block), max_chars):
            piece = block[i : i + max_chars].strip()
            if piece:
                chunks.append({"source": source, "text": piece})
    return chunks


def _summarize_corr_json(path: Path, title: str) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data.get("labels", [])
    corr = data.get("corr", [])
    pairs = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if j <= i:
                continue
            try:
                v = float(corr[i][j])
            except Exception:
                continue
            pairs.append((abs(v), a, b, v))
    pairs.sort(reverse=True)
    top = pairs[:8]
    lines = [
        f"{title}",
        f"区间: {data.get('start')} ~ {data.get('end')}",
        f"样本: {data.get('n_months') or data.get('n_weeks')} ",
        "最强相关对:",
    ]
    for _, a, b, v in top:
        lines.append(f"- {a} vs {b}: {v:+.2f}")
    return "\n".join(lines)


def _summarize_exposure_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        "因子暴露摘要",
        f"窗口: {data.get('window_start')} ~ {data.get('window_end')}",
        f"滚动窗口: {data.get('rolling_window_weeks')} 周",
        f"样本长度: {data.get('sample_length_weeks')} 周",
        f"Bootstrap: {data.get('bootstrap_samples')}",
        f"alpha_scale: {data.get('alpha_scale')}",
        f"因子: {', '.join(data.get('factors', []))}",
        "各资产 R方:",
    ]
    r2 = data.get("r_squared", {})
    for asset, value in r2.items():
        lines.append(f"- {asset}: {float(value):.3f}")

    matrix = data.get("matrix", {})
    notable = []
    for asset, row in matrix.items():
        for factor, coef in row.items():
            c = float(coef)
            if abs(c) >= 0.2:
                notable.append((abs(c), asset, factor, c))
    notable.sort(reverse=True)
    lines.append("较大暴露(|β|>=0.2):")
    for _, asset, factor, c in notable[:12]:
        lines.append(f"- {asset} × {factor}: {c:+.3f}")
    return "\n".join(lines)


def _summarize_regression_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    name = path.parent.name
    lines = [
        f"{name} 高频拟合回归结果",
        f"Y定义: {data.get('y_definition', '—')}",
        f"R²: {data.get('r_squared')}",
        f"Adj-R²: {data.get('adj_r_squared')}",
        f"样本数: {data.get('n_obs')}",
        f"资产: {', '.join(data.get('assets', []))}",
        f"权重: {json.dumps(data.get('weights', {}), ensure_ascii=False)}",
        f"滞后(月): {json.dumps(data.get('lags_months', {}), ensure_ascii=False)}",
    ]
    return "\n".join(lines)


def build_corpus() -> list[dict]:
    chunks: list[dict] = []
    for path in DOC_SOURCES:
        if not path.exists():
            continue
        rel = str(path.relative_to(ROOT))
        if path.suffix.lower() == ".md":
            chunks.extend(_chunk_markdown(path.read_text(encoding="utf-8"), rel))
        elif path.name == "macro_factor_corr.json":
            chunks.append({"source": rel, "text": _summarize_corr_json(path, "月频因子相关矩阵摘要")})
        elif path.name == "macro_hf_factor_corr.json":
            chunks.append({"source": rel, "text": _summarize_corr_json(path, "周频高频因子相关矩阵摘要")})
        elif path.name == "factor_exposure_latest.json":
            chunks.append({"source": rel, "text": _summarize_exposure_json(path)})
        elif path.name == "hf_regression_results.json":
            chunks.append({"source": rel, "text": _summarize_regression_json(path)})
    return chunks


_CORPUS: list[dict] | None = None


def get_corpus(refresh: bool = False) -> list[dict]:
    global _CORPUS
    if _CORPUS is None or refresh:
        _CORPUS = build_corpus()
    return _CORPUS


def retrieve(query: str, top_k: int = 4) -> list[dict]:
    tokens = set(_tokenize(query))
    if not tokens:
        return []
    q_lower = query.lower()
    scored = []
    for chunk in get_corpus():
        text = chunk["text"]
        text_l = text.lower()
        text_tokens = set(_tokenize(text))
        overlap = tokens & text_tokens
        if not overlap:
            continue
        # 更长 n-gram 权重更高；密度归一化，避免 README 长段落霸榜
        score = 0.0
        for t in overlap:
            w = 3.0 if len(t) >= 3 else 1.5
            score += w
            score += text_l.count(t) * 0.05
        score = score / max(1.0, (len(text) / 400) ** 0.5)

        src = chunk["source"]
        if "暴露" in q_lower and "exposure" in src:
            score += 8
        if ("相关" in q_lower or "矩阵" in q_lower) and "corr" in src:
            if "hf" in src and ("高频" in q_lower or "周" in q_lower):
                score += 8
            elif "hf" not in src and ("月" in q_lower or "低频" in q_lower or "高频" not in q_lower):
                score += 6
        if any(k in q_lower for k in ("地缘", "gpr", "黄金", "原油")) and "politics" in src:
            score += 10
        if any(k in q_lower for k in ("流动性", "社融", "m2")) and "mobility" in src:
            score += 10
        if any(k in q_lower for k in ("信用", "利差")) and "credit" in src:
            score += 8
        if any(k in q_lower for k in ("利率", "国债")) and "interest" in src:
            score += 8
        if any(k in q_lower for k in ("通胀", "商品")) and "inflasion" in src:
            score += 8
        if "readme" in src.lower() and any(k in q_lower for k in ("怎么", "如何", "构造", "方法", "定义")):
            score += 4

        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    # 先按来源去重，再补足 top_k
    out: list[dict] = []
    seen_src: set[str] = set()
    for _, chunk in scored:
        if chunk["source"] in seen_src:
            continue
        out.append(chunk)
        seen_src.add(chunk["source"])
        if len(out) >= top_k:
            return out
    for _, chunk in scored:
        if chunk in out:
            continue
        out.append(chunk)
        if len(out) >= top_k:
            break
    return out


def format_retrieval(chunks: list[dict]) -> str:
    if not chunks:
        return "（未检索到相关项目文档）"
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] 来源: {chunk['source']}\n{chunk['text']}")
    return "\n\n".join(parts)
