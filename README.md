# macro_quant

宏观量化因子研究项目：构建增长、通胀、利率、信用、汇率、地缘、流动性等宏观因子，计算因子相关性矩阵与资产因子暴露，并通过 Web 页面交互展示。

## 功能概览

Web 页面（`macro_factor_corr_interactive.html`）包含五个标签页：

| 标签页 | 内容 |
|--------|------|
| **因子矩阵** | 6 因子月频低频相关矩阵 + 周频高频相关矩阵（可拖动时间窗口） |
| **因子暴露** | 12 类资产对 7 个宏观因子的 LASSO + Bootstrap 暴露系数 |
| **模型预测** | LightGBM + BL 大类配置回测结果与图表（`model prediction/`） |
| **Agent 辩论** | 4 位专家多轮讨论 + RAG/工具 + 总协调人投研报告 |
| **AI 助手** | 基于本地 Ollama 的宏观量化问答（可选） |

浏览器访问：http://127.0.0.1:8765/

---

## 项目结构

```
macro_quant/
├── macro_factor_corr_interactive.html   # 主页面
├── plot_macro_factor_corr.py            # 月频相关矩阵
├── plot_macro_hf_corr.py                # 周频相关矩阵
├── macro_factor_monthly.csv             # 月频因子面板
├── macro_hf_factor_weekly.csv           # 周频因子面板
├── growth/                              # 增长因子
├── inflasion/                           # 通胀因子
├── interest rate/                       # 利率因子
├── credit/                              # 信用因子
├── exchange/                            # 汇率因子（美元指数）
├── politics/                            # 地缘因子（GPR）
├── mobility/                            # 流动性因子（M2-社融）
├── factor exposure/                     # 因子暴露计算
│   ├── fetch_timeseries.py              # 拉取资产价格
│   ├── compute_factor_exposure.py       # LASSO 暴露估计
│   └── data/combined_close.csv          # 资产收盘价
└── web/
    ├── app.py                           # FastAPI 服务
    └── requirements.txt
```

---

## 宏观因子定义

### 低频（月频，用于左侧热力图，6 因子）

| 因子 | 数据源 | 定义 |
|------|--------|------|
| 增长 | `growth/growth_factor.csv` | 宏观增长因子 |
| 通胀 | `inflasion/inflation_factor.csv` | CPI/PPI 合成通胀因子 |
| 利率 | `interest rate/rate_factor.csv` | **十年国债收益率绝对水平（%）** |
| 信用 | `credit/credit_factor.csv` | 3Y AA 中短票收益率 − 3Y 国开债收益率（利差水平，%） |
| 汇率 | `exchange/dxy_yahoo.csv` | **美元指数（DXY）月末绝对水平** |
| 地缘 | `politics/hf_geo_factor_synthetic.csv` | **沪金+布伦特原油绝对价格线性拟合**（月末值） |

### 高频（周频，用于右侧热力图，6 因子）

| 因子 | 数据源 | 定义 |
|------|--------|------|
| 增长 | `growth/hf_growth_factor_synthetic.csv` | `hf_yoy`，周末值 |
| 通胀 | `inflasion/hf_inflation_weekly.csv` | `hf_yoy_pct`，周末值 |
| 利率 | `interest rate/hf_rate_factor_daily.csv` | **中债国债总净价指数相反数**（`-index_net`），周末值 |
| 信用 | `credit/hf_credit_factor_daily.csv` | AA财富−国开财富 → HP去趋势 → 取反（指数形态 `hf_credit_factor`）；热力图用水平周末值，暴露用 `hf_mom_pct` 周求和 |
| 汇率 | `exchange/dxy_yahoo.csv` | **美元指数（DXY）周末绝对水平** |
| 地缘 | `politics/hf_geo_factor_synthetic.csv` | **沪金+布伦特原油绝对价格线性拟合的高频地缘因子**（周末值） |

### 因子暴露（周频回归自变量，7 因子）

暴露矩阵使用各因子的**高频环比序列**（周度聚合）：

- 增长 / 信用 / 利率：`hf_mom_pct` 周度求和
- 通胀：`hf_wow` 周末值
- 汇率：周度对数收益
- 地缘：黄金+原油合成高频因子的周度对数收益
- 流动性：`hf_mom_pct` 周度求和

---

## 环境安装

```bash
# Web 服务
cd web && pip install -r requirements.txt

# 因子暴露（拉取资产数据）
cd "../factor exposure" && pip install -r requirements.txt

# 各因子目录依赖（akshare、statsmodels、matplotlib 等）
pip install akshare pandas numpy statsmodels matplotlib scikit-learn yfinance
```

可选：本地 AI 助手需安装 [Ollama](https://ollama.com/) 并拉取模型（默认 `mistral`）。

---

## 数据更新流程

在项目根目录执行：

```bash
cd macro_quant
export MPLCONFIGDIR="$(pwd)/interest rate/.mplconfig"
```

### 1. 更新宏观因子

```bash
cd growth && python3 run_growth_pipeline.py && cd ..
cd inflasion && python3 update_all.py && cd ..
cd "interest rate" && python3 update_all.py && cd ..
cd credit && python3 update_all.py && cd ..
cd exchange && python3 fetch_dxy.py && cd ..
cd politics && python3 fetch_gpr.py && cd ..
cd mobility && python3 run_mobility_pipeline.py && cd ..
# 流动性默认联网：M2=东方财富，PE=乐咕乐股(沪深300/中证1000代理)；
# 社融存量同比仍读 mobility/中国_M2_同比.csv（Wind），需手工更新该列时请重导 Wind。
```

### 2. 更新资产价格，并合成地缘因子

```bash
cd "factor exposure"
python3 fetch_timeseries.py --start 2018-01-01
cd ../politics && python3 run_geo_pipeline.py && cd ..
```

### 3. 生成热力图 JSON

```bash
cd ..
python3 plot_macro_factor_corr.py    # 月频矩阵 → macro_factor_corr.json
python3 plot_macro_hf_corr.py        # 周频矩阵 → macro_hf_factor_corr.json
```

### 4. 生成因子暴露

```bash
cd "factor exposure"
python3 compute_factor_exposure.py \
  --bootstrap 3000 \
  --alpha-scale 0.5 \
  --rolling-window-weeks 315 \
  --sample-length-weeks 104
```

> `--rolling-window-weeks` 不要超过宏观高频共同样本周数；脚本在样本不足时会自动下调。Web 暴露页会读取 `factor_exposure_latest.json` 中的实际窗口，不再写死「3 年 / 413 周」。

### 一键更新

```bash
cd "/Users/xdw/Desktop/macro_quant"
python3 update_all_data.py
```

常用选项：

```bash
python3 update_all_data.py --skip-model          # 跳过较慢的模型预测回测
python3 update_all_data.py --skip-vol            # 跳过波动预测训练
python3 update_all_data.py --only factors,corr   # 只更新因子与相关矩阵
python3 update_all_data.py --continue-on-error   # 某步失败后继续
python3 update_all_data.py --dry-run             # 只打印命令
```

脚本会依次更新：宏观因子 → 资产价格与地缘合成 → 月/周相关矩阵 → 因子暴露 → 波动预测模型 → 模型预测回测。

#### 更新后，网页数据会不会变？

| 环境 | 会不会自动更新 | 怎么做 |
|------|----------------|--------|
| 本地 FastAPI（`web/app.py`） | 会 | 跑完后浏览器强刷即可（`Cmd + Shift + R`） |
| 本地 Streamlit | 会 | 跑完后刷新页面；若仍是旧数据，重启一次 `streamlit run streamlit_app.py` |
| Streamlit Cloud 线上站 | **不会** | 只读 GitHub 仓库；需把新数据 commit + push，Cloud 才会重新部署 |

线上网站同步示例：

```bash
cd "/Users/xdw/Desktop/macro_quant"

# 1) 本地更新全部数据
python3 update_all_data.py

# 2) 查看哪些数据文件变了
git status

# 3) 提交常见产出（按实际变更增删）
git add \
  macro_factor_monthly.csv macro_factor_corr.json macro_hf_factor_corr.json \
  macro_hf_factor_weekly.csv \
  "factor exposure/factor_exposure_latest.json" \
  "factor exposure/factor_exposure_latest.csv" \
  "factor exposure/data/combined_close.csv" \
  "model prediction/output" \
  "model prediction/models" \
  growth/ inflasion/ "interest rate"/ credit/ exchange/ politics/ mobility/ web/

git commit -m "$(cat <<'EOF'
Update project data for Streamlit Cloud.

Refresh factor panels, correlation matrices, exposure, and model outputs after update_all_data.
EOF
)"

# 4) 推到 GitHub → Streamlit Cloud 自动重新部署
git push origin main
```

说明：

- `update_all_data.py` **只改本机文件**，不会直接改线上网站。
- 线上站点要更新，关键是 **push 到 GitHub**；没有 push，Cloud 仍显示旧数据。
- 若只想更新矩阵/暴露、不重跑模型：`python3 update_all_data.py --skip-model`，再按上面步骤提交对应 JSON/CSV。

> 流动性在 `factors` 阶段会自动联网更新：M2（东方财富）+ 市盈率代理（乐咕乐股沪深300/中证1000）。社融存量同比仍依赖 `mobility/中国_M2_同比.csv`，公开源不稳定时需偶尔重导 Wind。

> 地缘合成依赖沪金/原油价格，已放在资产拉取之后（`assets` 阶段）。

模型预测默认使用 GBDT 排序/回归融合、动态 BL 观点置信度与 CVaR 优化；CVaR
失败时自动回退到均值方差/风险平价。可用
`model prediction/run_macro_strategy.py --experiment-name <name>` 独立保存 A/B
实验，并通过 `--boosting-type`、`--confidence-mode`、`--optimizer-mode` 切换组件。

Web「模型预测」页可切换三档激进程度（稳健/均衡/进取），只改组合权重上限与风险厌恶，不改 LightGBM 信号。非均衡档由 OOF 无重训快速生成：

```bash
cd "model prediction" && python3 run_aggression_profiles.py
```

若网络拉取失败，可在对应 `update_all.py` 中设 `UPDATE_DATA = False`，用本地已有 CSV 继续跑后续步骤。

### 手动分步更新（可选）

```bash
cd "/Users/xdw/Desktop/macro_quant"
export MPLCONFIGDIR="$(pwd)/interest rate/.mplconfig"

cd growth && python3 run_growth_pipeline.py && cd ..
cd inflasion && python3 update_all.py && cd ..
cd "interest rate" && python3 update_all.py && cd ..
cd credit && python3 update_all.py && cd ..
cd exchange && python3 fetch_dxy.py && cd ..
cd politics && python3 fetch_gpr.py && cd ..
cd mobility && python3 run_mobility_pipeline.py && cd ..

cd "factor exposure"
python3 fetch_timeseries.py --start 2018-01-01
cd ../politics && python3 run_geo_pipeline.py && cd ..

python3 plot_macro_factor_corr.py
python3 plot_macro_hf_corr.py

cd "factor exposure"
python3 compute_factor_exposure.py --bootstrap 3000 --alpha-scale 0.5 --rolling-window-weeks 315 --sample-length-weeks 104
cd ..

cd web && python3 train_vol_forecast.py && cd ..
cd "model prediction" && python3 run_all.py && cd ..
```
---

## 启动 Web 服务

```bash
# 首次：拉取中文默认模型（约 4.7GB）
ollama pull qwen2.5:7b

lsof -ti :8765 | xargs kill 2>/dev/null
cd "/Users/xdw/Desktop/macro_quant/web"
python3 app.py
```

打开 http://127.0.0.1:8765/ ，更新数据后 **强刷**（`Cmd + Shift + R`）即可，无需重启服务。

环境变量（可选）：`OLLAMA_HOST`、`OLLAMA_MODEL`（默认 `qwen2.5:7b`）。

### Streamlit 公开网站（无需 Render）

Streamlit Community Cloud 的 **Main file path** 填：

```text
streamlit_app.py
```

本地预览：

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

这是纯 Streamlit 版：直接调用 `web/` 下同一套数据模块，**不依赖 FastAPI / Render / 绑卡**。  
包含：因子矩阵、波动警报、因子暴露、模型预测。AI 助手与 Agent 辩论仍依赖本机 Ollama，云端暂不开放。

之后：本地修改 → `git push` → GitHub → Streamlit Cloud 自动重新部署。

数据更新注意：本地执行 `python3 update_all_data.py` **不会**自动更新线上站；需把更新后的数据文件 commit 并 `git push origin main`。完整指令见上文「数据更新流程 → 更新后，网页数据会不会变？」。

### 聊天助手能力

1. **实时上下文**：每次提问注入当前月频/周频相关矩阵摘要 + 最新因子暴露摘要  
2. **中文本地模型**：默认 Ollama `qwen2.5:7b`  
3. **简易检索**：对 `README.md` 与各模块 JSON 摘要做关键词 Top-K 检索  
4. **工具调用**：模型可调用 `get_corr_matrix` / `get_hf_corr_matrix` / `get_factor_exposure`

### 多 Agent 辩论系统

`web/debate/` 使用 LangGraph 编排 4 位专家与 1 位总协调人：

1. 宏观基本面专家
2. 技术量价专家
3. 情绪风险专家
4. 风控对冲专家
5. 总协调人（汇总完整辩论并生成《大类资产配置策略报告》）

每轮按上述顺序发言，后发言者可见此前全部记录；默认 2 轮，最多 3 轮。专家可调用
月/周频相关矩阵、因子暴露、波动监控、模型回测摘要和项目 RAG 检索工具。本地模型默认
使用 Ollama `qwen2.5:7b`。

命令行运行：

```bash
cd web
pip install -r requirements.txt
python -m debate.cli \
  --topic "当前增长、通胀与流动性组合下应如何制定大类资产配置框架？" \
  --asset-focus "沪深300、中债国债、沪金" \
  --rounds 2 \
  --output debate_report.md \
  --json debate_trace.json
```

只检查 RAG 和实时上下文、不调用 Ollama：

```bash
python -m debate.cli --topic "信用收缩风险" --dry-run
```

HTTP 调用：

```bash
curl -X POST http://127.0.0.1:8765/api/debate \
  -H 'Content-Type: application/json' \
  -d '{
    "topic": "当前宏观环境下如何制定大类资产配置与对冲框架？",
    "asset_focus": "权益、国债、黄金",
    "rounds": 2,
    "model": "qwen2.5:7b",
    "use_retrieval": true,
    "use_tools": true,
    "use_live_context": true
  }'
```

> 注意：`model prediction` 的 OOF 标签/权重只用于模型评估，不能当作实时交易信号。
> 多 Agent 输出是研究框架，不是投资建议。

### 常见问题

**端口 8765 已被占用**

```
ERROR: [Errno 48] address already in use
```

说明服务已在运行，直接刷新浏览器即可。如需重启：

```bash
lsof -ti :8765 | xargs kill
cd web && python3 app.py

### Streamlit 入口（保留原界面）

Main file path 请填：

```text
streamlit_app.py
```

本地：

```bash
streamlit run streamlit_app.py
```

Streamlit Community Cloud 需在 Secrets 配置 Render 后端地址（Cloud 无法单独暴露 FastAPI）：

```toml
BACKEND_URL = "https://你的-render-服务.onrender.com"
```

工作流：本地修改 → `git push` → GitHub → Render（主站）自动更新；Streamlit 只是全屏嵌入同一界面。
```

---

## 因子暴露方法论

- **因变量 Y**：12 类资产周度对数收益率（`combined_close.csv`）
- **自变量 X**：宏观因子高频环比序列
- **模型**：标准化 LASSO + 连续块 Bootstrap（默认 3000 次），取系数中位数
- **先验约束**：信用因子仅对债券类资产（中债国债、中债企业债、中证转债）估计，其余资产信用暴露置 0
- **默认参数**：`alpha_scale=0.5`，`rolling_window` 以共同样本为准（当前约 315 周，脚本会自动下调），`sample_length=104` 周

### 覆盖资产

上证50、沪深300、中证500、中证1000、恒生指数、中债国债、中债企业债、中证转债、布伦特原油、沪金、标普500、美元兑人民币

---

## API 接口

| 路径 | 说明 |
|------|------|
| `GET /` | 主页面 |
| `GET /macro_factor_corr.json` | 月频相关矩阵（默认区间） |
| `GET /api/corr?start=YYYY-MM&end=YYYY-MM` | 月频相关矩阵（自定义区间） |
| `GET /macro_hf_factor_corr.json` | 周频相关矩阵（默认区间） |
| `GET /api/hf-corr?start=YYYY-MM-DD&end=YYYY-MM-DD` | 周频相关矩阵（自定义区间） |
| `GET /factor_exposure_latest.json` | 最新因子暴露矩阵 |
| `GET /api/vol-monitor?window=13&shock_z=2` | 周频因子波动分位 + 本周冲击 + 暴露压力 + 树模型4周预测 |
| `GET /api/vol-forecast?factor=综合` | 单因子/综合：未来4周高波动概率（factor=增长因子/... 或 all） |
| `GET /api/model-prediction` | model prediction 回测指标 / 最新权重 / 图表列表 |
| `GET /model-prediction/figures/{name}.png` | 回测诊断图静态资源 |
| `GET /api/corr-pair-compare?factor_a=&factor_b=&start=&end=` | 高低频因子对差异拆解（序列/滚动相关/月末抽样） |
| `POST /api/chat` | Ollama 对话（上下文注入 + 检索 + 工具） |
| `POST /api/debate` | LangGraph：4 专家多轮辩论 + 总协调人报告 |
| `GET /api/retrieve?q=...` | 简易文档/JSON 检索调试 |
| `GET /api/health` | 服务与 Ollama 状态 |
| `GET /api/factor-exposure?end=YYYY-MM-DD` | 按结束周计算暴露 |
| `GET /api/factor-exposure/weeks` | 可选结束周列表 |

---

## 主要输出文件

| 文件 | 说明 |
|------|------|
| `macro_factor_corr.json` | 月频热力图数据 |
| `macro_hf_factor_corr.json` | 周频热力图数据 |
| `factor exposure/factor_exposure_latest.json` | 因子暴露表 |
| `interest rate/rate_factor_comparison.png` | 利率因子：收益率 vs 净价指数相反数 |
| `interest rate/cn10y_rate_factor.png` | 低频利率因子走势图 |
