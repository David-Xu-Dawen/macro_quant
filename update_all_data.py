#!/usr/bin/env python3
"""一键更新项目数据：宏观因子 → 资产/地缘合成 → 相关矩阵 → 因子暴露 → 波动模型 → 模型预测。

用法（在项目根目录）:
  python3 update_all_data.py
  python3 update_all_data.py --skip-model          # 跳过较慢的 LightGBM 回测
  python3 update_all_data.py --skip-vol            # 跳过波动预测训练
  python3 update_all_data.py --only factors,corr   # 只跑指定阶段
  python3 update_all_data.py --continue-on-error   # 某步失败后继续

阶段:
  factors   各宏观因子管线（含 GPR 拉取；不含地缘合成）
  assets    资产价格拉取 + 地缘合成（地缘依赖金/油价格）
  corr      月/周频相关矩阵 JSON
  exposure  LASSO 因子暴露
  vol       周频高波动预测模型
  model     LightGBM + BL 模型预测回测
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

STAGES = ("factors", "assets", "corr", "exposure", "vol", "model")


@dataclass
class Step:
    name: str
    cwd: Path
    argv: list[str]
    stage: str


def _step(name: str, rel_dir: str, script: str, *args: str, stage: str) -> Step:
    cwd = ROOT / rel_dir if rel_dir else ROOT
    return Step(
        name=name,
        cwd=cwd,
        argv=[PYTHON, script, *args],
        stage=stage,
    )


def build_steps(args: argparse.Namespace) -> list[Step]:
    exposure_args = [
        "--bootstrap",
        str(args.bootstrap),
        "--alpha-scale",
        str(args.alpha_scale),
        "--rolling-window-weeks",
        str(args.rolling_window_weeks),
        "--sample-length-weeks",
        str(args.sample_length_weeks),
    ]
    return [
        _step("增长因子", "growth", "run_growth_pipeline.py", stage="factors"),
        _step("通胀因子", "inflasion", "update_all.py", stage="factors"),
        _step("利率因子", "interest rate", "update_all.py", stage="factors"),
        _step("信用因子", "credit", "update_all.py", stage="factors"),
        _step("汇率因子", "exchange", "fetch_dxy.py", stage="factors"),
        _step("地缘 GPR", "politics", "fetch_gpr.py", stage="factors"),
        _step("流动性因子", "mobility", "run_mobility_pipeline.py", stage="factors"),
        # 地缘合成依赖沪金/原油，挂在 assets 阶段且紧跟价格拉取
        _step(
            "资产价格",
            "factor exposure",
            "fetch_timeseries.py",
            "--start",
            args.asset_start,
            stage="assets",
        ),
        _step("地缘合成", "politics", "run_geo_pipeline.py", stage="assets"),
        _step("月频相关矩阵", "", "plot_macro_factor_corr.py", stage="corr"),
        _step("周频相关矩阵", "", "plot_macro_hf_corr.py", stage="corr"),
        _step(
            "因子暴露",
            "factor exposure",
            "compute_factor_exposure.py",
            *exposure_args,
            stage="exposure",
        ),
        _step("波动预测训练", "web", "train_vol_forecast.py", stage="vol"),
        _step("模型预测回测", "model prediction", "run_all.py", stage="model"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键更新 macro_quant 全部数据")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help=f"只跑指定阶段，逗号分隔。可选: {','.join(STAGES)}",
    )
    parser.add_argument("--skip-factors", action="store_true")
    parser.add_argument("--skip-assets", action="store_true")
    parser.add_argument("--skip-corr", action="store_true")
    parser.add_argument("--skip-exposure", action="store_true")
    parser.add_argument("--skip-vol", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="某一步失败后继续后续步骤",
    )
    parser.add_argument("--asset-start", default="2018-01-01")
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--alpha-scale", type=float, default=0.5)
    parser.add_argument("--rolling-window-weeks", type=int, default=412)
    parser.add_argument("--sample-length-weeks", type=int, default=104)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的命令，不真正运行",
    )
    return parser.parse_args()


def selected_stages(args: argparse.Namespace) -> set[str]:
    if args.only.strip():
        stages = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = stages - set(STAGES)
        if unknown:
            raise SystemExit(f"未知阶段: {', '.join(sorted(unknown))}")
        return stages

    stages = set(STAGES)
    if args.skip_factors:
        stages.discard("factors")
    if args.skip_assets:
        stages.discard("assets")
    if args.skip_corr:
        stages.discard("corr")
    if args.skip_exposure:
        stages.discard("exposure")
    if args.skip_vol:
        stages.discard("vol")
    if args.skip_model:
        stages.discard("model")
    return stages


def run_step(step: Step, *, dry_run: bool) -> None:
    script_path = step.cwd / step.argv[1]
    cmd = " ".join(step.argv)
    print(f"\n=== [{step.stage}] {step.name} ===")
    print(f"$ cd {step.cwd}")
    print(f"$ {cmd}")
    if dry_run:
        return
    if not script_path.exists():
        raise FileNotFoundError(f"脚本不存在: {script_path}")
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / "interest rate" / ".mplconfig"))
    (ROOT / "interest rate" / ".mplconfig").mkdir(parents=True, exist_ok=True)
    subprocess.check_call(step.argv, cwd=step.cwd, env=env)


def main() -> int:
    args = parse_args()
    stages = selected_stages(args)
    steps = [s for s in build_steps(args) if s.stage in stages]
    if not steps:
        print("没有可执行的步骤。")
        return 1

    print(f"将执行 {len(steps)} 步，阶段: {', '.join(s for s in STAGES if s in stages)}")
    started = time.perf_counter()
    failures: list[str] = []

    for step in steps:
        try:
            run_step(step, dry_run=args.dry_run)
        except Exception as exc:
            msg = f"{step.name}: {exc}"
            failures.append(msg)
            print(f"\n!! 失败: {msg}")
            if not args.continue_on_error:
                print("已中止。可用 --continue-on-error 继续后续步骤。")
                return 1

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 60)
    if args.dry_run:
        print("Dry-run 完成，未实际执行。")
    elif failures:
        print(f"完成（有失败 {len(failures)} 步），耗时 {elapsed:.1f}s")
        for item in failures:
            print(f"  - {item}")
        return 1
    else:
        print(f"全部完成，耗时 {elapsed:.1f}s")
        print("刷新浏览器（Cmd+Shift+R）即可看到最新数据；Web 服务一般无需重启。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
