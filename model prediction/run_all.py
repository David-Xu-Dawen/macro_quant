"""一键运行宏观配置框架（需已有 data/raw）。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script: str, *args: str):
    cmd = [sys.executable, str(ROOT / script), *args]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)


if __name__ == "__main__":
    # 若 raw 不齐可先拉数
    raw = ROOT / "data" / "raw"
    needed = [
        "sse50", "csi300", "csi500", "csi1000",
        "bond_gov", "bond_corp", "csi_cb",
        "crude_sc", "gold_au", "spx",
    ]
    if not all((raw / f"{a}.csv").exists() for a in needed):
        run("fetch_akshare_data.py")

    run("run_macro_strategy.py", "--forward-days", "20", "--label-mode", "ranking")
    run("plot_results.py")
    print("\nDone. See output/ and models/macro_lgbm_bl.joblib")
