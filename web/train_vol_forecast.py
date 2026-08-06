#!/usr/bin/env python3
"""训练按因子的未来4周高波动预测模型。

用法:
  cd web && python3 train_vol_forecast.py
"""

from __future__ import annotations

import json

from vol_forecast import MODEL_PATH, META_PATH, train_model


def main() -> None:
    meta = train_model(save=True)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved meta  -> {META_PATH}")


if __name__ == "__main__":
    main()
