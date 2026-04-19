"""配置读取 — 直接加载 config.yaml，返回字典"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    加载 config.yaml，返回字典。
    优先级：显式传入 > 当前目录 > 项目根目录 > 内置默认值。
    """
    candidates: list[Path] = []

    if config_path:
        candidates.append(Path(config_path))

    candidates.append(Path.cwd() / "config.yaml")
    candidates.append(_DEFAULT_CONFIG_PATH)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if data is None:
                    continue
                if not isinstance(data, dict):
                    logger.warning("配置文件 %s 格式错误，期望顶层为字典", candidate)
                    continue
                return data
            except yaml.YAMLError as e:
                logger.error("配置文件 %s 解析失败: %s", candidate, e)
                raise SystemExit(f"配置文件解析失败: {e}")

    # 兜底默认值
    return {
        "llm": {
            "model": "Qwen/Qwen3-VL-4B-AWQ",
            "api_key": "abc123",
            "base_url": "http://localhost:7890/v1",
            "temperature": 0.0,
        },
        "embed": {
            "model": "text-embedding-v4",
            "api_key": "",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        "retrieval": {
            "top_k": 3,
            "score_threshold": 0.5,
        },
        "vector_store": {
            "persist_path": "./vector_store",
            "auto_rebuild": False,
        },
        "app": {
            "log_level": "INFO",
            "ship_db_path": "./data/ships.csv",
        },
    }
