"""配置读取 — 唯一配置源 config.yaml，返回字典"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

# ── 内置默认值 ──────────────────────────────

_DEFAULTS: dict[str, Any] = {
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
        "dimensions": 1024,
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


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并：base 提供默认值，override 覆盖。"""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    """加载单个 YAML 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件 {path} 格式错误，期望顶层为字典")
    return data


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    加载配置，内置默认值 + YAML 覆盖。

    搜索顺序：
      1. 显式传入的路径
      2. 当前工作目录下的 config.yaml
      3. 项目根目录下的 config.yaml

    缺失字段自动使用默认值，无需 YAML 写全。
    """
    user_dict: dict[str, Any] = {}

    if config_path:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        user_dict = _load_yaml(p)
    else:
        candidates = [
            Path.cwd() / "config.yaml",
            _DEFAULT_CONFIG_PATH,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                try:
                    user_dict = _load_yaml(candidate)
                    logger.debug("已加载配置: %s", candidate)
                except yaml.YAMLError as e:
                    logger.error("配置文件 %s 解析失败: %s", candidate, e)
                    raise SystemExit(f"配置文件解析失败: {e}")
                break

    return _deep_merge(_DEFAULTS, user_dict)
