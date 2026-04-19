"""
FPSMeter — 10秒滑动窗口 FPS 统计

在终端和日志中定期打印码流帧率和处理帧率。
"""

from __future__ import annotations

import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


class FPSMeter:
    """
    基于滑动窗口的 FPS 计算器。

    支持多个独立计数通道（如 "stream" 和 "process"）。
    """

    def __init__(self, window_seconds: float = 10.0):
        """
        Args:
            window_seconds: 滑动窗口时长（秒）。
        """
        self._window = window_seconds
        self._timestamps: dict[str, deque[float]] = {}
        self._last_print: dict[str, float] = {}
        self._print_interval = 5.0  # 每 5 秒打印一次

    def tick(self, channel: str = "default") -> None:
        """记录一次帧到达。"""
        now = time.monotonic()

        if channel not in self._timestamps:
            self._timestamps[channel] = deque()
            self._last_print[channel] = 0.0

        self._timestamps[channel].append(now)

        # 清理窗口外的旧数据
        cutoff = now - self._window
        while self._timestamps[channel] and self._timestamps[channel][0] < cutoff:
            self._timestamps[channel].popleft()

    def get_fps(self, channel: str = "default") -> float:
        """获取当前 FPS。"""
        if channel not in self._timestamps:
            return 0.0

        timestamps = self._timestamps[channel]
        if len(timestamps) < 2:
            return 0.0

        now = time.monotonic()
        cutoff = now - self._window
        # 清理旧数据
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

        if len(timestamps) < 2:
            return 0.0

        elapsed = timestamps[-1] - timestamps[0]
        if elapsed <= 0:
            return 0.0

        return (len(timestamps) - 1) / elapsed

    def should_print(self, channel: str = "default") -> bool:
        """判断是否应该打印 FPS（基于打印间隔）。"""
        now = time.monotonic()
        if channel not in self._last_print:
            self._last_print[channel] = now
            return False

        if now - self._last_print[channel] >= self._print_interval:
            self._last_print[channel] = now
            return True
        return False

    def print_fps(self, channel: str = "default", extra: str = "") -> str:
        """
        格式化打印 FPS 信息。

        Returns:
            格式化后的 FPS 字符串。
        """
        fps = self.get_fps(channel)
        parts = [f"[{channel}] FPS: {fps:.1f}"]
        if extra:
            parts.append(extra)
        msg = " | ".join(parts)
        logger.info(msg)
        return msg

    def get_all_fps(self) -> dict[str, float]:
        """获取所有通道的 FPS。"""
        return {ch: self.get_fps(ch) for ch in self._timestamps}

    def reset(self, channel: str | None = None) -> None:
        """重置计数器。"""
        if channel:
            self._timestamps.pop(channel, None)
            self._last_print.pop(channel, None)
        else:
            self._timestamps.clear()
            self._last_print.clear()
