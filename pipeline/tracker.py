"""
TrackManager — 跟踪状态管理

维护 YOLO track ID 与弦号识别结果的映射关系。
一旦某个 track ID 完成识别，后续帧沿用该结果，无需重复调用 Agent。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrackInfo:
    """单个 track 的状态信息。"""
    track_id: int
    hull_number: str = ""
    description: str = ""
    recognized: bool = False       # 是否已完成识别
    pending: bool = False          # 是否正在异步识别中
    first_seen_frame: int = 0      # 首次出现的帧号
    last_seen_frame: int = 0       # 最近一次出现的帧号
    db_match_id: str = ""          # 数据库匹配到的弦号（确认后）
    db_match_desc: str = ""        # 数据库匹配到的描述
    db_matched: bool = False       # 是否在数据库中匹配到


class TrackManager:
    """
    管理所有活跃的跟踪目标。

    功能：
    - 注册新 track ID
    - 查询 track 是否已识别
    - 绑定识别结果到 track ID
    - 清理长时间未出现的 track
    """

    def __init__(self, max_stale_frames: int = 300):
        """
        Args:
            max_stale_frames: 超过此帧数未出现的 track 将被清理。
        """
        self._tracks: dict[int, TrackInfo] = {}
        self._max_stale_frames = max_stale_frames

    def get_or_create(self, track_id: int, frame_id: int) -> TrackInfo:
        """获取或创建 track 记录。"""
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackInfo(
                track_id=track_id,
                first_seen_frame=frame_id,
                last_seen_frame=frame_id,
            )
            logger.debug("新 track 注册: id=%d, frame=%d", track_id, frame_id)
        else:
            self._tracks[track_id].last_seen_frame = frame_id
        return self._tracks[track_id]

    def needs_recognition(self, track_id: int) -> bool:
        """
        判断该 track 是否需要进行 Agent 识别。

        已识别 → False
        正在异步识别中 → False
        其他 → True
        """
        if track_id not in self._tracks:
            return True
        info = self._tracks[track_id]
        return not info.recognized and not info.pending

    def mark_pending(self, track_id: int) -> None:
        """标记 track 正在异步识别中。"""
        if track_id in self._tracks:
            self._tracks[track_id].pending = True

    def bind_result(
        self,
        track_id: int,
        hull_number: str,
        description: str,
    ) -> None:
        """将识别结果绑定到 track ID。"""
        if track_id not in self._tracks:
            logger.warning("尝试绑定结果到不存在的 track: %d", track_id)
            return

        info = self._tracks[track_id]
        info.hull_number = hull_number
        info.description = description
        info.recognized = True
        info.pending = False

        logger.info(
            "Track %d 识别完成: 弦号=%s, 描述=%s",
            track_id,
            hull_number or "(未识别)",
            description[:50] if description else "",
        )

    def bind_db_match(
        self,
        track_id: int,
        db_match_id: str,
        db_match_desc: str,
    ) -> None:
        """绑定数据库匹配结果。"""
        if track_id in self._tracks:
            info = self._tracks[track_id]
            info.db_match_id = db_match_id
            info.db_match_desc = db_match_desc
            info.db_matched = True

    def get_display_text(self, track_id: int) -> str:
        """
        获取用于在画面上显示的文字。

        已识别 + 数据库匹配 → "(库内确定id：XXX)"
        已识别 + 未匹配 → "(未知id：XXX - 描述)"
        未识别 → "(识别中...)"
        """
        if track_id not in self._tracks:
            return "(等待识别...)"

        info = self._tracks[track_id]

        if not info.recognized:
            if info.pending:
                return "(识别中...)"
            return "(等待识别...)"

        if info.db_matched:
            return f"(库内确定id：{info.db_match_id})"

        # 未匹配数据库
        label = info.hull_number or "未知"
        desc_short = info.description[:20] if info.description else ""
        if desc_short:
            return f"(未知id：{label} - {desc_short})"
        return f"(未知id：{label})"

    def cleanup_stale(self, current_frame: int) -> int:
        """
        清理长时间未出现的 track。

        Returns:
            清理的数量。
        """
        stale_ids = [
            tid for tid, info in self._tracks.items()
            if current_frame - info.last_seen_frame > self._max_stale_frames
        ]
        for tid in stale_ids:
            del self._tracks[tid]

        if stale_ids:
            logger.info("清理 %d 个过期 track: %s", len(stale_ids), stale_ids)

        return len(stale_ids)

    @property
    def active_tracks(self) -> dict[int, TrackInfo]:
        return dict(self._tracks)

    def __len__(self) -> int:
        return len(self._tracks)
