"""
ShipDetector — 基于 YOLO 的船只检测与跟踪

使用 ultralytics YOLO 原生追踪算法（ByteTrack），输出带 track ID 的检测框。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """单个检测结果。"""
    track_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    crop: np.ndarray | None = None   # 裁剪的图像区域


class ShipDetector:
    """
    YOLO 船只检测器（带原生跟踪）。

    使用 ultralytics YOLO 内置的 ByteTrack 追踪算法，
    为每个检测目标分配持久 track ID。
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "",
        conf_threshold: float = 0.25,
        tracker_type: str = "bytetrack",
        classes: list[int] | None = None,
    ):
        """
        Args:
            model_path: YOLO 模型路径或名称。若不存在会自动下载。
            device: 推理设备，"" 自动选择，"cpu" 强制 CPU，"0" 表示 GPU 0。
            conf_threshold: 检测置信度阈值。
            tracker_type: 追踪算法，"bytetrack" 或 "botsort"。
            classes: 只检测指定类别 ID 列表。None 表示检测所有类别。
                     COCO 中船的类别 ID 是 8。
        """
        from ultralytics import YOLO

        self._conf_threshold = conf_threshold
        self._tracker_type = tracker_type
        self._classes = classes

        logger.info("加载 YOLO 模型: %s (device=%s)", model_path, device or "auto")
        self._model = YOLO(model_path)
        self._device = device

        # 预热（忽略结果）
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.track(
                source=dummy,
                persist=True,
                tracker=f"{tracker_type}.yaml",
                verbose=False,
                device=device or None,
            )
        except Exception as e:
            logger.warning("YOLO 预热失败（不影响后续使用）: %s", e)

        logger.info("YOLO 模型加载完成，追踪器: %s", tracker_type)

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
    ) -> list[Detection]:
        """
        对单帧图像执行检测 + 跟踪。

        Args:
            frame: BGR 格式的图像帧 (H, W, 3)。
            frame_id: 帧编号（用于日志）。

        Returns:
            检测结果列表，每个包含 track_id, bbox, confidence, crop。
        """
        results = self._model.track(
            source=frame,
            persist=True,
            conf=self._conf_threshold,
            tracker=f"{self._tracker_type}.yaml",
            classes=self._classes,
            verbose=False,
            device=self._device or None,
        )

        detections: list[Detection] = []

        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes

        # 跟踪器未分配 ID 时，boxes.id 为 None
        if boxes.id is None:
            return detections

        for i in range(len(boxes)):
            # 获取 track ID
            track_id = int(boxes.id[i].item())

            # 获取 bbox (xyxy 格式)
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            # 确保 bbox 有效（x2 > x1, y2 > y1）
            if x2 <= x1 or y2 <= y1:
                continue

            # 获取置信度
            conf = float(boxes.conf[i].item())

            # 裁剪图像区域（加一点 padding）
            h, w = frame.shape[:2]
            pad = 10
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)
            crop = frame[cy1:cy2, cx1:cx2].copy()

            detections.append(Detection(
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                crop=crop,
            ))

        if detections:
            logger.debug(
                "帧 %d: 检测到 %d 个目标 (IDs: %s)",
                frame_id,
                len(detections),
                [d.track_id for d in detections],
            )

        return detections

    @property
    def model(self):
        return self._model
