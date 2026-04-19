"""
Pipeline 模块 — 船弦号识别视频处理流水线

核心组件：
  - ShipDetector: YOLO 船只检测与跟踪
  - AgentInference: Qwen3.5 VLM 弦号识别
  - TrackManager: 跟踪状态管理（track ID → 弦号绑定）
  - ShipPipeline: 主流水线编排（级联/并发双模式）
  - FPSMeter: 推理速度 FPS 统计
  - InputSource: 视频/相机/视频流输入
"""

from pipeline.detector import ShipDetector
from pipeline.agent_inference import AgentInference
from pipeline.tracker import TrackManager
from pipeline.pipeline import ShipPipeline
from pipeline.fps import FPSMeter

__all__ = [
    "ShipDetector",
    "AgentInference",
    "TrackManager",
    "ShipPipeline",
    "FPSMeter",
]
