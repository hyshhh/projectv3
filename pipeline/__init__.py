"""
Pipeline 模块 — 船弦号识别视频处理流水线

核心组件：
  - ShipDetector: YOLO 船只检测与跟踪
  - ShipHullAgent: LangGraph ReAct Agent 三步链路（识别→精确查找→语义检索）
  - TrackManager: 跟踪状态管理（track ID → 弦号绑定）
  - ShipPipeline: 主流水线编排（级联/并发双模式）
  - FPSMeter: 推理速度 FPS 统计
  - InputSource: 视频/相机/视频流输入
  - DemoRenderer: 可视化渲染
  - ScreenshotSaver: 截图保存
"""

from pipeline.detector import ShipDetector  # noqa: F401
from agent import ShipHullAgent, AgentResult  # noqa: F401
from pipeline.tracker import TrackManager  # noqa: F401
from pipeline.pipeline import ShipPipeline  # noqa: F401
from pipeline.fps import FPSMeter  # noqa: F401
from pipeline.video_input import InputSource  # noqa: F401
from pipeline.demo import DemoRenderer  # noqa: F401
from pipeline.output import ScreenshotSaver  # noqa: F401

__all__ = [
    "ShipDetector",
    "ShipHullAgent",
    "AgentResult",
    "TrackManager",
    "ShipPipeline",
    "FPSMeter",
    "InputSource",
    "DemoRenderer",
    "ScreenshotSaver",
]
