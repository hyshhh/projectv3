"""
ShipPipeline — 主流水线编排

级联模式（concurrent_mode=false）：
  YOLO 检测 → 等待 Agent 返回 → 绑定结果 → 绘制输出

并发模式（concurrent_mode=true）：
  YOLO 检测 → crop 送入队列 → Agent 独立线程异步推理
  → 结果按帧时间戳严格顺序出队 → 匹配到对应帧绘制输出

双层并发架构：
  外层：帧级任务队列（max_queued_frames 限制深度，防 OOM）
  内层：crop 级 API 并发（max_concurrent 控制）
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np

from pipeline.agent_inference import AgentInference, InferenceResult
from pipeline.detector import ShipDetector, Detection
from pipeline.demo import DemoRenderer
from pipeline.output import ScreenshotSaver
from pipeline.fps import FPSMeter
from pipeline.tracker import TrackManager
from pipeline.video_input import InputSource

logger = logging.getLogger(__name__)


class ShipPipeline:
    """
    船弦号识别视频处理流水线。

    整合 YOLO 检测、Agent 推理、跟踪管理，支持级联/并发双模式。
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Args:
            config: 全局配置字典。None 则从 config.yaml 加载。
        """
        if config is None:
            from config import load_config
            config = load_config()

        self._config = config

        # 读取 pipeline 相关配置
        pipe_cfg = config.get("pipeline", {})
        self._concurrent_mode: bool = pipe_cfg.get("concurrent_mode", False)
        self._max_concurrent: int = pipe_cfg.get("max_concurrent", 4)
        self._max_queued_frames: int = pipe_cfg.get("max_queued_frames", 30)
        self._process_every_n: int = max(1, pipe_cfg.get("process_every_n_frames", 1))
        self._prompt_mode: str = pipe_cfg.get("prompt_mode", "detailed")
        self._demo_enabled: bool = pipe_cfg.get("demo", False)

        # 读取 Agent 数据库配置
        from database import ShipDatabase
        self._db = ShipDatabase(config=config)

        # 初始化组件
        self._detector = ShipDetector(
            model_path=pipe_cfg.get("yolo_model", "yolov8n.pt"),
            device=pipe_cfg.get("device", ""),
            conf_threshold=pipe_cfg.get("conf_threshold", 0.25),
            tracker_type=pipe_cfg.get("tracker", "bytetrack"),
            tracker_params=pipe_cfg.get("tracker_params"),
            classes=pipe_cfg.get("detect_classes", [8]),  # COCO: 8=boat
        )

        self._agent = AgentInference(
            config=config,
            prompt_mode=self._prompt_mode,
            max_concurrent=self._max_concurrent,
        )

        self._tracker = TrackManager(
            max_stale_frames=pipe_cfg.get("max_stale_frames", 300),
        )

        self._fps = FPSMeter(window_seconds=10.0)

        # Demo 渲染器
        self._renderer = DemoRenderer(
            show_fps=True,
            show_track_id=True,
        )

        # 截图保存器
        output_dir = pipe_cfg.get("output_dir", "./output")
        self._saver = ScreenshotSaver(output_dir=output_dir)

        # 并发模式相关
        self._task_queue: queue.Queue = queue.Queue(
            maxsize=self._max_queued_frames
        )
        self._result_queue: queue.Queue = queue.Queue()
        self._agent_workers: list[threading.Thread] = []
        self._stop_event = threading.Event()

        # Agent 运行链路日志（限制最大条数防内存泄漏）
        self._agent_trace: list[dict[str, Any]] = []
        self._trace_lock = threading.Lock()
        self._max_trace_entries = 500

        logger.info(
            "ShipPipeline 初始化: mode=%s, max_concurrent=%d, max_queued=%d, process_every=%d, prompt=%s",
            "concurrent" if self._concurrent_mode else "cascade",
            self._max_concurrent,
            self._max_queued_frames,
            self._process_every_n,
            self._prompt_mode,
        )

    # ── Agent 链路日志 ──────────────────────────

    def _log_agent_trace(
        self,
        event_type: str,
        track_id: int,
        frame_id: int,
        content: str = "",
        **extra: Any,
    ) -> None:
        """记录 Agent 运行链路到 trace 日志。"""
        entry = {
            "type": event_type,
            "track_id": track_id,
            "frame_id": frame_id,
            "content": content,
            "timestamp": time.time(),
            **extra,
        }
        with self._trace_lock:
            self._agent_trace.append(entry)
            # 超过上限时截断，保留最近一半
            if len(self._agent_trace) > self._max_trace_entries:
                self._agent_trace = self._agent_trace[-(self._max_trace_entries // 2):]

        logger.info(
            "[AgentTrace] %s | track=%d frame=%d | %s%s",
            event_type.upper(),
            track_id,
            frame_id,
            content[:100] if content else "",
            f" | {extra}" if extra else "",
        )

    # ── 数据库查找 ──────────────────────────────

    def _db_lookup(self, hull_number: str, description: str) -> tuple[bool, list[str]]:
        """
        在数据库中查找匹配的弦号或描述。

        Returns:
            (exact_matched, semantic_match_ids)
            - exact_matched: hull_number 精确匹配到库内记录
            - semantic_match_ids: 语义检索匹配到的弦号列表（精确匹配时为空）
        """
        # 优先精确弦号匹配
        if hull_number:
            desc = self._db.lookup(hull_number)
            if desc is not None:
                return True, []

        # 语义检索（返回多个候选）
        semantic_ids: list[str] = []
        if description:
            try:
                results = self._db.semantic_search_filtered(description)
                semantic_ids = [r["hull_number"] for r in results if r.get("hull_number")]
            except Exception as e:
                logger.warning("语义检索异常: %s", e)

        return False, semantic_ids

    # ── 推理结果处理 ────────────────────────────

    def _handle_recognition_result(self, result: InferenceResult) -> None:
        """处理一次 Agent 推理结果：绑定到 track 并查询数据库。"""
        # 推理出错时也要解除 pending 状态
        if result.error:
            logger.warning(
                "推理出错 (track=%d, frame=%d): %s",
                result.track_id, result.frame_id, result.error,
            )
            self._tracker.cancel_pending(result.track_id)
            return

        self._log_agent_trace(
            "inference_result",
            track_id=result.track_id,
            frame_id=result.frame_id,
            content=f"弦号={result.hull_number or '(未识别)'} 描述={result.description[:50]}",
        )

        # 绑定识别结果
        self._tracker.bind_result(
            result.track_id,
            result.hull_number,
            result.description,
        )

        # 查询数据库
        exact_matched, semantic_ids = self._db_lookup(
            result.hull_number, result.description
        )

        if exact_matched:
            # 精确匹配 → 绿色
            desc = self._db.lookup(result.hull_number) or ""
            self._tracker.bind_db_match(result.track_id, result.hull_number, desc)
            self._log_agent_trace(
                "db_exact_match",
                track_id=result.track_id,
                frame_id=result.frame_id,
                content=f"库内确定id：{result.hull_number}",
            )
        elif semantic_ids:
            # 有语义匹配候选 → 黄色或红色
            self._tracker.bind_semantic_matches(result.track_id, semantic_ids)
            self._log_agent_trace(
                "db_semantic_match",
                track_id=result.track_id,
                frame_id=result.frame_id,
                content=f"语义匹配候选：{semantic_ids}",
            )
        else:
            self._log_agent_trace(
                "db_no_match",
                track_id=result.track_id,
                frame_id=result.frame_id,
                content=f"未匹配: {result.hull_number or '未知'}",
            )

    # ── 级联模式 ────────────────────────────────

    def _cascade_process(
        self,
        detections: list[Detection],
        frame_id: int,
    ) -> None:
        """级联模式：同步处理每个需要识别的检测目标。"""
        for det in detections:
            if not self._tracker.needs_recognition(det.track_id):
                continue

            if det.crop is None or det.crop.size == 0:
                continue

            self._tracker.mark_pending(det.track_id)

            self._log_agent_trace(
                "cascade_infer_start",
                track_id=det.track_id,
                frame_id=frame_id,
                content="同步推理开始",
            )

            result = self._agent.infer_single(
                crop=det.crop,
                track_id=det.track_id,
                frame_id=frame_id,
            )
            self._handle_recognition_result(result)

    # ── 并发模式 ────────────────────────────────

    def _concurrent_process(
        self,
        detections: list[Detection],
        frame_id: int,
    ) -> None:
        """并发模式：将 crop 送入队列，Agent 异步推理。"""
        for det in detections:
            if not self._tracker.needs_recognition(det.track_id):
                continue

            if det.crop is None or det.crop.size == 0:
                continue

            # 标记为 pending
            self._tracker.mark_pending(det.track_id)

            task = {
                "frame_id": frame_id,
                "timestamp": time.time(),
                "track_id": det.track_id,
                "crop": det.crop.copy(),
            }

            try:
                self._task_queue.put_nowait(task)
                self._log_agent_trace(
                    "concurrent_enqueue",
                    track_id=det.track_id,
                    frame_id=frame_id,
                    content=f"送入异步队列 (队列深度: {self._task_queue.qsize()})",
                )
            except queue.Full:
                logger.warning(
                    "任务队列已满 (%d)，丢弃 frame=%d track=%d",
                    self._max_queued_frames, frame_id, det.track_id,
                )
                # 取消 pending 状态
                self._tracker.cancel_pending(det.track_id)

    def _agent_worker_loop(self) -> None:
        """Agent 工作线程：从队列取任务并推理。"""
        while not self._stop_event.is_set():
            try:
                task = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            track_id = task["track_id"]
            frame_id = task["frame_id"]
            crop = task["crop"]

            self._log_agent_trace(
                "concurrent_infer_start",
                track_id=track_id,
                frame_id=frame_id,
                content="异步推理开始",
            )

            try:
                result = self._agent.infer_single(
                    crop=crop,
                    track_id=track_id,
                    frame_id=frame_id,
                )
            except Exception as e:
                logger.exception("Agent 推理异常 (track=%d, frame=%d)", track_id, frame_id)
                result = InferenceResult(
                    hull_number="",
                    description="",
                    track_id=track_id,
                    frame_id=frame_id,
                    error=str(e),
                )

            # 将结果放入结果队列
            self._result_queue.put({
                "frame_id": frame_id,
                "track_id": track_id,
                "result": result,
            })

    def _drain_results(self) -> int:
        """排空结果队列，处理所有已完成的异步推理结果。返回处理数量。"""
        count = 0
        while True:
            try:
                pending = self._result_queue.get_nowait()
                self._handle_recognition_result(pending["result"])
                count += 1
            except queue.Empty:
                break
        return count

    def _start_agent_workers(self) -> None:
        """启动 Agent 工作线程池。"""
        self._stop_event.clear()
        self._agent_workers.clear()
        for i in range(self._max_concurrent):
            worker = threading.Thread(
                target=self._agent_worker_loop,
                name=f"agent-worker-{i}",
                daemon=True,
            )
            worker.start()
            self._agent_workers.append(worker)
        logger.info("启动 %d 个 Agent 工作线程", self._max_concurrent)

    def _stop_agent_workers(self) -> None:
        """停止 Agent 工作线程，等待全部完成。"""
        self._stop_event.set()
        for worker in self._agent_workers:
            worker.join(timeout=10.0)
            if worker.is_alive():
                logger.warning("工作线程 %s 未在超时内退出", worker.name)
        self._agent_workers.clear()

        # workers 全部退出后再排空，避免竞态丢结果
        remaining_tasks = 0
        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()
                remaining_tasks += 1
            except queue.Empty:
                break
        if remaining_tasks:
            logger.info("清理 %d 个未处理任务", remaining_tasks)

        remaining_results = self._drain_results()
        if remaining_results:
            logger.info("处理 %d 个残留结果", remaining_results)

        logger.info("Agent 工作线程已停止")

    # ── 渲染 ────────────────────────────────────

    def _render_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        frame_id: int,
    ) -> np.ndarray:
        """通过 DemoRenderer 在帧上绘制检测框、识别结果和 HUD。"""
        return self._renderer.render(
            frame=frame,
            detections=detections,
            tracks=self._tracker.active_tracks,
            fps_info=self._fps.get_all_fps(),
            frame_id=frame_id,
            queue_depth=self._task_queue.qsize(),
            max_queue=self._max_queued_frames,
        )

    # ── 主流程 ──────────────────────────────────

    def process(
        self,
        source: str | int,
        output_path: str | None = None,
        display: bool = False,
        max_frames: int = 0,
        frame_callback: Callable[[np.ndarray, int], None] | None = None,
    ) -> dict[str, Any]:
        """
        运行完整的视频处理流水线。

        Args:
            source: 视频输入源（文件路径/相机号/RTSP URL）。
            output_path: 输出视频路径（可选）。
            display: 是否实时显示窗口（仅本地有显示器时有效）。
            max_frames: 最大处理帧数，0 表示不限制。
            frame_callback: 每帧处理完成后的回调函数 callback(frame, frame_id)。

        Returns:
            统计信息字典。
        """
        input_src = InputSource(source)
        video_writer = None
        last_detections: list[Detection] = []

        try:
            # 初始化视频写入器
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    output_path, fourcc,
                    input_src.source_fps,
                    (input_src.width, input_src.height),
                )
                if not video_writer.isOpened():
                    logger.error("无法创建输出视频: %s", output_path)
                    video_writer = None
                else:
                    logger.info("输出视频: %s", output_path)

            # 启动并发 worker
            if self._concurrent_mode:
                self._start_agent_workers()

            frame_id = 0
            total_detections = 0
            start_time = time.time()

            logger.info(
                "开始处理: source=%s, mode=%s, demo=%s",
                source,
                "concurrent" if self._concurrent_mode else "cascade",
                self._demo_enabled,
            )

            while True:
                ret, frame = input_src.read()
                if not ret:
                    logger.info("视频源结束或读取失败")
                    break

                frame_id += 1
                if max_frames > 0 and frame_id > max_frames:
                    break

                # FPS 统计
                self._fps.tick("stream")

                # ── 每帧都进行 YOLO 检测 + track 注册，保持 ByteTrack 连续 ──
                try:
                    detections = self._detector.detect(frame, frame_id)
                except Exception as e:
                    logger.error("YOLO 检测异常 (frame=%d): %s", frame_id, e)
                    detections = []

                last_detections = detections
                total_detections += len(detections)

                # 注册/更新 track（每帧执行，保持跟踪状态）
                for det in detections:
                    self._tracker.get_or_create(det.track_id, frame_id)

                # ── process_every_n_frames 控制 Agent 推理频率 ──
                should_process = (frame_id % self._process_every_n == 0)
                new_recognized = False

                if should_process:
                    # 记录当前已识别的 track ID 集合
                    prev_recognized = {
                        tid for tid, t in self._tracker.active_tracks.items() if t.recognized
                    }

                    # Agent 推理（级联或并发）
                    if self._concurrent_mode:
                        self._concurrent_process(detections, frame_id)
                    else:
                        self._cascade_process(detections, frame_id)

                    # 并发模式下排空已完成的结果
                    if self._concurrent_mode:
                        self._drain_results()

                    # 检查是否有新 track 被识别出弦号
                    for tid, t in self._tracker.active_tracks.items():
                        if t.recognized and tid not in prev_recognized and t.hull_number:
                            new_recognized = True
                            break

                # 清理过期 track
                self._tracker.cleanup_stale(frame_id)

                # 渲染输出
                if self._demo_enabled or output_path or display:
                    display_frame = self._render_frame(frame, last_detections, frame_id)
                else:
                    display_frame = frame

                # Agent 识别到弦号时保存截图
                if new_recognized:
                    self._saver.save(display_frame, frame_id)

                # 写入输出视频
                if video_writer:
                    video_writer.write(display_frame)

                # 回调
                if frame_callback:
                    frame_callback(display_frame, frame_id)

                # 实时显示
                if display:
                    cv2.imshow("Ship Pipeline", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("用户按下 q，停止处理")
                        break
                    elif key == ord("d"):
                        new_mode = "brief" if self._agent.prompt_mode == "detailed" else "detailed"
                        self._agent.set_prompt_mode(new_mode)
                        logger.info("提示词模式切换为: %s", new_mode)

                # 处理 FPS
                self._fps.tick("process")

                # 定期打印 FPS（每 5 秒）
                if self._fps.should_print("stream"):
                    elapsed = time.time() - start_time
                    self._fps.print_fps(
                        "stream",
                        extra=f"frames={frame_id}, elapsed={elapsed:.0f}s, tracks={len(self._tracker)}",
                    )
                    self._fps.print_fps("process")

                # 定期打印 Agent 运行链路摘要（每 100 帧）
                if frame_id % 100 == 0:
                    self._print_trace_summary()

            # ── 处理完成，收集统计 ──

            # 并发模式下最终排空结果
            if self._concurrent_mode:
                self._drain_results()

            elapsed = time.time() - start_time
            tracks = self._tracker.active_tracks
            total_recognized = sum(1 for t in tracks.values() if t.recognized)

            stats = {
                "total_frames": frame_id,
                "total_detections": total_detections,
                "total_tracks": len(tracks),
                "recognized_tracks": total_recognized,
                "elapsed_seconds": round(elapsed, 1),
                "avg_fps": round(frame_id / elapsed, 1) if elapsed > 0 else 0,
                "mode": "concurrent" if self._concurrent_mode else "cascade",
                "screenshots_saved": self._saver.saved_count,
            }

            logger.info("=" * 50)
            logger.info("处理完成统计:")
            logger.info("  总帧数: %d", stats["total_frames"])
            logger.info("  总检测数: %d", stats["total_detections"])
            logger.info("  跟踪目标数: %d", stats["total_tracks"])
            logger.info("  已识别: %d", stats["recognized_tracks"])
            logger.info("  耗时: %.1fs", stats["elapsed_seconds"])
            logger.info("  平均 FPS: %.1f", stats["avg_fps"])
            logger.info("  模式: %s", stats["mode"])
            logger.info("=" * 50)

            return stats

        except KeyboardInterrupt:
            logger.info("用户中断处理")
            elapsed = time.time() - start_time
            return {
                "total_frames": frame_id,
                "total_detections": total_detections,
                "total_tracks": len(self._tracker),
                "recognized_tracks": 0,
                "elapsed_seconds": round(elapsed, 1),
                "avg_fps": round(frame_id / elapsed, 1) if elapsed > 0 else 0,
                "mode": "concurrent" if self._concurrent_mode else "cascade",
                "interrupted": True,
            }

        finally:
            if self._concurrent_mode:
                self._stop_agent_workers()
            input_src.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()

    # ── 链路摘要 ────────────────────────────────

    def _print_trace_summary(self) -> None:
        """打印 Agent 运行链路摘要。"""
        with self._trace_lock:
            if not self._agent_trace:
                return
            recent = self._agent_trace[-20:]
            logger.info("=== Agent 运行链路摘要 (最近 %d 条) ===", len(recent))
            for entry in recent:
                logger.info(
                    "  [%s] track=%d frame=%d: %s",
                    entry["type"],
                    entry["track_id"],
                    entry["frame_id"],
                    entry.get("content", "")[:80],
                )

    @property
    def agent_trace(self) -> list[dict[str, Any]]:
        """获取完整的 Agent 运行链路日志。"""
        with self._trace_lock:
            return list(self._agent_trace)

    # ── 运行时控制 ──────────────────────────────

    def set_demo(self, enabled: bool) -> None:
        """设置 demo 开关。"""
        self._demo_enabled = enabled
        logger.info("Demo 模式: %s", "开启" if enabled else "关闭")

    def set_prompt_mode(self, mode: str) -> None:
        """切换提示词模式。"""
        self._agent.set_prompt_mode(mode)

    def switch_to_concurrent(self, enabled: bool) -> None:
        """动态切换级联/并发模式。"""
        self._concurrent_mode = enabled
        logger.info("切换为 %s 模式", "并发" if enabled else "级联")
