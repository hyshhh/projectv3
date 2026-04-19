# 🚢 Ship Hull Agent — 船弦号识别 Agent

基于 **LangChain + LangGraph + FAISS 向量库** 的智能船弦号识别系统，支持 **精确弦号匹配** 和 **RAG 语义检索** 两种识别模式。

## ✨ 功能特性

- **精确匹配**：输入弦号直接查字典，O(1) 响应
- **RAG 语义检索**：基于 FAISS 向量库，每条船记录（弦号+描述）作为一个 Document 建立索引
- **向量库持久化**：首次构建后缓存到磁盘，后续启动直接加载
- **Few-shot 引导**：内置示例对话，确保 Agent 严格遵循检索流程
- **CLI 工具**：命令行单次查询 / 交互式 REPL / 详细调用链模式
- **全参数配置化**：模型、检索、向量库所有参数通过 `.env` 配置

## 🏗️ 项目结构

```
ship-hull-agent/
├── config.py              # 配置读取：config.yaml + 内置默认值
├── config.yaml            # 全局配置文件
├── database/
│   └── __init__.py        # ShipDatabase：精确查找 + FAISS 向量语义检索
├── tools/
│   └── __init__.py        # LangChain @tool：lookup_by_hull_number / retrieve_by_description
├── agent/
│   └── __init__.py        # ShipHullAgent：ReAct Agent + Few-shot 示例
├── cli/
│   ├── __init__.py        # Rich CLI：单次查询 / 交互 REPL / --verbose 调用链
│   └── main.py            # ship-hull 入口
├── pipeline/              # 🎬 视频处理流水线
│   ├── __init__.py        # 模块导出
│   ├── __main__.py        # python -m pipeline 入口
│   ├── cli.py             # 命令行参数解析
│   ├── pipeline.py        # 主流水线编排（级联/并发双模式）
│   ├── detector.py        # YOLO 船只检测 + ByteTrack 跟踪
│   ├── agent_inference.py # Qwen3.5 VLM 弦号识别
│   ├── tracker.py         # 跟踪状态管理（线程安全）
│   ├── fps.py             # 10 秒滑动窗口 FPS 统计
│   ├── video_input.py     # 视频/相机/视频流统一输入
│   └── demo.py            # Demo 可视化渲染
├── build_db.py            # 批量建库脚本
├── data/
│   └── ships.csv          # 船只数据库
├── tests/
│   ├── __init__.py
│   ├── test_database.py   # 数据库单元测试
│   └── test_pipeline.py   # Pipeline 单元测试 + 并发压力测试
├── .env.example           # 环境变量模板
├── .gitignore
├── pyproject.toml         # 项目元数据 + 依赖声明
└── README.md
```

## 🔄 Agent 工作流程

```
用户输入
  │
  ├─ 包含弦号？
  │    → lookup_by_hull_number 精确查找
  │         ├─ found=true  → 直接返回结果
  │         └─ found=false → retrieve_by_description 语义检索
  │
  └─ 只有描述？
       → retrieve_by_description 语义检索
            ├─ FAISS 向量相似度匹配（top_k + 阈值过滤）
            └─ 返回最匹配的弦号 + 描述 + 相似度
```

### RAG 向量库设计

每条船记录构建为一个 Document：

```
page_content = "弦号 0014\n白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪"
metadata     = {"hull_number": "0014", "description": "白色大型客轮..."}
```

- `page_content` 用于 Embedding 向量化（弦号+描述一起编码，语义更丰富）
- `metadata` 用于提取结构化结果
- FAISS 索引构建后持久化到 `VECTOR_STORE_PATH`，下次启动直接加载

## 📸 构建船舶数据库（build_db.py）

通过图片自动识别船只，构建 CSV 数据库。脚本调用视觉模型对每张图片进行识别，生成弦号和描述后存入数据库。

### 前置条件

- Python 3.10+
- 已安装依赖：`pip install -e .`
- 视觉模型服务已启动（默认地址 `http://localhost:7890/v1`）

### 基本用法

```bash
python3 build_db.py <图片目录路径>
```

例如：

```bash
# 处理 images/ 目录下所有图片
python3 build_db.py ./images
```

支持的图片格式：`.jpg` `.jpeg` `.png` `.bmp` `.webp` `.gif`

### 处理流程

对目录中的每张图片，脚本会：

1. **调用视觉模型**识别船只，返回弦号 + 描述
2. **查重检查**：如果弦号已存在于数据库中，提示用户选择：
   - 按 `1`：跳过，保留原记录
   - 按 `2`：覆盖为新描述
   - 按 `3`：手动输入新弦号
3. **弦号确认**：如果是新弦号，提示用户确认是否正确：
   - 按 `1`：确认弦号
   - 按 `2`：手动输入正确的弦号
4. **立即写入 CSV**（每识别一张就保存，中断不丢数据）

### 交互示例

```
$ python3 build_db.py ./my_ship_photos

📦 已有数据库: ./data/ships.csv（4 条记录）
🖼️  找到 12 张图片，开始识别...

📡 使用模型: Qwen/Qwen3-VL-4B-AWQ
📡 服务地址: http://localhost:7890/v1

============================================================
[1/12] 处理: ship_001.jpg
============================================================

  📝 识别结果:
     弦号: 0014
     描述: 白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪

  ⚠️  弦号 [0014] 已存在于数据库中
     现有描述: 白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪
  按 1 跳过（保留原记录）
  按 2 覆盖为新描述
  按 3 手动输入新弦号
  请选择 [1/2/3] (1): 1
  ⏭️  已跳过

============================================================
[2/12] 处理: ship_002.jpg
============================================================

  📝 识别结果:
     弦号: (未识别)
     描述: 红色小型渔船，单层驾驶室，船尾有渔网架

  未识别到弦号
  按 1 跳过弦号（仅保存描述），按 2 手动输入弦号
  请选择 [1/2] (1): 2
  请输入正确弦号: F088
  ✅ 已保存弦号 [F088]

============================================================
📊 处理完成
   总计: 12 张图片
   成功: 10 条
   跳过: 2 条
   数据库: ./data/ships.csv（共 14 条记录）
============================================================
```

### 查重逻辑

脚本在所有场景下都会检查弦号重复：

| 场景 | 处理方式 |
|------|----------|
| 模型识别出弦号，数据库已存在 | 提示用户：跳过 / 覆盖 / 手动输入新弦号 |
| 用户手动输入弦号，数据库已存在 | 提示用户：跳过 / 覆盖 |
| 无弦号，用文件名作 fallback | 自动加后缀 `_2`、`_3` 避免覆盖 |

### 配置

脚本通过 `config.yaml` 配置视觉模型参数：

```yaml
llm:
  model: "Qwen/Qwen3-VL-4B-AWQ"
  api_key: "abc123"
  base_url: "http://localhost:7890/v1"
  temperature: 0.0

app:
  ship_db_path: "./data/ships.csv"  # 输出 CSV 路径
```

### 注意事项

- 每张图片识别后**立即写入 CSV**，中途按 `Ctrl+C` 不会丢失已处理的数据
- 写入失败（如磁盘满）会打印警告，数据暂存在内存中，下次成功写入时自动包含
- 图片过大可能导致内存占用较高，建议单张不超过 20MB

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/hyshhh/project.git
cd project
```

### 2. 启动视觉模型服务

使用 vLLM 部署模型（兼容 OpenAI API 格式）：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /media/ddc/新加卷/hys/hysnew/Qwen3.5-2B-AWQ \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 1024 \
  --port 7890 \
  --gpu-memory-utilization 0.15 \
  --max-num-seqs 10 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `--served-model-name` | `Qwen/Qwen3-VL-4B-AWQ` | 对外暴露的模型名称 |
| `--api-key` | `abc123` | API 密钥 |
| `--port` | `7890` | 服务端口 |
| `--max-model-len` | `1024` | 最大上下文长度 |
| `--gpu-memory-utilization` | `0.15` | GPU 显存占用比例 |
| `--max-num-seqs` | `10` | 最大并发序列数 |
| `--enable-auto-tool-choice` | — | 启用自动工具调用 |
| `--tool-call-parser` | `qwen` | 工具调用解析器 |

### 3. 安装依赖

```bash
pip install -e .
# 开发模式（含测试）
pip install -e ".[dev]"
```

### 4. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```env
# 对话模型（兼容 OpenAI API 格式）
CHAT_MODEL=Qwen/Qwen3-VL-4B-AWQ
LLM_API_KEY=your-llm-api-key
LLM_BASE_URL=http://localhost:7890/v1

# Embedding 模型（默认云端，也可本地部署，见下方"本地部署 Embedding 模型"章节）
EMBED_MODEL=text-embedding-v4
EMBED_API_KEY=your-embed-api-key
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# EMBED_DIMENSIONS=1024  # 部分模型需要（如 DashScope text-embedding-v4），不需要可注释掉

# RAG 检索参数
RETRIEVAL_TOP_K=3
RETRIEVAL_SCORE_THRESHOLD=0.5

# 向量库持久化路径
VECTOR_STORE_PATH=./vector_store
```

### 5. 运行

```bash
# 单次查询
ship-hull "帮我查一下弦号0014是什么船"

# 交互模式
ship-hull --interactive

# 详细模式（显示工具调用链）
ship-hull --verbose "弦号0256"
```

## 📖 使用示例

### 精确匹配

```
$ ship-hull "帮我查一下弦号0014是什么船"

识别结果：弦号 0014，描述：白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪
```

### 语义检索（弦号不存在）

```
$ ship-hull "弦号9999，这是一艘大型白色邮轮，船身有蓝色条纹装饰，有三个烟囱"

未找到对应弦号，根据描述检索到最相似的船：弦号 0123，描述：白色邮轮，船身有红蓝条纹装饰，三座烟囱（相似度：0.9234）
```

### 详细模式（调试）

```
$ ship-hull --verbose "我看到一艘灰色的军舰，外形很隐身"

┌─────────────────── 🔧 Agent 调用链 ───────────────────┐
│ # │ 类型     │ 内容                                   │
├───┼──────────┼────────────────────────────────────────┤
│ 0 │ human    │ 我看到一艘灰色的军舰，外形很隐身         │
│ 1 │ ai       │ → lookup_by_hull_number({"hull_number… │
│ 2 │ tool     │ ← {"found": false, "hull_number": ""}  │
│ 3 │ ai       │ → retrieve_by_description({"target_de… │
│ 4 │ tool     │ ← {"results": [{"hull_number": "0256"… │
│ 5 │ ai       │ 未找到对应弦号，根据描述检索到最相似的船… │
└─────────────────────────────────────────────────────────┘
```

### 作为 Python 库调用

```python
from agent import create_agent

agent = create_agent()
answer = agent.run("弦号0014是什么船")
print(answer)
```

## 🎬 Pipeline 视频处理流水线

基于 **YOLO + Qwen3.5 VLM + Agent** 的实时船只检测与弦号识别系统。支持视频文件、USB 相机、RTSP 视频流输入。

### 架构概览

```
视频输入 → YOLO 检测+跟踪 → 裁剪船只区域 → Qwen3.5 VLM 识别弦号+描述
                                                      │
                                          Agent 数据库匹配（精确/语义）
                                                      │
                                          绘制检测框+识别结果 → 输出
```

### 快速使用

```bash
# 处理视频文件
python -m pipeline.cli video.mp4

# USB 相机（设备号 0）
python -m pipeline.cli 0

# RTSP 视频流
python -m pipeline.cli rtsp://192.168.1.100/stream

# 开启 demo 可视化 + 输出结果视频
python -m pipeline.cli video.mp4 --demo --output result.mp4

# 并发模式 + 简略提示词
python -m pipeline.cli video.mp4 --concurrent --prompt-mode brief

# 详细日志
python -m pipeline.cli video.mp4 --verbose
```

### 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `source` | — | 输入源：文件路径 / 相机号 / RTSP URL | 必填 |
| `--output` | `-o` | 输出视频路径 | 无 |
| `--demo` | — | 开启 demo 可视化（检测框+识别结果叠加） | 关闭 |
| `--display` | — | 实时显示窗口（需有显示器） | 关闭 |
| `--concurrent` | `-c` | 使用并发模式（默认级联模式） | 关闭 |
| `--max-concurrent` | — | 最大并发 Agent 推理数 | `4` |
| `--max-queued-frames` | — | 并发模式最大队列深度（防 OOM） | `30` |
| `--process-every` | — | 每 N 帧处理一次 | `1` |
| `--prompt-mode` | — | 提示词模式：`detailed` / `brief` | `detailed` |
| `--max-frames` | — | 最大处理帧数（0=不限） | `0` |
| `--yolo-model` | — | YOLO 模型路径 | `yolov8n.pt` |
| `--device` | — | 推理设备（`cpu` / GPU 编号） | 自动 |
| `--conf` | — | 检测置信度阈值 | `0.25` |
| `--verbose` | `-v` | 详细日志输出 | 关闭 |

### 级联 vs 并发模式

```
级联模式（默认）：
  帧 N → YOLO 检测 → 等待 Agent 返回 → 绑定结果 → 渲染帧 N
  特点：简单可靠，帧间严格有序，速度受 Agent 延迟影响

并发模式（--concurrent）：
  帧 N → YOLO 检测 → crop 送入队列 → 立即处理帧 N+1
                    Agent 线程池异步推理 → 结果回填到对应 track
  特点：高吞吐，帧率不受 Agent 延迟影响，需要更多显存
```

### config.yaml 中的 Pipeline 配置

```yaml
pipeline:
  # 级联/并发双模式
  # false = 级联模式：同步等待 Agent 返回结果
  # true  = 并发模式：YOLO 不等待，Agent 异步推理
  concurrent_mode: false

  # 最大并发 Agent 推理数
  max_concurrent: 4

  # 最大队列深度（并发模式下防止 OOM）
  max_queued_frames: 30

  # 每 N 帧处理一次（1 = 每帧都处理）
  process_every_n_frames: 1

  # 提示词模式："detailed"（详细）或 "brief"（简略）
  prompt_mode: "detailed"

  # Demo 开关：true 开启视频 demo 可视化，false 关闭
  demo: false

  # YOLO 模型路径（不存在会自动下载）
  yolo_model: "yolov8n.pt"

  # 推理设备："" 自动选择，"cpu" 强制 CPU，"0" 表示 GPU 0
  device: ""

  # 检测置信度阈值
  conf_threshold: 0.25

  # 追踪算法："bytetrack" 或 "botsort"
  tracker: "bytetrack"

  # 只检测 COCO 类别 8（船），null 表示检测所有类别
  detect_classes:
    - 8

  # 超过此帧数未出现的 track 被清理
  max_stale_frames: 300
```

### 交互按键（display 模式下）

| 按键 | 动作 |
|------|------|
| `q` | 退出 |
| `d` | 切换提示词模式（详细 ↔ 简略） |
| `p` | 暂停/继续 |

### Python API 调用

```python
from config import load_config
from pipeline import ShipPipeline

config = load_config()
pipeline = ShipPipeline(config=config)

# 处理视频
stats = pipeline.process(
    source="video.mp4",
    output_path="result.mp4",
    display=False,
    max_frames=1000,
)
print(stats)
# {'total_frames': 1000, 'total_detections': 342, 'total_tracks': 8,
#  'recognized_tracks': 6, 'elapsed_seconds': 45.2, 'avg_fps': 22.1, 'mode': 'cascade'}

# 运行时控制
pipeline.set_demo(True)
pipeline.set_prompt_mode("brief")
pipeline.switch_to_concurrent(True)

# 查看 Agent 运行链路
trace = pipeline.agent_trace
```

### Pipeline 模块结构

```
pipeline/
├── __init__.py           # 模块导出
├── __main__.py           # python -m pipeline 入口
├── cli.py                # 命令行参数解析 + 启动
├── pipeline.py           # 主流水线编排（ShipPipeline）
├── detector.py           # YOLO 船只检测 + ByteTrack 跟踪
├── agent_inference.py    # Qwen3.5 VLM 弦号识别推理
├── tracker.py            # 跟踪状态管理（track ID ↔ 弦号绑定）
├── fps.py                # 10 秒滑动窗口 FPS 统计
├── video_input.py        # 视频/相机/视频流统一输入
└── demo.py               # Demo 可视化渲染（检测框+HUD）
```

### 跟踪与识别流程

```
新 track 出现 → 标记 pending → Agent 推理（VLM 识别弦号+描述）
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
              识别到弦号                       未识别到弦号
                    │                               │
            数据库精确匹配                    数据库语义检索
                    │                               │
              ┌─────┴─────┐                   ┌─────┴─────┐
              │           │                   │           │
           匹配成功    未匹配              匹配成功    未匹配
              │           │                   │           │
    (库内确定id：XXX) (未知id：XXX)   (库内确定id：XXX) (未知id：XXX)
```

后续帧：track ID 沿用已识别结果，不再调用 Agent。

## ⚙️ 配置说明

### 环境变量一览

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CHAT_MODEL` | 对话模型名称 | `Qwen/Qwen3-VL-4B-AWQ` |
| `LLM_API_KEY` | 对话模型 API Key | `abc123` |
| `LLM_BASE_URL` | 对话模型地址 | `http://localhost:7890/v1` |
| `LLM_TEMPERATURE` | 生成温度（0=确定性最高） | `0.0` |
| `EMBED_MODEL` | Embedding 模型名称 | `text-embedding-v4` |
| `EMBED_API_KEY` | Embedding API Key | — |
| `EMBED_BASE_URL` | Embedding 服务地址 | DashScope |
| `EMBED_DIMENSIONS` | Embedding 向量维度（部分模型不需要，如 Qwen3-Embedding-0.6B 设为 null） | `null`（不传） |
| `RETRIEVAL_TOP_K` | 语义检索返回条数 | `3` |
| `RETRIEVAL_SCORE_THRESHOLD` | 相似度阈值（低于此值的结果被过滤） | `0.5` |
| `VECTOR_STORE_PATH` | FAISS 索引持久化路径 | `./vector_store` |
| `VECTOR_STORE_AUTO_REBUILD` | 每次启动重建索引 | `false` |
| `SHIP_DB_PATH` | 自定义数据库 JSON 路径 | 内置默认数据 |
| `LOG_LEVEL` | 日志级别 | `INFO` |

### 自定义数据库

创建 JSON 文件（如 `data/ships.json`）：

```json
{
  "0014": "白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪",
  "A001": "你的自定义船只描述"
}
```

设置 `SHIP_DB_PATH=./data/ships.json`，并将 `VECTOR_STORE_AUTO_REBUILD=true` 以重建索引。

## 🧠 本地部署 Embedding 模型（可选）

默认使用 DashScope 云端 Embedding 服务（按 token 收费、有 batch 限制）。推荐本地部署，免费且无限制。

### 推荐模型

| 模型 | 参数量 | 显存占用 | 维度 | 特点 |
|------|--------|---------|------|------|
| **Qwen3-Embedding-0.6B** ⭐ | 0.6B | ~1.5GB | 1024 | 极轻量，中文效果好，Qwen 官方出品 |
| BGE-M3 | ~2.2GB | ~3GB | 1024 | 中英双语强，MTEB 榜单前列 |
| bge-large-zh-v1.5 | ~1.3GB | ~2GB | 1024 | 纯中文优化，模型小 |

推荐 **Qwen3-Embedding-0.6B**，显存占用极低，和你的 LLM 可以跑在同一张卡上。

### 部署步骤

```bash
# 1. 下载模型（ModelScope）
pip install modelscope
modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir ./models/Qwen3-Embedding-0.6B

# 2. 用 vLLM 部署为 OpenAI 兼容 API
python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen3-Embedding-0.6B \
  --api-key abc123 \
  --served-model-name Qwen3-Embedding-0.6B \
  --convert embed \
  --gpu-memory-utilization 0.15 \
  --max-model-len 2048 \
  --port 7891
```

启动成功后会看到类似输出：
```
INFO:     Uvicorn running on http://0.0.0.0:7891
```

### 修改配置

编辑 `config.yaml`，将 Embedding 指向本地服务：

```yaml
embed:
  model: "Qwen3-Embedding-0.6B"
  api_key: "abc123"                  # 本地部署随意填
  base_url: "http://localhost:7891/v1"  # 改为本地地址
  # dimensions: 1024  # Qwen3-Embedding-0.6B 不需要此参数，留空或注释掉
```

首次运行会自动构建 FAISS 向量库，后续启动直接加载缓存。

### 验证部署

```bash
# 测试 Embedding API 是否正常
curl http://localhost:7891/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer abc123" \
  -d '{"model": "Qwen3-Embedding-0.6B", "input": ["测试文本"]}'
```

返回包含 `embedding` 数组即表示部署成功。

### 本地 vs 云端对比

| | DashScope 云端 | 本地 Qwen3-Embedding-0.6B |
|---|---|---|
| 延迟 | ~200-500ms（网络） | ~20-50ms（本地） |
| 费用 | 按 token 收费 | 免费 |
| batch 限制 | 10 条/次 | 无限制 |
| 网络依赖 | 需联网 | 完全离线 |
| 显存占用 | 0 | ~1.5GB |

## 🧪 测试

```bash
pytest -v                          # 运行全部测试（20 个）
pytest tests/test_database.py -v   # 指定文件
```

## 🛠️ 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| LLM 编排 | LangChain + LangGraph | ReAct Agent 模式 |
| 向量库 | FAISS (faiss-cpu) | 语义检索索引 |
| Embedding | OpenAI Embeddings API | 文本向量化 |
| 视觉模型 | Qwen3.5 VLM (OpenAI API 兼容) | 船只图像弦号识别 |
| 目标检测 | ultralytics YOLO | 船只检测 |
| 跟踪 | ByteTrack (YOLO 内置) | 多目标跟踪 |
| 视频处理 | OpenCV (cv2) | 视频读写、图像处理 |
| 并发 | threading + queue.Queue | 级联/并发双模式推理 |
| 配置 | pydantic-settings | 环境变量管理 |
| 向量计算 | NumPy | 余弦相似度 |
| HTTP | httpx | API 调用 |
| CLI | Rich | 终端美化输出 |
| 测试 | pytest | 单元测试 + 并发压力测试 |

## 📝 开发指南

### 换 LLM Provider

任何兼容 OpenAI API 格式的服务都行：

```env
CHAT_MODEL=gpt-4o
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.openai.com/v1
```

### 添加新工具

在 `tools/__init__.py` 中添加 `@tool` 函数，`build_tools()` 返回列表中加上即可。Agent 会自动识别。

### 数据变更后重建索引

```bash
VECTOR_STORE_AUTO_REBUILD=true ship-hull "触发一次即可重建"
```

或直接删除 `VECTOR_STORE_PATH` 目录，下次启动自动重建。

## 📄 License

MIT
