## 性能压测与监控检查指南

### 概览

本文档用于指导对 Simple Inference Server 进行高并发压测与监控检查，主要目标：

- 验证在目标硬件上，embeddings / chat / audio 在高并发下的 **吞吐量、延迟、错误率**。
- 利用 `/metrics` 与 `/health` 观测 **限流/队列、批处理、缓存、warmup** 的运行状况。
- 及时发现配置不当、模型负载过高或潜在资源问题（CPU/GPU 饱和、内存/线程异常增长）。

本指南只描述方法和建议命令，你可以在需要时再执行。

---

## 环境准备

### 服务启动

1. 安装依赖：

```bash
uv sync
```

2. 预下载模型（可选，若保留 `AUTO_DOWNLOAD_MODELS=1` 可跳过）：

```bash
MODELS=BAAI/bge-m3,meta-llama/Llama-3.2-1B-Instruct uv run python scripts/download_models.py
MODELS=openai/whisper-tiny uv run python scripts/download_models.py
```

3. 启动服务（示例）：

```bash
# 仅 embeddings + chat
MODELS=BAAI/bge-m3,meta-llama/Llama-3.2-1B-Instruct \
MODEL_DEVICE=auto \
MAX_CONCURRENT=2 \
uv run python scripts/run_dev.py --device auto

# 加上 Whisper 音频
MODELS=BAAI/bge-m3,meta-llama/Llama-3.2-1B-Instruct,openai/whisper-tiny \
MODEL_DEVICE=auto \
MAX_CONCURRENT=2 \
AUDIO_MAX_CONCURRENT=1 \
uv run python scripts/run_dev.py --device auto
```

### 基本健康检查

- `GET /health`：检查模型列表、warmup 是否完成、是否有失败模型；
- `GET /metrics`：Prometheus 指标抓取端点（本文所有指标名称均来源于此）。

---

## 压测场景设计

### 1. Embeddings 压测（高 QPS 高频调用的核心路径）

**推荐脚本**：`scripts/benchmark_embeddings.py`  
**建议目标**：观察在不同 `MAX_CONCURRENT`、`BATCH_WINDOW_MS`、`BATCH_WINDOW_MAX_SIZE` 下的吞吐与 p95/p99。

示例命令：

```bash
# 单模型、并发 8、请求数 200
uv run python scripts/benchmark_embeddings.py \
  --models BAAI/bge-m3 \
  --n-requests 200 \
  --concurrency 8 \
  --base-url http://localhost:8000
```

**可尝试的维度**：

- `MAX_CONCURRENT = 1 / 2 / 4`
- `BATCH_WINDOW_MS = 0 / 4 / 6 / 10`
- `BATCH_WINDOW_MAX_SIZE = 8 / 16 / 32`

**关注点**：

- 吞吐量随 `MAX_CONCURRENT` 和 batch window 变化是否有明显提升；
- 429（Too Many Requests）是否大量出现（说明队列配置过小或模型过慢）；
- tail latency（p95/p99）是否在可接受范围内。

---

### 2. Chat 文本压测（LLM 短对话）

**推荐脚本**：`scripts/benchmark_chat.py`  
**建议目标**：验证 `ENABLE_CHAT_BATCHING` 下的批处理效果，以及 prompt 限制与回退行为。

示例命令：

```bash
uv run python scripts/benchmark_chat.py \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --prompt "Explain FP8 quantization" \
  --n-requests 100 \
  --concurrency 8 \
  --base-url http://localhost:8000
```

**可尝试的维度**：

- `CHAT_BATCH_WINDOW_MS = 4 / 8 / 10`
- `CHAT_BATCH_MAX_SIZE = 4 / 8`
- `CHAT_MAX_PROMPT_TOKENS` 是否足够大；过小会导致 400。

**关注点**：

- 在批处理开启时，是否能观察到明显更高吞吐；
- 429 是否主要来自 chat 批队列（而不是全局 limiter），可通过指标区分；
- prompt 超长时是否返回友好的 400 错误。

---

### 3. Chat + Vision（Qwen-VL）功能验证性压测

这类调用通常更重，不建议高并发+长压，只需要：

- 低并发（例如 2–4）；
- 少量请求（例如 10–20），验证：

  - 远程图片关闭时（`ALLOW_REMOTE_IMAGES=0`），使用 `data:` / 本地路径是否稳定；
  - 开启远程图片（`ALLOW_REMOTE_IMAGES=1` 且配置 `REMOTE_IMAGE_HOST_ALLOWLIST`）时，大图片或非法 host 能否被拒绝。

样例 curl（参考 README 中 Qwen-VL 示例）即可，不必额外写脚本。

---

### 4. Whisper 音频压测

**推荐脚本**：`scripts/benchmark_audio.py`  
**建议目标**：验证音频路径不会拖垮 embeddings/chat，`AUDIO_MAX_CONCURRENT` 与 `AUDIO_MAX_QUEUE_SIZE` 是否合适。

示例命令：

```bash
BASE_URL=http://localhost:8000 \
MODEL_NAME=openai/whisper-tiny \
uv run python scripts/benchmark_audio.py \
  -- --n-requests 40 --concurrency 4
```

**关注点**：

- 如果 CPU-only 或低性能 GPU，`AUDIO_MAX_CONCURRENT` 建议从 1 开始；
- 音频请求不应显著拉高 embeddings/chat 的队列等待时间。

---

## 关键监控指标与预期

### 1. 请求级别指标

**Embeddings**

- `embedding_requests_total{model,status}`
- `embedding_request_latency_seconds{model}`
- `embedding_request_queue_wait_seconds{model}`

**Chat**

- `chat_requests_total{model,status}`
- `chat_request_latency_seconds{model}`
- `chat_request_queue_wait_seconds{model}`

**Audio**

- `audio_requests_total{model,status}`
- `audio_request_latency_seconds{model}`
- `audio_request_queue_wait_seconds{model}`

**全局队列拒绝**

- `embedding_queue_rejections_total`（由 limiter / audio_limiter 共享）

**建议检查**：

- status=200 的计数是否与压测工具统计的成功请求数吻合；
- status=429 / 503 是否在可接受比例内（短时间调参时允许小比例，长期运行应尽量减少）；
- 队列等待时间直方图中是否有大量 >0.1s 或 >1s 的样本。

---

### 2. 批处理与缓存指标

**Chat 批处理**

- `chat_batch_queue_size{model}`：当前 chat batch 队列深度
- `chat_batch_size{model}`：单次批的请求数分布
- `chat_batch_wait_seconds{model}`：从 enqueue 到 batch 执行的等待时间
- `chat_batch_oom_retries_total{model}`：批处理 OOM 重试次数
- `chat_batch_queue_rejections_total{model}`：因队列限制被拒绝的请求数
- `chat_batch_requeues_total{model}`：因配置不兼容/回退导致的重入队列次数
- `chat_count_pool_size`：token counting 线程池大小

**Embedding 批处理与缓存**

- `embedding_batch_wait_seconds{model}`：embedding 批处理等待时间
- `embedding_cache_hits_total{model}`
- `embedding_cache_misses_total{model}`

**建议检查**：

- chat 批大小是否集中在合理范围（如 2–8）；
- 批等待时间是否与 `CHAT_BATCH_WINDOW_MS` 大致对应，不应经常远超窗口；
- cache hit/miss 比例是否符合预期：高重复内容场景期待较高 hit；纯随机文本则 miss 为主。

---

### 3. Warmup 与健康状态

**Warmup 指标**

- `warmup_pool_ready_workers{model,capability,executor}`

**Health 信息（`GET /health`）**

- `status`：`ok` / `unhealthy`
- `warmup.ok_models`、`warmup.failures`
- `warmup.capabilities[model][capability]`
- `chat_batch_queues` / `embedding_batch_queues`：队列深度与 max size
- `runtime_config`：当前生效的关键参数快照

**建议检查**：

- 启动后 warmup 是否对预期模型与能力标记为 `True`；
- 若关闭或限制 warmup（通过 `ENABLE_WARMUP=0` 或 allowlist/skiplist），`/health` 输出是否与配置一致；
- 压测期间 `chat_batch_queues` / `embedding_batch_queues` 是否长期接近上限（若是，说明限流/批窗口/模型瓶颈需调整）。

---

## 典型问题与调参思路（Checklist）

1. **大量 429（Too Many Requests）**
   - 看 `embedding_queue_rejections_total`、`chat_batch_queue_rejections_total`：
     - 若是 limiter 导致 → 适当增大 `MAX_QUEUE_SIZE` 或 `MAX_CONCURRENT`，同时观察 p99；
     - 若是 chat batch 队列导致 → 调大 `CHAT_BATCH_QUEUE_SIZE`，或减小 batch window/批大小，避免积压。

2. **p99 延迟过高但 GPU/CPU 利用率不高**
   - 检查：
     - `BATCH_WINDOW_MS` 是否设置过大；
     - chat/embedding 队列等待直方图是否拉长；
   - 调整策略：
     - 降低 `BATCH_WINDOW_MS` / `CHAT_BATCH_WINDOW_MS`；
     - 在单机场景下尝试 `MAX_CONCURRENT=1` 或 2，通过批处理抬高吞吐，而不是单纯加并发。

3. **音频请求拖慢整体响应**
   - 查看：
     - `audio_request_queue_wait_seconds` 是否显著升高；
     - embeddings/chat 的 queue wait 是否同步恶化。
   - 调整：
     - 降低 `AUDIO_MAX_CONCURRENT`，并确保 `MAX_CONCURRENT` 主要服务 embeddings/chat；
     - 必要时将音频单独部署一个实例。

4. **warmup 阶段启动时间过长或偶发 OOM**
   - 降低：
     - `WARMUP_BATCH_SIZE`、`WARMUP_STEPS`；
   - 或限制：
     - `WARMUP_VRAM_BUDGET_MB`、`WARMUP_VRAM_PER_WORKER_MB`，让 warmup 使用更保守的 worker 数；
   - 通过 `/health` 与 `warmup_pool_ready_workers` 确认 warmup 覆盖与失败模型。

---

## 附录：常用命令速查

- 列出模型：

```bash
curl http://localhost:8000/v1/models
```

- 健康检查：

```bash
curl http://localhost:8000/health | jq .
```

- 抓取部分 Metrics（手看）：

```bash
curl http://localhost:8000/metrics | grep -E "embedding_request_latency|chat_request_latency|audio_request_latency"
```

- 快速 Embeddings 手工测试：

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"BAAI/bge-m3","input":["hello","world"]}'
```

你可以按本文件的步骤，在需要压测时逐条执行和对照监控。

