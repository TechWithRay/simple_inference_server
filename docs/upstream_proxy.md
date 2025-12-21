# 上游代理模型（OpenAI / vLLM）

本服务可以作为“单一 OpenAI-compatible 网关”：你的应用只配置一个推理服务地址（本服务），但可以把部分 `model` 请求转发到上游 OpenAI 兼容服务（例如 **vLLM** 或 **OpenAI**），同时代理流量与本地流量的限流/队列相互隔离。

## 你能得到什么

- **显式路由**：只有在 `model_config` 里声明为“代理模型”的条目会被转发。
- **原样透传**：代理路径不做严格字段校验，`tools`、`tool` role、厂商扩展字段等会被保留；上游响应也原样返回。
- **独立流控**：OpenAI 代理与 vLLM 代理各自有独立的 limiter（不与本地 chat/embedding/vision/audio 限流互相影响）。
- **流式透传**：代理 chat 支持 `stream=true`，会将上游的 SSE（`text/event-stream`）直接转发给客户端（本地模型仍保持不支持 streaming 的现状）。

## 快速开始

1. 在 `configs/model_config.local.yaml` 添加代理模型条目（推荐使用 overlay，且该文件已在 `.gitignore` 中忽略）。
2. `MODELS` 里填写代理模型的 **name**（若未写 `name`，则使用 `hf_repo_id` 作为模型 id）。
3. 配置上游地址/鉴权（环境变量或单模型覆盖），启动服务即可。

## 关键概念：`name` vs `hf_repo_id`（非常重要）

本服务加载模型时：

- **`name`（可选）**：客户端/应用看到的模型 ID（请求 `model` 字段）以及 `MODELS` 里使用的 ID。
- **`hf_repo_id`（必填）**：
  - 对本地模型：它就是 Hugging Face repo id。
  - 对代理模型：我们复用该字段作为**上游 model id**（即网关转发给上游时使用的 `model`）。

也就是说，代理模型的典型配置是：

- 应用侧使用：`model: "<name>"`（例如 `proxy-chat`）
- 网关转发上游使用：`model: "<hf_repo_id>"`（例如 `gpt-4o-mini`）

## 支持的端点

### `POST /v1/chat/completions`

- **本地模型**：保持现有行为（目前不支持 streaming；请求 schema 更严格；本地结构化输出可能会做校验与重试）。
- **代理模型**：原样透传到上游 `/v1/chat/completions`。
  - 如果 `stream=true`，网关会把上游 SSE 流转发给客户端。

### `POST /v1/embeddings`

- **本地模型**：保持现有行为。
- **代理模型**：原样透传到上游 `/v1/embeddings`（非流式）。

## 配置代理模型（YAML）

推荐在 `configs/model_config.local.yaml` 添加：

### OpenAI chat 代理

```yaml
models:
  - name: "proxy-chat"
    hf_repo_id: "gpt-4o-mini"  # 上游 model id
    handler: "app.models.openai_proxy.OpenAIChatProxyModel"
    # 可选覆盖（单模型粒度）：
    # upstream_base_url: "https://api.openai.com/v1"
    # upstream_api_key_env: "OPENAI_API_KEY"
    # upstream_timeout_sec: 60
    # upstream_headers:
    #   X-My-Header: "foo"
```

### vLLM chat 代理

```yaml
models:
  - name: "heavy-qwen"
    hf_repo_id: "Qwen3-32B-Instruct"
    handler: "app.models.vllm_proxy.VLLMChatProxyModel"
    upstream_base_url: "http://127.0.0.1:8001/v1"
```

启动时设置：

- `MODELS=proxy-chat,heavy-qwen,...`

## 代理 handler 一览

- **OpenAI 上游**：
  - `app.models.openai_proxy.OpenAIChatProxyModel`
  - `app.models.openai_proxy.OpenAIEmbeddingProxyModel`
- **vLLM 上游**：
  - `app.models.vllm_proxy.VLLMChatProxyModel`
  - `app.models.vllm_proxy.VLLMEmbeddingProxyModel`

## 单模型可用字段（会被传给 handler 的 `config=`）

- **`upstream_base_url`**：上游 base URL，可写 `http://host` 或 `http://host/v1`，网关会规范化到 `/v1`。
  - vLLM 代理如果没写该字段，则必须提供 `VLLM_BASE_URL`。
- **`upstream_timeout_sec`**：上游请求超时（秒）。
- **`upstream_api_key_env`**：从哪个环境变量读取 API key（推荐，避免把 key 写进 YAML）。
- **`upstream_api_key`**：直接写 key（支持但不推荐提交到 git）。
- **`upstream_headers`**：附加的自定义请求头（例如给 ingress/路由使用）。
- **`skip_download`**：为 true 时，`AUTO_DOWNLOAD_MODELS` 和 `scripts/download_models.py` 会跳过该条目。
  - 代理 handler 默认也会被自动跳过下载，这个字段更多是给其他非 HF handler 用。

## 环境变量

### OpenAI

- `OPENAI_BASE_URL`（默认 `https://api.openai.com/v1`）
- `OPENAI_API_KEY`（可选）
- `OPENAI_PROXY_TIMEOUT_SEC`（默认 `60`）
- `OPENAI_PROXY_MAX_CONCURRENT` / `OPENAI_PROXY_MAX_QUEUE_SIZE` / `OPENAI_PROXY_QUEUE_TIMEOUT_SEC`（可选；不填则回退到全局 `MAX_CONCURRENT` / `MAX_QUEUE_SIZE` / `QUEUE_TIMEOUT_SEC`）

### vLLM

- `VLLM_BASE_URL`（未在单模型里设置 `upstream_base_url` 时必填）
- `VLLM_API_KEY`（可选）
- `VLLM_PROXY_TIMEOUT_SEC`（默认 `60`）
- `VLLM_PROXY_MAX_CONCURRENT` / `VLLM_PROXY_MAX_QUEUE_SIZE` / `VLLM_PROXY_QUEUE_TIMEOUT_SEC`（可选；不填则回退到全局）

## 鉴权行为（重要）

网关发往上游的 `Authorization` 逻辑：

- 如果服务端配置了 key（例如 `OPENAI_API_KEY`，或单模型 `upstream_api_key_env` / `upstream_api_key`），网关会使用：
  - `Authorization: Bearer <key>`
- 否则，如果客户端请求带了 `Authorization`，网关会**透传该 header**到上游。

安全建议：

- 如果你不希望客户端“自带上游 key”，请在网关侧固定配置 `OPENAI_API_KEY`（并对外加一层鉴权/网络隔离），不要把网关裸露在公网。

## 独立限流与常见错误码

代理流量使用独立 limiter：

- OpenAI 代理：`OPENAI_PROXY_*`
- vLLM 代理：`VLLM_PROXY_*`

常见返回：

- `429 Too Many Requests`（队列满或等待超时），并带 `Retry-After`
- `503 Service Unavailable`（服务正在 shutdown/drain）
- `504 Gateway Timeout`（上游请求超时）
- `502 Bad Gateway`（上游 HTTP 层失败）

## warmup 与下载行为

- 代理模型会被 **自动跳过下载**（避免将 `gpt-4o-mini` 之类当作 HF repo 去下载）。
- 代理模型会被 **默认跳过 warmup**（避免默认启动就依赖上游可用/有 key）。
- 如果你把代理模型加入 `WARMUP_ALLOWLIST`，warmup 会对上游做一次最小请求；若 warmup 开启且失败，可能触发启动 fail-fast（与本地模型一致）。

## 可观测性

- `GET /v1/models` 会返回 `owned_by`（`local` / `openai` / `vllm`）
- 日志：`chat_proxy_request`、`embedding_proxy_request` 会记录上游类型、状态码、耗时

## Ruby SDK 使用（可选）

Ruby SDK 只需要把 `base_url` 指向本服务，然后把 `model` 设为代理模型的 `name` 即可。

流式示例（代理模型会真正走 SSE；本地模型则 SDK 会自动 fallback 到非流式）：

```ruby
client.chat_completions_stream(
  model: "proxy-chat",
  messages: [{ "role" => "user", "content" => "hi" }]
) do |event|
  # event 为解析后的 SSE JSON chunk
end
```


