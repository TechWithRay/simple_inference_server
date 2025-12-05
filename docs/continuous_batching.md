# Static Request Batching (Chat)

Scope and priorities
- Phase 1 (implemented): HF-backed **static request batching** for text-only chat models. This raises throughput by grouping compatible requests into a single inference call, but does not support continuous/iteration-level scheduling (joining mid-generation).
- Phase 2 (later): add true continuous batching with streaming responses and user-side abort.

Why
- A queue + scheduler lets concurrent chat requests share prefill/decoding kernels, improving GPU/CPU utilization.
- Transformers >=4.46 ships batching primitives, but for simplicity, we currently use a windowed grouping approach.

Phase 1 behavior (Static/Windowed Batching)
- **Grouping Strategy**: A per-model worker collects requests within `CHAT_BATCH_WINDOW_MS` (e.g., 10ms).
- **Compatibility Constraint**: Requests are only batched if they share **identical decoding parameters** (temperature, top_p, max_new_tokens, stop sequences).
- **Execution**: Uses `model.batched_generate` (or `batched_generate_prepared`). The batch runs until all requests in it are finished (static batching).
- **Limits**: Prompt length guard (`CHAT_MAX_PROMPT_TOKENS`) and `CHAT_MAX_NEW_TOKENS` ceiling reject oversize requests early.
- **OOM Handling**: On CUDA OOM, the batcher automatically halves the batch size and retries, providing graceful degradation.

Testing
- Unit tests cover the scheduler (batch windowing, config grouping, overflow rejections).
- Integration tests verify throughput improvements and OOM recovery.
