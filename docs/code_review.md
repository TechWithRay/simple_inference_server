# Codebase assessment

This document captures the current state of the codebase after enabling eager loading and broad warmup coverage. It highlights strengths, risks, and opportunities for further optimization.

## What works well

- **Startup contracts and fail-fast behavior**: The main startup path enforces model allowlisting, downloads when enabled, and validates dependencies like `ffmpeg`; optional warmup now fans out across all capabilities (embeddings, chat, vision) with per-worker execution and VRAM budgeting to avoid oversubscription.【F:app/main.py†L14-L91】【F:app/warmup.py†L98-L182】
- **Backpressure and safety rails**: A bounded limiter couples `_queue` and a semaphore so requests either acquire capacity or receive clear 429/503-style errors without unbounded buffering; chat batching adds its own bounded queue and prompt-length guard before scheduling heavy work.【F:app/concurrency/limiter.py†L21-L97】【F:app/chat_batching.py†L47-L175】
- **Batching and caching for high-frequency paths**: Embedding and chat handlers share configurable executors sized to `MAX_CONCURRENT`, leverage micro-batching with windowed coalescing, and embedder models include per-request no-grad guards plus an LRU cache to avoid redundant computation (see individual model implementations).【F:app/threadpool.py†L1-L46】【F:app/batching.py†L1-L180】

## Engineering notes / conventions

- When adding internal tasks that use `limiter` / `audio_limiter`, always set a queue label (model or task name) via `set_queue_label` / `set_audio_queue_label` so queue-wait metrics stay attributable instead of falling back to `generic`.
- New audio/vision handlers must be thread-safe if they run in a shared thread pool; either design them as thread-safe or protect non-thread-safe resources (tokenizers, pipelines, HTTP clients) with locks similar to `WhisperASR`.
- **Requeue path on chat batching**: When batching splits by generation parameters, leftover items are now requeued with bounded exponential backoff and a per-item deadline. This smooths burstiness without unbounded retries; items that cannot be requeued in time surface as 429s and are counted via `CHAT_BATCH_REQUEUES` / `CHAT_BATCH_QUEUE_REJECTIONS` metrics.【F:app/chat_batching.py†L171-L365】
- **Visibility of warmup coverage**: Warmup metrics record pool readiness and `/health` exposes `warmup` details including per-model capability success plus `warmup_failures`. Operators can see which models/capabilities skipped or failed warmup without inspecting logs.【F:app/api.py†L1076-L1136】【F:app/warmup.py†L140-L185】

## Documentation alignment

- README now describes warmup across all capabilities plus configuration toggles for budgets, allow/skip lists, and fail-fast behavior so operators can keep eager-loading guarantees consistent with runtime flags.【F:README.md†L180-L210】
