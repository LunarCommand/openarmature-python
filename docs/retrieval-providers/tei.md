# Self-hosting TEI

[Text Embeddings Inference](https://huggingface.co/docs/text-embeddings-inference)
(TEI) is HuggingFace's Rust serving stack for encoder models: embedding
bi-encoders and reranking cross-encoders. It is the self-hosted backend
for `TeiEmbeddingProvider` and `TeiRerankProvider`, and the right tool for
this workload where a generation server like vLLM is not. vLLM is built
for autoregressive decoding (KV cache, token streaming), none of which an
encoder uses; TEI serves `/embed` and `/rerank` directly, with Flash
Attention and batching, at lower latency because the architecture matches.

Self-hosting keeps your corpus and queries on your own hardware and takes
the per-token cost of a hosted embedding API to zero, at the price of
running two containers.

## The 30-second version

TEI serves exactly one model per container, so embedding and reranking are
two containers on two ports. A working pair:

```bash
# Embeddings: a bi-encoder on port 8083
docker run --rm --name tei-embed \
  --gpus '"device=0"' \
  -p 8083:80 \
  -v $HOME/.cache/huggingface:/data \
  ghcr.io/huggingface/text-embeddings-inference:86-1.9 \
  --model-id BAAI/bge-small-en-v1.5 \
  --port 80

# Reranking: a cross-encoder on port 8082
docker run --rm --name tei-reranker \
  --gpus '"device=0"' \
  -p 8082:80 \
  -v $HOME/.cache/huggingface:/data \
  ghcr.io/huggingface/text-embeddings-inference:86-1.9 \
  --model-id BAAI/bge-reranker-v2-m3 \
  --port 80 \
  --auto-truncate false
```

TEI listens on port 80 inside the container; `-p 8083:80` maps it to a
host port. The `-v` mount is TEI's model cache, so a restart reloads from
disk instead of re-downloading. First boot pulls the model (seconds for a
small embedder, a minute or two for a larger reranker); later boots are
sub-second.

## One model per container

A single TEI process loads one model, and the endpoints it exposes follow
that model's architecture. An embedding bi-encoder maps one text to one
vector and answers `/embed` and `/v1/embeddings`. A reranking cross-encoder
scores a `(query, document)` pair to one number and answers `/rerank`.
They are different model families, so a full retrieval stack runs two
containers. They share the image and the cache mount and coexist on one
GPU with room to spare (a small embedder needs a few hundred MB, a
large reranker a couple of GB).

| Endpoint | Container | Purpose |
|---|---|---|
| `POST /embed` | embedding | Dense vector(s) for the input text(s) |
| `POST /v1/embeddings` | embedding | OpenAI-compatible embeddings surface |
| `POST /rerank` | reranker | Score documents against a query |
| `GET /health` | both | Liveness (200 OK, empty body) |
| `GET /info` | both | Reports `model_id`, model type, `max_input_length` |
| `GET /metrics` | both | Prometheus metrics |

## Choosing the image

TEI publishes **GPU-architecture-specific images**; the wrong tag will not
start. Pick by your card's CUDA compute capability:

| Tag prefix | Architecture | Example cards |
|---|---|---|
| `cpu-` | none | CPU-only, no GPU |
| (plain, no prefix) | Ampere 8.0 | A100, A30 |
| `86-` | Ampere 8.6 | RTX 3090, A10, A40 |
| `89-` | Ada Lovelace | RTX 4090, L4, L40S |
| `hopper-` | Hopper 9.0 | H100 |

The suffix is the TEI version line (`1.9` in the examples, current as of
mid-2026). So an RTX 3090 uses `:86-1.9`, an RTX 4090 uses `:89-1.9`. The
tag is architecture-specific, not model-specific, so both containers use
the same image and it is cached once. HuggingFace publishes the full
matrix in the [TEI supported-models docs](https://huggingface.co/docs/text-embeddings-inference/supported_models).

## Wiring the providers

Point the TEI providers at the host ports. `base_url` is the instance root
(the provider appends `/embed` or `/rerank` itself), and `model` is the
model you loaded, so the observability layer reports the right identifier:

```python
from openarmature.retrieval import TeiEmbeddingProvider, TeiRerankProvider

embedder = TeiEmbeddingProvider(
    base_url="http://localhost:8083",
    model="BAAI/bge-small-en-v1.5",
)
reranker = TeiRerankProvider(
    base_url="http://localhost:8082",
    model="BAAI/bge-reranker-v2-m3",
)
```

TEI reports no token usage on either surface, so `response.usage` is
`None` for TEI calls; that is the nullable-usage contract, not a bug.

`chunk_size` (default 32) is TEI's per-request batch cap, the server's
`--max-client-batch-size`. `embed()` splits a longer input list into
consecutive chunks under this size and stitches the vectors back in input
order, and `rerank()` chunks a large candidate pool the same way. Set it
to match the server if you raise TEI's limit:

```python
embedder = TeiEmbeddingProvider(
    base_url="http://localhost:8083",
    model="BAAI/bge-small-en-v1.5",
    chunk_size=64,  # match TEI's --max-client-batch-size if you raise it
)
```

## Fail loud on over-length input

The reranker command sets `--auto-truncate false`. By default TEI silently
clips an input that exceeds the model's token window, which quietly changes
what you scored; with auto-truncate off, an over-length `(query, document)`
pair returns a validation error instead. That surfaces through the provider
as `ProviderInvalidRequest`, so an over-length call fails loudly at the
boundary rather than returning a score computed on truncated text. Prefer
it for retrieval, where a silently-clipped document is a correctness bug,
not a warning. The embedding container defaults are usually fine, since a
single short text rarely overflows the window.

## Readiness and smoke tests

`ready()` on either provider issues a minimal probe against its endpoint
(TEI serves no model catalog, so it is an actual embed / rerank call).
Before wiring the providers, confirm the containers directly:

```bash
# Liveness (both): 200 OK, empty body
curl http://localhost:8083/health
curl http://localhost:8082/health

# Model identity and max input length
curl -s http://localhost:8083/info | jq
curl -s http://localhost:8082/info | jq

# Embed: one vector per input; index [0] is the vector
curl -s http://localhost:8083/embed \
  -H 'Content-Type: application/json' \
  -d '{"inputs": "water ice in permanently shadowed lunar craters"}' | jq '.[0] | length'

# Rerank: on-topic docs score high, off-topic near zero, sorted descending
curl -s http://localhost:8082/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "where is there water on the Moon?",
    "texts": [
      "Ice sits in permanently shadowed craters at the lunar poles.",
      "The Sea of Tranquility was the Apollo 11 landing site."
    ]
  }' | jq
```

## Production deployment

Run each container under a process supervisor so it restarts on failure
and boots with the host. A systemd unit that wraps `docker run` in the
foreground (no `-d`) gives you `systemctl start/stop` and `journalctl`
logs, matching how you would run any other serving container:

```ini
# /etc/systemd/system/tei-embed.service
[Unit]
Description=TEI embeddings (bge-small-en-v1.5)
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
# Foreground docker run so systemd owns the lifecycle. --rm cleans up on
# stop; the ExecStartPre clears any stale container a hard crash left behind.
ExecStartPre=-/usr/bin/docker rm -f tei-embed
ExecStart=/usr/bin/docker run --rm --name tei-embed \
  --gpus '"device=0"' \
  -p 8083:80 \
  -v /home/youruser/.cache/huggingface:/data \
  ghcr.io/huggingface/text-embeddings-inference:86-1.9 \
  --model-id BAAI/bge-small-en-v1.5 \
  --port 80
ExecStop=/usr/bin/docker stop tei-embed
Restart=always
RestartSec=5
# First boot pulls the image + model before the port opens; raise the
# start timeout so systemd does not kill the unit mid-download.
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
```

The reranker unit is the same shape with its own name, port, model, and
the `--auto-truncate false` flag. `systemctl enable --now tei-embed` boots
it and starts it; a second unit does the reranker.

A few production notes:

- **Wait for the network.** `network-online.target` in the unit matters:
  the first boot pulls the model from HuggingFace, and pairing it with a
  generous `TimeoutStartSec` keeps systemd from killing the unit while the
  download is still in flight (a small embedder is quick; a large reranker
  can take a minute or two cold).
- **GPU pinning.** `--gpus '"device=0"'` binds a container to one GPU by
  index. On a multi-GPU host where indices can reorder across reboots, pin
  by UUID instead (`--gpus '"device=GPU-..."'`, from `nvidia-smi -L`) so a
  container always lands on the intended card.
- **VRAM planning.** Both containers fit comfortably on a single 24 GB
  card (a small embedder plus a large reranker is a few GB total), leaving
  headroom for other work. Check placement with `nvidia-smi` after start.
- **Cache volume.** Point the `/data` mount at the HuggingFace cache that
  holds the weights (an absolute path, since a `multi-user.target` unit
  runs as root and `%h` would resolve to `/root`). Keep it on fast local
  disk; it is the difference between a sub-second restart and a
  re-download.
- **Health checks.** Point your orchestrator's liveness probe at
  `GET /health` (200 OK) and readiness at `GET /info` (confirms the model
  loaded).

## Where to next

- **[Retrieval Providers](index.md)**: the bundled providers, the
  protocol contract, and the error categories.
- **[Retrieval concept page](../concepts/retrieval.md)**: embedding,
  reranking, `input_type`, chunking, and the nullable-usage contract.
- **[Authoring a Provider](authoring.md)**: implement the protocol for a
  backend TEI does not cover.
