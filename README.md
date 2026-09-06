# SLM-125M — a 125M-parameter language model built from scratch

This repository is the complete build pipeline for **slm-125m**: a small,
Llama-style language model trained *from nothing* on US legal and financial text,
plus the web frontend that serves it.

Nothing here is a fine-tune of an existing model. The tokenizer, the weights, the
fine-tuning data, and the retrieval index are all built by the code in this repo,
mostly on rented cloud GPUs, for roughly **$20 total**.

**Live demo:** https://slm-125m-psi.vercel.app
**Models on Hugging Face** (private): `IndraniBera/slm-125m-base`, `-sft`, `-raft`

---

## What the model is

| Property | Value |
|---|---|
| Architecture | Llama-style decoder (RoPE, SwiGLU, RMSNorm, no biases, tied embeddings) |
| Size | ~125.8M parameters |
| Layers / width / heads | 12 layers / 768 hidden / 12 attention heads |
| Vocabulary | 16,384 tokens, byte-level BPE trained on this corpus |
| Context length | 1,024 tokens |
| Training data | ~2.04B tokens: US case law, SEC filings, and a slice of educational web text |
| Final pretraining loss | 9.107 → 2.4832 (1 epoch, ~3,877 steps) |

It is a **base (completion) model** first. It is then fine-tuned twice — once for
question-answering (SFT), once to read retrieved passages (RAFT) — and paired with
a keyword search index so it can answer questions with citations (RAG).

---

## How the pipeline works

The build runs in numbered phases. Everything runs on [Modal](https://modal.com)
(serverless cloud compute); datasets are streamed from Hugging Face and never
fully downloaded.

| Phase | Stage | Where | What happens |
|---|---|---|---|
| 0 | setup | CPU | Smoke test + measure how many usable tokens each source yields |
| 1 | clean | CPU | Stream each source, run the deterministic cleaner, keep ~698K docs |
| 2 | dedup | CPU | MinHash/LSH near-duplicate removal + exact dedup + strip anything resembling the eval sets → ~670K docs |
| 3 | tokenizer | CPU | Train a fresh 16,384-token byte-level BPE tokenizer on the clean corpus |
| 4 | tokenize | CPU | Encode + pack the corpus into 1,024-token `uint16` windows, 99/1 train/val split |
| 5 | pretrain | 8×H100 | Train the base model — 1 epoch, cosine LR 6e-4→6e-5, 524K-token batches, resumable from checkpoint |
| 6 | deploy | CPU | Push the base model + tokenizer to the Hugging Face Hub |
| 7–10 | SFT | 1×L4 | Generate 1,196 grounded QA pairs with `gpt-5.4-nano`, tokenize them loss-masked, fine-tune the base model, evaluate on CaseHOLD |
| 11–12 | RAFT | 1×L4 | Rebuild the same QA pairs with 3 retrieved-style passages each (some with the answer, some without), continue-train from the SFT checkpoint |
| 13 | rag_index | CPU | Build an in-memory BM25 keyword index over 6,000 corpus chunks (no vector DB) |
| — | serve | CPU | Modal HTTP endpoints run inference for the three playgrounds; the Vercel app calls them |

**Not built yet:** RLAIF (preference tuning) is planned only — see
`instruction-QA SFT + RLAIF`. No reward-model or DPO code exists.

### The data mix

Legal-first, because the legal sources are the point and they cap out around 2B
tokens on their own:

| Source | Hugging Face ID | Role | Realized share |
|---|---|---|---|
| Case law | `HFforLegal/case-law` (`us`) | US court opinions (some OCR noise) | ~35% |
| SEC filings | `PleIAs/SEC` | 10-Ks etc., born-digital | ~42% |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | general fluency filler | ~23% |

Anything resembling `casehold/casehold` or `coastalcph/lex_glue` is stripped from
training so the CaseHOLD evaluation stays honest.

---

## The Python files

There are only four, and `config.py` drives all of them.

### `config.py` — the single source of truth

Every number in the build lives here as a frozen dataclass: model shape
(`ModelConfig`), cleaning thresholds (`CleanConfig`), the optimizer and schedule
for each training stage (`TrainConfig`, `SFTConfig`, `RAFTConfig`), the dataset
mix (`DATA_MIX`), the QA-generation mix (`QA_MIX`), all filesystem paths on the
Modal volume, the Hugging Face repo names, and the ordered list of pipeline
stages (`STAGES`). No other file hard-codes these values — they all import from
here.

```bash
python config.py   # prints the model's parameter count, token budget, and stage list
```

### `cleaning.py` — the deterministic document cleaner (Phase 1)

Pure functions, no side effects, so the corpus is reproducible. Given a raw
document it:

- drops short or symbol-heavy lines (`filter_lines`)
- removes known boilerplate — "Table of Contents", "Form 10-K", SEC address blocks, signature lines (`strip_boilerplate`)
- rejects documents that loop the same phrases (`is_repetitive`)
- rejects non-English text — fast ASCII-ratio check first, `langdetect` only for the ambiguous band (`is_english`)
- for scanned sources, rejects OCR garble by checking what fraction of words aren't in the system dictionary (`is_ocr_garble`)

`clean_document()` runs the whole chain and returns a `CleanResult` (kept or not,
plus the reason and character counts).

### `dedup.py` — deduplication + decontamination helpers (Phase 2)

Small pure helpers used to find and remove duplicates:

- `normalize` / `words` — lowercase, collapse whitespace, tokenize
- `exact_hash` — a blake2b fingerprint for exact-duplicate detection
- `word_ngrams` / `shingles` — n-gram and shingle sets that feed MinHash/LSH near-duplicate detection, and also the "does this doc look like the eval set" contamination check

### `modal_app.py` — the orchestrator (everything)

The one large file. It defines the Modal app, the container images (CPU, GPU,
inference, etc.), and **one function per phase** — cleaning shards, building
near-dup clusters, training the tokenizer, packing tokens, the DDP pretraining
loop, the Hugging Face pushes, QA-pair generation via OpenAI, the SFT and RAFT
training loops, the CaseHOLD and perplexity evals, and the BM25 index builder.

Each phase has a `@app.local_entrypoint()` you invoke with `modal run`:

```bash
modal run modal_app.py::clean            # Phase 1
modal run modal_app.py::dedup            # Phase 2
modal run modal_app.py::tokenizer        # Phase 3
modal run modal_app.py::tokenize         # Phase 4
modal run modal_app.py::pretrain_run     # Phase 5
modal run modal_app.py::deploy           # Phase 6
modal run modal_app.py::generate_qa      # Phase 7
modal run modal_app.py::sft              # Phase 9
modal run modal_app.py::casehold         # Phase 9.7  (base vs SFT eval)
modal run modal_app.py::deploy_sft       # Phase 10
modal run modal_app.py::raft_prepare     # Phase 11
modal run modal_app.py::raft             # Phase 11.6
modal run modal_app.py::deploy_raft      # Phase 12
modal run modal_app.py::rag_index        # Phase 13
```

It also defines the three always-on inference endpoints that back the website:

- `Playground` — base-model text completion
- `PlaygroundSFT` — QA answers from the SFT model
- `PlaygroundRAG` — retrieves 3 chunks with BM25, feeds them to the RAFT model, returns an answer plus its sources

**Prerequisites:** a Modal account, plus Modal secrets named `huggingface-token`,
`openai-token`, and `playground-api-key`.

---

## Other files

| File | What it is |
|---|---|
| `Replication_Guide.md` | A self-contained brief for reproducing the data pipeline (Phases 0–4) from scratch, with exact commands and expected output |
| `index.json` | The token manifest written by Phase 4 — 2.04B train tokens, 1,991,282 windows, per-shard counts |
| `instruction-QA-fine-tuning.txt`, `instruction-RAFT.txt`, `instruction-QA SFT + RLAIF` | The original step-by-step task prompts that drove the SFT, RAFT, and (planned) RLAIF work |
| `SLM-125M-build-flow.pdf` | One-page visual summary of the whole pipeline and its results |
| `web/` | The Next.js frontend, deployed to Vercel |

---

## The web app (`web/`)

A [Next.js](https://nextjs.org) app deployed at
**https://slm-125m-psi.vercel.app**. Its API routes (`web/app/api/*`) are thin
proxies that add the API key and forward requests to the Modal endpoints.

| Route | What it shows |
|---|---|
| `/` | Base-model completion playground, plus architecture / corpus / loss-curve visualizations |
| `/sft` | QA playground (SFT model) with dataset and evaluation stats |
| `/rag` | RAG playground — RAFT model answering with live BM25 retrieval |

```bash
cd web
npm install
npm run dev      # http://localhost:3000
```

Local development needs `web/.env.local` with the Modal endpoint URLs and
`PLAYGROUND_API_KEY` (not committed).

---

## Results at a glance

| Stage | Metric | Value |
|---|---|---|
| Pretraining | train loss | 9.107 → 2.4832 |
| SFT | assistant-token perplexity | 8.11 |
| SFT | CaseHOLD accNorm (base → SFT) | 0.170 → 0.166 |
| RAFT | assistant-token perplexity | 2.821 |
| RAG index | chunks / unique terms | 6,000 / 42,127 |
