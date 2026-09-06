"""Single source of truth for the from-scratch 125M SLM build."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# Project identity / paths (all paths are relative to the Modal Volume mount).
PROJECT = "slm-125m"
HF_REPO = "IndraniBera/slm-125m-base"  # set to your own HF namespace for Phase 6
HF_REPO_SFT = "IndraniBera/slm-125m-sft"  # Phase 10: SFT model push

VOLUME_NAME = "slm-125m"
DATA_ROOT = "/data"
CLEAN_DIR = f"{DATA_ROOT}/clean"          # Phase 1
CORPUS_DIR = f"{DATA_ROOT}/corpus"        # Phase 2
TOKENIZER_DIR = f"{DATA_ROOT}/tokenizer"  # Phase 3
TOKENS_DIR = f"{DATA_ROOT}/tokens"        # Phase 4
TRAIN_TOKENS_DIR = f"{TOKENS_DIR}/train"
VAL_TOKENS_DIR = f"{TOKENS_DIR}/val"
CKPT_DIR = f"{DATA_ROOT}/checkpoints"     # Phase 5 (not in this brief)
BASE_CKPT_DIR = f"{CKPT_DIR}/base"
RESUME_CKPT_PATH = f"{CKPT_DIR}/ckpt.pt"
METRICS_PATH = f"{CKPT_DIR}/metrics.jsonl"

SFT_DIR = f"{DATA_ROOT}/sft"              # Phase 7
SFT_RAW_DIR = f"{SFT_DIR}/raw"
SFT_TOKENS_DIR = f"{SFT_DIR}/tokens"      # Phase 8
SFT_INDEX_PATH = f"{SFT_DIR}/sft_index.json"
SFT_CKPT_DIR = f"{CKPT_DIR}/sft"          # Phase 9

HF_SECRET_NAME = "huggingface-token"
OPENAI_SECRET_NAME = "openai-token"
QA_MODEL = "gpt-5.4-nano"


@dataclass(frozen=True)
class ModelConfig:
    """Maps 1:1 to transformers.LlamaConfig. ~125.8M params with tied embeddings."""

    vocab_size: int = 16_384
    hidden_size: int = 768
    intermediate_size: int = 3_072        # SwiGLU inner
    num_hidden_layers: int = 12
    num_attention_heads: int = 12         # head dim 64
    num_key_value_heads: int = 12         # == heads -> MHA
    max_position_embeddings: int = 1_024  # context length
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"              # SwiGLU
    tie_word_embeddings: bool = True
    attention_bias: bool = False

    def to_llama_kwargs(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "hidden_act": self.hidden_act,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_bias": self.attention_bias,
        }

    def approx_params(self) -> int:
        e = self.vocab_size * self.hidden_size
        h, i = self.hidden_size, self.intermediate_size
        kv = self.num_key_value_heads * (h // self.num_attention_heads)
        attn = h * h + 2 * (h * kv) + h * h
        mlp = 3 * h * i
        per_layer = attn + mlp + 2 * h
        return e + self.num_hidden_layers * per_layer


MODEL = ModelConfig()

SPECIAL_TOKENS: Mapping[str, str] = {
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
    "pad_token": "<|pad|>",
    "unk_token": "<|unk|>",
}
EXTRA_CHAT_TOKENS: tuple[str, ...] = ("<|user|>", "<|assistant|>", "<|system|>")


@dataclass(frozen=True)
class Source:
    name: str
    hf_id: str
    token_budget: int      # stop streaming this source at ~this many clean tokens
    text_field: str
    split: str = "train"
    config_name: str | None = None
    strict_ocr: bool = False


# Legal-first mix (Choice A). Legal sources cap ~2B tokens, so take all of them
# and cap web at 0.5B. NOT 70/20/10. Realized ~40/40/20 (case-law/sec/web).
DATA_MIX: tuple[Source, ...] = (
    Source("case-law", "HFforLegal/case-law", 1_000_000_000, "document",
           split="us", strict_ocr=True),
    Source("sec", "PleIAs/SEC", 1_300_000_000, "text", split="train"),
    Source("fineweb-edu", "HuggingFaceFW/fineweb-edu", 500_000_000, "text",
           split="train", config_name="sample-10BT"),
)

TARGET_TOKENS: int = 2_500_000_000
CHARS_PER_TOKEN: float = 4.0

# Held OUT of training; Phase 2 strips docs resembling these.
EVAL_HOLDOUT: tuple[str, ...] = ("coastalcph/lex_glue", "casehold/casehold")


@dataclass(frozen=True)
class QASource:
    name: str
    corpus_subdir: str       # Source.name whose Phase-2 .txt files we sample excerpts from
    num_pairs: int           # total QA pairs to generate for this domain
    pairs_per_call: int = 5  # QA pairs requested per OpenAI call (amortizes excerpt cost)


# 1,200 pairs total, weighted to match the legal-first pretraining mix (~55/25/20).
QA_MIX: tuple[QASource, ...] = (
    QASource("legal", "case-law", 660),
    QASource("sec", "sec", 300),
    QASource("general", "fineweb-edu", 240),
)

QA_EXCERPT_CHARS: int = 4_800   # ~1,200 tokens at 4 chars/token, fed as generation context
QA_SYSTEM_PROMPT: str = (
    "You write concise question-answer training pairs for a small language model "
    "with a 1024-token context window. Ground every pair ONLY in the excerpt "
    "provided; never invent facts absent from it. "
    'Return strict JSON of the form {"pairs": [{"question": str, "answer": str}]}. '
    "Questions must be answerable from the excerpt alone. Answers must be under "
    "~150 words, self-contained, and factual."
)


@dataclass(frozen=True)
class CleanConfig:
    min_line_chars: int = 40
    max_nonalnum_ratio: float = 0.30
    min_doc_chars: int = 600
    repetition_top_k: int = 10
    max_repetition_ratio: float = 0.50
    ngram_n: int = 4
    lang_sample_chars: int = 5_000
    nonword_ratio_max: float = 0.20     # OCR gate: drop if >20% words non-dictionary
    ocr_min_tokens: int = 50
    dict_path: str = "/usr/share/dict/words"


CLEAN = CleanConfig()

SEQ_LEN: int = 1_024
VAL_EVERY_N_WINDOWS: int = 100          # every 100th window -> val (99/1 split)
TOKENS_DTYPE: str = "uint16"


@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = SEQ_LEN
    micro_batch_size: int = 32
    global_batch_tokens: int = 524_288
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_tokens: int = 200_000_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    ckpt_every_steps: int = 500
    log_every_steps: int = 20
    eval_every_steps: int = 1_000
    seed: int = 1337


TRAIN = TrainConfig()

PRETRAIN_GPU = "H100"
PRETRAIN_GPU_COUNT = 8
BUDGET_CAP_USD = 40.0


@dataclass(frozen=True)
class SFTConfig:
    seq_len: int = SEQ_LEN
    epochs: int = 2
    micro_batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 20
    ckpt_every_steps: int = 100
    log_every_steps: int = 10
    seed: int = 1337


SFT = SFTConfig()
SFT_GPU = "L4"  # single GPU: SFT set is ~1000x smaller than the pretrain corpus

# ---- RAFT: continue-train the SFT checkpoint to read noisy retrieved context ----
RAFT_DIR = f"{DATA_ROOT}/raft"
RAFT_RAW_DIR = f"{RAFT_DIR}/raw"
RAFT_TOKENS_DIR = f"{RAFT_DIR}/tokens"
RAFT_INDEX_PATH = f"{RAFT_DIR}/raft_index.json"
RAFT_CKPT_DIR = f"{CKPT_DIR}/raft"          # separate from SFT_CKPT_DIR; SFT weights untouched
HF_REPO_RAFT = "IndraniBera/slm-125m-raft"  # Phase 12: RAFT model push

RAFT_CHUNK_CHARS = 880       # ~220 tokens at CHARS_PER_TOKEN
RAFT_CHUNK_MIN_CHARS = 150
RAFT_GOLDEN_PRESENT_PROB = 0.8  # RAFT paper default: 20% golden-omitted


@dataclass(frozen=True)
class RAFTConfig:
    seq_len: int = SEQ_LEN
    epochs: int = 4              # bumped from 2: 148 steps was too little signal
    micro_batch_size: int = 16
    lr: float = 3e-5            # bumped from 2e-5, still below SFT.lr (second continuation pass)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 20
    ckpt_every_steps: int = 100
    log_every_steps: int = 10
    seed: int = 1337


RAFT = RAFTConfig()
RAFT_GPU = "L4"

# ---- RAG: BM25 retrieval index served alongside the RAFT checkpoint ----
RAG_DIR = f"{DATA_ROOT}/rag"
RAG_CHUNKS_PATH = f"{RAG_DIR}/chunks.jsonl"
RAG_BM25_PATH = f"{RAG_DIR}/bm25_index.json"
RAG_CHUNK_CHARS = 880
RAG_CHUNK_MIN_CHARS = 150
RAG_TARGET_CHUNKS = 6_000    # proportioned across domains to match QA_MIX weights
RAG_RETRIEVE_K = 3           # matches RAFT's training-time chunk count

STAGES: tuple[str, ...] = (
    "setup", "clean", "dedup", "tokenizer", "tokenize", "pretrain", "deploy",
    "qa_generate", "sft_tokenize", "sft_train", "raft_prepare", "raft_tokenize",
    "raft_train", "rag_index",
)


if __name__ == "__main__":
    p = MODEL.approx_params()
    print(f"{PROJECT}")
    print(f"model: {p:,} params (~{p/1e6:.1f}M) | vocab {MODEL.vocab_size} | "
          f"{MODEL.num_hidden_layers}L/{MODEL.hidden_size}d/"
          f"{MODEL.num_attention_heads}h kv={MODEL.num_key_value_heads}")
    print(f"target tokens: {TARGET_TOKENS/1e9:.1f}B (~{TARGET_TOKENS/p:.0f} tok/param)")
    print(f"stages: {' -> '.join(STAGES)}")
