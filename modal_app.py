"""Modal App for the from-scratch 125M SLM build (Phases 0 to 4)."""

from __future__ import annotations

import modal

import config

app = modal.App(config.PROJECT)

# CPU base. All pip/apt build steps MUST come before add_local_* (Modal rule).
_cpu_base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wamerican")  # /usr/share/dict/words for the OCR gate
    .pip_install(
        "datasets==3.6.0",
        "huggingface_hub==0.34.4",
        "langdetect==1.0.9",
        "pyarrow==17.0.0",
        "datasketch==1.6.5",
    )
)
cpu_image = _cpu_base.add_local_python_source("config", "cleaning", "dedup")

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}


def _stream_source(source: "config.Source", n: int):
    from datasets import load_dataset

    ds = load_dataset(source.hf_id, source.config_name, split=source.split, streaming=True)
    for i, record in enumerate(ds):
        if i >= n:
            break
        yield record


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 15)
def smoke_test(n_per_source: int = 10) -> dict:
    from cleaning import clean_document

    summary: dict[str, dict] = {}
    for source in config.DATA_MIX:
        print("\n" + "=" * 78)
        print(f"SOURCE: {source.name}  ({source.hf_id}, split={source.split}, "
              f"field='{source.text_field}')")
        print("=" * 78)
        kept = 0
        reasons: dict[str, int] = {}
        for i, record in enumerate(_stream_source(source, n_per_source)):
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            result = clean_document(text)
            reasons[result.reason] = reasons.get(result.reason, 0) + 1
            kept += int(result.kept)
            excerpt = (result.text[:240] if result.kept else text[:160]).replace("\n", " / ")
            print(f"\n[{source.name} #{i}] raw={result.raw_chars:>7} clean={result.clean_chars:>7} "
                  f"-> {result.reason.upper()}")
            print(f"    {excerpt}")
        summary[source.name] = {"streamed": n_per_source, "kept": kept, "reasons": reasons}
    print("\nSMOKE TEST SUMMARY")
    for name, s in summary.items():
        print(f"  {name:<12} kept {s['kept']}/{s['streamed']}  reasons={s['reasons']}")
    return summary


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20)
def measure_sources(n_per_source: int = 2000) -> dict:
    from cleaning import clean_document

    TOTAL_ROWS = {"case-law": 282_390, "sec": 48_543, "fineweb-edu": 9_670_000}
    out: dict[str, dict] = {}
    for source in config.DATA_MIX:
        clean_chars = kept = 0
        for record in _stream_source(source, n_per_source):
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            r = clean_document(text)
            if r.kept:
                kept += 1
                clean_chars += r.clean_chars
        avg_clean = clean_chars / n_per_source if n_per_source else 0
        total = TOTAL_ROWS[source.name]
        est = total * avg_clean / config.CHARS_PER_TOKEN
        out[source.name] = {"est_clean_tokens": int(est), "keep_rate": round(kept / n_per_source, 3)}
        print(f"{source.name:<12} keep={kept/n_per_source:.0%}  avg_clean={avg_clean:>7.0f} ch/doc  "
              f"rows={total:>9,}  est_clean_tokens={est/1e9:.2f}B")
    print(f"TOTAL est clean tokens: {sum(v['est_clean_tokens'] for v in out.values())/1e9:.2f}B")
    return out


# ---- Phase 1: stream + clean, one worker per parquet shard ----
_SOURCE_BY_NAME = {s.name: s for s in config.DATA_MIX}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 60)
def clean_shard(source_name: str, url: str, shard_index: int, token_cap: int) -> dict:
    import os

    from datasets import load_dataset

    from cleaning import clean_document

    source = _SOURCE_BY_NAME[source_name]
    out_dir = f"{config.CLEAN_DIR}/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/shard-{shard_index:03d}.txt"
    ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
    streamed = kept = clean_chars = 0
    reasons: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as fh:
        for record in ds:
            streamed += 1
            text = record.get(source.text_field) or ""
            if not isinstance(text, str):
                text = str(text)
            r = clean_document(text, strict_ocr=source.strict_ocr)
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
            if r.kept:
                fh.write(r.text.replace("\n", " ").strip() + "\n")
                kept += 1
                clean_chars += r.clean_chars
                if clean_chars / config.CHARS_PER_TOKEN >= token_cap:
                    break
    volume.commit()
    est_tokens = int(clean_chars / config.CHARS_PER_TOKEN)
    print(f"[{source_name} shard {shard_index:03d}] streamed={streamed} kept={kept} "
          f"est_tokens={est_tokens/1e6:.1f}M reasons={reasons}")
    return {"source": source_name, "shard": shard_index, "streamed": streamed,
            "kept": kept, "est_tokens": est_tokens, "reasons": reasons}


def _parquet_urls(hf_id: str, config_name: str, split: str) -> list[str]:
    import json
    import urllib.request

    api = f"https://datasets-server.huggingface.co/parquet?dataset={hf_id}"
    req = urllib.request.Request(api, headers={"User-Agent": "slm-125m"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return [f["url"] for f in data.get("parquet_files", [])
            if f.get("config") == config_name and f.get("split") == split]


@app.local_entrypoint()
def clean(fineweb_shards: int = 1, only: str = ""):
    def cfg(s):
        return s.config_name or "default"

    sources = [s for s in config.DATA_MIX if not only or s.name == only]
    work = []
    for s in sources:
        urls = _parquet_urls(s.hf_id, cfg(s), s.split)
        if s.name == "fineweb-edu":
            urls = urls[:fineweb_shards]
        per_shard_cap = s.token_budget // max(1, len(urls))
        for i, url in enumerate(urls):
            work.append((s.name, url, i, per_shard_cap))
        print(f"{s.name:<12} {len(urls)} shard(s), per-shard cap ~{per_shard_cap/1e6:.0f}M tokens")
    print(f"Launching {len(work)} clean workers...")
    results = list(clean_shard.starmap(work))
    report: dict[str, dict] = {}
    for r in results:
        agg = report.setdefault(r["source"], {"streamed": 0, "kept": 0, "est_tokens": 0, "reasons": {}})
        agg["streamed"] += r["streamed"]
        agg["kept"] += r["kept"]
        agg["est_tokens"] += r["est_tokens"]
        for k, v in r["reasons"].items():
            agg["reasons"][k] = agg["reasons"].get(k, 0) + v
    print("PHASE 1 DROP REPORT")
    total = 0
    for name, a in report.items():
        total += a["est_tokens"]
        print(f"  {name:<12} streamed={a['streamed']:>8} kept={a['kept']:>8} "
              f"est_tokens={a['est_tokens']/1e9:.2f}B drops={a['reasons']}")
    print(f"  TOTAL est_clean_tokens={total/1e9:.2f}B")
    save_report.remote(report)


ocr_image = cpu_image

# ---- Phase 2: dedup + contamination strip ----
SHINGLE_K = 5
MINHASH_PERM = 32
MINHASH_THRESHOLD = 0.8
DECONTAM_NGRAM = 13
SIG_DIR = f"{config.DATA_ROOT}/tmp/minhash_sigs"
NEAR_DUPS_PATH = f"{config.DATA_ROOT}/tmp/near_dups.json"
DECONTAM_SOURCES = {"case-law", "sec"}
CLEAN_SHARDS = {"case-law": 10, "sec": 5, "fineweb-edu": 5}


def _build_contamination_ngrams() -> set:
    from datasets import load_dataset

    from dedup import word_ngrams, words

    grams: set = set()
    for hf_id, cfg_name in [("casehold/casehold", None), ("coastalcph/lex_glue", "case_hold")]:
        try:
            urls = _parquet_urls(hf_id, cfg_name or "default", "test")
            if not urls:
                urls = _parquet_urls(hf_id, cfg_name or "default", "train")
            ds = load_dataset("parquet", data_files=urls, split="train", streaming=True)
            for rec in ds:
                text = " ".join(str(v) for v in rec.values() if isinstance(v, str))
                grams |= word_ngrams(words(text), DECONTAM_NGRAM)
        except Exception as e:
            print(f"  [decontam] could not load {hf_id}: {e}")
    print(f"  [decontam] {len(grams):,} eval 13-grams loaded")
    return grams


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20, cpu=4.0, memory=4_096)
def minhash_shard(shard_basename: str) -> dict:
    import os

    import numpy as np
    from datasketch import MinHash

    from dedup import shingles, words

    path = f"{config.CLEAN_DIR}/case-law/{shard_basename}"
    sigs, idxs = [], []
    with open(path, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.rstrip("\n")
            if not line:
                continue
            m = MinHash(num_perm=MINHASH_PERM)
            sh = list(shingles(words(line), SHINGLE_K))
            if sh:
                m.update_batch(sh)
            sigs.append(m.hashvalues.astype(np.uint64))
            idxs.append(idx)
    os.makedirs(SIG_DIR, exist_ok=True)
    np.savez(f"{SIG_DIR}/{shard_basename}.npz",
             sigs=np.vstack(sigs), idxs=np.asarray(idxs, dtype=np.int64))
    volume.commit()
    print(f"[minhash {shard_basename}] {len(idxs):,} docs")
    return {"shard": shard_basename, "n": len(idxs)}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20, memory=8_192)
def build_near_dups() -> int:
    import glob
    import json
    import os

    import numpy as np
    from datasketch import MinHash, MinHashLSH

    near: dict[str, list[int]] = {}
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_PERM)
    for npz_path in sorted(glob.glob(f"{SIG_DIR}/*.npz")):
        shard = os.path.basename(npz_path)[: -len(".npz")]
        data = np.load(npz_path)
        for row, idx in zip(data["sigs"], data["idxs"]):
            m = MinHash(num_perm=MINHASH_PERM, hashvalues=row)
            if lsh.query(m):
                near.setdefault(shard, []).append(int(idx))
            else:
                lsh.insert(f"{shard}:{int(idx)}", m)
    os.makedirs(os.path.dirname(NEAR_DUPS_PATH), exist_ok=True)
    with open(NEAR_DUPS_PATH, "w", encoding="utf-8") as fh:
        json.dump(near, fh)
    volume.commit()
    total = sum(len(v) for v in near.values())
    print(f"[near-dups] {total:,} case-law near-duplicates")
    return total


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 30, cpu=4.0, memory=8_192)
def write_corpus_shard(source_name: str, shard_basename: str) -> dict:
    import json
    import os

    from dedup import exact_hash, word_ngrams, words

    near: set[int] = set()
    if source_name == "case-law":
        with open(NEAR_DUPS_PATH, encoding="utf-8") as fh:
            near = set(json.load(fh).get(shard_basename, []))
    contam = _build_contamination_ngrams() if source_name in DECONTAM_SOURCES else None
    in_path = f"{config.CLEAN_DIR}/{source_name}/{shard_basename}"
    out_dir = f"{config.CORPUS_DIR}/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    seen: set[str] = set()
    kept = clean_chars = 0
    reasons = {"near_dup": 0, "exact_dup": 0, "contaminated": 0, "kept": 0}
    with open(in_path, encoding="utf-8") as fin, \
            open(f"{out_dir}/{shard_basename}", "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            text = line.rstrip("\n")
            if not text:
                continue
            if idx in near:
                reasons["near_dup"] += 1
                continue
            h = exact_hash(text)
            if h in seen:
                reasons["exact_dup"] += 1
                continue
            if contam and (word_ngrams(words(text), DECONTAM_NGRAM) & contam):
                reasons["contaminated"] += 1
                continue
            seen.add(h)
            fout.write(text + "\n")
            kept += 1
            clean_chars += len(text)
            reasons["kept"] += 1
    volume.commit()
    print(f"[corpus {source_name}/{shard_basename}] kept={kept} drops={reasons}")
    return {"source": source_name, "shard": shard_basename, "kept": kept,
            "est_tokens": int(clean_chars / config.CHARS_PER_TOKEN), "reasons": reasons}


@app.function(image=cpu_image, volumes=VOLUMES)
def write_phase2_report(results: list) -> dict:
    import json

    report: dict[str, dict] = {}
    for r in results:
        if not r:
            continue
        agg = report.setdefault(r["source"], {"kept": 0, "est_tokens": 0,
              "reasons": {"near_dup": 0, "exact_dup": 0, "contaminated": 0, "kept": 0}})
        agg["kept"] += r["kept"]
        agg["est_tokens"] += r["est_tokens"]
        for k, v in r["reasons"].items():
            agg["reasons"][k] = agg["reasons"].get(k, 0) + v
    total = sum(v["est_tokens"] for v in report.values())
    print("PHASE 2 REPORT")
    for name, a in report.items():
        print(f"  {name:<12} kept={a['kept']:>8} est_tokens={a['est_tokens']/1e9:.2f}B drops={a['reasons']}")
    print(f"  TOTAL corpus est tokens: {total/1e9:.2f}B")
    with open(f"{config.CORPUS_DIR}/phase2_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()
    return report


@app.local_entrypoint()
def dedup(compute_sigs: bool = True):
    if compute_sigs:
        names = [f"shard-{i:03d}.txt" for i in range(CLEAN_SHARDS["case-law"])]
        print(f"1/3 MinHash signatures for {len(names)} case-law shards...")
        list(minhash_shard.map(names))
    print("2/3 building near-dup set (LSH)...")
    build_near_dups.remote()
    work = [(src, f"shard-{i:03d}.txt") for src, n in CLEAN_SHARDS.items() for i in range(n)]
    print(f"3/3 writing final corpus ({len(work)} shards, parallel)...")
    results = list(write_corpus_shard.starmap(work))
    write_phase2_report.remote(results)


# ---- Phase 3: train the 16K byte-level BPE tokenizer ----
ml_image = _cpu_base.pip_install("transformers==4.46.3").add_local_python_source(
    "config", "cleaning", "dedup")


def _corpus_line_iter():
    import glob

    for path in sorted(glob.glob(f"{config.CORPUS_DIR}/*/*.txt")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line:
                    yield line


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 40, cpu=8.0, memory=16_384)
def train_tokenizer() -> dict:
    import os

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    specials = list(config.SPECIAL_TOKENS.values()) + list(config.EXTRA_CHAT_TOKENS)
    tok = Tokenizer(models.BPE(unk_token=config.SPECIAL_TOKENS["unk_token"]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.MODEL.vocab_size, special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    print("training BPE...")
    tok.train_from_iterator(_corpus_line_iter(), trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=config.SPECIAL_TOKENS["bos_token"],
        eos_token=config.SPECIAL_TOKENS["eos_token"],
        pad_token=config.SPECIAL_TOKENS["pad_token"],
        unk_token=config.SPECIAL_TOKENS["unk_token"],
        additional_special_tokens=list(config.EXTRA_CHAT_TOKENS))
    os.makedirs(config.TOKENIZER_DIR, exist_ok=True)
    fast.save_pretrained(config.TOKENIZER_DIR)
    volume.commit()
    for s in ["The plaintiff shall bear the burden of proof by a preponderance of the evidence.",
              "The Company's net revenues increased 12% year over year pursuant to the agreement."]:
        ids = fast.encode(s)
        print(f"  '{s[:40]}...' -> {len(ids)} tokens | roundtrip={fast.decode(ids).strip() == s}")
    print(f"vocab_size={fast.vocab_size}")
    return {"vocab_size": fast.vocab_size}


@app.local_entrypoint()
def tokenizer():
    train_tokenizer.remote()


# ---- Phase 4: tokenize + pack into uint16 1024-token windows, split 99/1 ----
TOKENIZE_SHARDS = {"case-law": 4, "sec": 6, "fineweb-edu": 4}
ENCODE_BATCH = 1_000


@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 40, cpu=8.0, memory=16_384)
def tokenize_shard(source_name: str, shard_index: int, num_shards: int) -> dict:
    import glob
    import os

    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    eos_id = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
    seq_len = config.SEQ_LEN
    os.makedirs(config.TRAIN_TOKENS_DIR, exist_ok=True)
    os.makedirs(config.VAL_TOKENS_DIR, exist_ok=True)
    train_path = f"{config.TRAIN_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin"
    val_path = f"{config.VAL_TOKENS_DIR}/{source_name}-{shard_index:03d}.bin"
    buf: list[int] = []
    win_count = n_train = n_val = 0
    corpus_files = sorted(glob.glob(f"{config.CORPUS_DIR}/{source_name}/*.txt"))

    def _doc_iter():
        for path in corpus_files:
            with open(path, encoding="utf-8") as fh:
                for idx, line in enumerate(fh):
                    if idx % num_shards == shard_index:
                        line = line.rstrip("\n")
                        if line:
                            yield line

    with open(train_path, "wb") as ftr, open(val_path, "wb") as fva:
        batch: list[str] = []

        def _flush():
            nonlocal win_count, n_train, n_val
            if not batch:
                return
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                buf.extend(ids)
                buf.append(eos_id)
            while len(buf) >= seq_len:
                window = np.asarray(buf[:seq_len], dtype=np.uint16)
                del buf[:seq_len]
                if win_count % config.VAL_EVERY_N_WINDOWS == 0:
                    window.tofile(fva)
                    n_val += 1
                else:
                    window.tofile(ftr)
                    n_train += 1
                win_count += 1

        for doc in _doc_iter():
            batch.append(doc)
            if len(batch) >= ENCODE_BATCH:
                _flush()
                batch = []
        _flush()
    volume.commit()
    print(f"[{source_name} {shard_index:03d}] train_win={n_train} val_win={n_val} "
          f"train_tok={n_train*seq_len/1e6:.1f}M")
    return {"source": source_name, "shard": shard_index, "train_windows": n_train,
            "val_windows": n_val, "train_tokens": n_train * seq_len, "val_tokens": n_val * seq_len}


@app.function(image=ml_image, volumes=VOLUMES)
def write_token_index(results: list) -> dict:
    import json

    shards = [r for r in results if r]
    total = {"seq_len": config.SEQ_LEN, "dtype": config.TOKENS_DTYPE,
             "train_windows": sum(r["train_windows"] for r in shards),
             "val_windows": sum(r["val_windows"] for r in shards),
             "train_tokens": sum(r["train_tokens"] for r in shards),
             "val_tokens": sum(r["val_tokens"] for r in shards), "shards": shards}
    with open(f"{config.TOKENS_DIR}/index.json", "w", encoding="utf-8") as fh:
        json.dump(total, fh, indent=2)
    volume.commit()
    print(f"index: train={total['train_tokens']/1e9:.2f}B tok ({total['train_windows']} win), "
          f"val={total['val_tokens']/1e6:.1f}M tok ({total['val_windows']} win)")
    return total


@app.local_entrypoint()
def tokenize():
    work = [(name, i, n) for name, n in TOKENIZE_SHARDS.items() for i in range(n)]
    print(f"Launching {len(work)} tokenize workers...")
    results = list(tokenize_shard.starmap(work))
    write_token_index.remote(results)


# ---- Phase 5 (fast path): DDP pretrain on 8xH100, one process per GPU ----
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "numpy==1.26.4")
    .add_local_python_source("config")
)
WORLD_SIZE = 8


def _shard_maps(seq_len: int):
    import glob

    import numpy as np

    paths = sorted(glob.glob(f"{config.TRAIN_TOKENS_DIR}/*.bin"))
    maps = []
    for p in paths:
        m = np.memmap(p, dtype=np.uint16, mode="r")
        n = (len(m) // seq_len) * seq_len
        maps.append(m[:n].reshape(-1, seq_len))
    return maps


def _train_worker(rank: int, world_size: int, port: int, epochs: int,
                   max_seconds: int, result: dict) -> None:
    import math
    import os
    import time

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, LlamaConfig

    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                             init_method=f"tcp://127.0.0.1:{port}")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    seq_len = config.SEQ_LEN
    maps = _shard_maps(seq_len)
    index = [(si, ri) for si, m in enumerate(maps) for ri in range(m.shape[0])]
    rng = np.random.default_rng(config.TRAIN.seed)
    rng.shuffle(index)
    my_index = index[rank::world_size]

    llama_cfg = LlamaConfig(**config.MODEL.to_llama_kwargs())
    model = AutoModelForCausalLM.from_config(llama_cfg, attn_implementation="sdpa")
    model = model.to(device=device, dtype=torch.bfloat16)

    # Resume support: every rank loads the same checkpoint independently (shared
    # volume mount) *before* the DDP wrap, so all replicas start identical.
    resume_step = 0
    ckpt = None
    if os.path.exists(config.RESUME_CKPT_PATH):
        ckpt = torch.load(config.RESUME_CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
        resume_step = ckpt["step"]
        if rank == 0:
            print(f"resuming from checkpoint at step {resume_step}")

    model = DDP(model, device_ids=[rank])

    opt = torch.optim.AdamW(model.parameters(), lr=config.TRAIN.lr,
                             betas=(config.TRAIN.beta1, config.TRAIN.beta2),
                             weight_decay=config.TRAIN.weight_decay)
    if ckpt is not None:
        opt.load_state_dict(ckpt["optimizer"])

    micro_bs = config.TRAIN.micro_batch_size
    windows_per_global_step = config.TRAIN.global_batch_tokens // seq_len
    grad_accum = max(1, windows_per_global_step // (micro_bs * world_size))
    warmup_steps = max(1, config.TRAIN.warmup_tokens // config.TRAIN.global_batch_tokens)
    steps_per_epoch = len(my_index) // (micro_bs * grad_accum)
    total_steps = steps_per_epoch * epochs

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return config.TRAIN.lr * step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return config.TRAIN.min_lr + 0.5 * (config.TRAIN.lr - config.TRAIN.min_lr) * (
            1 + math.cos(math.pi * min(prog, 1.0)))

    start = time.time()
    step = resume_step
    ptr = 0
    model.train()
    stop = False
    for _epoch in range(epochs):
        for _ in range(steps_per_epoch):
            if step >= total_steps:
                stop = True
                break
            if time.time() - start > max_seconds:
                stop = True
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for _ in range(grad_accum):
                batch = my_index[ptr:ptr + micro_bs]
                ptr = (ptr + micro_bs) % len(my_index)
                rows = np.stack([maps[si][ri] for si, ri in batch]).astype(np.int64)
                ids = torch.from_numpy(rows).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=ids, labels=ids)
                    loss = out.loss / grad_accum
                loss.backward()
                loss_acc += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.grad_clip)
            opt.step()
            step += 1
            if rank == 0 and step % config.TRAIN.log_every_steps == 0:
                print(f"step {step}/{total_steps} loss={loss_acc:.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e} elapsed={time.time()-start:.0f}s")
            if rank == 0 and step % config.TRAIN.ckpt_every_steps == 0:
                os.makedirs(config.CKPT_DIR, exist_ok=True)
                torch.save({"model": model.module.state_dict(),
                            "optimizer": opt.state_dict(), "step": step},
                           config.RESUME_CKPT_PATH)
                volume.commit()
        if stop:
            break

    dist.barrier()
    if rank == 0:
        os.makedirs(config.BASE_CKPT_DIR, exist_ok=True)
        model.module.save_pretrained(config.BASE_CKPT_DIR)
        result["steps"] = step
        result["elapsed"] = time.time() - start
    dist.destroy_process_group()


# Modal H100 = $3.9492/hr; 8x = $31.5936/hr = $0.008776/s (checked 2026-07-14).
# max_seconds=900 -> ~$7.90 typical; hard timeout=1050 -> ~$9.21 worst case if the
# cooperative check is skipped, keeping a single attempt safely under a $10 cap.
@app.function(image=gpu_image, gpu=f"H100:{WORLD_SIZE}", volumes=VOLUMES, timeout=1050)
def pretrain(epochs: int = 1, max_seconds: int = 900) -> dict:
    import torch.multiprocessing as mp

    manager = mp.Manager()
    result = manager.dict()
    mp.spawn(_train_worker, args=(WORLD_SIZE, 29501, epochs, max_seconds, result),
              nprocs=WORLD_SIZE, join=True)
    volume.commit()
    return dict(result)


@app.local_entrypoint()
def pretrain_run(epochs: int = 1, max_seconds: int = 900):
    r = pretrain.remote(epochs, max_seconds)
    print(f"done: {r}")


# ---- Phase 6: push base model + tokenizer to HF Hub ----
@app.function(image=cpu_image, volumes=VOLUMES,
              secrets=[modal.Secret.from_name(config.HF_SECRET_NAME)], timeout=60 * 15)
def deploy_to_hf(private: bool = True) -> dict:
    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
    api.create_repo(config.HF_REPO, private=private, exist_ok=True)
    api.upload_folder(repo_id=config.HF_REPO, folder_path=config.BASE_CKPT_DIR)
    api.upload_folder(repo_id=config.HF_REPO, folder_path=config.TOKENIZER_DIR)
    print(f"pushed to https://huggingface.co/{config.HF_REPO} (private={private})")
    return {"repo": config.HF_REPO, "private": private}


@app.local_entrypoint()
def deploy(private: bool = True):
    r = deploy_to_hf.remote(private)
    print(f"done: {r}")


# ---- Phase 7: generate SFT QA pairs with gpt-5.4-nano, grounded in our own corpus ----
openai_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("openai==1.59.6")
    .add_local_python_source("config")
)


def _sample_excerpts(corpus_subdir: str, n: int, chars: int) -> list[str]:
    import glob
    import random

    files = sorted(glob.glob(f"{config.CORPUS_DIR}/{corpus_subdir}/*.txt"))
    rng = random.Random(config.TRAIN.seed)
    excerpts: list[str] = []
    attempts = 0
    while len(excerpts) < n and attempts < n * 5 and files:
        attempts += 1
        path = rng.choice(files)
        with open(path, encoding="utf-8") as fh:
            lines = [line for line in fh if len(line) > 200]
        if not lines:
            continue
        excerpts.append(rng.choice(lines).strip()[:chars])
    return excerpts


@app.function(image=openai_image, volumes=VOLUMES,
              secrets=[modal.Secret.from_name(config.OPENAI_SECRET_NAME)], timeout=60 * 30)
def generate_qa_domain(domain_name: str, corpus_subdir: str, num_pairs: int,
                        pairs_per_call: int) -> dict:
    import json
    import os

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    n_calls = -(-num_pairs // pairs_per_call)  # ceil
    excerpts = _sample_excerpts(corpus_subdir, n_calls, config.QA_EXCERPT_CHARS)

    os.makedirs(config.SFT_RAW_DIR, exist_ok=True)
    out_path = f"{config.SFT_RAW_DIR}/{domain_name}.jsonl"
    written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, excerpt in enumerate(excerpts):
            if written >= num_pairs:
                break
            want = min(pairs_per_call, num_pairs - written)
            try:
                resp = client.chat.completions.create(
                    model=config.QA_MODEL,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": config.QA_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Generate exactly {want} QA pairs "
                                                     f"from this excerpt:\n\n{excerpt}"},
                    ],
                )
                payload = json.loads(resp.choices[0].message.content)
                pairs = payload.get("pairs", [])[:want]
            except Exception as e:
                print(f"[{domain_name} #{i}] generation failed: {e}")
                continue
            for p in pairs:
                q, a = p.get("question", "").strip(), p.get("answer", "").strip()
                if not q or not a:
                    continue
                fh.write(json.dumps({"prompt": q, "response": a, "domain": domain_name}) + "\n")
                written += 1
            if (i + 1) % 20 == 0:
                print(f"[{domain_name}] {written}/{num_pairs} pairs")
    volume.commit()
    print(f"[{domain_name}] done: {written}/{num_pairs} pairs -> {out_path}")
    return {"domain": domain_name, "requested": num_pairs, "written": written}


@app.local_entrypoint()
def generate_qa():
    work = [(s.name, s.corpus_subdir, s.num_pairs, s.pairs_per_call) for s in config.QA_MIX]
    results = list(generate_qa_domain.starmap(work))
    total = sum(r["written"] for r in results)
    print(f"Total QA pairs generated: {total}")
    for r in results:
        print(f"  {r['domain']}: {r['written']}/{r['requested']}")


# ---- Phase 8: tokenize QA pairs into loss-masked, padded SFT examples ----
@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 10, cpu=4.0, memory=8_192)
def tokenize_sft() -> dict:
    import glob
    import json
    import os

    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    user_id = tok.convert_tokens_to_ids("<|user|>")
    asst_id = tok.convert_tokens_to_ids("<|assistant|>")
    eos_id = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
    pad_id = tok.pad_token_id
    seq_len = config.SEQ_LEN

    os.makedirs(config.SFT_TOKENS_DIR, exist_ok=True)
    ids_list: list["np.ndarray"] = []
    mask_list: list["np.ndarray"] = []
    domain_counts: dict[str, int] = {}
    dropped = 0

    for path in sorted(glob.glob(f"{config.SFT_RAW_DIR}/*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                q_ids = tok(rec["prompt"], add_special_tokens=False)["input_ids"]
                a_ids = tok(rec["response"], add_special_tokens=False)["input_ids"]
                ids = [user_id] + q_ids + [asst_id] + a_ids + [eos_id]
                if len(ids) > seq_len:
                    dropped += 1
                    continue
                # Loss mask: only the assistant span (+ trailing eos) contributes to loss.
                mask = [0] * (2 + len(q_ids)) + [1] * (len(a_ids) + 1)
                pad = seq_len - len(ids)
                ids_list.append(np.asarray(ids + [pad_id] * pad, dtype=np.uint16))
                mask_list.append(np.asarray(mask + [0] * pad, dtype=np.uint8))
                dom = rec.get("domain", "unknown")
                domain_counts[dom] = domain_counts.get(dom, 0) + 1

    ids_arr = np.stack(ids_list)
    mask_arr = np.stack(mask_list)
    ids_arr.tofile(f"{config.SFT_TOKENS_DIR}/input_ids.bin")
    mask_arr.tofile(f"{config.SFT_TOKENS_DIR}/loss_mask.bin")

    index = {"seq_len": seq_len, "dtype": "uint16", "mask_dtype": "uint8",
             "num_examples": len(ids_list), "dropped_too_long": dropped,
             "domain_counts": domain_counts}
    with open(config.SFT_INDEX_PATH, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    volume.commit()
    print(f"SFT tokenize: {len(ids_list)} examples packed, {dropped} dropped (too long)")
    return index


@app.local_entrypoint()
def sft_tokenize():
    r = tokenize_sft.remote()
    print(f"done: {r}")


# ---- Phase 9: SFT on a single GPU, starting from the pretrained base checkpoint ----
@app.function(image=gpu_image, gpu=config.SFT_GPU, volumes=VOLUMES, timeout=60 * 30)
def sft_train(epochs: int = config.SFT.epochs, max_seconds: int = 1500) -> dict:
    import json
    import os
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda")
    with open(config.SFT_INDEX_PATH, encoding="utf-8") as fh:
        sft_index = json.load(fh)
    n_examples = sft_index["num_examples"]
    seq_len = config.SEQ_LEN

    ids = np.memmap(f"{config.SFT_TOKENS_DIR}/input_ids.bin", dtype=np.uint16,
                     mode="r").reshape(n_examples, seq_len)
    mask = np.memmap(f"{config.SFT_TOKENS_DIR}/loss_mask.bin", dtype=np.uint8,
                      mode="r").reshape(n_examples, seq_len)

    model = AutoModelForCausalLM.from_pretrained(config.BASE_CKPT_DIR)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=config.SFT.lr,
                             weight_decay=config.SFT.weight_decay)

    micro_bs = config.SFT.micro_batch_size
    steps_per_epoch = max(1, n_examples // micro_bs)
    total_steps = steps_per_epoch * epochs

    def lr_at(step: int) -> float:
        if step < config.SFT.warmup_steps:
            return config.SFT.lr * (step + 1) / config.SFT.warmup_steps
        return config.SFT.lr

    rng = np.random.default_rng(config.SFT.seed)
    order = np.arange(n_examples)
    start = time.time()
    step = 0
    stop = False
    for _epoch in range(epochs):
        rng.shuffle(order)
        for b in range(steps_per_epoch):
            if time.time() - start > max_seconds:
                stop = True
                break
            batch_idx = order[b * micro_bs:(b + 1) * micro_bs]
            input_ids = torch.from_numpy(ids[batch_idx].astype(np.int64)).to(device)
            loss_mask = torch.from_numpy(mask[batch_idx].astype(np.float32)).to(device)

            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids).logits[:, :-1, :]
                targets = input_ids[:, 1:]
                tmask = loss_mask[:, 1:]
                loss_tok = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                            targets.reshape(-1), reduction="none")
                loss = (loss_tok * tmask.reshape(-1)).sum() / tmask.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.SFT.grad_clip)
            opt.step()
            step += 1
            if step % config.SFT.log_every_steps == 0:
                print(f"step {step}/{total_steps} loss={loss.item():.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e} elapsed={time.time()-start:.0f}s")
            if step % config.SFT.ckpt_every_steps == 0:
                os.makedirs(config.SFT_CKPT_DIR, exist_ok=True)
                model.save_pretrained(config.SFT_CKPT_DIR)
                volume.commit()
        if stop:
            break

    os.makedirs(config.SFT_CKPT_DIR, exist_ok=True)
    model.save_pretrained(config.SFT_CKPT_DIR)
    volume.commit()
    elapsed = time.time() - start
    print(f"SFT done: {step} steps in {elapsed:.0f}s -> {config.SFT_CKPT_DIR}")
    return {"steps": step, "elapsed": elapsed}


@app.local_entrypoint()
def sft(epochs: int = config.SFT.epochs, max_seconds: int = 1500):
    r = sft_train.remote(epochs, max_seconds)
    print(f"done: {r}")


# ---- Phase 9.5: perplexity of the fine-tuned model on the SFT set (assistant tokens only) ----
# NOTE: sft_train has no held-out split (all examples are trained on across all epochs), so
# this is a train-set perplexity, not a generalization measure.
@app.function(image=gpu_image, gpu=config.SFT_GPU, volumes=VOLUMES, timeout=60 * 10)
def eval_sft_perplexity(batch_size: int = 32) -> dict:
    import json
    import math

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda")
    with open(config.SFT_INDEX_PATH, encoding="utf-8") as fh:
        sft_index = json.load(fh)
    n_examples = sft_index["num_examples"]
    seq_len = config.SEQ_LEN

    ids = np.memmap(f"{config.SFT_TOKENS_DIR}/input_ids.bin", dtype=np.uint16,
                     mode="r").reshape(n_examples, seq_len)
    mask = np.memmap(f"{config.SFT_TOKENS_DIR}/loss_mask.bin", dtype=np.uint8,
                      mode="r").reshape(n_examples, seq_len)

    model = AutoModelForCausalLM.from_pretrained(config.SFT_CKPT_DIR)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    total_loss = 0.0
    total_tokens = 0.0
    with torch.no_grad():
        for b in range(0, n_examples, batch_size):
            input_ids = torch.from_numpy(ids[b:b + batch_size].astype(np.int64)).to(device)
            loss_mask = torch.from_numpy(mask[b:b + batch_size].astype(np.float32)).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids).logits[:, :-1, :]
            targets = input_ids[:, 1:]
            tmask = loss_mask[:, 1:]
            loss_tok = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                        targets.reshape(-1), reduction="none")
            total_loss += (loss_tok * tmask.reshape(-1)).sum().item()
            total_tokens += tmask.sum().item()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    print(f"SFT perplexity eval: {n_examples} examples, {int(total_tokens)} scored tokens, "
          f"loss={avg_loss:.4f} perplexity={ppl:.3f}")
    return {"num_examples": n_examples, "num_scored_tokens": int(total_tokens),
            "loss": avg_loss, "perplexity": ppl}


@app.local_entrypoint()
def perplexity(batch_size: int = 32):
    r = eval_sft_perplexity.remote(batch_size)
    print(f"done: {r}")


# ---- Phase 9.7: base -> SFT head-to-head on CaseHOLD (held out of pretraining, see
# _build_contamination_ngrams / config.EVAL_HOLDOUT) ----
# 5-way multiple choice: pick which of 5 candidate holdings completes the <HOLDING> span
# in a real citing sentence. Scored by length-normalized log-likelihood of each candidate
# under each checkpoint (no chat template on either side, so it's an apples-to-apples
# comparison of the same completion-likelihood task pre- and post-SFT).
casehold_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "numpy==1.26.4",
                 "datasets==3.6.0", "huggingface_hub==0.34.4", "pyarrow==17.0.0")
    .add_local_python_source("config")
)
CASEHOLD_GPU = "L4"


@app.function(image=casehold_image, gpu=CASEHOLD_GPU, volumes=VOLUMES, timeout=60 * 20)
def eval_casehold(n_examples: int = 500, seed: int = 1337, batch_size: int = 16) -> dict:
    import random

    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    bos_id = tok.bos_token_id
    seq_len = config.SEQ_LEN

    urls = _parquet_urls("casehold/casehold", "all", "test")
    ds = load_dataset("parquet", data_files=urls, split="train")
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(ds)), min(n_examples, len(ds))))
    examples = [ds[i] for i in idxs]

    def build_options(example):
        prefix = example["citing_prompt"].split("<HOLDING>")[0]
        prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        options = []
        for i in range(5):
            holding_ids = tok(" " + example[f"holding_{i}"], add_special_tokens=False)["input_ids"]
            holding_ids = holding_ids[: seq_len - 2]  # guarantee room for >=1 ctx token + bos
            budget = seq_len - 1 - len(holding_ids)
            ctx = prefix_ids[-budget:] if budget > 0 else []
            options.append(([bos_id] + ctx + holding_ids, len(holding_ids)))
        return options

    flat = []  # (example_idx, option_idx, token_ids, n_score)
    for ex_idx, ex in enumerate(examples):
        for opt_idx, (ids, n_score) in enumerate(build_options(ex)):
            flat.append((ex_idx, opt_idx, ids, n_score))

    def score_checkpoint(ckpt_dir: str) -> dict:
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir)
        model = model.to(device=device, dtype=torch.bfloat16)
        model.eval()

        sum_logprob: dict[tuple[int, int], float] = {}
        with torch.no_grad():
            for b in range(0, len(flat), batch_size):
                batch = flat[b:b + batch_size]
                maxlen = max(len(x[2]) for x in batch)
                input_ids = torch.full((len(batch), maxlen), tok.pad_token_id, dtype=torch.long)
                for j, (_, _, ids, _) in enumerate(batch):
                    input_ids[j, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                input_ids = input_ids.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :]
                targets = input_ids[:, 1:]
                tok_logprob = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                for j, (ex_idx, opt_idx, ids, n_score) in enumerate(batch):
                    L = len(ids)
                    start = L - 1 - n_score
                    sum_logprob[(ex_idx, opt_idx)] = tok_logprob[j, start:L - 1].sum().item()
                del logits, tok_logprob, targets, input_ids

        correct_raw = correct_norm = 0
        for ex_idx, ex in enumerate(examples):
            gold = int(ex["label"])
            raw = [sum_logprob[(ex_idx, i)] for i in range(5)]
            norm = [sum_logprob[(ex_idx, i)] / max(flat[ex_idx * 5 + i][3], 1) for i in range(5)]
            correct_raw += int(np.argmax(raw) == gold)
            correct_norm += int(np.argmax(norm) == gold)

        del model
        torch.cuda.empty_cache()
        return {"acc": correct_raw / len(examples), "acc_norm": correct_norm / len(examples)}

    base = score_checkpoint(config.BASE_CKPT_DIR)
    sft = score_checkpoint(config.SFT_CKPT_DIR)
    result = {
        "task": "casehold (5-way, held out of pretraining corpus)",
        "n_examples": len(examples),
        "random_baseline": 0.2,
        "base": base,
        "sft": sft,
        "delta_acc_norm": sft["acc_norm"] - base["acc_norm"],
    }
    print(f"CaseHOLD head-to-head ({len(examples)} held-out examples):")
    print(f"  base: acc={base['acc']:.3f} acc_norm={base['acc_norm']:.3f}")
    print(f"  sft:  acc={sft['acc']:.3f} acc_norm={sft['acc_norm']:.3f}")
    print(f"  delta (sft - base), acc_norm: {result['delta_acc_norm']:+.3f}")
    return result


@app.local_entrypoint()
def casehold(n_examples: int = 500, seed: int = 1337, batch_size: int = 16):
    r = eval_casehold.remote(n_examples, seed, batch_size)
    print(f"done: {r}")


# ---- Phase 9.6: export SFT dataset stats + a few sample QA pairs, for the frontend ----
@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 5)
def export_sft_stats(n_samples_per_domain: int = 4) -> dict:
    import glob
    import json
    import random

    with open(config.SFT_INDEX_PATH, encoding="utf-8") as fh:
        sft_index = json.load(fh)

    rng = random.Random(config.TRAIN.seed)
    samples = []
    for path in sorted(glob.glob(f"{config.SFT_RAW_DIR}/*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh]
        samples.extend(rng.sample(recs, min(n_samples_per_domain, len(recs))))

    result = {"index": sft_index, "samples": samples}
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def sft_stats(n_samples_per_domain: int = 4):
    export_sft_stats.remote(n_samples_per_domain)


# ---- Phase 10: push SFT model + tokenizer to HF Hub ----
@app.function(image=cpu_image, volumes=VOLUMES,
              secrets=[modal.Secret.from_name(config.HF_SECRET_NAME)], timeout=60 * 15)
def deploy_sft_to_hf(private: bool = True) -> dict:
    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
    api.create_repo(config.HF_REPO_SFT, private=private, exist_ok=True)
    api.upload_folder(repo_id=config.HF_REPO_SFT, folder_path=config.SFT_CKPT_DIR)
    api.upload_folder(repo_id=config.HF_REPO_SFT, folder_path=config.TOKENIZER_DIR)
    print(f"pushed to https://huggingface.co/{config.HF_REPO_SFT} (private={private})")
    return {"repo": config.HF_REPO_SFT, "private": private}


@app.local_entrypoint()
def deploy_sft(private: bool = True):
    r = deploy_sft_to_hf.remote(private)
    print(f"done: {r}")


# ---- Phase 11: RAFT data prep ----
# Reuses the exact same 1,196 SFT prompt/response pairs verbatim (no new OpenAI calls).
# _sample_excerpts is a pure function of (corpus_subdir, n, chars, config.TRAIN.seed), so
# replaying it here deterministically reconstructs the same small excerpt pool each QA
# domain was generated from -- letting us recover a "golden chunk" for every existing QA
# pair by lexical best-match, without needing to persist anything at generation time.
def _chunk_text(text: str, target_chars: int = 880, min_chars: int = 150) -> list[str]:
    import re

    chunks: list[str] = []
    pos, n = 0, len(text)
    while pos < n:
        end = min(pos + target_chars, n)
        if end < n:
            window = text[end:end + 200]
            dot = window.find(". ")
            nl = window.find("\n")
            if dot != -1 and (nl == -1 or dot < nl):
                end = end + dot + 1
            elif nl != -1:
                end = end + nl
        chunk = text[pos:end].strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        pos = end
    return chunks or ([text.strip()] if len(text.strip()) >= min_chars else [])


def _word_counts(text: str) -> dict:
    import re
    from collections import Counter

    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20, cpu=4.0, memory=4_096)
def raft_prepare_data() -> dict:
    import glob
    import json
    import math
    import os
    import random

    rng = random.Random(config.RAFT.seed)

    # Rebuild each domain's candidate golden-chunk pool from the same excerpts the QA
    # generation pipeline drew from (deterministic replay, zero new API calls).
    domain_pools: dict[str, list[str]] = {}
    for src in config.QA_MIX:
        n_calls = -(-src.num_pairs // src.pairs_per_call)
        excerpts = _sample_excerpts(src.corpus_subdir, n_calls, config.QA_EXCERPT_CHARS)
        pool: list[str] = []
        for ex in excerpts:
            pool.extend(_chunk_text(ex, config.RAFT_CHUNK_CHARS, config.RAFT_CHUNK_MIN_CHARS))
        domain_pools[src.name] = pool
        print(f"[raft-prep] {src.name}: {len(excerpts)} excerpts -> {len(pool)} candidate chunks")

    domain_names = list(domain_pools.keys())
    os.makedirs(config.RAFT_RAW_DIR, exist_ok=True)
    stats = {d: {"total": 0, "golden_present": 0, "golden_absent": 0} for d in domain_names}

    for domain in domain_names:
        pool = domain_pools[domain]
        pool_word_counts = [_word_counts(c) for c in pool]

        # IDF over this domain's pool: rare/distinctive words (party names, figures,
        # defined terms) should count for more than common words when matching a
        # response back to its source chunk -- raw overlap-count over-weights "the",
        # "court", "company", etc.
        df: dict[str, int] = {}
        for wc in pool_word_counts:
            for w in wc:
                df[w] = df.get(w, 0) + 1
        n_pool = len(pool_word_counts)
        idf = {w: math.log(1 + n_pool / d) for w, d in df.items()}

        out_path = f"{config.RAFT_RAW_DIR}/{domain}.jsonl"
        src_path = f"{config.SFT_RAW_DIR}/{domain}.jsonl"
        with open(src_path, encoding="utf-8") as fh, open(out_path, "w", encoding="utf-8") as out:
            for line in fh:
                rec = json.loads(line)
                prompt, response = rec["prompt"], rec["response"]

                resp_words = set(_word_counts(response))
                scores = [sum(idf.get(w, 0.0) for w in (resp_words & set(wc)))
                          for wc in pool_word_counts]
                golden_idx = max(range(len(pool)), key=lambda i: scores[i]) if pool else None

                golden_present = golden_idx is not None and rng.random() < config.RAFT_GOLDEN_PRESENT_PROB
                if golden_present:
                    # Hard in-domain distractor: the runner-up by the same IDF score --
                    # genuinely topically-similar-but-wrong, not just "some other chunk."
                    ranked = sorted(range(len(pool)), key=lambda i: -scores[i])
                    hard_idx = next((i for i in ranked if i != golden_idx), golden_idx)
                    other_domains = [d for d in domain_names if d != domain]
                    easy_domain = rng.choice(other_domains)
                    easy_chunk = rng.choice(domain_pools[easy_domain])
                    chunk_texts = [pool[golden_idx], pool[hard_idx], easy_chunk]
                    true_pos = 0
                else:
                    # Golden-omitted: distractors should stay easy/random here -- a hard
                    # near-match would half-answer the question while we're teaching the
                    # model to say "not found," which would be contradictory signal.
                    candidates = [i for i in range(len(pool)) if i != golden_idx] \
                        if golden_idx is not None else list(range(len(pool)))
                    if len(candidates) >= 2:
                        d1_idx, d2_idx = rng.sample(candidates, 2)
                    else:
                        d1_idx = d2_idx = rng.choice(candidates) if candidates else 0
                    other_domains = [d for d in domain_names if d != domain]
                    d3_domain = rng.choice(other_domains)
                    d3 = rng.choice(domain_pools[d3_domain])
                    chunk_texts = [pool[d1_idx], pool[d2_idx], d3]
                    true_pos = None

                order = list(range(len(chunk_texts)))
                rng.shuffle(order)
                shuffled = [chunk_texts[i] for i in order]
                golden_shown_pos = order.index(true_pos) if true_pos is not None else None

                ctx_lines = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(shuffled))
                user_text = f"Context:\n{ctx_lines}\nQuestion: {prompt}"
                if golden_shown_pos is not None:
                    assistant_text = f"Based on passage {golden_shown_pos + 1}: {response}"
                    stats[domain]["golden_present"] += 1
                else:
                    assistant_text = f"None of the provided passages address this question directly. {response}"
                    stats[domain]["golden_absent"] += 1
                stats[domain]["total"] += 1

                out.write(json.dumps({"prompt": user_text, "response": assistant_text,
                                       "domain": domain}) + "\n")
        print(f"[raft-prep] {domain}: {stats[domain]}")

    volume.commit()
    print(f"RAFT data prep done: {stats}")
    return stats


@app.local_entrypoint()
def raft_prepare():
    r = raft_prepare_data.remote()
    print(f"done: {r}")


# ---- Phase 11.5: tokenize RAFT examples (mirrors tokenize_sft exactly) ----
@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 10, cpu=4.0, memory=8_192)
def tokenize_raft() -> dict:
    import glob
    import json
    import os

    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    user_id = tok.convert_tokens_to_ids("<|user|>")
    asst_id = tok.convert_tokens_to_ids("<|assistant|>")
    eos_id = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
    pad_id = tok.pad_token_id
    seq_len = config.SEQ_LEN

    os.makedirs(config.RAFT_TOKENS_DIR, exist_ok=True)
    ids_list: list["np.ndarray"] = []
    mask_list: list["np.ndarray"] = []
    domain_counts: dict[str, int] = {}
    dropped = 0

    for path in sorted(glob.glob(f"{config.RAFT_RAW_DIR}/*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                q_ids = tok(rec["prompt"], add_special_tokens=False)["input_ids"]
                a_ids = tok(rec["response"], add_special_tokens=False)["input_ids"]
                ids = [user_id] + q_ids + [asst_id] + a_ids + [eos_id]
                if len(ids) > seq_len:
                    dropped += 1
                    continue
                mask = [0] * (2 + len(q_ids)) + [1] * (len(a_ids) + 1)
                pad = seq_len - len(ids)
                ids_list.append(np.asarray(ids + [pad_id] * pad, dtype=np.uint16))
                mask_list.append(np.asarray(mask + [0] * pad, dtype=np.uint8))
                dom = rec.get("domain", "unknown")
                domain_counts[dom] = domain_counts.get(dom, 0) + 1

    ids_arr = np.stack(ids_list)
    mask_arr = np.stack(mask_list)
    ids_arr.tofile(f"{config.RAFT_TOKENS_DIR}/input_ids.bin")
    mask_arr.tofile(f"{config.RAFT_TOKENS_DIR}/loss_mask.bin")

    index = {"seq_len": seq_len, "dtype": "uint16", "mask_dtype": "uint8",
             "num_examples": len(ids_list), "dropped_too_long": dropped,
             "domain_counts": domain_counts}
    with open(config.RAFT_INDEX_PATH, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    volume.commit()
    print(f"RAFT tokenize: {len(ids_list)} examples packed, {dropped} dropped (too long)")
    return index


@app.local_entrypoint()
def raft_tokenize():
    r = tokenize_raft.remote()
    print(f"done: {r}")


# ---- Phase 11.6: RAFT training -- continues from the SFT checkpoint, saves to a new,
# separate checkpoint dir. checkpoints/sft is never opened for writing here. ----
@app.function(image=gpu_image, gpu=config.RAFT_GPU, volumes=VOLUMES, timeout=60 * 30)
def raft_train(epochs: int = config.RAFT.epochs, max_seconds: int = 1500) -> dict:
    import json
    import os
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda")
    with open(config.RAFT_INDEX_PATH, encoding="utf-8") as fh:
        raft_index = json.load(fh)
    n_examples = raft_index["num_examples"]
    seq_len = config.SEQ_LEN

    ids = np.memmap(f"{config.RAFT_TOKENS_DIR}/input_ids.bin", dtype=np.uint16,
                     mode="r").reshape(n_examples, seq_len)
    mask = np.memmap(f"{config.RAFT_TOKENS_DIR}/loss_mask.bin", dtype=np.uint8,
                      mode="r").reshape(n_examples, seq_len)

    model = AutoModelForCausalLM.from_pretrained(config.SFT_CKPT_DIR)  # continue from SFT, not base
    model = model.to(device=device, dtype=torch.bfloat16)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=config.RAFT.lr,
                             weight_decay=config.RAFT.weight_decay)

    micro_bs = config.RAFT.micro_batch_size
    steps_per_epoch = max(1, n_examples // micro_bs)
    total_steps = steps_per_epoch * epochs

    def lr_at(step: int) -> float:
        if step < config.RAFT.warmup_steps:
            return config.RAFT.lr * (step + 1) / config.RAFT.warmup_steps
        return config.RAFT.lr

    rng = np.random.default_rng(config.RAFT.seed)
    order = np.arange(n_examples)
    start = time.time()
    step = 0
    stop = False
    for _epoch in range(epochs):
        rng.shuffle(order)
        for b in range(steps_per_epoch):
            if time.time() - start > max_seconds:
                stop = True
                break
            batch_idx = order[b * micro_bs:(b + 1) * micro_bs]
            input_ids = torch.from_numpy(ids[batch_idx].astype(np.int64)).to(device)
            loss_mask = torch.from_numpy(mask[batch_idx].astype(np.float32)).to(device)

            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids).logits[:, :-1, :]
                targets = input_ids[:, 1:]
                tmask = loss_mask[:, 1:]
                loss_tok = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                            targets.reshape(-1), reduction="none")
                loss = (loss_tok * tmask.reshape(-1)).sum() / tmask.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.RAFT.grad_clip)
            opt.step()
            step += 1
            if step % config.RAFT.log_every_steps == 0:
                print(f"step {step}/{total_steps} loss={loss.item():.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e} elapsed={time.time()-start:.0f}s")
            if step % config.RAFT.ckpt_every_steps == 0:
                os.makedirs(config.RAFT_CKPT_DIR, exist_ok=True)
                model.save_pretrained(config.RAFT_CKPT_DIR)
                volume.commit()
        if stop:
            break

    os.makedirs(config.RAFT_CKPT_DIR, exist_ok=True)
    model.save_pretrained(config.RAFT_CKPT_DIR)
    volume.commit()
    elapsed = time.time() - start
    print(f"RAFT done: {step} steps in {elapsed:.0f}s -> {config.RAFT_CKPT_DIR}")
    return {"steps": step, "elapsed": elapsed}


@app.local_entrypoint()
def raft(epochs: int = config.RAFT.epochs, max_seconds: int = 1500):
    r = raft_train.remote(epochs, max_seconds)
    print(f"done: {r}")


# ---- Phase 11.7: perplexity of the RAFT model on the RAFT set (assistant tokens only) ----
# Same methodology as eval_sft_perplexity: train-set perplexity (no held-out split), scored
# on the loss-masked assistant span only. Numbers aren't directly comparable to SFT's
# perplexity though -- RAFT's assistant span includes the "Based on passage N" / "None of
# the provided passages..." wrapper text, not just the raw answer.
@app.function(image=gpu_image, gpu=config.RAFT_GPU, volumes=VOLUMES, timeout=60 * 10)
def eval_raft_perplexity(batch_size: int = 32) -> dict:
    import json
    import math

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda")
    with open(config.RAFT_INDEX_PATH, encoding="utf-8") as fh:
        raft_index = json.load(fh)
    n_examples = raft_index["num_examples"]
    seq_len = config.SEQ_LEN

    ids = np.memmap(f"{config.RAFT_TOKENS_DIR}/input_ids.bin", dtype=np.uint16,
                     mode="r").reshape(n_examples, seq_len)
    mask = np.memmap(f"{config.RAFT_TOKENS_DIR}/loss_mask.bin", dtype=np.uint8,
                      mode="r").reshape(n_examples, seq_len)

    model = AutoModelForCausalLM.from_pretrained(config.RAFT_CKPT_DIR)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    total_loss = 0.0
    total_tokens = 0.0
    with torch.no_grad():
        for b in range(0, n_examples, batch_size):
            input_ids = torch.from_numpy(ids[b:b + batch_size].astype(np.int64)).to(device)
            loss_mask = torch.from_numpy(mask[b:b + batch_size].astype(np.float32)).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :]
            targets = input_ids[:, 1:]
            tmask = loss_mask[:, 1:]
            loss_tok = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                        targets.reshape(-1), reduction="none")
            total_loss += (loss_tok * tmask.reshape(-1)).sum().item()
            total_tokens += tmask.sum().item()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    print(f"RAFT perplexity eval: {n_examples} examples, {int(total_tokens)} scored tokens, "
          f"loss={avg_loss:.4f} perplexity={ppl:.3f}")
    return {"num_examples": n_examples, "num_scored_tokens": int(total_tokens),
            "loss": avg_loss, "perplexity": ppl}


@app.local_entrypoint()
def raft_perplexity(batch_size: int = 32):
    r = eval_raft_perplexity.remote(batch_size)
    print(f"done: {r}")


# ---- Phase 12: push RAFT model + tokenizer to HF Hub ----
@app.function(image=cpu_image, volumes=VOLUMES,
              secrets=[modal.Secret.from_name(config.HF_SECRET_NAME)], timeout=60 * 15)
def deploy_raft_to_hf(private: bool = True) -> dict:
    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
    api.create_repo(config.HF_REPO_RAFT, private=private, exist_ok=True)
    api.upload_folder(repo_id=config.HF_REPO_RAFT, folder_path=config.RAFT_CKPT_DIR)
    api.upload_folder(repo_id=config.HF_REPO_RAFT, folder_path=config.TOKENIZER_DIR)
    print(f"pushed to https://huggingface.co/{config.HF_REPO_RAFT} (private={private})")
    return {"repo": config.HF_REPO_RAFT, "private": private}


@app.local_entrypoint()
def deploy_raft(private: bool = True):
    r = deploy_raft_to_hf.remote(private)
    print(f"done: {r}")


# ---- Phases 14-17: DPO / RLAIF -- preference-tune the SFT checkpoint on AI-labelled pairs.
# Continues from checkpoints/sft into checkpoints/dpo; SFT + RAFT weights are never opened
# for writing. base -> SFT -> {RAFT, DPO}. Whole pipeline budgeted at <=30 min wall-clock.
# See DPO-plan.md. ----
rlaif_eval_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "numpy==1.26.4", "openai==1.59.6")
    .add_local_python_source("config")
)


def _count_domains(recs: list) -> dict:
    c: dict[str, int] = {}
    for r in recs:
        c[r["domain"]] = c.get(r["domain"], 0) + 1
    return c


def _rlaif_qa_records() -> list[dict]:
    """The 1,196 SFT QA pairs, verbatim, as {prompt, response, domain}."""
    import glob
    import json

    recs: list[dict] = []
    for path in sorted(glob.glob(f"{config.SFT_RAW_DIR}/*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                recs.append({"prompt": rec["prompt"], "response": rec["response"],
                             "domain": rec.get("domain", "unknown")})
    return recs


def _rlaif_domain_chunk_pools() -> dict:
    """domain -> [chunk, ...], rebuilt from the exact excerpts QA generation drew from
    (deterministic _sample_excerpts replay -- identical to raft_prepare_data)."""
    pools: dict[str, list[str]] = {}
    for src in config.QA_MIX:
        n_calls = -(-src.num_pairs // src.pairs_per_call)
        excerpts = _sample_excerpts(src.corpus_subdir, n_calls, config.QA_EXCERPT_CHARS)
        pool: list[str] = []
        for ex in excerpts:
            pool.extend(_chunk_text(ex, config.RAFT_CHUNK_CHARS, config.RAFT_CHUNK_MIN_CHARS))
        pools[src.name] = pool
    return pools


def _rlaif_ground(records: list, pools: dict) -> None:
    """Attach rec['excerpt'] = the best-matching chunk for rec['response'] (per-domain
    IDF-weighted word overlap, same scoring as raft_prepare_data)."""
    import math

    for domain in {r["domain"] for r in records}:
        pool = pools.get(domain, [])
        if not pool:
            for r in records:
                if r["domain"] == domain:
                    r["excerpt"] = ""
            continue
        pool_wc = [_word_counts(c) for c in pool]
        df: dict[str, int] = {}
        for wc in pool_wc:
            for w in wc:
                df[w] = df.get(w, 0) + 1
        n_pool = len(pool_wc)
        idf = {w: math.log(1 + n_pool / d) for w, d in df.items()}
        for r in records:
            if r["domain"] != domain:
                continue
            rw = set(_word_counts(r["response"]))
            scores = [sum(idf.get(w, 0.0) for w in (rw & set(wc))) for wc in pool_wc]
            r["excerpt"] = pool[max(range(len(pool)), key=lambda i: scores[i])]


def _rlaif_split(records: list) -> None:
    """Deterministic, domain-stratified held-out split; sets rec['split'] and rec['id']."""
    import random

    rng = random.Random(config.DPO.seed)
    by_dom: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_dom.setdefault(r["domain"], []).append(i)
    total = len(records)
    heldout: set[int] = set()
    for idxs in by_dom.values():
        k = round(config.RLAIF_HELDOUT_N * len(idxs) / max(total, 1))
        heldout.update(rng.sample(idxs, min(k, len(idxs))))
    for i, r in enumerate(records):
        r["id"] = i
        r["split"] = "heldout" if i in heldout else "train"


def _rlaif_perturb(answer: str, kind: str, foreign: str) -> str:
    """A deliberately worse version of a grounded answer, for Track-B preference pairs."""
    import re

    toks = answer.split()
    if kind == "truncate" and len(toks) >= 6:
        return " ".join(toks[: max(3, int(len(toks) * 0.55))]).rstrip(".,;:") + "."
    if kind == "unsupported" and foreign:
        return answer.rstrip() + " " + foreign
    if kind == "entity":
        m = re.search(r"\b(19|20)\d{2}\b", answer)
        if m:
            return answer[: m.start()] + str(int(m.group()) + 7) + answer[m.end():]
        m = re.search(r"\b\d[\d,]*(?:\.\d+)?\b", answer)
        if m:
            return answer[: m.start()] + "roughly double that" + answer[m.end():]
    return " ".join(toks[: max(3, len(toks) // 2)]).rstrip(".,;:") + "."


def _rlaif_generate(model, tok, user_prompts: list, device, *, n: int, temp: float,
                    top_p: float, max_new_tokens: int) -> list:
    """Left-padded batched sampling. Returns [[str]*n, ...] aligned with user_prompts."""
    import torch

    user_id = tok.convert_tokens_to_ids("<|user|>")
    asst_id = tok.convert_tokens_to_ids("<|assistant|>")
    eos_id = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
    pad_id = tok.pad_token_id
    prompt_cap = config.SEQ_LEN - max_new_tokens - 40
    out: list = [[] for _ in user_prompts]
    B = config.RLAIF_GEN_BATCH
    for s in range(0, len(user_prompts), B):
        chunk = user_prompts[s:s + B]
        seqs = [[user_id] + tok(p, add_special_tokens=False)["input_ids"][:prompt_cap] + [asst_id]
                for p in chunk]
        maxlen = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, x in enumerate(seqs):
            ids[i, maxlen - len(x):] = torch.tensor(x, dtype=torch.long)
            attn[i, maxlen - len(x):] = 1
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids.to(device), attention_mask=attn.to(device),
                do_sample=True, temperature=temp, top_p=top_p, num_return_sequences=n,
                max_new_tokens=max_new_tokens, pad_token_id=pad_id, eos_token_id=eos_id)
        gen = gen[:, maxlen:]
        for i in range(len(seqs)):
            for j in range(n):
                out[s + i].append(tok.decode(gen[i * n + j], skip_special_tokens=True).strip())
    return out


# ---- Phase 14: prep -- held-out split, on-policy candidates, Track-B perturbation pairs ----
@app.function(image=gpu_image, gpu=config.DPO_GPU, volumes=VOLUMES, timeout=60 * 20)
def prep_rlaif_data() -> dict:
    import json
    import os
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    records = _rlaif_qa_records()
    _rlaif_ground(records, _rlaif_domain_chunk_pools())
    _rlaif_split(records)

    os.makedirs(config.RLAIF_RAW_DIR, exist_ok=True)
    with open(config.RLAIF_PROMPTS_PATH, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    train = [r for r in records if r["split"] == "train"]
    heldout = [r for r in records if r["split"] == "heldout"]

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(config.SFT_CKPT_DIR)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    samples = _rlaif_generate(
        model, tok, [r["prompt"] for r in train], device,
        n=config.RLAIF_SAMPLES_PER_PROMPT, temp=config.RLAIF_SAMPLE_TEMP,
        top_p=config.RLAIF_SAMPLE_TOP_P, max_new_tokens=config.RLAIF_SAMPLE_MAX_NEW_TOKENS)

    with open(config.RLAIF_CANDIDATES_PATH, "w", encoding="utf-8") as fh:
        for r, s in zip(train, samples):
            fh.write(json.dumps({"id": r["id"], "domain": r["domain"], "prompt": r["prompt"],
                                 "excerpt": r["excerpt"], "samples": s}) + "\n")

    # Track B: deterministic perturbation pairs. chosen = the grounded gold answer,
    # rejected = a corruption of it (truncated / unsupported-sentence / wrong-entity).
    rng = random.Random(config.DPO.seed)
    kinds = ["truncate", "unsupported", "entity"]
    dom_sents: dict[str, list[str]] = {}
    for r in records:
        for sent in r["response"].replace("\n", " ").split(". "):
            sent = sent.strip()
            if len(sent) > 25:
                dom_sents.setdefault(r["domain"], []).append(sent.rstrip(".") + ".")
    foreign_pool = {d: [x for dd, ss in dom_sents.items() if dd != d for x in ss] for d in dom_sents}

    tb = 0
    with open(config.RLAIF_PAIRS_TRACKB_PATH, "w", encoding="utf-8") as fh:
        for i, r in enumerate(train):
            kind = kinds[i % len(kinds)]
            pool = foreign_pool.get(r["domain"]) or [""]
            bad = _rlaif_perturb(r["response"], kind, rng.choice(pool))
            if bad.strip() and bad.strip() != r["response"].strip():
                fh.write(json.dumps({"prompt": r["prompt"], "chosen": r["response"],
                                     "rejected": bad, "domain": r["domain"], "track": "B",
                                     "kind": kind, "excerpt": r["excerpt"]}) + "\n")
                tb += 1

    volume.commit()
    summary = {"records": len(records), "train": len(train), "heldout": len(heldout),
               "heldout_by_domain": _count_domains(heldout),
               "candidate_prompts": len(samples), "samples_each": config.RLAIF_SAMPLES_PER_PROMPT,
               "track_b_pairs": tb}
    print(f"rlaif prep: {summary}")
    return summary


@app.local_entrypoint()
def rlaif_prep():
    r = prep_rlaif_data.remote()
    print(f"done: {r}")


# ---- Phase 15: AI preference labelling -- sharded across RLAIF_JUDGE_SHARDS containers ----
@app.function(image=openai_image, volumes=VOLUMES,
              secrets=[modal.Secret.from_name(config.OPENAI_SECRET_NAME)], timeout=60 * 45)
def rlaif_judge_shard(shard_index: int, num_shards: int) -> list:
    import json
    import os
    import random
    import time

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    max_retries=config.RLAIF_JUDGE_MAX_RETRIES, timeout=60.0)
    rows: list[dict] = []
    with open(config.RLAIF_CANDIDATES_PATH, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i % num_shards == shard_index:
                rows.append(json.loads(line))

    def _letter_idx(v, letters) -> int:
        s = str(v).strip().upper()
        for k, L in enumerate(letters):
            if L in s:
                return k
        return -1

    def judge(excerpt, question, answers, order):
        letters = [chr(65 + k) for k in range(len(answers))]
        shown = "\n".join(f"[{letters[k]}] {answers[order[k]]}" for k in range(len(answers)))
        user = f"Excerpt:\n{excerpt}\n\nQuestion: {question}\n\nCandidate answers:\n{shown}"
        resp = client.chat.completions.create(
            model=config.PREF_JUDGE_MODEL, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": config.RLAIF_JUDGE_SYSTEM_PROMPT},
                      {"role": "user", "content": user}])
        p = json.loads(resp.choices[0].message.content)
        bi, wi = _letter_idx(p.get("best"), letters), _letter_idx(p.get("worst"), letters)
        if bi < 0 or wi < 0:
            raise ValueError(f"unparseable verdict: {p}")
        return order[bi], order[wi], float(p.get("confidence", 0.0))

    out: list[dict] = []
    n_err = 0
    for rec in rows:
        answers = [a for a in rec["samples"] if a.strip()]
        if len(answers) < 2:
            continue
        n = len(answers)
        o1 = list(range(n))
        random.Random(rec["id"]).shuffle(o1)
        o2 = list(range(n))
        random.Random(rec["id"] + 99_991).shuffle(o2)
        try:
            b1, w1, c1 = judge(rec["excerpt"], rec["prompt"], answers, o1)
            time.sleep(config.RLAIF_JUDGE_SLEEP)
            b2, w2, c2 = judge(rec["excerpt"], rec["prompt"], answers, o2)
            time.sleep(config.RLAIF_JUDGE_SLEEP)
        except Exception as e:
            n_err += 1
            if n_err <= 3:
                print(f"[judge {shard_index}] id={rec['id']} failed: {e}")
            continue
        if b1 == b2 and w1 == w2 and b1 != w1 and min(c1, c2) >= config.RLAIF_JUDGE_MIN_CONF:
            out.append({"prompt": rec["prompt"], "chosen": answers[b1], "rejected": answers[w1],
                        "domain": rec["domain"], "track": "A",
                        "judge_conf": round(min(c1, c2), 3), "excerpt": rec["excerpt"]})
    print(f"[judge {shard_index}/{num_shards}] {len(rows)} prompts -> {len(out)} kept pairs "
          f"({n_err} calls errored)")
    return out


@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 10)
def rlaif_collect_pairs(track_a_lists: list) -> dict:
    import json
    import random

    track_a = [p for sub in track_a_lists for p in (sub or [])]
    track_b: list[dict] = []
    with open(config.RLAIF_PAIRS_TRACKB_PATH, encoding="utf-8") as fh:
        for line in fh:
            track_b.append(json.loads(line))

    rng = random.Random(config.DPO.seed)
    rng.shuffle(track_a)
    rng.shuffle(track_b)
    target = config.RLAIF_TARGET_PAIRS
    n_b = min(len(track_b), round(target * config.RLAIF_TRACK_B_FRAC))
    n_a = min(len(track_a), target - n_b)
    n_b = min(len(track_b), target - n_a)  # backfill from B if A came up short
    pairs = track_a[:n_a] + track_b[:n_b]
    rng.shuffle(pairs)

    domain_counts: dict[str, int] = {}
    track_counts: dict[str, int] = {}
    with open(config.RLAIF_PAIRS_PATH, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
            domain_counts[p["domain"]] = domain_counts.get(p["domain"], 0) + 1
            track_counts[p["track"]] = track_counts.get(p["track"], 0) + 1
    volume.commit()
    summary = {"total_pairs": len(pairs), "track_a_available": len(track_a),
               "track_b_available": len(track_b), "track_counts": track_counts,
               "domain_counts": domain_counts}
    print(f"rlaif collect pairs: {summary}")
    return summary


@app.local_entrypoint()
def rlaif_judge():
    shards = [(i, config.RLAIF_JUDGE_SHARDS) for i in range(config.RLAIF_JUDGE_SHARDS)]
    print(f"Launching {len(shards)} judge shards ({config.PREF_JUDGE_MODEL})...")
    results = list(rlaif_judge_shard.starmap(shards))
    r = rlaif_collect_pairs.remote(results)
    print(f"done: {r}")


# Rebuild pairs.jsonl from the on-disk sources without re-judging -- used to re-mix the
# Track-A / Track-B ratio (via RLAIF_TRACK_B_FRAC / RLAIF_TARGET_PAIRS) after inspection.
@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 5)
def rebuild_rlaif_pairs(track_b_only: bool = False) -> dict:
    import json
    import os
    import random

    track_b = [json.loads(x) for x in open(config.RLAIF_PAIRS_TRACKB_PATH, encoding="utf-8")]
    track_a: list[dict] = []
    if not track_b_only and os.path.exists(config.RLAIF_PAIRS_PATH):
        track_a = [r for r in (json.loads(x) for x in open(config.RLAIF_PAIRS_PATH, encoding="utf-8"))
                   if r.get("track") == "A"]

    rng = random.Random(config.DPO.seed)
    rng.shuffle(track_a)
    rng.shuffle(track_b)
    target = config.RLAIF_TARGET_PAIRS
    n_a = 0 if track_b_only else min(len(track_a), target - round(target * config.RLAIF_TRACK_B_FRAC))
    pairs = track_a[:n_a] + track_b[:target - n_a]
    rng.shuffle(pairs)

    domain_counts: dict[str, int] = {}
    track_counts: dict[str, int] = {}
    with open(config.RLAIF_PAIRS_PATH, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
            domain_counts[p["domain"]] = domain_counts.get(p["domain"], 0) + 1
            track_counts[p["track"]] = track_counts.get(p["track"], 0) + 1
    volume.commit()
    out = {"total_pairs": len(pairs), "track_counts": track_counts, "domain_counts": domain_counts,
           "track_a_on_disk": len(track_a), "track_b_on_disk": len(track_b)}
    print(f"rebuild pairs: {out}")
    return out


@app.local_entrypoint()
def rlaif_rebuild_pairs(track_b_only: bool = False):
    r = rebuild_rlaif_pairs.remote(track_b_only)
    print(f"done: {r}")


# ---- Phase 15.5: tokenize preference pairs (mirrors tokenize_sft, doubled for chosen/rejected) ----
@app.function(image=ml_image, volumes=VOLUMES, timeout=60 * 10, cpu=4.0, memory=8_192)
def tokenize_rlaif() -> dict:
    import json
    import os

    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    user_id = tok.convert_tokens_to_ids("<|user|>")
    asst_id = tok.convert_tokens_to_ids("<|assistant|>")
    eos_id = tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
    pad_id = tok.pad_token_id
    seq_len = config.SEQ_LEN

    def encode(prompt: str, answer: str):
        q = tok(prompt, add_special_tokens=False)["input_ids"]
        a = tok(answer, add_special_tokens=False)["input_ids"]
        ids = [user_id] + q + [asst_id] + a + [eos_id]
        if len(ids) > seq_len:
            return None
        mask = [0] * (2 + len(q)) + [1] * (len(a) + 1)
        pad = seq_len - len(ids)
        return (np.asarray(ids + [pad_id] * pad, dtype=np.uint16),
                np.asarray(mask + [0] * pad, dtype=np.uint8))

    os.makedirs(config.RLAIF_TOKENS_DIR, exist_ok=True)
    c_ids, c_mask, r_ids, r_mask = [], [], [], []
    domain_counts: dict[str, int] = {}
    track_counts: dict[str, int] = {}
    dropped = 0
    with open(config.RLAIF_PAIRS_PATH, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            ce, re_ = encode(rec["prompt"], rec["chosen"]), encode(rec["prompt"], rec["rejected"])
            if ce is None or re_ is None:
                dropped += 1
                continue
            c_ids.append(ce[0])
            c_mask.append(ce[1])
            r_ids.append(re_[0])
            r_mask.append(re_[1])
            d, t = rec.get("domain", "unknown"), rec.get("track", "?")
            domain_counts[d] = domain_counts.get(d, 0) + 1
            track_counts[t] = track_counts.get(t, 0) + 1

    np.stack(c_ids).tofile(f"{config.RLAIF_TOKENS_DIR}/chosen_input_ids.bin")
    np.stack(c_mask).tofile(f"{config.RLAIF_TOKENS_DIR}/chosen_loss_mask.bin")
    np.stack(r_ids).tofile(f"{config.RLAIF_TOKENS_DIR}/rejected_input_ids.bin")
    np.stack(r_mask).tofile(f"{config.RLAIF_TOKENS_DIR}/rejected_loss_mask.bin")

    index = {"seq_len": seq_len, "dtype": "uint16", "mask_dtype": "uint8",
             "num_pairs": len(c_ids), "dropped_too_long": dropped,
             "domain_counts": domain_counts, "track_counts": track_counts}
    with open(config.RLAIF_INDEX_PATH, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    volume.commit()
    print(f"RLAIF tokenize: {len(c_ids)} pairs packed, {dropped} dropped (too long)")
    return index


@app.local_entrypoint()
def rlaif_tokenize():
    r = tokenize_rlaif.remote()
    print(f"done: {r}")


# ---- Phase 16: DPO training -- continues from the SFT checkpoint, writes checkpoints/dpo.
# checkpoints/sft is never opened for writing. Cooperative time cap + resumable, like sft_train. ----
@app.function(image=gpu_image, gpu=config.DPO_GPU, volumes=VOLUMES, timeout=config.DPO_TIMEOUT)
def dpo_train(epochs: int = config.DPO.epochs,
              max_seconds: int = config.DPO_MAX_SECONDS) -> dict:
    import json
    import os
    import time

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda")
    with open(config.RLAIF_INDEX_PATH, encoding="utf-8") as fh:
        idx = json.load(fh)
    n = idx["num_pairs"]
    seq_len = config.SEQ_LEN
    d = config.RLAIF_TOKENS_DIR

    def mm(name, dtype):
        return np.memmap(f"{d}/{name}", dtype=dtype, mode="r").reshape(n, seq_len)

    c_ids, c_mask = mm("chosen_input_ids.bin", np.uint16), mm("chosen_loss_mask.bin", np.uint8)
    r_ids, r_mask = mm("rejected_input_ids.bin", np.uint16), mm("rejected_loss_mask.bin", np.uint8)

    policy = AutoModelForCausalLM.from_pretrained(config.SFT_CKPT_DIR).to(device, torch.bfloat16)
    ref = AutoModelForCausalLM.from_pretrained(config.SFT_CKPT_DIR).to(device, torch.bfloat16)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    policy.train()
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable(  # trade compute for memory -- L4 is tight
        gradient_checkpointing_kwargs={"use_reentrant": False})

    opt = torch.optim.AdamW(policy.parameters(), lr=config.DPO.lr,
                            weight_decay=config.DPO.weight_decay)

    micro_bs = config.DPO.micro_batch_size
    steps_per_epoch = max(1, n // micro_bs)
    total_steps = steps_per_epoch * epochs
    beta, lam = config.DPO.beta, config.DPO.nll_weight

    def lr_at(step: int) -> float:
        if step < config.DPO.warmup_steps:
            return config.DPO.lr * (step + 1) / config.DPO.warmup_steps
        return config.DPO.lr

    def seq_logp(model, ids, mask):
        # per-token logprob without materialising a full (B,T,V) softmax (TRL-style).
        # chosen and rejected are scored in separate forwards to halve the peak logits tensor.
        out = []
        tok = []
        for half_ids, half_m in ((ids[:micro_bs], mask[:micro_bs]),
                                 (ids[micro_bs:], mask[micro_bs:])):
            logits = model(input_ids=half_ids, use_cache=False).logits[:, :-1, :].float()
            lp = (torch.gather(logits, -1, half_ids[:, 1:, None]).squeeze(-1)
                  - torch.logsumexp(logits, dim=-1))
            del logits
            m = half_m[:, 1:]
            out.append((lp * m).sum(-1))
            tok.append(m.sum(-1))
        return torch.cat(out), torch.cat(tok)

    rng = np.random.default_rng(config.DPO.seed)
    order = np.arange(n)
    start = time.time()
    step = 0
    stop = False
    for _epoch in range(epochs):
        rng.shuffle(order)
        for b in range(steps_per_epoch):
            if time.time() - start > max_seconds:
                stop = True
                break
            bi = order[b * micro_bs:(b + 1) * micro_bs]
            cj = torch.from_numpy(c_ids[bi].astype(np.int64)).to(device)
            rj = torch.from_numpy(r_ids[bi].astype(np.int64)).to(device)
            cm = torch.from_numpy(c_mask[bi].astype(np.float32)).to(device)
            rm = torch.from_numpy(r_mask[bi].astype(np.float32)).to(device)
            ids = torch.cat([cj, rj])
            msk = torch.cat([cm, rm])

            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pol, pol_tok = seq_logp(policy, ids, msk)
                with torch.no_grad():
                    rf, _ = seq_logp(ref, ids, msk)
            pol_c, pol_r = pol.chunk(2)
            rf_c, rf_r = rf.chunk(2)
            ntok_c = pol_tok.chunk(2)[0].clamp(min=1)
            logits_dpo = (pol_c - rf_c) - (pol_r - rf_r)
            loss_dpo = -F.logsigmoid(beta * logits_dpo).mean()
            loss_nll = -(pol_c / ntok_c).mean()
            loss = loss_dpo + lam * loss_nll
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.DPO.grad_clip)
            opt.step()
            step += 1

            if step % config.DPO.log_every_steps == 0:
                with torch.no_grad():
                    rw, rl = beta * (pol_c - rf_c), beta * (pol_r - rf_r)
                    acc = (rw > rl).float().mean().item()
                    margin = (rw - rl).mean().item()
                    clen, rlen = cm[:, 1:].sum(-1).mean().item(), rm[:, 1:].sum(-1).mean().item()
                print(f"step {step}/{total_steps} loss={loss.item():.4f} "
                      f"dpo={loss_dpo.item():.4f} nll={loss_nll.item():.4f} acc={acc:.3f} "
                      f"margin={margin:+.3f} len(c/r)={clen:.0f}/{rlen:.0f} "
                      f"lr={opt.param_groups[0]['lr']:.2e} elapsed={time.time()-start:.0f}s")
            if step % config.DPO.ckpt_every_steps == 0:
                os.makedirs(config.RLAIF_CKPT_DIR, exist_ok=True)
                policy.save_pretrained(config.RLAIF_CKPT_DIR)
                volume.commit()
        if stop:
            break

    os.makedirs(config.RLAIF_CKPT_DIR, exist_ok=True)
    policy.save_pretrained(config.RLAIF_CKPT_DIR)
    volume.commit()
    elapsed = time.time() - start
    print(f"DPO done: {step}/{total_steps} steps in {elapsed:.0f}s -> {config.RLAIF_CKPT_DIR}")
    return {"steps": step, "total_steps": total_steps, "elapsed": elapsed}


@app.local_entrypoint()
def dpo(epochs: int = config.DPO.epochs, max_seconds: int = config.DPO_MAX_SECONDS):
    r = dpo_train.remote(epochs, max_seconds)
    print(f"done: {r}")


# ---- Phase 16.5: eval -- DPO perplexity on the SFT set + DPO-vs-SFT win-rate on held-out prompts ----
@app.function(image=rlaif_eval_image, gpu=config.DPO_GPU, volumes=VOLUMES,
              secrets=[modal.Secret.from_name(config.OPENAI_SECRET_NAME)], timeout=60 * 20)
def evaluate_rlaif(batch_size: int = 32) -> dict:
    import json
    import math
    import os
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from openai import OpenAI
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    tok.padding_side = "left"
    sft = AutoModelForCausalLM.from_pretrained(config.SFT_CKPT_DIR).to(device, torch.bfloat16).eval()
    dpo_m = AutoModelForCausalLM.from_pretrained(config.RLAIF_CKPT_DIR).to(device, torch.bfloat16).eval()

    # (a) assistant-token perplexity on the SFT set -- identical method to eval_sft_perplexity.
    with open(config.SFT_INDEX_PATH, encoding="utf-8") as fh:
        sidx = json.load(fh)
    ne, seq_len = sidx["num_examples"], config.SEQ_LEN
    ids = np.memmap(f"{config.SFT_TOKENS_DIR}/input_ids.bin", dtype=np.uint16, mode="r").reshape(ne, seq_len)
    msk = np.memmap(f"{config.SFT_TOKENS_DIR}/loss_mask.bin", dtype=np.uint8, mode="r").reshape(ne, seq_len)

    def ppl(model) -> tuple:
        tl = tt = 0.0
        with torch.no_grad():
            for b in range(0, ne, batch_size):
                x = torch.from_numpy(ids[b:b + batch_size].astype(np.int64)).to(device)
                m = torch.from_numpy(msk[b:b + batch_size].astype(np.float32)).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(input_ids=x).logits[:, :-1, :]
                lt = F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(),
                                     x[:, 1:].reshape(-1), reduction="none")
                tl += (lt * m[:, 1:].reshape(-1)).sum().item()
                tt += m[:, 1:].sum().item()
        return tl / tt, math.exp(tl / tt)

    dpo_loss, dpo_ppl = ppl(dpo_m)
    sft_loss, sft_ppl = ppl(sft)
    ppl_rise = (dpo_ppl - sft_ppl) / sft_ppl

    # (b) win-rate: one answer each from SFT and DPO on the held-out prompts, judged pairwise.
    heldout: list[dict] = []
    with open(config.RLAIF_PROMPTS_PATH, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["split"] == "heldout":
                heldout.append(r)
    prompts = [r["prompt"] for r in heldout]
    gk = dict(n=1, temp=config.RLAIF_EVAL_SAMPLE_TEMP, top_p=0.95,
              max_new_tokens=config.RLAIF_SAMPLE_MAX_NEW_TOKENS)
    sft_ans = [a[0] for a in _rlaif_generate(sft, tok, prompts, device, **gk)]
    dpo_ans = [a[0] for a in _rlaif_generate(dpo_m, tok, prompts, device, **gk)]

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    max_retries=config.RLAIF_JUDGE_MAX_RETRIES, timeout=60.0)

    def winner(excerpt, q, a_dpo, a_sft) -> str:
        def call(A, B) -> str:
            user = f"Excerpt:\n{excerpt}\n\nQuestion: {q}\n\nAnswer A:\n{A}\n\nAnswer B:\n{B}"
            resp = client.chat.completions.create(
                model=config.PREF_JUDGE_MODEL, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": config.RLAIF_WINRATE_SYSTEM_PROMPT},
                          {"role": "user", "content": user}])
            return str(json.loads(resp.choices[0].message.content).get("winner", "tie")).strip().upper()
        try:
            w1 = {"A": "dpo", "B": "sft"}.get(call(a_dpo, a_sft), "tie")   # A = dpo
            time.sleep(config.RLAIF_JUDGE_SLEEP)
            w2 = {"A": "sft", "B": "dpo"}.get(call(a_sft, a_dpo), "tie")   # A = sft
            time.sleep(config.RLAIF_JUDGE_SLEEP)
        except Exception as e:
            print(f"winrate judge failed: {e}")
            return "tie"
        return w1 if w1 == w2 else "tie"

    tally = {"dpo": 0, "sft": 0, "tie": 0}
    per_dom: dict[str, dict] = {}
    for r, ad, asf in zip(heldout, dpo_ans, sft_ans):
        v = winner(r["excerpt"], r["prompt"], ad, asf)
        tally[v] += 1
        per_dom.setdefault(r["domain"], {"dpo": 0, "sft": 0, "tie": 0})[v] += 1

    n_h = max(len(heldout), 1)
    win_rate = (tally["dpo"] + 0.5 * tally["tie"]) / n_h
    dom_rates = {d: round((v["dpo"] + 0.5 * v["tie"]) / max(sum(v.values()), 1), 3)
                 for d, v in per_dom.items()}
    passed = (win_rate >= config.RLAIF_WINRATE_MIN
              and all(x >= 0.50 for x in dom_rates.values())
              and ppl_rise <= config.RLAIF_PPL_MAX_RISE)

    result = {
        "sft_loss": round(sft_loss, 4), "dpo_loss": round(dpo_loss, 4),
        "sft_perplexity": round(sft_ppl, 3), "dpo_perplexity": round(dpo_ppl, 3),
        "perplexity_rise": round(ppl_rise, 4), "n_heldout": len(heldout), "tally": tally,
        "win_rate_vs_sft": round(win_rate, 3), "win_rate_by_domain": dom_rates,
        "mean_answer_words": {
            "dpo": round(sum(len(a.split()) for a in dpo_ans) / n_h, 1),
            "sft": round(sum(len(a.split()) for a in sft_ans) / n_h, 1)},
        "passed": passed,
    }
    os.makedirs(config.RLAIF_DIR, exist_ok=True)
    with open(config.RLAIF_WINRATE_PATH, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    volume.commit()
    print(json.dumps(result, indent=2))
    print(f"\nRLAIF eval {'PASSED' if passed else 'DID NOT PASS'}: "
          f"win_rate={win_rate:.3f} (>= {config.RLAIF_WINRATE_MIN}), "
          f"ppl {sft_ppl:.2f} -> {dpo_ppl:.2f} ({ppl_rise:+.1%}, cap {config.RLAIF_PPL_MAX_RISE:.0%})")
    return result


@app.local_entrypoint()
def rlaif_eval(batch_size: int = 32):
    r = evaluate_rlaif.remote(batch_size)
    print(f"done: {r}")


# ---- Phase 17: push DPO model + tokenizer to HF Hub ----
@app.function(image=cpu_image, volumes=VOLUMES,
              secrets=[modal.Secret.from_name(config.HF_SECRET_NAME)], timeout=60 * 15)
def deploy_rlaif_to_hf(private: bool = True) -> dict:
    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HUGGINGFACE_TOKEN"])
    api.create_repo(config.HF_REPO_RLAIF, private=private, exist_ok=True)
    api.upload_folder(repo_id=config.HF_REPO_RLAIF, folder_path=config.RLAIF_CKPT_DIR)
    api.upload_folder(repo_id=config.HF_REPO_RLAIF, folder_path=config.TOKENIZER_DIR)
    print(f"pushed to https://huggingface.co/{config.HF_REPO_RLAIF} (private={private})")
    return {"repo": config.HF_REPO_RLAIF, "private": private}


@app.local_entrypoint()
def deploy_rlaif(private: bool = True):
    r = deploy_rlaif_to_hf.remote(private)
    print(f"done: {r}")


# ---- Playground inference endpoint: backs the Vercel front end ----
infer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "fastapi[standard]==0.115.4")
    .add_local_python_source("config")
)


@app.cls(image=infer_image, volumes=VOLUMES, cpu=2.0, memory=4_096, timeout=60 * 3,
         secrets=[modal.Secret.from_name("playground-api-key")], scaledown_window=60 * 10)
class Playground:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(config.BASE_CKPT_DIR)
        self.model.eval()

    @modal.fastapi_endpoint(method="POST")
    def complete(self, item: dict) -> dict:
        import os

        from fastapi import HTTPException

        if item.get("api_key") != os.environ.get("PLAYGROUND_API_KEY"):
            raise HTTPException(status_code=401, detail="unauthorized")

        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt required")
        max_new_tokens = max(1, min(int(item.get("max_new_tokens", 40)), 80))

        ids = self.tok(prompt, return_tensors="pt")
        ids.pop("token_type_ids", None)
        with self.torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.8, top_p=0.95, pad_token_id=self.tok.pad_token_id,
            )
        full_text = self.tok.decode(out[0], skip_special_tokens=True)
        completion = full_text[len(prompt):]
        return {"prompt": prompt, "completion": completion}


@app.cls(image=infer_image, volumes=VOLUMES, cpu=2.0, memory=4_096, timeout=60 * 3,
         secrets=[modal.Secret.from_name("playground-api-key")], scaledown_window=60 * 10)
class PlaygroundSFT:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
        self.user_id = self.tok.convert_tokens_to_ids("<|user|>")
        self.asst_id = self.tok.convert_tokens_to_ids("<|assistant|>")
        self.eos_id = self.tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
        self.model = AutoModelForCausalLM.from_pretrained(config.SFT_CKPT_DIR)
        self.model.eval()

    @modal.fastapi_endpoint(method="POST")
    def complete(self, item: dict) -> dict:
        import os

        from fastapi import HTTPException

        if item.get("api_key") != os.environ.get("PLAYGROUND_API_KEY"):
            raise HTTPException(status_code=401, detail="unauthorized")

        question = str(item.get("question", "")).strip()
        if not question:
            raise HTTPException(status_code=400, detail="question required")
        max_new_tokens = max(1, min(int(item.get("max_new_tokens", 120)), 200))

        q_ids = self.tok(question, add_special_tokens=False)["input_ids"]
        input_ids = self.torch.tensor([[self.user_id] + q_ids + [self.asst_id]])
        with self.torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.95, pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.eos_id,
            )
        gen_ids = out[0][input_ids.shape[1]:]
        answer = self.tok.decode(gen_ids, skip_special_tokens=True).strip()
        return {"question": question, "answer": answer}


# ---- Phase 13: RAG index -- BM25 over a sampled multi-domain slice of the corpus.
# Same chunker as RAFT data prep, so the noise profile the model was trained on matches
# what it actually sees at inference. ----
@app.function(image=cpu_image, volumes=VOLUMES, timeout=60 * 20, cpu=4.0, memory=4_096)
def build_rag_index() -> dict:
    import glob
    import json
    import math
    import os
    import random

    rng = random.Random(config.RAFT.seed)
    total_pairs = sum(s.num_pairs for s in config.QA_MIX)

    chunks: list[dict] = []  # {domain, text}
    for src in config.QA_MIX:
        target = max(1, round(config.RAG_TARGET_CHUNKS * src.num_pairs / total_pairs))
        files = sorted(glob.glob(f"{config.CORPUS_DIR}/{src.corpus_subdir}/*.txt"))
        rng.shuffle(files)
        got = 0
        for path in files:
            if got >= target:
                break
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if got >= target:
                        break
                    if len(line) <= 200:
                        continue
                    for c in _chunk_text(line.strip(), config.RAG_CHUNK_CHARS, config.RAG_CHUNK_MIN_CHARS):
                        chunks.append({"domain": src.name, "text": c})
                        got += 1
                        if got >= target:
                            break
        print(f"[rag-index] {src.name}: {got}/{target} chunks")

    # BM25 stats: inverted index term -> [[chunk_idx, tf], ...], doc frequency, doc lengths.
    postings: dict[str, list[list[int]]] = {}
    df: dict[str, int] = {}
    doc_len: list[int] = []
    for idx, c in enumerate(chunks):
        counts = _word_counts(c["text"])
        doc_len.append(sum(counts.values()))
        for term, tf in counts.items():
            postings.setdefault(term, []).append([idx, tf])
            df[term] = df.get(term, 0) + 1

    N = len(chunks)
    avg_doc_len = sum(doc_len) / max(N, 1)
    bm25 = {"N": N, "avg_doc_len": avg_doc_len, "doc_len": doc_len,
            "postings": postings, "df": df}

    os.makedirs(config.RAG_DIR, exist_ok=True)
    with open(config.RAG_CHUNKS_PATH, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")
    with open(config.RAG_BM25_PATH, "w", encoding="utf-8") as fh:
        json.dump(bm25, fh)
    volume.commit()
    print(f"RAG index done: {N} chunks, {len(postings)} unique terms")
    return {"num_chunks": N, "num_terms": len(postings)}


@app.local_entrypoint()
def rag_index():
    r = build_rag_index.remote()
    print(f"done: {r}")


@app.cls(image=infer_image, volumes=VOLUMES, cpu=2.0, memory=4_096, timeout=60 * 3,
         secrets=[modal.Secret.from_name("playground-api-key")], scaledown_window=60 * 10)
class PlaygroundRAG:
    @modal.enter()
    def load(self):
        import json
        import math
        import re

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.re = re
        self.math = math
        self.tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
        self.user_id = self.tok.convert_tokens_to_ids("<|user|>")
        self.asst_id = self.tok.convert_tokens_to_ids("<|assistant|>")
        self.eos_id = self.tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
        self.model = AutoModelForCausalLM.from_pretrained(config.RAFT_CKPT_DIR)
        self.model.eval()

        with open(config.RAG_BM25_PATH, encoding="utf-8") as fh:
            self.bm25 = json.load(fh)
        self.chunks = []
        with open(config.RAG_CHUNKS_PATH, encoding="utf-8") as fh:
            for line in fh:
                self.chunks.append(json.loads(line))

    def _retrieve(self, query: str, k: int) -> list[int]:
        terms = set(self.re.findall(r"[a-z0-9]+", query.lower()))
        N = self.bm25["N"]
        avg_len = self.bm25["avg_doc_len"]
        doc_len = self.bm25["doc_len"]
        k1, b = 1.5, 0.75
        scores: dict[int, float] = {}
        for t in terms:
            postings = self.bm25["postings"].get(t)
            if not postings:
                continue
            dft = self.bm25["df"][t]
            idf = self.math.log(1 + (N - dft + 0.5) / (dft + 0.5))
            for idx, tf in postings:
                dl = doc_len[idx]
                denom = tf + k1 * (1 - b + b * dl / avg_len)
                scores[idx] = scores.get(idx, 0.0) + idf * (tf * (k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [idx for idx, _ in ranked[:k]]

    @modal.fastapi_endpoint(method="POST")
    def complete(self, item: dict) -> dict:
        import os

        from fastapi import HTTPException

        if item.get("api_key") != os.environ.get("PLAYGROUND_API_KEY"):
            raise HTTPException(status_code=401, detail="unauthorized")

        question = str(item.get("question", "")).strip()
        if not question:
            raise HTTPException(status_code=400, detail="question required")
        max_new_tokens = max(1, min(int(item.get("max_new_tokens", 150)), 220))

        idxs = self._retrieve(question, config.RAG_RETRIEVE_K)
        retrieved = [self.chunks[i] for i in idxs]
        if retrieved:
            ctx_lines = "\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(retrieved))
            prompt = f"Context:\n{ctx_lines}\nQuestion: {question}"
        else:
            prompt = f"Context:\nQuestion: {question}"

        q_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        input_ids = self.torch.tensor([[self.user_id] + q_ids + [self.asst_id]])
        with self.torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.95, pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.eos_id,
            )
        gen_ids = out[0][input_ids.shape[1]:]
        answer = self.tok.decode(gen_ids, skip_special_tokens=True).strip()
        return {
            "question": question,
            "answer": answer,
            "sources": [{"domain": c["domain"], "text": c["text"]} for c in retrieved],
        }


# ---- OCR-threshold analysis (optional; informs config.CLEAN.nonword_ratio_max) ----
@app.function(image=ocr_image, timeout=60 * 15)
def ocr_sample(n_docs: int = 3000) -> dict:
    import re

    from cleaning import clean_document

    with open("/usr/share/dict/words", encoding="utf-8", errors="ignore") as fh:
        vocab = {w.strip().lower() for w in fh if w.strip().isalpha()}
    tokre = re.compile(r"[A-Za-z]{3,}")
    source = _SOURCE_BY_NAME["case-law"]
    ratios: list[float] = []
    for record in _stream_source(source, n_docs):
        text = record.get(source.text_field) or ""
        if not isinstance(text, str):
            text = str(text)
        r = clean_document(text)
        if not r.kept:
            continue
        toks = [t.lower() for t in tokre.findall(r.text)]
        if len(toks) < 50:
            continue
        ratios.append(sum(1 for t in toks if t not in vocab) / len(toks))
    ratios.sort()
    n = len(ratios)
    for t in [0.10, 0.15, 0.20, 0.25, 0.30]:
        d = sum(1 for x in ratios if x > t)
        print(f"  drop if non-word ratio >{int(t*100)}%: {d} docs ({d/n:.1%})")
    return {"scored": n}


@app.local_entrypoint()
def ocr(n_docs: int = 3000):
    ocr_sample.remote(n_docs)


@app.function(image=cpu_image, volumes=VOLUMES)
def save_report(report: dict) -> None:
    import json

    with open(f"{config.CLEAN_DIR}/phase1_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    volume.commit()


@app.local_entrypoint()
def main(n_per_source: int = 10):
    smoke_test.remote(n_per_source)


@app.local_entrypoint()
def measure(n_per_source: int = 2000):
    measure_sources.remote(n_per_source)
