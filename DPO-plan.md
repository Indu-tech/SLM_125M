# DPO on top of the QA-SFT checkpoint — implementation plan

**Status:** Step 1 (land the code) ✅ done — `config.py` + `modal_app.py`, syntax
and Modal registration verified, **nothing run yet**. Steps 2–7 execute one at a
time with a confirmation checkpoint after each (same convention as
`instruction-QA-fine-tuning.txt` / `instruction-RAFT.txt`). See §10 for the gate list.

**Goal:** preference-tune `checkpoints/sft` (`IndraniBera/slm-125m-sft`) with
**DPO** using **AI-labelled** preference pairs (RLAIF), producing a new, separate
checkpoint `checkpoints/dpo` (`IndraniBera/slm-125m-dpo`). The SFT weights are
never modified. RAFT is untouched — DPO branches off SFT in parallel to RAFT, so
the lineage becomes:

**Budget:** whole pipeline **≤ 30 min wall-clock, ≈ $3–4** — see §7 for how the
serial ~1–1.5 hr first draft was compressed (parallel judging, merged entry
points, tighter training, deferred CaseHOLD).

```
base ──▶ SFT ──┬──▶ RAFT  (reads noisy retrieved context)
               └──▶ DPO   (preference-tuned QA answers)   ◀── this plan
```

---

## 1. Why DPO (recap of the decision)

| | DPO (chosen) | PPO / GRPO |
|---|---|---|
| Extra models in memory | 1 (frozen reference) | 3 (reference + reward + value) |
| Online generation loop | none — offline static dataset | yes, every step |
| API-in-the-loop reward | no | yes (or a distilled local RM) |
| Stability at 125M | high | fragile |
| Fits the existing `sft_train` / `raft_train` mould | almost line-for-line | no |
| All-in cost | ≈ $3–6 | ≈ $15–40 (online) / ≈ $5–10 (local RM) |

DPO is a single hand-rolled loss added to a training loop that already exists
twice in `modal_app.py`. It needs no new library — `gpu_image`
(`torch==2.5.1`, `transformers==4.46.3`, `numpy`) already has everything.

---

## 2. The DPO objective (what the training loop computes)

For a preference triple `(x, y_w, y_l)` — prompt, chosen ("win"), rejected ("lose"):

```
r_w  = β · ( logπ_θ(y_w│x) − logπ_ref(y_w│x) )      # implicit reward, chosen
r_l  = β · ( logπ_θ(y_l│x) − logπ_ref(y_l│x) )      # implicit reward, rejected
L_DPO = − log σ( r_w − r_l )
```

- `logπ(y│x)` = sum of per-token log-probabilities over the **response span only**
  (identical loss mask to SFT: prompt tokens contribute nothing).
- `π_ref` = the frozen SFT checkpoint. `π_θ` = the same weights, trainable.
- `β` (KL strength) = **0.1**. Lower β → policy drifts further from SFT.

**Auxiliary NLL term (recommended for a 125M model).** Pure DPO on a tiny model
can trade fluency for preference margin. Add a small supervised loss on the
chosen response:

```
L = L_DPO + λ_nll · NLL_θ(y_w│x)          λ_nll = 0.1
```

This is the "RPO/DPO-positive" trick and is a single extra `cross_entropy` call
on tensors already in hand. Configurable; default on.

**Per-step metrics to log** (all cheap, all from tensors already computed):
`loss`, `reward_chosen` (`r_w.mean()`), `reward_rejected` (`r_l.mean()`),
`reward_margin` (`(r_w − r_l).mean()`), `reward_acc`
(`(r_w > r_l).float().mean()`), `chosen_len` / `rejected_len` mean token counts
(length-bias watchdog).

---

## 3. Preference dataset construction

### 3.1 Prompts

- **Reuse the 1,196 QA prompts** already in `sft/raw/*.jsonl`. No new OpenAI QA
  generation.
- **Reserve a held-out split first:** `RLAIF_HELDOUT_N = 120` prompts
  (deterministic by `seed=1337`, stratified across the 3 domains) are set aside
  and **never** used to build training pairs — they back the win-rate eval in §6.
- Training pairs are built from the remaining ~1,076 prompts (~55% judged Track A,
  ~45% Track B).
- Each prompt keeps its **source excerpt**, recovered by the same deterministic
  `_sample_excerpts` replay that RAFT already uses (`raft_prepare_data`,
  `modal_app.py:1180`). The excerpt is the grounding the judge sees.

### 3.2 Candidate answers — two tracks

**Track A — on-policy, AI-judged (~55% of pairs).**
For each training prompt, sample **3** completions from the SFT model
(`temperature=1.0`, `top_p=0.95`, `max_new_tokens=180`, `eos` = `<|eos|>`), on
the L4 in **batches of 256** (all ~3,100 generations finish in ~2 min). The judge
(see §3.3) reads the excerpt + question + the 3 answers in randomised order and
returns `{best, worst, confidence, reason}`. `(chosen, rejected)` = `(best,
worst)`.

**Track B — anti-hallucination perturbation (~45%, no API cost, deterministic).**
`chosen` = the SFT model's greedy answer (or the original gold QA answer from
`sft/raw`); `rejected` = a programmatic corruption of it, one of:
- truncate to the first ~55% of tokens (incompleteness);
- append one plausible but excerpt-unsupported sentence (hallucination);
- perturb a number / named entity (factual drift).

Track B needs no model and no API — it is a pure function of the SFT answers, so
it runs in seconds and gives an unambiguous "grounding matters" signal. Raising
its share to ~45% is the single biggest lever for cutting judging wall-clock.

### 3.3 The judge — cheap OpenAI model, fanned out

- **Model:** `PREF_JUDGE_MODEL = "gpt-5.4-mini"` (better-calibrated pairwise
  preference than `-nano`; still cheap). Falls back to `gpt-5.4-nano` via config
  flag if cost/latency is the priority.
- **Prompt:** system message fixes the criteria — *(1) grounded in the excerpt,
  (2) complete, (3) no unsupported claims, (4) concise when otherwise equal* —
  and forces `response_format={"type": "json_object"}` (same pattern as
  `generate_qa_domain`, `modal_app.py:735`).
- **Debiasing:** call the judge **twice** per prompt with different answer
  orderings. Keep the pair only if **both calls agree** on the underlying
  best/worst candidates **and** `min(confidence) ≥ RLAIF_JUDGE_MIN_CONF (0.6)`.
- **Parallelism (the speed fix):** the ~1,150 judged prompts (≈ 2,300 calls) are
  sharded across **`RLAIF_JUDGE_SHARDS = 24`** Modal containers via `.starmap`
  (same mechanism `generate_qa` already uses, `modal_app.py:764`). ~48 prompts ×
  2 calls × ~1.5 s per container → **all judging done in ~3 min wall-clock**
  instead of ~30 min serial.
- Expected yield after the agreement + confidence filter: ~60–75% of Track-A
  prompts. Track B tops the dataset up to **`RLAIF_TARGET_PAIRS ≈ 1,200`**.

### 3.4 Stored artefacts (on the Modal volume)

```
/data/rlaif/
  prompts.jsonl        {prompt, domain, excerpt, split: "train"|"heldout"}
  candidates.jsonl     {prompt_id, domain, answers: [...], source: "sft@t1.0"}
  pairs.jsonl          {prompt, chosen, rejected, domain, track: "A"|"B", judge_conf}
  tokens/
    chosen_input_ids.bin   rejected_input_ids.bin      (uint16, N×1024)
    chosen_loss_mask.bin    rejected_loss_mask.bin     (uint8,  N×1024)
  rlaif_index.json     {seq_len, num_pairs, dropped_too_long, domain_counts,
                        track_counts, heldout_n}
```

---

## 4. Config additions (`config.py`) — ✅ landed

The DPO/RLAIF block sits after the RAFT config (`config.py`, between `RAFT_GPU`
and the RAG section). Key values as landed:

```python
RLAIF_DIR / RLAIF_RAW_DIR / RLAIF_TOKENS_DIR / RLAIF_INDEX_PATH
RLAIF_PROMPTS_PATH / RLAIF_CANDIDATES_PATH / RLAIF_PAIRS_TRACKB_PATH / RLAIF_PAIRS_PATH
RLAIF_WINRATE_PATH
RLAIF_CKPT_DIR   = f"{CKPT_DIR}/dpo"          # separate from SFT_CKPT_DIR / RAFT_CKPT_DIR
HF_REPO_RLAIF    = "IndraniBera/slm-125m-dpo"

PREF_JUDGE_MODEL          = "gpt-5.4-mini"
RLAIF_HELDOUT_N           = 120
RLAIF_SAMPLES_PER_PROMPT  = 3
RLAIF_SAMPLE_TEMP / TOP_P = 1.0 / 0.95
RLAIF_SAMPLE_MAX_NEW_TOKENS = 180
RLAIF_GEN_BATCH           = 64      # prompts per generate(); x3 samples run concurrently on L4
RLAIF_JUDGE_SHARDS        = 24
RLAIF_JUDGE_MIN_CONF      = 0.6
RLAIF_TARGET_PAIRS        = 1_200
RLAIF_TRACK_B_FRAC        = 0.45
RLAIF_EVAL_SAMPLE_TEMP    = 0.7     # matches the playground endpoints
RLAIF_WINRATE_MIN         = 0.60
RLAIF_PPL_MAX_RISE        = 0.15
RLAIF_JUDGE_SYSTEM_PROMPT / RLAIF_WINRATE_SYSTEM_PROMPT   # both force JSON output

class DPOConfig:  epochs=2, micro_batch_size=16, beta=0.1, lr=5e-6, nll_weight=0.1,
                  weight_decay=0.0, grad_clip=1.0, warmup_steps=20,
                  ckpt_every_steps=100, log_every_steps=10, seed=1337
DPO = DPOConfig();  DPO_GPU = "L4"
DPO_MAX_SECONDS = 600       # cooperative cap on the training loop
DPO_TIMEOUT     = 60 * 15   # Modal hard kill on dpo_train

STAGES += ("rlaif_prep", "rlaif_judge", "rlaif_tokenize", "rlaif_train", "rlaif_eval")
```

`~1,200 pairs / batch 16 = 75 steps/epoch × 2 = ~150 steps ≈ 3–5 min on L4` —
far inside the 10-min cap.

**End-to-end wall-clock budget: ≤ 30 min** (see §7). The three levers that got it
there from ~1–1.5 hr: (1) judging fanned out across 24 Modal containers instead of
one serial loop; (2) five merged Modal entry points instead of ten, so ~4 fewer
cold starts; (3) DPO at 2 epochs / batch 16 and CaseHOLD dropped from the default
path (it was never a success metric).

---

## 5. New phases in `modal_app.py` — 5 merged entry points

Fewer, wider functions than SFT/RAFT had: each `modal run` pays a ~20–40 s
container cold start, so merging 10 steps into 5 saves ~3 min of pure overhead.
Each still mirrors existing code and gets one `@app.local_entrypoint()`.

| # | Entry point → function | Image / HW | Merges / mirrors | What it does |
|---|---|---|---|---|
| 2 | `rlaif_prep` → `prep_rlaif_data` | `gpu_image`, `DPO_GPU` | `raft_prepare_data` + `PlaygroundSFT` | **(a)** load the 1,196 QA prompts, replay `_sample_excerpts` + per-domain IDF match for the grounding chunk, cut the deterministic 120-prompt held-out split; **(b)** load the SFT model once, generate `RLAIF_SAMPLES_PER_PROMPT` (3) sampled answers per training prompt, `RLAIF_GEN_BATCH` (64) prompts per `generate()` call; **(c)** build all Track-B perturbation pairs from the gold answers (no API). Writes `prompts.jsonl`, `candidates.jsonl`, `pairs_trackB.jsonl`. |
| 3 | `rlaif_judge` → `rlaif_judge_shard` ×24 + `rlaif_collect_pairs` | `openai_image` / `cpu_image` | `generate_qa_domain` + `.starmap` (`modal_app.py:764`) | Shard the Track-A prompts across `RLAIF_JUDGE_SHARDS` (24) containers; 2 order-swapped `gpt-5.4-mini` ranking calls each; keep pairs where both agree on best/worst and `min(conf) ≥ RLAIF_JUDGE_MIN_CONF`; concatenate with Track B at `RLAIF_TRACK_B_FRAC`, trim to `RLAIF_TARGET_PAIRS`. Writes `pairs.jsonl`. |
| 4 | `rlaif_tokenize` → `tokenize_rlaif` | `ml_image` | `tokenize_sft` (`modal_app.py:773`) | Build `[<user> q <assistant> a <eos>]` for **both** chosen and rejected, loss-mask the response span, pad to 1024, drop pairs where either side > 1024. Writes 4 `.bin` files + `rlaif_index.json`. |
| 5 | `dpo` → `dpo_train(epochs=2, max_seconds=600)` | `gpu_image`, `DPO_GPU`, `timeout=DPO_TIMEOUT` (15 min) | `raft_train` (`modal_app.py:1352`) | Load policy **and** frozen reference, both from `SFT_CKPT_DIR`; DPO + λ·NLL loss (`seq_logp` over the masked span, `−logσ(β·[(π_c−ref_c)−(π_r−ref_r)])`); constant LR, 20-step warmup; checkpoint every 100 steps to `RLAIF_CKPT_DIR`; `stop`-flag **10-min cooperative cap** + resumable like `sft_train`/`raft_train`. Logs `acc`, `margin`, `len(c/r)`. **Never opens `SFT_CKPT_DIR` for writing.** |
| 6 | `rlaif_eval` → `evaluate_rlaif` | `rlaif_eval_image` (torch+transformers+openai), `DPO_GPU` + OpenAI | `eval_sft_perplexity` (`:929`) + new win-rate | One L4, SFT + DPO loaded once: **(a)** assistant-token perplexity of DPO on the SFT set + SFT's for comparison; **(b)** 120 held-out prompts, one answer each from SFT and DPO (`temp=RLAIF_EVAL_SAMPLE_TEMP`), `gpt-5.4-mini` picks the winner order-swapped → DPO **win / tie / loss** rate overall + per domain, mean answer length. Writes `winrate.json`, prints **PASSED / DID NOT PASS** against `RLAIF_WINRATE_MIN` (0.60) and `RLAIF_PPL_MAX_RISE` (0.15). |
| 7 | `deploy_rlaif` → `deploy_rlaif_to_hf` | `cpu_image` + HF secret | `deploy_sft_to_hf` (`modal_app.py:1123`) | Push `RLAIF_CKPT_DIR` + tokenizer to `HF_REPO_RLAIF` (`IndraniBera/slm-125m-dpo`, private). Separate step so the HF push stays behind an explicit confirmation, exactly like `deploy_sft` / `deploy_raft`. |

**Deferred, off the 30-min path (run later if wanted):**

| Entry point | Cost | Why deferred |
|---|---|---|
| `casehold` (edit: add DPO as a 3rd column next to `base`/`sft`) | +~6 min L4 | Never a success metric — SFT itself went 0.170 → 0.166. Nice-to-have for the README table. |
| `PlaygroundDPO` + `/dpo` route in `web/` | frontend work | Only if DPO gets its own playground. |

### `dpo_train` loop sketch

```python
policy = AutoModelForCausalLM.from_pretrained(config.SFT_CKPT_DIR).to(device, torch.bfloat16)
ref    = AutoModelForCausalLM.from_pretrained(config.SFT_CKPT_DIR).to(device, torch.bfloat16)
ref.eval();  [p.requires_grad_(False) for p in ref.parameters()]
opt = torch.optim.AdamW(policy.parameters(), lr=config.DPO.lr,
                        weight_decay=config.DPO.weight_decay)

def seq_logp(model, ids, mask):                      # sum log-prob over masked response span
    logits = model(input_ids=ids).logits[:, :-1, :]
    lp = F.log_softmax(logits.float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return (lp * mask[:, 1:]).sum(-1)                 # (B,)

start = time.time()
for step in ...:
    if time.time() - start > max_seconds:        # cooperative 10-min cap (== sft_train)
        model.save_pretrained(config.RLAIF_CKPT_DIR); volume.commit(); break
    cj, cm, rj, rm = <batch: chosen ids/mask, rejected ids/mask>   # each (B,1024)
    ids  = torch.cat([cj, rj]);  msk = torch.cat([cm, rm])         # (2B,1024)
    pol  = seq_logp(policy, ids, msk)
    with torch.no_grad():
        rf = seq_logp(ref, ids, msk)
    pol_c, pol_r = pol.chunk(2);  rf_c, rf_r = rf.chunk(2)
    logits_dpo = (pol_c - rf_c) - (pol_r - rf_r)
    loss_dpo = -F.logsigmoid(config.DPO.beta * logits_dpo).mean()
    loss_nll = -(pol_c / cm[:, 1:].sum(-1).clamp(min=1)).mean()     # length-norm NLL on chosen
    loss = loss_dpo + config.DPO.nll_weight * loss_nll
    loss.backward();  clip_grad_norm_(policy.parameters(), config.DPO.grad_clip)
    opt.step();  opt.zero_grad(set_to_none=True)
```

Memory on L4 (24 GB): policy params+grads+AdamW ≈ 1.5 GB, ref ≈ 0.25 GB,
activations for 32×1024 still small — `micro_batch_size = 16` fits comfortably.

---

## 6. Evaluation & success criteria

| Metric | Where | Expectation for "it worked" |
|---|---|---|
| **DPO win-rate vs SFT** (120 held-out prompts, AI judge) | `rlaif_eval` | **≥ 60% overall**, no domain **< 50%** |
| Assistant-token perplexity on SFT set | `rlaif_eval` | rise **< ~15%** (8.11 → ≲ 9.3) |
| Mean answer length (DPO vs SFT) | `rlaif_eval` | within **±25%** — guards against DPO length-inflation |
| Qualitative spot-check | 10–15 held-out prompts by hand | fewer unsupported claims, tighter grounding to the excerpt |
| CaseHOLD accNorm (base / sft / dpo) | `casehold` *(deferred)* | likely **flat** (SFT was 0.170→0.166); not a success criterion |

If win-rate < 55% or perplexity balloons or outputs get repetitive/degenerate:
lower `lr` to 2e-6, raise `beta` to 0.2, raise `nll_weight` to 0.25, or drop to
1 epoch — then re-run `::dpo` from the SFT checkpoint (~4 min, cheap).

---

## 7. Steps performed — full run in ≤ 30 minutes

Modal L4 @ $0.80/hr. OpenAI prices are **estimates** from current nano/mini tiers
(nano ≈ $0.05 / $0.40, mini ≈ $0.25 / $2.00 per 1M in/out tokens) — confirm
against live pricing before running; token is in `.env.local`. "Clock" below is
cumulative elapsed wall-time including each `modal run` cold start.

| # | Step | Command | HW | What runs | Duration | Clock | Cost |
|---|---|---|---|---|---|---|---|
| 1 | Land config + phase code ✅ done | *(edit `config.py`, `modal_app.py`)* | local | `DPOConfig` + `RLAIF_*` consts; 7 functions + 6 entrypoints; syntax + Modal registration verified | — | — | $0 |
| 2 | Prep: prompts + candidates + Track B | `modal run modal_app.py::rlaif_prep` | 1×L4 | held-out split; SFT loaded once; 3 samples/prompt, 64 prompts/`generate()`; all perturbation pairs | **~5–8 min** | 0:08 | ~$0.15 |
| 3 | AI preference labelling (8× parallel) | `modal run modal_app.py::rlaif_judge` | OpenAI ×8 + CPU | ~2,150 order-swapped `gpt-5.4-mini` calls, sharded 8 ways with `max_retries=6` + 0.25 s pacing (org limits: 500 RPM / 200K TPM — 24 shards thrashed on 429); agree+confidence filter; merge with Track B → ~1,200 pairs | **~8–12 min** | 0:18 | ~$1.5–2.5 |
| 4 | Tokenize pairs | `modal run modal_app.py::rlaif_tokenize` | CPU | chosen/rejected, loss-masked, padded to 1024 → 4 `.bin` files + index | **~2 min** | 0:14 | ~$0.05 |
| 5 | **DPO fine-tuning** | `modal run modal_app.py::dpo` | 1×L4 | ~150 steps / 2 epochs / batch 16; `max_seconds=600` cap, 15-min `timeout`, ckpt every 100 steps | **~5–8 min**<br>(hard cap 10 min) | 0:21 | ~$0.10 |
| 6 | Eval (perplexity + win-rate) | `modal run modal_app.py::rlaif_eval` | 1×L4 + OpenAI | both checkpoints loaded once; ~240 judge calls; `winrate.json` + PASS/FAIL vs thresholds | **~5–7 min** | **0:27** | ~$0.35 |
| 7 | Deploy — on PASS + your OK | `modal run modal_app.py::deploy_rlaif` | CPU + HF | push checkpoint + tokenizer to `slm-125m-dpo` (private) | ~2–3 min | 0:30 | ~$0.02 |
| — | **Data → train → eval (steps 2–6)** | | | | | **≈ 25–27 min** | **≈ $3–4** |
| opt | CaseHOLD 3-way | `modal run modal_app.py::casehold` | 1×L4 | base / sft / dpo accNorm | +~6 min | — | +$0.08 |

### What changed to get from ~1–1.5 hr to ≤ 30 min

| Lever | Before | After | Saved |
|---|---|---|---|
| **Judging** | one serial loop, ~2,100 calls @ ~1.5 s | `.starmap` across **24 containers** (`RLAIF_JUDGE_SHARDS`) | ~30 min → ~3–5 min |
| **Cold starts** | 10 `modal run` invocations | **6 entry points** (5 on the critical path) | ~4 × ~40 s ≈ 3 min |
| **Track B share** | 30% (rest need judge calls) | **45%** (deterministic, instant) | ~30% fewer judge calls |
| **Candidates** | 4 samples/prompt, `max_new_tokens=200` | **3 samples**, `180`, 64/batch | ~10–20 min → ~5 min |
| **DPO training** | 3 epochs, batch 8, cap 25 min | **2 epochs, batch 16, cap 10 min** | ~10 min → ~7 min |
| **CaseHOLD** | in the default path | **deferred** (never a success metric) | ~6 min |
| **Evals** | 3 separate L4 runs | **1 run** (`rlaif_eval`), both models loaded once | ~2 cold starts + reloads |

### How step 5 stays capped

- **Cooperative cap:** `if time.time() - start > max_seconds: break` inside the epoch loop, identical to `sft_train` (`modal_app.py:881`) / `raft_train` (`:1398`). `max_seconds = 600` → stops and saves at **10 min**.
- **Hard kill:** the function is declared `timeout=60*15` → container force-stopped at **15 min**.
- **Resumable:** checkpoint every 100 steps to `RLAIF_CKPT_DIR`; re-invoke `::dpo` to continue.
- **Expected actual:** ~150 steps × ~1.5–2.5 s/step on L4 (policy fwd+bwd on 32 seqs + ref fwd) ≈ **230–380 s loop** + ~60–90 s to load two checkpoints → **~5–8 min**. Comparable to RAFT's 192 s run.

Cost is the same order as RAFT (`~$0.14` GPU + `~$0.10` CPU) plus the one-time
judging spend, comparable to the original QA-pair generation.

---

## 8. Files touched

| File | Change | Status |
|---|---|---|
| `config.py` | DPO/RLAIF block (63 lines): `DPOConfig` + `DPO*` consts, `RLAIF_*` paths/consts, `HF_REPO_RLAIF`, `PREF_JUDGE_MODEL`, 2 judge prompts; 5 new names appended to `STAGES` | ✅ landed |
| `modal_app.py` | `rlaif_eval_image`; helpers `_count_domains`, `_rlaif_qa_records`, `_rlaif_domain_chunk_pools`, `_rlaif_ground`, `_rlaif_split`, `_rlaif_perturb`, `_rlaif_generate`; functions `prep_rlaif_data`, `rlaif_judge_shard`, `rlaif_collect_pairs`, `tokenize_rlaif`, `dpo_train`, `evaluate_rlaif`, `deploy_rlaif_to_hf`; entrypoints `rlaif_prep`, `rlaif_judge`, `rlaif_tokenize`, `dpo`, `rlaif_eval`, `deploy_rlaif` | ✅ landed |
| `README.md` | move RLAIF out of "Not built yet"; pipeline-table rows, `modal run` commands, results row | after eval passes |
| `eval_casehold` (edit) | add DPO as a 3rd column | optional / deferred |
| `web/lib/dpo-stats.ts`, `web/app/dpo/page.tsx`, `PlaygroundDPO` | a 4th playground | optional |
| `DPO-plan.md` | this file | ✅ |

No changes to `cleaning.py`, `dedup.py`, the pretraining path, the SFT path, or
the RAFT path.

---

## 9. Risks specific to a 125M model

| Risk | Mitigation in this plan |
|---|---|
| Both on-policy candidates are weak → noisy preference labels | 3 samples (best-vs-worst spread) + double judge + confidence filter; Track B (45%) has an unambiguous winner |
| DPO trades fluency for reward margin | `nll_weight = 0.1` auxiliary SFT loss; perplexity guardrail; `lr = 5e-6` |
| Length inflation (DPO's known failure mode) | judge criterion prefers concision on ties; length-normalised NLL term; length watchdog in metrics + eval; IPO / length-normalised DPO available as a fallback |
| Reward hacking / degeneration | `beta = 0.1`, frozen reference KL anchor, grad clip, early stop on perplexity/repetition |
| Fewer pairs (1,200) + 2 epochs → weaker signal | Track B raises the *usable* signal density; if win-rate is marginal, bump `RLAIF_JUDGE_SHARDS` stays, raise `RLAIF_TARGET_PAIRS` to 1,800 and `epochs` to 3 — adds ~6 min, still < 40 min |
| Single-container judge bias from parallel shards | each shard is independent and deterministic-seeded; order-swap + agreement filter is unchanged by sharding |
| CaseHOLD doesn't move | expected (SFT didn't either); win-rate is the real metric — CaseHOLD is deferred, not on the critical path |

---

## 10. Execution order (each step gated on your confirmation)

| # | Command | Check before continuing | ~Clock |
|---|---|---|---|
| 1 | ✅ Land `config.py` + `modal_app.py` (7 fns / 6 entrypoints), no runs | code review — syntax + Modal registration already verified | — |
| 2 | `modal run modal_app.py::rlaif_prep` | held-out split size + domain balance, a few 3-sample tuples, Track-B pair count | 0:08 |
| 3 | `modal run modal_app.py::rlaif_judge` | yield %, confidence distribution, 5 sample pairs, final track/domain mix | 0:12 |
| 4 | `modal run modal_app.py::rlaif_tokenize` | `rlaif_index.json` — `num_pairs`, `dropped_too_long`, `track_counts` | 0:14 |
| 5 | `modal run modal_app.py::dpo` | `acc` → ~0.7–0.9, `margin` ↑ positive, `len(c/r)` stays balanced | 0:21 |
| 6 | `modal run modal_app.py::rlaif_eval` | `winrate.json`: win-rate ≥ 60% overall & ≥ 50% per domain, PPL rise ≤ 15% → prints PASSED | 0:27 |
| 7 | *(on PASS + your OK)* `modal run modal_app.py::deploy_rlaif` | HF repo `slm-125m-dpo` populated | 0:30 |
| 7b | *(if win-rate marginal)* tune per §6, re-run `::dpo` + `::rlaif_eval` | — | +0:12 |
| 8 | *(optional, off-path)* `modal run modal_app.py::casehold` | 3-way accNorm for the README table | +0:06 |
| 9 | *(optional)* wire `PlaygroundDPO` + `/dpo` into `web/` | — | — |

Steps 2–6 (data → train → eval): **≈ 25–27 min**. Deploy adds ~3 min.
