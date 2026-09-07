"""Regenerate SLM-125M-build-flow.pdf.

The original one-page-per-two-stages diagram had no committed source; this
reconstructs the same layout (numbered colour-coded stage panels, rounded
sub-step pills, RESULT sidebar, connector arrows) so it can be regenerated.
Run: python build_flow_pdf.py   (needs `pip install reportlab`).
"""
import os

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SLM-125M-build-flow.pdf")
PAGE_W, PAGE_H = letter
MARGIN = 50
CONTENT_W = PAGE_W - 2 * MARGIN
SIDEBAR_W = 168
GAP = 14
MAIN_W = CONTENT_W - SIDEBAR_W - GAP
PILL_PAD_X = 9
NAVY = HexColor("#1e293b")
GRAY = HexColor("#64748b")

DATE = "2026-09-07"

STAGES = [
    dict(color="#2563eb", fill="#eff6ff", num="1", title="DATA PIPELINE",
         meta="Phases 0-4  ·  CPU  ·  ~$0.18",
         steps=[
             "Smoke test + per-source token-yield measurement",
             "Stream & deterministic clean - 718K -> 698K docs kept",
             "MinHash / LSH near-dedup + exact dedup + CaseHOLD / LexGLUE decontamination -> ~670K docs",
             "Train fresh 16,384 byte-level BPE tokenizer on the corpus",
             "Tokenize + pack into 1024-token uint16 windows, 99 / 1 train-val split",
         ],
         result=["2.04B train tokens", "1,991,282 train windows",
                 "realized mix: SEC 42% / case-law 35% / FineWeb-Edu 23%"]),
    dict(color="#7c3aed", fill="#f5f3ff", num="2", title="PRETRAINING",
         meta="Phase 5  ·  8xH100 DDP (Modal)  ·  ~$8-9",
         steps=[
             "1 epoch · 3,877 steps · ~2.03B tokens seen · 524K-token global batch",
             "AdamW (beta 0.9 / 0.95, wd 0.1), LR 6e-4 -> 6e-5 cosine, 200M-token warmup",
             "Completed in 2 runs - hit $10 budget cap at step 3064, resumed from checkpoint",
         ],
         result=["train loss 9.107 -> 2.4832"]),
    dict(color="#0d9488", fill="#f0fdfa", num="3", title="BASE MODEL DEPLOY",
         meta="Phase 6",
         steps=["Push base model + tokenizer to the Hugging Face Hub"],
         result=["HF: IndraniBera/slm-125m-base (private)"]),
    dict(color="#16a34a", fill="#f0fdf4", num="4", title="QA SUPERVISED FINE-TUNING",
         meta="Phases 7-10  ·  1xL4",
         steps=[
             "Generate 1,196 QA pairs with gpt-5.4-nano, grounded in pretrain-corpus excerpts",
             "Domain split: Legal 656 / SEC 300 / General 240",
             "Tokenize into loss-masked, padded SFT examples",
             "SFT-train from the base checkpoint · 2 epochs · LR 5e-5",
             "Base-vs-SFT head-to-head on CaseHOLD (held out of pretraining)",
         ],
         result=["assistant-token perplexity 8.11", "CaseHOLD accNorm 0.170 -> 0.166",
                 "HF: IndraniBera/slm-125m-sft (private)"]),
    dict(color="#ea580c", fill="#fff7ed", num="5", title="RAFT  (retrieval-aware fine-tuning)",
         meta="Phases 11-12  ·  1xL4  ·  ~$0.14  ·  continues from the SFT checkpoint",
         steps=[
             "Reformat the same 1,196 pairs with 3 retrieved-style passages each",
             "1 golden + 2 distractors ~79% of the time · 3 distractors, no golden ~21%",
             "golden-present 942 / golden-absent 254",
             "Tokenize RAFT examples (mirrors the SFT tokenizer exactly)",
             "RAFT-train from the SFT checkpoint · 4 epochs · 296 steps · LR 3e-5",
         ],
         result=["assistant-token perplexity 2.821", "HF: IndraniBera/slm-125m-raft (private)"]),
    dict(color="#dc2626", fill="#fef2f2", num="6", title="RLAIF  —  DPO preference tuning",
         meta="Phases 14-17  ·  1xL4  ·  ~$4  ·  branches from the SFT checkpoint (parallel to RAFT)",
         steps=[
             "Sample 3 on-policy SFT answers per training prompt (temp 0.8) as preference candidates",
             "gpt-5.4-mini judge, 8-shard parallel, order-swapped + confidence-filtered (>= 0.6)",
             "On-policy Track A dropped - 125M samples too weak to rank; 1,076 Track-B pairs kept",
             "Track B: grounded gold answer vs. targeted corruption (truncate / unsupported / wrong entity)",
             "DPO-train from the SFT checkpoint · 4 epochs · 536 steps · LR 1.5e-5 · beta 0.05 · +0.1 aux-NLL",
             "Held-out win-rate eval - DPO vs SFT on 120 prompts, gpt-5.4-mini judge",
         ],
         result=["reward margin +0.03 -> +1.3 (healthy DPO curve)",
                 "win-rate vs SFT 0.525  (target >= 0.60 - not met)",
                 "assistant-token perplexity 8.11 -> 8.97",
                 "HF: IndraniBera/slm-125m-dpo (private)",
                 "verdict: no measurable gain - 125M capacity limit"]),
    dict(color="#db2777", fill="#fdf2f8", num="7", title="RAG INDEX",
         meta="Phase 13  ·  CPU",
         steps=[
             "Build BM25 lexical inverted index (in-memory, no vector DB)",
             "6,000 chunks - legal 3,300 / SEC 1,500 / general 1,200",
             "~880-char paragraph-snapped chunker (same noise profile as RAFT training)",
         ],
         result=["42,127 unique terms", "retrieve k = 3"]),
    dict(color="#334155", fill="#f1f5f9", num="8", title="FRONTEND & SERVING",
         meta="Vercel + Modal FastAPI endpoints",
         steps=[
             "Next.js app deployed on Vercel (project slm-125m)",
             "/  base-model completion playground + architecture / corpus / loss-curve viz - LIVE",
             "/sft  QA playground (SFT model) + dataset & eval stats - built locally, deploy pending",
             "/rag  RAG playground - RAFT model + live BM25 retrieval - built locally, deploy pending",
             "Modal inference endpoints: Playground (live) · PlaygroundSFT · PlaygroundRAG",
         ],
         result=["LIVE: https://slm-125m-psi.vercel.app",
                 "base playground deployed; /sft + /rag not yet pushed"]),
]


def wrap(text, font, size, max_w):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = w if not cur else cur + " " + w
        if stringWidth(trial, font, size) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def pill_lines(step):
    return wrap(step, "Helvetica", 8.4, MAIN_W - 2 * PILL_PAD_X)


PILL1 = 15.0      # 1-line pill height
PILLN = 10.0      # extra per wrapped line
PGAP = 4.0        # gap between pills
HEAD = 38.0       # panel top -> first pill
BOTPAD = 9.0
ARROW = 15.0


def stage_height(s):
    h = HEAD
    for st in s["steps"]:
        h += PILL1 + PILLN * (len(pill_lines(st)) - 1) + PGAP
    return h - PGAP + BOTPAD


def draw_stage(c, s, top):
    h = stage_height(s)
    x = MARGIN
    accent, fill = HexColor(s["color"]), HexColor(s["fill"])
    # panel
    c.setFillColor(fill)
    c.setStrokeColor(accent)
    c.setLineWidth(1)
    c.roundRect(x, top - h, CONTENT_W, h, 9, stroke=1, fill=1)
    c.setFillColor(accent)
    c.rect(x, top - h, 4.5, h, stroke=0, fill=1)  # left accent bar

    # header
    c.setFillColor(accent)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x + 16, top - 18, f"{s['num']}  ·  {s['title']}")
    c.setFillColor(GRAY)
    c.setFont("Helvetica", 7.3)
    c.drawString(x + 16, top - 28, s["meta"])
    c.setStrokeColor(accent)
    c.setLineWidth(0.5)
    c.line(x + 16, top - 32, x + 16 + MAIN_W, top - 32)

    # sidebar
    sx = x + 16 + MAIN_W + GAP
    c.setFillColor(HexColor("#ffffff"))
    c.setStrokeColor(accent)
    c.setLineWidth(0.5)
    c.roundRect(sx, top - h + 7, SIDEBAR_W - 20, h - 14, 5, stroke=1, fill=1)
    c.setFillColor(accent)
    c.setFont("Helvetica-Bold", 7.3)
    c.drawString(sx + 8, top - 18, "RESULT")
    c.setFillColor(NAVY)
    c.setFont("Helvetica", 7.2)
    ry = top - 28
    for r in s["result"]:
        for ln in wrap(r, "Helvetica", 7.2, SIDEBAR_W - 36):
            c.drawString(sx + 8, ry, ln)
            ry -= 9.2

    # step pills
    py = top - HEAD
    for st in s["steps"]:
        lines = pill_lines(st)
        ph = PILL1 + PILLN * (len(lines) - 1)
        c.setFillColor(HexColor("#ffffff"))
        c.setStrokeColor(accent)
        c.setLineWidth(0.6)
        c.roundRect(x + 16, py - ph, MAIN_W, ph, 3.5, stroke=1, fill=1)
        c.setFillColor(NAVY)
        c.setFont("Helvetica", 8.2)
        ty = py - 9.8
        for ln in lines:
            c.drawString(x + 16 + PILL_PAD_X, ty, ln)
            ty -= PILLN
        py -= ph + PGAP
    return h


def draw_arrow(c, cx, top):
    c.setStrokeColor(HexColor("#94a3b8"))
    c.setFillColor(HexColor("#94a3b8"))
    c.setLineWidth(1.3)
    c.line(cx, top, cx, top - 9)
    p = c.beginPath()
    p.moveTo(cx - 3.5, top - 9)
    p.lineTo(cx + 3.5, top - 9)
    p.lineTo(cx, top - 14)
    p.close()
    c.drawPath(p, stroke=0, fill=1)


def main():
    c = canvas.Canvas(OUT, pagesize=letter)
    # title (page 1 only)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(MARGIN, PAGE_H - 58, "SLM-125M  —  Completed Build Pipeline")
    c.setFillColor(GRAY)
    c.setFont("Helvetica", 7.6)
    c.drawString(MARGIN, PAGE_H - 71,
                 "From-scratch 125.8M-param Llama-style legal / financial SLM  ·  12L / 768d / 12h  ·  "
                 "16,384 vocab  ·  1,024 context  ·  " + DATE)
    y = PAGE_H - 88
    first_on_page = True
    for s in STAGES:
        h = stage_height(s)
        need = h + (0 if first_on_page else ARROW)
        if y - need < MARGIN + 20:
            c.showPage()
            y = PAGE_H - 50
            first_on_page = True
        if not first_on_page:
            draw_arrow(c, MARGIN + CONTENT_W / 2, y)
            y -= ARROW
        draw_stage(c, s, y)
        y -= h
        first_on_page = False

    # footer note
    if y - 42 < MARGIN:
        c.showPage()
        y = PAGE_H - 50
    c.setFillColor(HexColor("#fffbeb"))
    c.setStrokeColor(HexColor("#f59e0b"))
    c.setLineWidth(0.7)
    c.roundRect(MARGIN, y - 34, CONTENT_W, 34, 5, stroke=1, fill=1)
    c.setFillColor(HexColor("#b45309"))
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(MARGIN + 12, y - 13, "NOTE")
    c.setFillColor(NAVY)
    c.setFont("Helvetica", 7.6)
    c.drawString(MARGIN + 12, y - 24,
                 "RLAIF / DPO (stage 6) was run end-to-end and the checkpoint published, but did not clear the "
                 "0.60 win-rate bar - it is not a strict upgrade over SFT.")
    c.showPage()
    c.save()
    print("wrote", OUT)


main()
