// Real numbers from config.py, the RAFT/RAG index on the Modal volume, and the actual
// training/eval runs (modal_app.py::raft_train, eval_raft_perplexity, build_rag_index).

export const raftTraining = {
  epochs: 4,
  lr: 3e-5,
  microBatchSize: 16,
  optimizer: "AdamW (weight_decay=0.1)",
  gpuType: "1x NVIDIA L4 (Modal)",
  startedFrom: "SFT checkpoint (slm-125m-sft), not the base model",
  steps: 296,
};

export const raftDataset = {
  totalExamples: 1_196,
  domainCounts: [
    { domain: "Legal", key: "legal", count: 656 },
    { domain: "SEC filings", key: "sec", count: 300 },
    { domain: "General", key: "general", count: 240 },
  ],
  goldenPresent: 942,
  goldenAbsent: 254,
  chunksPerExample: 3,
  note: "Reuses the same 1,196 SFT prompt/response pairs verbatim (no new generation). Each is reformatted with 3 retrieved-style passages -- 1 golden + 2 distractors (1 hard in-domain, 1 easy cross-domain) ~79% of the time, 3 distractors and no golden the other ~21% -- so the model learns to identify the relevant passage, cite it, and say so when none apply.",
};

export const raftPerplexity = {
  value: 2.821,
  loss: 1.037,
  scoredTokens: 108_144,
  numExamples: 1_196,
  note: "Computed on assistant-response tokens only, same methodology as the SFT perplexity number. Not directly comparable to it though: the scored span here includes the templated “Based on passage N” / “None of the provided passages...” wrapper text, which is more predictable than SFT's freeform answers, so a lower number is partly a format artifact.",
};

export const cost = {
  raftTrainSeconds: 191.635,
  ratePerHourL4: 0.80,
  get raftTrainUsd() {
    return (this.raftTrainSeconds * this.ratePerHourL4) / 3600;
  },
  cpuEstimateUsd: 0.10,
  get totalUsd() {
    return this.raftTrainUsd + this.cpuEstimateUsd;
  },
  note: "GPU cost is measured directly from training-loop seconds on Modal L4 ($0.80/hr). CPU cost (golden-chunk matching, RAG index build) is a conservative estimate -- those jobs aren't separately metered, but are small CPU-only Modal runs bounded by a fixed chunk quota, not a full corpus scan.",
};

export const ragIndex = {
  totalChunks: 6_000,
  domainCounts: [
    { domain: "Legal", key: "legal", count: 3_300 },
    { domain: "SEC filings", key: "sec", count: 1_500 },
    { domain: "General", key: "general", count: 1_200 },
  ],
  uniqueTerms: 42_127,
  retrieveK: 3,
  method: "BM25 (lexical, in-memory inverted index -- no vector DB)",
  note: "Chunked with the same ~880-char, paragraph-snapped chunker used for RAFT training data, so the retrieval noise the model sees here matches what it was trained to handle.",
};

export const raftModelMeta = {
  hfRepo: "IndraniBera/slm-125m-raft",
  hfUrl: "https://huggingface.co/IndraniBera/slm-125m-raft",
  hfPrivate: true,
  baseRepo: "IndraniBera/slm-125m-sft",
};

export type RAGSample = { domain: string; question: string };

export const ragSamples: RAGSample[] = [
  { domain: "legal", question: "What did the court say about the admission of a sawed-off shotgun as evidence?" },
  { domain: "legal", question: "How did the District Court instruct the jury regarding expert testimony?" },
  { domain: "sec", question: "What ATM-related services does the Company provide?" },
  { domain: "sec", question: "When was the Company incorporated, and in which state?" },
  { domain: "general", question: "What is hyperglycemia, and why is sustained hyperglycemia concerning?" },
  { domain: "general", question: "What is the capital of France?" },
];
