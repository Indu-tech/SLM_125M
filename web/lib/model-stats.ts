// Real numbers pulled from config.py, index.json, and the two training run logs
// (bu55rk993.output, by9qp1w4l.output). No estimates except where labeled.

export const architecture = {
  family: "Llama decoder-only transformer",
  paramsTotal: 125_847_552,
  vocabSize: 16_384,
  hiddenSize: 768,
  intermediateSize: 3_072,
  numLayers: 12,
  numHeads: 12,
  numKeyValueHeads: 12, // == numHeads -> standard multi-head attention (no GQA)
  maxPositionEmbeddings: 1_024,
  ropeTheta: 10_000,
  activation: "SwiGLU (silu)",
  tiedEmbeddings: true,
  attentionBias: false,
  rmsNormEps: 1e-5,
};

export const training = {
  epochs: 1,
  epochsNote: "completed across two runs: first stopped at step 3064/3877 on the $10 budget cap, resumed from checkpoint and finished the remaining steps",
  totalSteps: 3877,
  globalBatchTokens: 524_288,
  seqLen: 1_024,
  microBatchSize: 32,
  lrMax: 6e-4,
  lrMin: 6e-5,
  optimizer: "AdamW (beta1=0.9, beta2=0.95, weight_decay=0.1)",
  gpuType: "8x NVIDIA H100 (Modal, DDP)",
  tokensTrained: 3877 * 524_288, // 2,032,664,576
  finalTrainLoss: 2.4832,
  initialTrainLoss: 9.107,
};

export const cost = {
  run1Seconds: 900.4414029121399,
  run2Seconds: 247.06048893928528,
  ratePerHour8xH100: 8 * 3.9492,
  get run1Usd() {
    return (this.run1Seconds * this.ratePerHour8xH100) / 3600;
  },
  get run2Usd() {
    return (this.run2Seconds * this.ratePerHour8xH100) / 3600;
  },
  get totalUsd() {
    return this.run1Usd + this.run2Usd;
  },
  budgetTargetUsd: 10,
  note: "Estimated from GPU-seconds elapsed inside the training loop; actual billed total is likely marginally higher since container/DDP startup overhead per run isn't included.",
};

export const corpusSplit = [
  { source: "SEC filings", shortName: "sec", windows: 839_869, tokens: 0.860e9, pct: 42.2 },
  { source: "Case law", shortName: "case-law", windows: 698_104, tokens: 0.715e9, pct: 35.1 },
  { source: "FineWeb-Edu", shortName: "fineweb-edu", windows: 453_309, tokens: 0.464e9, pct: 22.8 },
];

export const corpusTotals = {
  trainWindows: 1_991_282,
  trainTokens: 2.039072768e9,
  valWindows: 20_119,
};

export const lossCurve: { step: number; loss: number; lr: number }[] = [{"step": 20, "loss": 9.107, "lr": 2.99e-05}, {"step": 40, "loss": 8.2027, "lr": 6.14e-05}, {"step": 60, "loss": 7.3977, "lr": 9.29e-05}, {"step": 80, "loss": 7.0307, "lr": 0.000124}, {"step": 100, "loss": 6.5862, "lr": 0.000156}, {"step": 120, "loss": 6.0915, "lr": 0.000187}, {"step": 140, "loss": 5.8683, "lr": 0.000219}, {"step": 160, "loss": 5.505, "lr": 0.00025}, {"step": 180, "loss": 5.2639, "lr": 0.000282}, {"step": 200, "loss": 5.3193, "lr": 0.000313}, {"step": 220, "loss": 4.8469, "lr": 0.000345}, {"step": 240, "loss": 4.7114, "lr": 0.000376}, {"step": 260, "loss": 4.5802, "lr": 0.000408}, {"step": 280, "loss": 4.5073, "lr": 0.000439}, {"step": 300, "loss": 4.3467, "lr": 0.000471}, {"step": 320, "loss": 4.3043, "lr": 0.000502}, {"step": 340, "loss": 4.2606, "lr": 0.000534}, {"step": 360, "loss": 4.082, "lr": 0.000565}, {"step": 380, "loss": 3.9197, "lr": 0.000597}, {"step": 400, "loss": 3.8289, "lr": 0.0006}, {"step": 420, "loss": 3.6634, "lr": 0.0006}, {"step": 440, "loss": 3.6902, "lr": 0.0006}, {"step": 460, "loss": 3.5656, "lr": 0.000599}, {"step": 480, "loss": 3.3878, "lr": 0.000599}, {"step": 500, "loss": 3.4514, "lr": 0.000598}, {"step": 520, "loss": 3.4539, "lr": 0.000598}, {"step": 540, "loss": 3.1768, "lr": 0.000597}, {"step": 560, "loss": 3.137, "lr": 0.000597}, {"step": 580, "loss": 3.115, "lr": 0.000596}, {"step": 600, "loss": 3.0734, "lr": 0.000595}, {"step": 620, "loss": 3.11, "lr": 0.000594}, {"step": 640, "loss": 3.1264, "lr": 0.000593}, {"step": 660, "loss": 3.0744, "lr": 0.000592}, {"step": 680, "loss": 3.0502, "lr": 0.00059}, {"step": 700, "loss": 3.1445, "lr": 0.000589}, {"step": 720, "loss": 2.9053, "lr": 0.000588}, {"step": 740, "loss": 2.9405, "lr": 0.000586}, {"step": 760, "loss": 3.0921, "lr": 0.000585}, {"step": 780, "loss": 2.9331, "lr": 0.000583}, {"step": 800, "loss": 2.9273, "lr": 0.000581}, {"step": 820, "loss": 2.8914, "lr": 0.000579}, {"step": 840, "loss": 2.7131, "lr": 0.000577}, {"step": 860, "loss": 2.9807, "lr": 0.000575}, {"step": 880, "loss": 2.8569, "lr": 0.000573}, {"step": 900, "loss": 2.9544, "lr": 0.000571}, {"step": 920, "loss": 2.9694, "lr": 0.000569}, {"step": 940, "loss": 2.9123, "lr": 0.000567}, {"step": 960, "loss": 2.8259, "lr": 0.000564}, {"step": 980, "loss": 2.7827, "lr": 0.000562}, {"step": 1000, "loss": 2.9155, "lr": 0.000559}, {"step": 1020, "loss": 2.8634, "lr": 0.000557}, {"step": 1040, "loss": 2.6703, "lr": 0.000554}, {"step": 1060, "loss": 2.8591, "lr": 0.000551}, {"step": 1080, "loss": 2.7014, "lr": 0.000549}, {"step": 1100, "loss": 2.7213, "lr": 0.000546}, {"step": 1120, "loss": 2.7739, "lr": 0.000543}, {"step": 1140, "loss": 2.7326, "lr": 0.00054}, {"step": 1160, "loss": 2.8086, "lr": 0.000537}, {"step": 1180, "loss": 2.7323, "lr": 0.000534}, {"step": 1200, "loss": 2.6901, "lr": 0.00053}, {"step": 1220, "loss": 2.6534, "lr": 0.000527}, {"step": 1240, "loss": 2.7762, "lr": 0.000524}, {"step": 1260, "loss": 2.6741, "lr": 0.00052}, {"step": 1280, "loss": 2.7367, "lr": 0.000517}, {"step": 1300, "loss": 2.7152, "lr": 0.000513}, {"step": 1320, "loss": 2.7857, "lr": 0.00051}, {"step": 1340, "loss": 2.5978, "lr": 0.000506}, {"step": 1360, "loss": 2.7726, "lr": 0.000502}, {"step": 1380, "loss": 2.6344, "lr": 0.000499}, {"step": 1400, "loss": 2.7431, "lr": 0.000495}, {"step": 1420, "loss": 2.6926, "lr": 0.000491}, {"step": 1440, "loss": 2.6706, "lr": 0.000487}, {"step": 1460, "loss": 2.5167, "lr": 0.000483}, {"step": 1480, "loss": 2.6629, "lr": 0.000479}, {"step": 1500, "loss": 2.7527, "lr": 0.000475}, {"step": 1520, "loss": 2.6378, "lr": 0.000471}, {"step": 1540, "loss": 2.5444, "lr": 0.000467}, {"step": 1560, "loss": 2.5699, "lr": 0.000462}, {"step": 1580, "loss": 2.5358, "lr": 0.000458}, {"step": 1600, "loss": 2.7061, "lr": 0.000454}, {"step": 1620, "loss": 2.5767, "lr": 0.000449}, {"step": 1640, "loss": 2.669, "lr": 0.000445}, {"step": 1660, "loss": 2.6344, "lr": 0.000441}, {"step": 1680, "loss": 2.5427, "lr": 0.000436}, {"step": 1700, "loss": 2.6054, "lr": 0.000432}, {"step": 1720, "loss": 2.3887, "lr": 0.000427}, {"step": 1740, "loss": 2.679, "lr": 0.000423}, {"step": 1760, "loss": 2.6109, "lr": 0.000418}, {"step": 1780, "loss": 2.4468, "lr": 0.000414}, {"step": 1800, "loss": 2.7236, "lr": 0.000409}, {"step": 1820, "loss": 2.6555, "lr": 0.000404}, {"step": 1840, "loss": 2.4935, "lr": 0.0004}, {"step": 1860, "loss": 2.6003, "lr": 0.000395}, {"step": 1880, "loss": 2.6451, "lr": 0.00039}, {"step": 1900, "loss": 2.5564, "lr": 0.000385}, {"step": 1920, "loss": 2.5344, "lr": 0.000381}, {"step": 1940, "loss": 2.5271, "lr": 0.000376}, {"step": 1960, "loss": 2.834, "lr": 0.000371}, {"step": 1980, "loss": 2.4095, "lr": 0.000366}, {"step": 2000, "loss": 2.605, "lr": 0.000361}, {"step": 2020, "loss": 2.5259, "lr": 0.000357}, {"step": 2040, "loss": 2.4874, "lr": 0.000352}, {"step": 2060, "loss": 2.5639, "lr": 0.000347}, {"step": 2080, "loss": 2.5751, "lr": 0.000342}, {"step": 2100, "loss": 2.4852, "lr": 0.000337}, {"step": 2120, "loss": 2.4253, "lr": 0.000332}, {"step": 2140, "loss": 2.473, "lr": 0.000328}, {"step": 2160, "loss": 2.555, "lr": 0.000323}, {"step": 2180, "loss": 2.3786, "lr": 0.000318}, {"step": 2200, "loss": 2.5218, "lr": 0.000313}, {"step": 2220, "loss": 2.6105, "lr": 0.000308}, {"step": 2240, "loss": 2.5746, "lr": 0.000303}, {"step": 2260, "loss": 2.4974, "lr": 0.000299}, {"step": 2280, "loss": 2.4746, "lr": 0.000294}, {"step": 2300, "loss": 2.4941, "lr": 0.000289}, {"step": 2320, "loss": 2.497, "lr": 0.000284}, {"step": 2340, "loss": 2.2834, "lr": 0.000279}, {"step": 2360, "loss": 2.4484, "lr": 0.000275}, {"step": 2380, "loss": 2.4819, "lr": 0.00027}, {"step": 2400, "loss": 2.3632, "lr": 0.000265}, {"step": 2420, "loss": 2.44, "lr": 0.00026}, {"step": 2440, "loss": 2.4998, "lr": 0.000256}, {"step": 2460, "loss": 2.5817, "lr": 0.000251}, {"step": 2480, "loss": 2.318, "lr": 0.000246}, {"step": 2500, "loss": 2.3806, "lr": 0.000242}, {"step": 2520, "loss": 2.5635, "lr": 0.000237}, {"step": 2540, "loss": 2.5225, "lr": 0.000233}, {"step": 2560, "loss": 2.3475, "lr": 0.000228}, {"step": 2580, "loss": 2.5443, "lr": 0.000224}, {"step": 2600, "loss": 2.4925, "lr": 0.000219}, {"step": 2620, "loss": 2.4036, "lr": 0.000215}, {"step": 2640, "loss": 2.3203, "lr": 0.000211}, {"step": 2660, "loss": 2.4626, "lr": 0.000206}, {"step": 2680, "loss": 2.428, "lr": 0.000202}, {"step": 2700, "loss": 2.3471, "lr": 0.000198}, {"step": 2720, "loss": 2.4681, "lr": 0.000193}, {"step": 2740, "loss": 2.3738, "lr": 0.000189}, {"step": 2760, "loss": 2.5194, "lr": 0.000185}, {"step": 2780, "loss": 2.423, "lr": 0.000181}, {"step": 2800, "loss": 2.5007, "lr": 0.000177}, {"step": 2820, "loss": 2.4809, "lr": 0.000173}, {"step": 2840, "loss": 2.366, "lr": 0.000169}, {"step": 2860, "loss": 2.4268, "lr": 0.000165}, {"step": 2880, "loss": 2.468, "lr": 0.000161}, {"step": 2900, "loss": 2.4247, "lr": 0.000158}, {"step": 2920, "loss": 2.2165, "lr": 0.000154}, {"step": 2940, "loss": 2.3969, "lr": 0.00015}, {"step": 2960, "loss": 2.5851, "lr": 0.000147}, {"step": 2980, "loss": 2.388, "lr": 0.000143}, {"step": 3000, "loss": 2.3924, "lr": 0.00014}, {"step": 3020, "loss": 2.4511, "lr": 0.000136}, {"step": 3040, "loss": 2.4115, "lr": 0.000133}, {"step": 3060, "loss": 2.2791, "lr": 0.00013}, {"step": 3080, "loss": 2.4378, "lr": 0.000126}, {"step": 3100, "loss": 2.4771, "lr": 0.000123}, {"step": 3120, "loss": 2.3742, "lr": 0.00012}, {"step": 3140, "loss": 2.4849, "lr": 0.000117}, {"step": 3160, "loss": 2.4985, "lr": 0.000114}, {"step": 3180, "loss": 2.4796, "lr": 0.000111}, {"step": 3200, "loss": 2.6881, "lr": 0.000109}, {"step": 3220, "loss": 2.3991, "lr": 0.000106}, {"step": 3240, "loss": 2.4332, "lr": 0.000103}, {"step": 3260, "loss": 2.4001, "lr": 0.000101}, {"step": 3280, "loss": 2.5092, "lr": 9.81e-05}, {"step": 3300, "loss": 2.3631, "lr": 9.56e-05}, {"step": 3320, "loss": 2.4611, "lr": 9.32e-05}, {"step": 3340, "loss": 2.459, "lr": 9.09e-05}, {"step": 3360, "loss": 2.4217, "lr": 8.87e-05}, {"step": 3380, "loss": 2.3985, "lr": 8.66e-05}, {"step": 3400, "loss": 2.3559, "lr": 8.45e-05}, {"step": 3420, "loss": 2.3109, "lr": 8.25e-05}, {"step": 3440, "loss": 2.4275, "lr": 8.06e-05}, {"step": 3460, "loss": 2.3565, "lr": 7.88e-05}, {"step": 3480, "loss": 2.2879, "lr": 7.71e-05}, {"step": 3500, "loss": 2.4462, "lr": 7.54e-05}, {"step": 3520, "loss": 2.4742, "lr": 7.39e-05}, {"step": 3540, "loss": 2.2515, "lr": 7.24e-05}, {"step": 3560, "loss": 2.287, "lr": 7.09e-05}, {"step": 3580, "loss": 2.243, "lr": 6.96e-05}, {"step": 3600, "loss": 2.3004, "lr": 6.84e-05}, {"step": 3620, "loss": 2.3265, "lr": 6.72e-05}, {"step": 3640, "loss": 2.3843, "lr": 6.62e-05}, {"step": 3660, "loss": 2.3889, "lr": 6.52e-05}, {"step": 3680, "loss": 2.4022, "lr": 6.43e-05}, {"step": 3700, "loss": 2.5128, "lr": 6.34e-05}, {"step": 3720, "loss": 2.3071, "lr": 6.27e-05}, {"step": 3740, "loss": 2.312, "lr": 6.21e-05}, {"step": 3760, "loss": 2.5051, "lr": 6.15e-05}, {"step": 3780, "loss": 2.3394, "lr": 6.1e-05}, {"step": 3800, "loss": 2.4054, "lr": 6.07e-05}, {"step": 3820, "loss": 2.3521, "lr": 6.04e-05}, {"step": 3840, "loss": 2.2113, "lr": 6.02e-05}, {"step": 3860, "loss": 2.4832, "lr": 6e-05}];

export const samplePrompts: { source: string; prompt: string }[] = [
  { source: "Case law", prompt: "William Fugate was arrested for operating a motor vehicle on a" },
  { source: "Case law", prompt: "David Cary Ford was admitted to the practice of law in the Commonwealth of" },
  { source: "SEC filings", prompt: "Firstar Corporation is a registered bank holding company incorporated in" },
  { source: "SEC filings", prompt: "All references to \"Notes\" are to Notes to Consolidated Financial" },
  { source: "FineWeb-Edu", prompt: "HIV can be passed on when infected bodily fluid, such as" },
  { source: "FineWeb-Edu", prompt: "For all the love, romance and scandal in Jane Austen's books, what they are really about is freedom and" },
];

export const modelMeta = {
  hfRepo: "IndraniBera/slm-125m-base",
  hfUrl: "https://huggingface.co/IndraniBera/slm-125m-base",
  hfPrivate: true,
};
