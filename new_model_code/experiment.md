# B4DL Q3D Experiment Log

Last updated: 2026-04-30. Maintained by hand — append new runs at the bottom.

This is the working record for the Qwen3-0.6B + 3D-AVS SMAP + Q3D quantisation
pipeline that lives in [new_model_code/](.). Use it to pick up where the
last session left off without re-deriving context.

## TL;DR (current state)

- **Pipeline**: 3D-AVS SMAP feature extraction → projector warmup (1a) →
  full SFT (1b) → GRPO LoRA fine-tune (stage 2). All on one A6000 (eval) /
  two A6000s (training).
- **Box format**: every 7-DOF AABB+yaw is now serialised as 8 Q3D coord
  tokens between `<|box_start|>` / `<|box_end|>` markers. The codebook is
  1024 entries shared across axes, with distance-aware xy / log-space size
  / sin–cos yaw schemes.
- **Status**:
  - **Stage 1a, 1b done**, checkpoints saved.
  - **GRPO unsafe run** (lr=1e-5, kl=0.05) ran 170 steps, KL diverged at
    step 170 → killed. step_100 / step_150 adapters preserved.
  - **GRPO safer run** (lr=5e-6, kl=0.20) is currently running (last seen
    step ~120, r_mean ~0.41, kl=-0.43). step_50, step_100 saved.
  - **Eval framework** (eval_q3d.py) reproduces LiDAR-LLM metrics
    (ACC-10, ACC-5, BEV mIoU per-class, Car-only mIoU) and works with or
    without a LoRA adapter.
- **Headline numbers** (500 val samples):
  - Classification: ACC-10 **97.4%** (SFT-1b) → **98.2%** (GRPO step_100).
    Beats nothing published explicitly but on par with what LiDAR-LLM
    suggests qualitatively.
  - Visual-Grounding **Car BEV mIoU** (true det_area task — model has to
    localise from text alone): **0.56%** for both SFT-1b and GRPO step_100,
    versus LiDAR-LLM's **14.3%**. Mode collapse.
  - det_object Car BEV mIoU: **63.1%** SFT, **63.8%** GRPO. The model copies
    the input box well; it's not an interesting metric.

The bottleneck is unambiguously **det_area localisation**. GRPO can fix
format compliance and squeeze a bit more out of the easy task but cannot
teach the SFT base to ground text → 3D. Next move is a det_area-only
"stage 1c" continued SFT (script drafted, not yet run).

## Repository layout

```
new_model_code/
├── qwen_mm/
│   ├── __init__.py
│   ├── data.py            # SMAPDataset + Collator (Qwen ChatML, masking)
│   ├── model.py           # MMQwen: Qwen3ForCausalLM + projector + Q3D aux
│   └── quantizer.py       # 1024-entry codebook, encode/decode/parse
├── scripts/
│   ├── stage1a_warmup.sh  # frozen LLM, projector-only warmup
│   ├── stage1b_full.sh    # full FT with coord_aux_weight=0.5
│   ├── stage1c_detarea.sh # NOT YET RUN — det_area-only continued SFT
│   ├── stage2_grpo.sh     # GRPO LoRA, currently set to "safer" hps
│   └── auto_launch_stage1.sh
├── extract_smap_features.py
├── build_combined_stage1.py     # legacy text-form combiner (still useful)
├── convert_q3d_data.py          # text-form combined.json → Q3D json
├── train_qwen_sft.py            # SFT entry, supports tune_mm_only flag
├── train_qwen_grpo.py           # custom GRPO (Dr.GRPO, K=4 rollouts)
├── rewards_lidar.py             # parse_boxes (handles Q3D + legacy)
├── eval_q3d.py                  # LiDAR-LLM-comparable evaluator
└── experiment.md                # this file
```

Large data / checkpoints live under `/data1/byounggun/3davs_b4dl/`.

## Q3D quantisation design

Each 7-DOF box `[xmin, xmax, ymin, ymax, zmin, zmax, yaw]` becomes 8 tokens
between `<|box_start|>` and `<|box_end|>`:

```
<|box_start|><X><Y><Z><W><L><H><SIN><COS><|box_end|>
```

Codebook: 1024 entries `<coord_0>` ... `<coord_1023>`, **shared across all
8 axes**. Each axis applies a different decoding to recover the metre /
unit value:

| Axis | Range | Bins | Resolution |
|------|-------|------|-----------|
| X, Y (centre) | distance-aware: \|c\|<20m / 20–80m | 800 / 224 | **5 cm near, ~54 cm far** |
| Z (centre) | linear [-3, 3] m | 1024 | ~6 mm |
| W, L, H (AABB extent) | log-space [0.1, 20] m | 1024 | ~0.5% relative |
| SIN, COS (yaw) | linear [-1, 1] | 1024 | ~0.2% |

Why these schemes:
- **distance-aware xy**: nuScenes objects are concentrated near the ego —
  uniform 1024 bins across [-80, 80] would waste resolution on empty
  far-field cells. Near-field 5 cm is enough to localise pedestrians; far
  bins still cover the fall-off range.
- **log-space sizes**: a 0.5 m pedestrian and a 15 m trailer get the same
  *relative* precision; under uniform 1024-bin in [0, 20], pedestrians
  would all collapse to one bin.
- **sin/cos yaw**: avoids the cyclic discontinuity at ±π that an
  angle-token scheme would have.

Round-trip error from `convert_q3d_data.py` over 354,486 boxes:
- centre near (\|d\|<20 m): **mean 3.75 cm, p95 6 cm**
- centre mid (20–50 m): **mean 26 cm, p95 52 cm**
- centre far (≥50 m): mean 2.27 m, p95 15.5 m (clamping at 80 m)
- size sum-of-3-axes: mean 1.1 cm
- yaw: mean 0.5 mrad

## Auxiliary loss

`MMQwen.forward` adds a per-axis **L1 loss on the differentiable expected
coord** during SFT. For every coord-token position in `labels` (after the
splice has expanded image tokens), we:

1. Take the model's logits at that position, sliced to the contiguous
   coord-vocab range `[coord_token_min, coord_token_max]`.
2. Softmax to get a 1024-dim distribution over the codebook.
3. Compute `expected = softmax @ bin_value_table[axis]` — the metre / unit
   value the model is "predicting" in expectation.
4. L1 against `bin_value_table[axis, gt_token_id]` (the GT bin's value).

Per-axis weighting in `_coord_aux_loss`:
- xy / z / sin / cos: 1.0
- w / l / h (log-space): 0.5

This complements the standard CE — CE tells the model to pick the right
token, L1 tells it that *neighbouring* tokens are nearly as good (smooth
gradient).

`bin_value_table` is a non-persistent buffer; the table is re-derived
from `quantizer.bin_value_table()` at every model `__init__`, so saved
checkpoints don't need to ship it.

## Data files

All under `/data1/byounggun/3davs_b4dl/data/`:

- `stage1_combined.json` (211 MB) — **legacy text-form**, kept for reference
- `stage1_combined_q3d.json` (226 MB) — **active SFT data**, all boxes
  quantised. 367,824 samples = 161,845 nuCaption + 123,248 det_object +
  82,731 det_area
- `3dtesting_train_q3d.json` (111 MB) — nuGrounding-only subset of the
  above (205,979 samples), used by GRPO
- `3dtesting_val_q3d.json` (22.4 MB) — Q3D-quantised val.json,
  43,448 samples, used by `eval_q3d.py`
- `stage1c_detarea_q3d.json` (42.2 MB) — **just det_area** (82,731 samples).
  Built but not yet trained on.

Features: `/data1/byounggun/3davs_b4dl/features/smap_lidar12/*.pt`,
~34 k scenes, each `(1, 12, 512)` fp16 tensor.

## Training runs

### Stage 1a — projector warmup (DONE)

Script: [scripts/stage1a_warmup.sh](scripts/stage1a_warmup.sh)
Output: `/data1/byounggun/3davs_b4dl/checkpoints/qwen_stage1a_q3d/`

- GPUs: 0, 1 (DDP, no ZeRO)
- Frozen LLM, only `mm_projector` trainable (525 k / 597 M params = 0.09 %)
- bf16 + AdamW, lr 1e-3, batch 16/GPU, 1 epoch (11,495 steps)
- Coord aux loss **off** (frozen LLM ⇒ coord embeddings can't move)
- Runtime: **3877 s (~64 min)**, 94.9 samples/sec, 2.97 steps/sec
- Loss: 4.94 (step 20) → **2.51** (final), monotone decrease

Output: `mm_projector.bin` + `trainer_state.json`. Picked up directly by
stage 1b via `--pretrain_mm_projector`.

### Stage 1b — full FT (DONE)

Script: [scripts/stage1b_full.sh](scripts/stage1b_full.sh)
Output: `/data1/byounggun/3davs_b4dl/checkpoints/qwen_stage1b_q3d/`
(checkpoints at step 5000, 10000, 11495 + final)

- GPUs: 0, 1
- All params trainable, including the 1025 newly-added embedding rows
- bf16 + AdamW, lr 5e-5 cosine warmup 0.03, batch 8/GPU × grad-accum 2 = 32 effective
- gradient_checkpointing on (bs 8 needs it for 0.6 B + 152 k vocab)
- **coord_aux_weight = 0.5**
- Runtime: **6429 s (~107 min)**, 57.2 samples/sec, 1.79 steps/sec
- Loss: 13.6 (step 10) → ~3.0 mid → **3.65 mean over epoch**

Final checkpoint contains: model.safetensors (1.2 GB), tokenizer.json
(11.6 MB) with all 1025 special tokens, mm_projector.bin, config.json
with `coord_token_min=151670`, `coord_token_max=152693`. So loading via
`MMQwen.from_pretrained(...)` gives a fully-wired model — no extra calls
needed besides `set_image_token_id` (which `eval_q3d.py` does explicitly).

### Stage 2 GRPO — first run (DONE, KL diverged)

Script: [scripts/stage2_grpo.sh](scripts/stage2_grpo.sh) — *was* this hp set:
- lr 1e-5, kl_coef 0.05
- temperature 1.0, top_p 0.9, max_new_tokens 128
- LoRA r=32, alpha=64 on q/k/v/o (9.1 M trainable)
- num_rollouts K=4, gradient_accumulation 4, sample_ratio 0.25 of nuGrounding

Output: `/data1/byounggun/3davs_b4dl/checkpoints/qwen_stage2_grpo_q3d/`
(adapters: step_50, step_100, step_150)

Trajectory (rolling-window 40 means):

| step | r_mean | det_area | det_object | KL | bev | notes |
|------|--------|----------|------------|----|----|-------|
| 10 | +0.545 | +0.03 (n=11) | +0.74 (n=29) | -0.004 | 0.000 | start |
| 50 | +0.427 | +0.15 (n=75) | +0.67 (n=125) | -0.082 | 0.000 | det_area improving fast |
| 100 | +0.469 | +0.18 (n=159) | +0.68 (n=241) | -0.345 | 0.004 | **best general checkpoint** |
| 150 | +0.468 | +0.18 (n=233) | +0.67 (n=367) | +0.583 | 0.010 | KL flipped sign |
| 160 | +0.424 | +0.17 (n=251) | +0.68 (n=389) | +1.225 | 0.010 | KL exploding |
| 170 | +0.546 | +0.17 (n=264) | +0.68 (n=416) | +3.260 | 0.007 | killed |

Diagnosis: kl_coef=0.05 too small. The Dr.GRPO advantage (no std norm) is
fine; the issue is LR + KL together let the policy drift.

### Stage 2 GRPO — safer run (DONE, also diverged at step ~190)

Same script after editing the hp block (lr 5e-6, kl_coef 0.20).
Output: `/data1/byounggun/3davs_b4dl/checkpoints/qwen_stage2_grpo_q3d_safe/`
(adapters: step_50, step_100, step_150).

| step | r_mean | det_area | det_object | KL | bev |
|------|--------|----------|------------|----|----|
| 10 | +0.590 | +0.09 (n=11) | +0.78 (n=29) | -0.012 | 0.000 |
| 50 | +0.437 | +0.14 (n=75) | +0.71 (n=125) | -0.084 | 0.002 |
| 100 | +0.461 | +0.15 (n=159) | +0.72 (n=241) | -0.359 | 0.005 |
| 150 | +0.413 | +0.16 (n=233) | +0.70 (n=367) | -0.558 | 0.004 |
| 170 | +0.553 | +0.16 (n=264) | +0.69 (n=416) | +0.510 | 0.009 |
| 190 | +0.661 | +0.16 (n=281) | +0.69 (n=479) | **+2.490** | 0.002 |

Observations:
- Safer run **starts higher** (+0.59 vs +0.55) and **stays around**
  unsafe-run levels through step 100.
- KL flips sign around step 170 (similar to unsafe), but the magnitude
  at the same step is ~6× lower (+2.5 vs +3.3) thanks to the higher
  kl_coef.
- Has not delivered any det_area improvement over SFT-1b, just more
  format compliance.

Killed at step 190 with SIGINT; last saved adapter is step_150.

### Stage 1c — det_area continued SFT (NOT YET RUN)

Script written: [scripts/stage1c_detarea.sh](scripts/stage1c_detarea.sh)
Plan:
- Init from `qwen_stage1b_q3d` (model + tokenizer + projector all loaded
  via `--model_name_or_path qwen_stage1b_q3d`)
- Train **only on det_area** (82,731 samples), 2 epochs
- lr 2e-5 (lower than 1b's 5e-5 to avoid catastrophic forgetting on
  classification), warmup 0.03, cosine
- **coord_aux_weight = 1.0** (double 1b's weight; the L1 distance signal
  is what we most want for localisation)
- bs 8/GPU × grad-accum 2, gradient_checkpointing on
- Expected runtime: ~25–30 min for 2 epochs (82 k samples × 2 / 32
  effective batch / 1.79 steps-per-sec ≈ 1.4 h; halve for 2 GPUs ≈ 45 min)
- Output: `/data1/byounggun/3davs_b4dl/checkpoints/qwen_stage1c_detarea_q3d/`

Hypothesis: removing the det_object echo distractor will let the model
allocate capacity to text→region grounding. Even if it overfits to
det_area phrasing, GRPO afterwards can re-balance.

## Evaluation framework

[eval_q3d.py](eval_q3d.py) loads either the bare SFT checkpoint or
SFT + LoRA adapter (`--lora_adapter`), runs greedy generation on the
quantised val set, and reports:

```
=== Q3D eval summary ===
[det_object] n= 234  parse_rate=...  iou_mean=...  cd_mean=...  acc10=...  acc5=...
[  det_area] n= 266  parse_rate=...  iou_mean=...  cd_mean=...  acc10=...  acc5=...

=== LiDAR-LLM comparable metrics ===
ACC-10 (exact)        : ...%
ACC-5  (supercategory): ...%
BEV mIoU (all matched): ...
BEV mIoU per class: ...
Car BEV mIoU breakdown:
  all     :
  det_obj : (echo task — not what LiDAR-LLM measures)
  det_area: (true VG; LiDAR-LLM 14.3)
```

ACC-10 is exact match against the 10 nuScenes detection classes.
ACC-5 buckets into {vehicle, two_wheeler, pedestrian, barrier,
traffic_cone} (see `SUPER5` in eval_q3d.py).

Reproducing the runs in this doc:

```bash
# SFT-1b baseline
CUDA_VISIBLE_DEVICES=1 python -u eval_q3d.py \
  --sft_dir /data1/byounggun/3davs_b4dl/checkpoints/qwen_stage1b_q3d \
  --max_samples 500 \
  --out_jsonl /tmp/q3d_logs/eval_sft1b_500.jsonl

# GRPO step_100 (unsafe run)
CUDA_VISIBLE_DEVICES=1 python -u eval_q3d.py \
  --sft_dir /data1/byounggun/3davs_b4dl/checkpoints/qwen_stage1b_q3d \
  --lora_adapter /data1/byounggun/3davs_b4dl/checkpoints/qwen_stage2_grpo_q3d/step_100 \
  --max_samples 500 \
  --out_jsonl /tmp/q3d_logs/eval_grpo100_500.jsonl
```

## Eval results (500 samples)

### SFT-1b baseline

```
[det_object] n= 234  parse_rate=97.44%  iou_mean=0.498  iou_p50=0.529  cd_mean=1.25m  acc10=94.44%  acc5=97.44%
[  det_area] n= 266  parse_rate=62.03%  iou_mean=0.003  iou_p50=0.000  cd_mean=25.83m  acc10=100.0%  acc5=100.0%

ACC-10 (exact)        : 97.40%  (487/500)
ACC-5  (supercategory): 98.80%  (494/500)
BEV mIoU (all matched): 0.258  (n=442)

BEV mIoU per class:
  bicycle      : mean=0.208  p50=0.016  n=17
  bus          : mean=0.385  p50=0.322  n=48
  car          : mean=0.279  p50=0.000  n=176
  pedestrian   : mean=0.142  p50=0.000  n=142
  traffic_cone : mean=0.405  p50=0.172  n=46
  truck        : mean=0.327  p50=0.095  n=13

Car BEV mIoU breakdown:
  all     : mean=27.93  n=176
  det_obj : mean=63.12  n=77   (echo task)
  det_area: mean= 0.56  n=99   (true VG; LiDAR-LLM 14.3)
```

### GRPO step_100 (unsafe run, lora applied)

```
[det_object] n= 234  parse_rate=91.88%  iou_mean=0.545  iou_p50=0.662  cd_mean=1.11m  acc10=96.15%  acc5=97.86%
[  det_area] n= 266  parse_rate=76.69%  iou_mean=0.002  iou_p50=0.000  cd_mean=23.89m  acc10=100.0%  acc5=100.0%

ACC-10 (exact)        : 98.20%  (491/500)
ACC-5  (supercategory): 99.00%  (495/500)
BEV mIoU (all matched): 0.242  (n=487)

Car BEV mIoU breakdown:
  all     : mean=25.78  n=183
  det_obj : mean=63.78  n=73   (echo task)
  det_area: mean= 0.56  n=110  (true VG; LiDAR-LLM 14.3)
```

### Side-by-side (500 val samples)

| Metric | SFT-1b | GRPO unsafe step_100 | GRPO safer step_100 |
|--------|--------|----------|----------|
| ACC-10 | 97.40% | **98.20%** | 97.80% |
| ACC-5 | 98.80% | **99.00%** | 98.60% |
| BEV mIoU (all) | 0.258 | 0.242 | 0.259 |
| det_obj IoU mean | 0.498 | **0.545** | 0.544 |
| det_obj IoU p50 | 0.529 | **0.662** | 0.665 |
| det_obj parse | 97.4% | 91.9% | 92.7% |
| **det_area parse** | 62.0% | **76.7%** | 68.8% |
| det_area missed | 144 | **86** | 117 |
| det_obj Car mIoU | 63.12 | 63.78 | **63.97** |
| **det_area Car mIoU (VG)** | **0.56** | **0.56** | **0.54** |

Reading: GRPO (either hp set) improves format compliance and pushes the
easy task a bit higher, but the localisation gap (0.56 vs 14.3) is
unmoved. Safer GRPO and unsafe GRPO produce nearly identical eval
numbers at step_100; the safer run's value is that it survives further
training without catastrophic divergence (KL stays bounded ~6× lower).
For the GRPO checkpoint to actually use, **either step_100 is fine** —
unsafe gives the bigger parse-rate gain, safer is more conservative.

### Sanity-check on the IoU number itself

When the det_area Car number came in at 0.56 % we questioned the metric.
Verified end-to-end in `/tmp/q3d_logs/eval_*.jsonl`:

- Synthetic checks (`bev_iou` in `rewards_lidar.py`): identical box → 1.0,
  50 %-shifted box → 0.333, far-apart → 0.0. ✓
- Data format: nuGrounding raw answers store boxes as
  `[xmin, xmax, ymin, ymax, zmin, zmax, yaw]` — i.e., the **AABB of the
  rotated 3D box plus its yaw**. We compute IoU on the AABB extents,
  which is the same convention LiDAR-LLM evaluates against (same HF
  dataset).
- Real det_area Car evidence (from `eval_sft1b_500.jsonl`): 122 Car
  prompts, 71 with both pred + GT parsed, **only 5/71 have any positive
  IoU**. Best single match IoU = 0.353. Average across the 71 matched
  samples = 0.79 % (the headline 0.56 % includes parse-failures as
  IoU = 0).
- Pred-side fingerprint: 122 Car predictions → only 42 unique coord
  sequences. Single most common output is empty `<|box_start|><|box_end|>`
  (31×); next is the 8-token "average car" sequence
  `<coord_404><coord_0><coord_242><coord_578><coord_739><coord_578><coord_506><coord_0>`
  (18×). 31 singletons account for the rest. The model is mode-collapsed,
  not mis-evaluated.
- Predicted centre distribution vs. GT (n=110 vs n=171):
  - pred  median (4.8, **−20.0**), x range [−20, 32], y range [−44, 34]
  - gt    median (5.7, **+17.0**), x range [−70, 73], y range [−74, 80]
  - the pred y is locked into a narrow strip; GT spans the full ±80 m.

### Why we trail LiDAR-LLM (apples-to-apples)

LiDAR-LLM Visual-Grounding is the *exact* same det_area task on the same
HF dataset, so the data is comparable. Differences:

| Knob | LiDAR-LLM | Ours |
|------|-----------|------|
| LLM size | 7 B (LLaMA-2 / Vicuna) | 0.6 B (Qwen3) |
| SFT epochs (grounding) | unspecified, multi-stage | 1 |
| Box format | text floats | Q3D 1024-bin discrete tokens |
| Reported Car BEV mIoU | 14.3 % | 0.56 % |

So the 0.56 vs 14.3 gap is roughly: ~10× model capacity + ~3× training,
plus our model has to learn the discrete codebook from scratch (LiDAR-LLM
just regresses floats with the LM). Q3D's payoff is supposed to come
later — cleaner aux loss and better RL signal for refinement — but
1 epoch of vanilla SFT isn't enough to get the discretisation paying
rent.

### Failure mode samples (det_area)

SFT-1b mode collapse — same coord prefix repeated across very different prompts:
```
GT  : The pedestrian is located at <|box_start|><coord_808><coord_129>...<|box_end|>
PRED: The pedestrian is located at <|box_start|><coord_552><coord_0><coord_314><coord_358>...<|box_end|>

GT  : The 2 cars are at [<|box_start|><coord_425><coord_976>...,<|box_start|><coord_503>...]
PRED: The 2 cars are at [<|box_start|><coord_404><coord_0><coord_267><coord_578>...,<|box_start|>(same again)]
```

The model has effectively memorised "average pedestrian box" / "average
car box" and emits it regardless of the question. SFT loss can't punish
this because the L2-on-expected-coord penalty is small once the bin index
is in the right neighbourhood; CE only cares about exact-bin match.

GRPO step_100 keeps the same collapse, just emits valid 8-token sequences
more reliably (no spurious early `<|box_end|>`).

## What changed vs the prior (text-form) pipeline

| Item | Before | After |
|------|--------|-------|
| Box format | text `[xmin,...,yaw]` | 8 Q3D coord tokens |
| Tokenizer vocab | 151,670 (Qwen3 + `<image>`) | 152,694 (+ 1024 coord) |
| SFT loss | CE only | CE + per-axis L1 on expected coord |
| Reward parser | regex on bracketed text | quantizer.parse_quantized_boxes + legacy fallback |
| GRPO data | `3dtesting_dataset/train.json` | `3dtesting_train_q3d.json` |

Prior text-form SFT (vicuna baseline, before model swap) showed similar
det_area collapse — the failure is not a Q3D artefact but a fundamental
SFT/data-balance issue. Q3D has *helped* by making the parse rate climb
from ~25% (text) to 60–75%.

## Key files & paths

| Purpose | Path |
|---|---|
| Stage-1b SFT model | `/data1/byounggun/3davs_b4dl/checkpoints/qwen_stage1b_q3d/` |
| GRPO unsafe step_100 adapter | `/data1/byounggun/3davs_b4dl/checkpoints/qwen_stage2_grpo_q3d/step_100/` |
| GRPO safer step_100 adapter | `/data1/byounggun/3davs_b4dl/checkpoints/qwen_stage2_grpo_q3d_safe/step_100/` |
| Logs (cleared on reboot) | `/tmp/q3d_logs/` |
| Latest eval JSONLs | `/tmp/q3d_logs/eval_sft1b_500.jsonl`, `eval_grpo100_500.jsonl` |
| SMAP features | `/data1/byounggun/3davs_b4dl/features/smap_lidar12/*.pt` |

## Next session — pick up here

Priority order:

1. **Run stage 1c** (`bash scripts/stage1c_detarea.sh`). This is the
   most-likely-to-help-most experiment we have queued. ~45 min on
   2× A6000. After it finishes, eval with
   `eval_q3d.py --sft_dir qwen_stage1c_detarea_q3d`.
2. **Decide on safer GRPO**: it's running but tracking the unsafe run.
   If by step ~150 it's still around r_mean 0.4 with KL trending past
   -0.5, kill it and accept step_100 as the GRPO checkpoint.
3. **GRPO from stage 1c**: re-run [scripts/stage2_grpo.sh](scripts/stage2_grpo.sh)
   pointing `--sft_dir` at the stage 1c output. Stage 1c should give
   GRPO better starting rollouts on det_area, which is where GRPO is
   currently signal-starved.
4. **Optional improvements** (only if 1–3 don't crack 5% on Car
   det_area mIoU):
   - **per-task data weighting**: triple-up det_area in
     `stage1_combined_q3d.json` and re-run a fresh stage 1b. More
     expensive than 1c but more principled.
   - **rotated BEV IoU loss**: implement self-contained rotated-rect
     polygon clipping (no mmdet) and add as a third aux term. The
     log on this is already discussed — design is in `quantizer.py`
     comments.
   - **prompt restructuring**: many det_area prompts read "The car is
     at" — model can't disambiguate which car. Add `[front view]` /
     `[back view]` prefixes (some samples already have them) more
     consistently in the data prep.

## Open issues / gotchas

- `bin_value_table` is a non-persistent buffer. If you ever change the
  codebook layout (axis ranges), make sure to delete cached `.bin`
  buffers and confirm `quantizer.py` and the model agree.
- `train_qwen_grpo.py` log message `[grpo] sampled 51494 of 51494
  (ratio=0.25)` is misleading: 51,494 is the post-sample count
  (25 % of 205,979 nuGrounding entries). The "of 51,494" bit is a
  display bug, not a sampling bug.
- The eval Car det_area count is small (n≈100 in 500 val samples). For
  publishable numbers, run with `--max_samples 5000` (~50 min on 1
  A6000) — most of our 0.56 is one-decimal-place noise.
- Two GPUs (2, 3) on the host are still occupied by another user
  (`donguk`). Default scripts use GPUs 0, 1. Override with
  `GPUS=0,1 bash scripts/...` to be explicit.

## Commands cheat sheet

```bash
# 0) sanity: round-trip the quantizer
python -c "from qwen_mm.quantizer import encode_box_indices, decode_box_indices; \
  print(decode_box_indices(encode_box_indices([10.5, 14.5, -5.0, -3.2, -1.8, -0.4, 1.57])))"

# 1) (one-time) build the Q3D data files
python convert_q3d_data.py
python -c "import json; d=json.load(open('/data1/byounggun/3davs_b4dl/data/stage1_combined_q3d.json')); \
  json.dump([x for x in d if x.get('task')=='nugrounding'], \
            open('/data1/byounggun/3davs_b4dl/data/3dtesting_train_q3d.json','w'))"

# 2) train
bash scripts/stage1a_warmup.sh        # ~64 min, GPUs 0,1
bash scripts/stage1b_full.sh          # ~107 min, GPUs 0,1
bash scripts/stage1c_detarea.sh       # NOT YET RUN
GPU=0 bash scripts/stage2_grpo.sh     # ~per-step 5 s × N

# 3) eval
CUDA_VISIBLE_DEVICES=1 python -u eval_q3d.py --sft_dir <CKPT> --max_samples 500
# add --lora_adapter <ADAPTER_DIR> for any GRPO checkpoint
```
