# B4DL VoxelNeXt 파이프라인 재현 가이드

다른 서버에서 처음부터 데이터 준비 + 학습까지 그대로 따라할 수 있도록 정리.
모든 경로는 환경변수로 묶어놨으니 본인 서버에 맞게 바꾸면 됨.

## 0. 최종 디렉토리 레이아웃

```
$REPO_ROOT/                             # B4DL 레포
├── new_model_code/                     # 학습 코드 (이 가이드의 출발점)
├── 3dtesting_dataset/                  # nuGrounding 원본 (LiDAR-LLM)
│   ├── raw/                            #   HF에서 받은 raw json
│   ├── train.json   val.json           #   ChatML로 변환된 raw metres 형식
│   └── build_3dtesting.py
├── lidarllm_only_dataset/              # nuCaption 원본 (B4DL HF)
│   ├── stage1_lidarllm_mm.json
│   └── stage1_train_converted.json     #   ChatML 변환본
└── (...)

$DATA_ROOT/                             # 학습용 데이터/체크포인트 보관 (큰 디스크)
├── data/
│   ├── stage1_combined.json            # nuCaption + nuGrounding (raw metres)
│   ├── stage1_combined_q3d.json        # ↑ 의 Q3D 양자화본 (학습용)
│   ├── 3dtesting_train_q3d.json        # nuGrounding only, Q3D 양자화 (GRPO/eval)
│   └── 3dtesting_val_q3d.json
├── features/
│   └── voxelnext_q256/                 # 샘플당 .pt blob 34,013개
└── checkpoints/
    ├── qwen_stage1a_vxnxt/
    ├── qwen_stage1b_vxnxt/
    └── qwen_stage2_grpo_vxnxt/

$NUSCENES_ROOT/                         # nuScenes 원본 (별도 디스크 OK)
├── v1.0-trainval/
├── samples/   sweeps/   maps/
```

권장 환경변수:

```bash
export REPO_ROOT=/home/<user>/B4DL
export DATA_ROOT=/data1/<user>/3davs_b4dl
export NUSCENES_ROOT=/data/nuscenes      # nuScenes 받은 위치
export VOXELNEXT_ROOT=/data1/<user>/voxelnext_work/VoxelNeXt
export VOXELNEXT_CKPT=/data1/<user>/voxelnext_work/ckpt/voxelnext_nuscenes_kernel1.pth
```

---

## 1. 사전 준비

### 1.1 nuScenes 다운로드

[nuscenes.org](https://www.nuscenes.org/nuscenes#download) 에서 다음 받기:
- `v1.0-trainval` (메타데이터 + 1000 scene)
- `samples`, `sweeps` (LiDAR/카메라 데이터)

10-sweep 누적으로 LiDAR feature를 뽑으니 **`sweeps/LIDAR_TOP/` 가 반드시 필요**.

### 1.2 Conda envs

학습용 (`qwen_mm`)과 VoxelNeXt 추출용 (`voxelnext`)은 **반드시 분리** — spconv 버전이 충돌함.

#### qwen_mm (학습/평가)

```bash
conda create -p $DATA_ROOT/conda_envs/qwen_mm python=3.10 -y
conda activate $DATA_ROOT/conda_envs/qwen_mm

# torch 2.4.x + cu121 권장
pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.0 accelerate datasets peft trl
pip install nuscenes-devkit pyquaternion wandb numpy einops sentencepiece
```

#### voxelnext (feature 추출 전용)

```bash
conda create -p $DATA_ROOT/conda_envs/voxelnext python=3.10 -y
conda activate $DATA_ROOT/conda_envs/voxelnext

pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121

# 핵심: spconv는 cu121 빌드 (cu120은 torch 2.4와 silent segfault)
pip install spconv-cu121 cumm-cu121

pip install numpy easydict pyyaml numba llvmlite tqdm scikit-image SharedArray \
            opencv-python tensorboardX nuscenes-devkit pyquaternion av2 \
            "kornia<0.7" matplotlib gdown

# pcdet (OpenPCDet fork)
git clone https://github.com/JIA-Lab-research/VoxelNeXt $VOXELNEXT_ROOT
cd $VOXELNEXT_ROOT
pip install --no-build-isolation -e .

# pcdet 빌드가 cuda_bf16 헤더로 깨지면:
# conda install -c nvidia cuda-cccl=12.1
```

### 1.3 VoxelNeXt pretrained ckpt

```bash
mkdir -p $(dirname $VOXELNEXT_CKPT)
gdown "https://drive.google.com/uc?id=1IV7e7G9X-61KXSjMGtQo579pzDNbhwvf" -O $VOXELNEXT_CKPT
# 32 MB, mAP 60.5 / NDS 66.6 on nuScenes val
```

---

## 2. 데이터 준비 (4 단계)

### 2.1 nuGrounding raw 다운로드 → ChatML 변환

LiDAR-LLM의 3D box grounding 데이터.

```bash
cd $REPO_ROOT/3dtesting_dataset
python build_3dtesting.py --feat_folder ""    # feature 필터링 끄기 (없을 때)
```

`build_3dtesting.py`가 내부적으로 다음을 함:
1. `raw/LiDAR-LLM-Nu-Grounding-{train,val}.json` 을 HF `Senqiao/LiDAR-LLM-Nu-Grounding` 에서 받음
2. ChatML format (`<image>\n[<view> view] question`, GPT answer)로 변환
3. `train.json` (~206k entry), `val.json` 으로 저장

샘플:
```json
{
  "scene_id": "<sample_token>",
  "split": "train",
  "view": "back",
  "template_type": "det_object",
  "conversations": [
    {"from": "human", "value": "<image>\n[back view] What is at the location [-16.33,-16.03,-1.32,...,1.36]?"},
    {"from": "gpt",   "value": "There is a traffic_cone at the location [-16.33,-16.03,...,1.36]."}
  ]
}
```

### 2.2 nuCaption raw → ChatML 변환

B4DL의 nuCaption (LiDAR-LLM 1단계 caption 데이터).

```bash
cd $REPO_ROOT/lidarllm_only_dataset
# stage1_lidarllm_mm.json 받기 (B4DL HF)
huggingface-cli download ccho4702/nuScenes-B4DL stage1_lidarllm_mm.json --local-dir .
```

raw mm → converted 변환 (질문/답을 conversations로):
```python
# convert_nucaption.py
import json
src = json.load(open("stage1_lidarllm_mm.json"))
out = [{
    "scene_id": e["sample_token"],
    "sample_token": e["sample_token"],
    "split": e["split"],
    "conversations": [
        {"from": "human", "value": f"<image>\n{e['question']}"},
        {"from": "gpt",   "value": e["answer"]},
    ],
} for e in src]
json.dump([e for e in out if e["split"]=="train"],
          open("stage1_train_converted.json", "w"), indent=2)
json.dump([e for e in out if e["split"]=="val"],
          open("stage1_val_converted.json",   "w"), indent=2)
```

→ `stage1_train_converted.json` (~162k)

### 2.3 합치고 Q3D 양자화

**Q3D 양자화**가 핵심: raw `[xmin,xmax,...,yaw]` 7-tuple metres → `<coord_a><coord_b>...<coord_h>` 8 token (1024-bin codebook).
훈련은 next-token classification으로 동작하고, 추론 후엔 regex로 metres 복구.

상세 스킴은 [qwen_mm/quantizer.py](qwen_mm/quantizer.py) 참고:
- X, Y centre: 거리적응형 (±20m: 5cm bin, ±20~80m: ~54cm bin)
- Z centre: 선형 ±3m
- W, L, H: log-space [0.1, 20]m
- sin yaw, cos yaw: 선형 [-1, 1]

#### Step A — combine (nuCaption + nuGrounding, raw metres 유지하면서 box span만 마킹)

```bash
mkdir -p $DATA_ROOT/data
cd $REPO_ROOT/new_model_code
python build_combined_stage1.py \
    --nucaption $REPO_ROOT/lidarllm_only_dataset/stage1_train_converted.json \
    --nugrounding $REPO_ROOT/3dtesting_dataset/train.json \
    --out $DATA_ROOT/data/stage1_combined.json
```

→ `stage1_combined.json` (~368k entry, raw metres + `<|box_start|>...<|box_end|>` 마킹)

#### Step B — Q3D 양자화 (run-once)

```bash
# stage1 학습용 (caption + grounding)
python convert_q3d_data.py \
    --in_path  $DATA_ROOT/data/stage1_combined.json \
    --out_path $DATA_ROOT/data/stage1_combined_q3d.json

# GRPO + eval용 (grounding only)
python convert_q3d_data.py \
    --in_path  $REPO_ROOT/3dtesting_dataset/train.json \
    --out_path $DATA_ROOT/data/3dtesting_train_q3d.json
python convert_q3d_data.py \
    --in_path  $REPO_ROOT/3dtesting_dataset/val.json \
    --out_path $DATA_ROOT/data/3dtesting_val_q3d.json
```

박스 부분이 다음처럼 바뀜:
```
before: <|box_start|>[-16.33,-16.03,-1.32,-1.03,-1.61,-0.87,1.36]<|box_end|>
after:  <|box_start|><coord_76><coord_376><coord_300><coord_212><coord_205><coord_386><coord_1012><coord_619><|box_end|>
```

학습 dataloader는 이 양자화본을 그대로 토크나이즈만 함 — 런타임 변환 없음.
디코딩(token→metres)은 **추론 후** `parse_quantized_boxes()` regex로.

### 2.4 sample_tokens_union.json (feature 추출 대상)

학습/eval에 등장하는 모든 sample_token의 합집합. VoxelNeXt feature를 이 토큰들에 대해서만 뽑음.

```python
# build_token_union.py
import json
tokens = set()
for p in [
    f"{DATA_ROOT}/data/stage1_combined_q3d.json",
    f"{DATA_ROOT}/data/3dtesting_train_q3d.json",
    f"{DATA_ROOT}/data/3dtesting_val_q3d.json",
]:
    for e in json.load(open(p)):
        tokens.add(e["sample_token"])
json.dump(sorted(tokens), open("$REPO_ROOT/new_model_code/sample_tokens_union.json", "w"))
```

→ 약 34,013 token.

---

## 3. VoxelNeXt feature 추출

scene 하나당 `.pt` blob 하나 (K=256 sparse voxel queries).

```bash
conda activate $DATA_ROOT/conda_envs/voxelnext

mkdir -p $DATA_ROOT/features/voxelnext_q256

CUDA_VISIBLE_DEVICES=0 python -u $REPO_ROOT/new_model_code/extract_voxelnext_features.py \
    --nuscenes_root $NUSCENES_ROOT \
    --token_list $REPO_ROOT/new_model_code/sample_tokens_union.json \
    --ckpt $VOXELNEXT_CKPT \
    --save_dir $DATA_ROOT/features/voxelnext_q256 \
    --top_k 256
```

각 blob (`<sample_token>.pt`) 구조:
```python
{
  "feat":  fp16 (256, 128),   # backbone feature
  "xyz":   fp16 (256, 3),      # metres in keyframe ego frame
  "score": fp16 (256,),        # sigmoid(max-class hm logit)
  "cls":   int8 (256,),
  "mask":  bool (256,),        # True where a real voxel (rest is zero-pad)
}
```

샘플당 ~150 KB. 34k 샘플 = **약 5 GB** 디스크.
1× A6000 기준 **~16 it/s**, 전체 약 30분~1시간. 디스크 IO 병목 시 GPU 2개로 분할 권장 (`--start --end`).

---

## 4. 학습 실행

[scripts/run_all_voxelnext.sh](scripts/run_all_voxelnext.sh) 가 stage 1a → 1b → 2를 순서대로 돌림:

```bash
conda activate $DATA_ROOT/conda_envs/qwen_mm

GPUS=2,3 bash $REPO_ROOT/new_model_code/scripts/run_all_voxelnext.sh
```

옵션:
- `GPUS=2,3` — 사용 GPU. stage 2(GRPO)는 첫 번째 GPU만 사용.
- `SKIP_STAGE1A=1` / `SKIP_STAGE1B=1` / `SKIP_STAGE2=1` — 단계 건너뛰기 (이전 산출물 있어야 함).

각 단계 산출물:
| 단계 | 출력 디렉토리 | 핵심 파일 |
|---|---|---|
| 1a (projector warmup) | `$DATA_ROOT/checkpoints/qwen_stage1a_vxnxt/` | `mm_projector.bin` (feat + pos projector) |
| 1b (full FT)          | `$DATA_ROOT/checkpoints/qwen_stage1b_vxnxt/` | HF model + tokenizer |
| 2 (GRPO LoRA)         | `$DATA_ROOT/checkpoints/qwen_stage2_grpo_vxnxt/` | LoRA adapter |

### 평가

```bash
CUDA_VISIBLE_DEVICES=0 python -u $REPO_ROOT/new_model_code/eval_q3d.py \
    --sft_dir $DATA_ROOT/checkpoints/qwen_stage1b_vxnxt \
    --val_path $DATA_ROOT/data/3dtesting_val_q3d.json \
    --feat_folder $DATA_ROOT/features/voxelnext_q256 \
    --max_samples 500
```

LiDAR-LLM 비교 지표 (BEV mIoU on Car for det_area, ACC-10/ACC-5)도 함께 출력됨.

---

## 5. 디스크 / 시간 견적

| 항목 | 크기 | 비고 |
|---|---|---|
| nuScenes v1.0-trainval | ~480 GB | sweeps + samples 포함 |
| 3dtesting raw + 변환본 | ~120 MB | |
| nuCaption raw + 변환본 | ~250 MB | |
| `stage1_combined_q3d.json` | ~250 MB | 학습 dataloader 입력 |
| VoxelNeXt feature blobs | ~5 GB | 34k 샘플 × ~150 KB |
| stage 1a checkpoint | ~250 KB | projector만 저장 |
| stage 1b checkpoint | ~3 GB | Qwen3-0.6B full FT |
| stage 2 LoRA adapter | ~50 MB | r=32, alpha=64 |

| 단계 | A6000×2 시간 | 비고 |
|---|---|---|
| feature 추출 | 30분~1시간 | I/O bound |
| stage 1a | ~1시간 | 22989 step × 1.5 it/s |
| stage 1b | 4~5시간 | K=256 splice → batch 4 × grad_accum 4 |
| stage 2 GRPO | 6~8시간 | 1 GPU, K=4 rollouts × max_new=128 |

---

## 6. 트러블슈팅

### spconv 관련

- **silent segfault after MeanVFE** → `spconv-cu120`이 설치됨. `pip uninstall spconv-cu120 cumm-cu120 && pip install spconv-cu121 cumm-cu121`.
- **`__NV_IS_DEVICE` undefined** (cuda_bf16 헤더) → `conda install -c nvidia cuda-cccl=12.1`.

### pcdet 빌드

- **PEP 517 build isolation: torch not visible** → `pip install --no-build-isolation -e .`.
- **kornia 0.8.x argo2 import 깨짐** → `pip install "kornia<0.7"`.
- **matplotlib numpy ABI mismatch** → `pip install -U matplotlib`.

### VoxelNeXt config

- **`_BASE_CONFIG_` 못 찾음** → 추출 스크립트가 `cfg_from_yaml_file` 부르기 전에 `VoxelNeXt/tools/` 로 chdir. 이미 코드에 들어있음.
- **dataset 없이 `build_network` 깨짐** → `_DummyDataset` + `_DummyFeatureEncoder(num_point_features=5)` 로 우회. 이미 코드에 들어있음.

### 학습

- **stage 1a `trainable=132,096` (132k)** → pos_projector freeze 버그. `freeze_llm_except_projector` 가 두 projector 모두 unfreeze 하는지 확인. 수정본에서는 ~136,192 (136k)이 정상.
- **`<|box_start|><|box_end|>` 빈 박스 출력 mode collapse** → encoder가 spatial 신호를 못 주는 신호. SMAP에서 이게 50%까지 가면 voxelnext로 갈아탈 시기.

---

## 7. 빠른 새 서버 부트스트랩

```bash
# 0. 변수 정의
export REPO_ROOT=/home/<user>/B4DL
export DATA_ROOT=/data1/<user>/3davs_b4dl
export NUSCENES_ROOT=/data/nuscenes
export VOXELNEXT_ROOT=/data1/<user>/voxelnext_work/VoxelNeXt
export VOXELNEXT_CKPT=/data1/<user>/voxelnext_work/ckpt/voxelnext_nuscenes_kernel1.pth

# 1. 레포 + nuScenes
git clone <B4DL repo> $REPO_ROOT
# nuScenes 다운로드는 별도

# 2. 환경 (§1.2)
# qwen_mm + voxelnext 따로

# 3. 데이터 (§2)
cd $REPO_ROOT/3dtesting_dataset && python build_3dtesting.py --feat_folder ""
# nuCaption raw 받고 변환 (§2.2)
cd $REPO_ROOT/new_model_code
python build_combined_stage1.py --out $DATA_ROOT/data/stage1_combined.json
python convert_q3d_data.py --in_path $DATA_ROOT/data/stage1_combined.json \
                           --out_path $DATA_ROOT/data/stage1_combined_q3d.json
python convert_q3d_data.py --in_path $REPO_ROOT/3dtesting_dataset/train.json \
                           --out_path $DATA_ROOT/data/3dtesting_train_q3d.json
python convert_q3d_data.py --in_path $REPO_ROOT/3dtesting_dataset/val.json \
                           --out_path $DATA_ROOT/data/3dtesting_val_q3d.json
python build_token_union.py    # §2.4

# 4. feature 추출 (§3)
conda activate $DATA_ROOT/conda_envs/voxelnext
CUDA_VISIBLE_DEVICES=0 python -u extract_voxelnext_features.py \
    --nuscenes_root $NUSCENES_ROOT \
    --token_list sample_tokens_union.json \
    --ckpt $VOXELNEXT_CKPT \
    --save_dir $DATA_ROOT/features/voxelnext_q256

# 5. 학습 (§4)
conda activate $DATA_ROOT/conda_envs/qwen_mm
GPUS=0,1 bash scripts/run_all_voxelnext.sh
```
