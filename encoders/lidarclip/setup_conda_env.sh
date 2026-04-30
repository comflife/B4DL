#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${1:-lidarclip}"

conda create -y -n "${ENV_NAME}" python=3.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

conda install -y -c pytorch -c conda-forge \
  pytorch=1.9.0 torchvision=0.10.0 cudatoolkit=11.1

pip install --upgrade pip setuptools==59.5.0 wheel ninja

pip install \
  einops==0.8.1 \
  ipdb==0.13.13 \
  lyft_dataset_sdk==0.0.8 \
  matplotlib==3.5.2 \
  mmdet==2.14.0 \
  mmengine==0.10.6 \
  mmsegmentation==0.14.1 \
  numba==0.48.0 \
  numpy==1.19.5 \
  nuscenes-devkit==1.1.10 \
  open3d==0.17.0 \
  openai_clip==1.0.1 \
  opencv-python==4.11.0.86 \
  pandas==1.3.5 \
  Pillow==9.5.0 \
  pycocotools==2.0.7 \
  pyquaternion==0.9.9 \
  pytest==7.4.4 \
  pytorch-lightning==1.6.2 \
  scikit-image==0.19.3 \
  scipy==1.7.3 \
  seaborn==0.13.2 \
  shapely==1.8.5 \
  terminaltables==3.1.10 \
  tqdm==4.65.2 \
  trimesh==2.35.39

pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

cd /home/byounggun/B4DL/encoders/lidarclip/sst
pip install -v -e .

cd /home/byounggun/B4DL/encoders/lidarclip
echo "Environment '${ENV_NAME}' is ready."
