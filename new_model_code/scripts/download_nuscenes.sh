#!/usr/bin/env bash
# nuScenes v1.0-trainval downloader.
#
# nuScenes requires registration + signed URLs. There is no public direct
# download — you must:
#   1. Register at https://www.nuscenes.org/sign-up
#   2. Log in and visit https://www.nuscenes.org/nuscenes#download
#   3. Click each "Trainval" .tgz link, copy the *signed* URL (valid ~24h),
#      and paste them into the URLS=( ... ) block below.
#
# Total size: ~480 GB across 11 tarballs:
#   - v1.0-trainval_meta.tgz                             (~400 MB)
#   - v1.0-trainval{01..10}_blobs.tgz                    (~46 GB each)
#
# Sweeps are inside the same trainval blobs (no separate sweeps tarball).
# After extraction, layout will be:
#   $NUSCENES_ROOT/v1.0-trainval/
#   $NUSCENES_ROOT/samples/{LIDAR_TOP,CAM_FRONT,...}/
#   $NUSCENES_ROOT/sweeps/{LIDAR_TOP,...}/
#   $NUSCENES_ROOT/maps/

set -euo pipefail

if [[ -z "${NUSCENES_ROOT:-}" ]]; then
    echo "ERROR: NUSCENES_ROOT not set. source ~/.bashrc.b4dl first." >&2
    exit 1
fi

mkdir -p "$NUSCENES_ROOT" "$NUSCENES_ROOT/_tarballs"
cd "$NUSCENES_ROOT/_tarballs"

# Paste signed download URLs here (one per line, https://...). Empty by default.
URLS=(
    https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/can_bus.zip
    https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz
    https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval01_blobs.tgz
    https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval02_blobs.tgz
    https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval03_blobs.tgz
    https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval04_blobs.tgz
    https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval05_blobs.tgz
    https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval06_blobs.tgz
    https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval07_blobs.tgz
    https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval08_blobs.tgz
    https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval09_blobs.tgz
    https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval10_blobs.tgz
    https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-test_meta.tgz
    https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-test_blobs.tgz
)

if [[ ${#URLS[@]} -eq 0 ]]; then
    cat <<EOF
URLS array is empty. Edit this script and paste the 11 signed nuScenes URLs
from https://www.nuscenes.org/nuscenes#download into the URLS=( ... ) block.

Tarballs:
  - v1.0-trainval_meta.tgz                  (metadata, ~400 MB)
  - v1.0-trainval01_blobs.tgz ... 10_blobs  (samples + sweeps, ~46 GB each)

Then re-run:  bash $0
EOF
    exit 1
fi

echo "Downloading ${#URLS[@]} tarballs to $PWD ..."
for url in "${URLS[@]}"; do
    fname=$(basename "${url%%\?*}")
    if [[ -f "$fname" ]]; then
        echo "  exists, skipping: $fname"
        continue
    fi
    echo "  downloading: $fname"
    # -c resume, --tries handle flaky CDN
    wget -c --tries=10 --timeout=60 -O "$fname" "$url"
done

echo
echo "Extracting ..."
cd "$NUSCENES_ROOT"
for f in _tarballs/*.tgz; do
    [[ -f "$f" ]] || continue
    echo "  tar xzf $f"
    tar xzf "$f" -C "$NUSCENES_ROOT"
done
for f in _tarballs/*.zip; do
    [[ -f "$f" ]] || continue
    echo "  unzip $f"
    unzip -q -o "$f" -d "$NUSCENES_ROOT"
done

echo
echo "Done. Verify layout:"
echo "  $NUSCENES_ROOT/v1.0-trainval/   (json metadata)"
echo "  $NUSCENES_ROOT/samples/LIDAR_TOP/"
echo "  $NUSCENES_ROOT/sweeps/LIDAR_TOP/"
echo
echo "Tarballs left in $NUSCENES_ROOT/_tarballs (rm to save ~480 GB)."
