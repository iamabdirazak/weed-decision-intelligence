#!/usr/bin/env bash
# Downloads CropAndWeed with its official tools and maps it to the binary
# CropOrWeed2 variant. Needs about 17 GB of free disk. Run from the repository.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p data
if [ ! -d data/tools ]; then
  git clone --depth 1 https://github.com/cropandweed/cropandweed-dataset data/tools
fi
pip install -q -r data/tools/cnw/requirements.txt
(
  cd data/tools/cnw
  python setup.py --data_root ../../cropandweed --no_mapping
  python map_dataset.py --bboxes ../../cropandweed/bboxes --labelids ../../cropandweed/labelIds \
                        --dataset_target CropOrWeed2
)
echo "Done. Next: python -m src.preparation.plants"
