#!/bin/bash
set -euo pipefail

WORKDIR="${HOME}/ml-benchmark"

echo "[1/4] Install system packages"
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv unzip

echo "[2/4] Upgrade pip and install Python dependencies"
python3 -m pip install --upgrade pip
python3 -m pip install lightgbm scikit-learn pandas numpy kaggle

echo "[3/4] Prepare working directory"
mkdir -p "${WORKDIR}"

echo "[4/4] Done"
echo "Working directory: ${WORKDIR}"
echo "Next: place Kaggle key at ~/.kaggle/kaggle.json then run:"
echo "  kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ${WORKDIR}"
echo "  python3 ${WORKDIR}/benchmark.py --data ${WORKDIR}/creditcard.csv --output ${WORKDIR}/benchmark_result.json"
