#!/bin/bash
set -e

URL_BASE="https://huggingface.co/datasets/nicklashansen/tdmpc2/resolve/main/mt80/chunk"

for i in $(seq 0 19); do
    FILE="chunk_${i}.pt"
    URL="${URL_BASE}_${i}.pt"
    if [ -f "${FILE}" ]; then
        echo "[${i+1}/20] Skipping ${FILE} (already present)"
        continue
    fi
    echo "[${i+1}/20] Downloading ${FILE} from ${URL}"
    wget -O "${FILE}" "${URL}"
    echo "[${i+1}/20] Finished ${FILE}"
done
