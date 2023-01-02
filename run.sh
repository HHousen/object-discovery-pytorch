#!/bin/bash
set -e
set -x

poetry install

chmod +x download_clevr.sh
./download_clevr.sh /tmp/CLEVR

python -m slot_attention.train
