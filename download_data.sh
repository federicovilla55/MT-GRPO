#!/bin/bash

WMT_URL="https://github.com/wmt-conference/wmt25-general-mt/raw/refs/heads/main/data/wmt25-genmt-humeval.jsonl"
TATOEBA_URL="https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2"
DATA_DIR="data"
WMT_FILE_PATH="$DATA_DIR/wmt25-genmt-humeval.jsonl"
TATOEBA_FILE_PATH="$DATA_DIR/eng_sentences.tsv"

mkdir -p "$DATA_DIR"

if curl -L "$WMT_URL" -o "$WMT_FILE_PATH"; then
    echo "Successfully downloaded WMT data."
else
    echo "ERROR: Failed to download WMT data."
    exit 1
fi

if curl -L "$TATOEBA_URL" | bzip2 -d > "$TATOEBA_FILE_PATH"; then
    echo "Successfully downloaded and decompressed Tatoeba data."
else
    echo "ERROR: Failed to download or decompress Tatoeba data."
    rm -f "$TATOEBA_FILE_PATH"
    exit 1
fi

echo "Result:"
ls -lh "$DATA_DIR"