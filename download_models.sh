#!/bin/bash

MODELS=(
    "Helsinki-NLP/opus-mt-en-it:helsinki-nlp"
    "google/madlad400-3b-mt:madlad-google"
    "facebook/nllb-200-distilled-1.3B:nllb-200"
)

for entry in "${MODELS[@]}"; do
    REPO="${entry%%:*}"
    DIR="${entry#*:}"
    
    echo "Downloading model: $REPO to $DIR..."
    if hf download "$REPO" --local-dir "$DIR"; then
        echo "Downloaded $REPO."
    else
        echo "ERROR with $REPO." >&2
        exit 1
    fi
done