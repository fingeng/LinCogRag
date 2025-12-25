#!/bin/bash

echo "Downloading biomedical-ner-all model manually..."

# Create model directory
mkdir -p models/biomedical-ner-all

cd models/biomedical-ner-all

# Download files using wget (works better in China)
echo "Downloading config.json..."
wget https://hf-mirror.com/d4data/biomedical-ner-all/resolve/main/config.json

echo "Downloading pytorch_model.bin..."
wget https://hf-mirror.com/d4data/biomedical-ner-all/resolve/main/pytorch_model.bin

echo "Downloading tokenizer files..."
wget https://hf-mirror.com/d4data/biomedical-ner-all/resolve/main/tokenizer_config.json
wget https://hf-mirror.com/d4data/biomedical-ner-all/resolve/main/vocab.txt
wget https://hf-mirror.com/d4data/biomedical-ner-all/resolve/main/special_tokens_map.json
wget https://hf-mirror.com/d4data/biomedical-ner-all/resolve/main/tokenizer.json

echo "✅ Model downloaded to models/biomedical-ner-all"
