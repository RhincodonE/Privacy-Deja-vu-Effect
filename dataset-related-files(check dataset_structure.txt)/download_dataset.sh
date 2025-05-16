#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Download and prepare Tiny-ImageNet-200 and IMDb in the desired structure
# -----------------------------------------------------------------------------

# Base directory for datasets
DATA_ROOT="$(pwd)/datasets"
mkdir -p "$DATA_ROOT"

## 1. Tiny-ImageNet-200 -------------------------------------------------------

TINYNET_DIR="$DATA_ROOT/Tiny-ImageNet-200"
if [ ! -d "$TINYNET_DIR" ]; then
  echo "Downloading Tiny-ImageNet-200..."
  wget -q http://cs231n.stanford.edu/tiny-imagenet-200.zip -O /tmp/tiny-imagenet-200.zip

  echo "Extracting..."
  unzip -q /tmp/tiny-imagenet-200.zip -d "$DATA_ROOT"
  mv "$DATA_ROOT/tiny-imagenet-200" "$TINYNET_DIR"
  rm /tmp/tiny-imagenet-200.zip

  echo "Reorganizing validation set..."
  pushd "$TINYNET_DIR/val" > /dev/null
  # Each line: <img> <wnid> ...
  while read IMG WNID _; do
    mkdir -p images/"$WNID"
    mv images/"$IMG" images/"$WNID"/"$IMG"
  done < val_annotations.txt
  popd > /dev/null

  echo "Copying helper files..."
  # Assumes these live next to this script
  cp build-superclass.py superclasses.json wnids.txt words.txt "$TINYNET_DIR/"
fi

## 2. IMDb -------------------------------------------------------------------

IMDB_DIR="$DATA_ROOT/IMDb"
if [ ! -d "$IMDB_DIR" ]; then
  echo "Downloading IMDb sentiment data..."
  wget -q http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz -O /tmp/aclImdb_v1.tar.gz

  echo "Extracting..."
  tar -xzf /tmp/aclImdb_v1.tar.gz -C "$DATA_ROOT"
  mv "$DATA_ROOT/aclImdb" "$IMDB_DIR"
  rm /tmp/aclImdb_v1.tar.gz

  echo "Renaming test → val and cleaning up..."
  mv "$IMDB_DIR/test" "$IMDB_DIR/val"
  # remove unsup folder if present
  rm -rf "$IMDB_DIR/train/unsup"
fi

echo "All done. Datasets are under:"
echo "  $TINYNET_DIR"
echo "  $IMDB_DIR"
