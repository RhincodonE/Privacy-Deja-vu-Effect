#!/usr/bin/env python3
"""
Generate higher-level superclasses for Tiny-ImageNet-200 using WordNet hypernyms.

Assumes this script is placed directly in the tiny-imagenet-200 folder.
"""

import os
import json
from collections import defaultdict

import nltk
from nltk.corpus import wordnet as wn
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────────

# Base directory is wherever this script resides
BASE_DIR    = os.path.dirname(os.path.realpath(__file__))

WNIDS_FILE  = os.path.join(BASE_DIR, 'wnids.txt')
WORDS_FILE  = os.path.join(BASE_DIR, 'words.txt')

# How many steps up the hypernym tree to go. 
# Larger = coarser (fewer, broader superclasses)
DEPTH       = 6

# Minimum number of fine-classes per superclass to keep it
MIN_MEMBERS = 3

# Output mapping file
OUT_FILE    = os.path.join(BASE_DIR, 'superclasses.json')


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_wnids(path):
    """Read wnids.txt (one WNID per line)."""
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_words(path):
    """
    Read words.txt, which is formatted like:
      n01443537 tench, Tinca tinca
    Returns dict { wnid: human_readable_name }
    """
    d = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                wnid, name = parts
                d[wnid] = name
    return d

def wnid_to_synset(wnid):
    """
    Convert a Tiny-ImageNet WNID (e.g. 'n01443537') to a NLTK WordNet synset.
    The offset is the integer part after dropping the leading 'n'.
    """
    offset = int(wnid[1:])
    return wn.synset_from_pos_and_offset('n', offset)

def get_superclass(syn, depth):
    """
    Given a synset, follow its longest hypernym_paths() up 'depth' steps
    (counting from the leaf). Returns the lemma name of that hypernym.
    """
    paths = syn.hypernym_paths()
    if not paths:
        return syn.name().split('.')[0]
    path = max(paths, key=len)
    idx = max(0, len(path) - depth)
    sup_syn = path[idx]
    return sup_syn.name().split('.')[0]

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # ensure WordNet is downloaded
    nltk.download('wordnet', quiet=True)
    
    print(f"Using BASE_DIR = {BASE_DIR}")
    print("Loading WNIDs and words…")
    wnids = load_wnids(WNIDS_FILE)
    words = load_words(WORDS_FILE)
    
    superclass_map = defaultdict(list)
    print(f"Mapping {len(wnids)} classes to depth-{DEPTH} hypernyms…")
    for wnid in tqdm(wnids):
        try:
            syn = wnid_to_synset(wnid)
        except Exception:
            # skip if no mapping
            continue
        
        sup = get_superclass(syn, DEPTH)
        superclass_map[sup].append(wnid)
    
    # filter tiny groups
    filtered = {
        sup: ids 
        for sup, ids in superclass_map.items() 
        if len(ids) >= MIN_MEMBERS
    }
    
    print(f"Kept {len(filtered)} superclasses with ≥ {MIN_MEMBERS} members each.")
    for sup, ids in list(filtered.items())[:5]:
        print(f"  {sup:20s}: {len(ids)} classes → {ids[:3]} …")
    
    # write out JSON
    with open(OUT_FILE, 'w') as f:
        json.dump(filtered, f, indent=2)
    print(f"\nWrote mapping to {OUT_FILE}")

if __name__ == '__main__':
    main()

