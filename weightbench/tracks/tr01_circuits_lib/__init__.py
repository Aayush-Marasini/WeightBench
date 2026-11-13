"""Helpers for TR-01 Embeddings-Stripped Circuit Discovery (TR01-ESC-*).

Modules (import lazily; they pull in torch):
    model     character-level pre-RMSNorm RoPE Transformer, numpy export, symmetry transforms
    data      corpus download/build (Python stdlib, English, German, Romaji)
    train     deterministic training + shared on-disk cache
    analysis  activation-based ground truth (private) and static weight-only methods
"""
