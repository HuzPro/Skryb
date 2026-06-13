"""Clean-room, NeMo-free cache-aware streaming Parakeet (FastConformer + RNN-T).

Runs ``nvidia/nemotron-speech-streaming-en-0.6b`` on PyTorch (incl. ROCm via
the ``cuda`` alias) by reusing nano_parakeet's FastConformer building blocks
and swapping in the streaming model's differences: LayerNorm causal conv,
causal dw-striding subsampling, and an RNN-T joint (vocab 1024).
"""
