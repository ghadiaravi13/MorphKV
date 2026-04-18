"""
MorphKV: A library for efficient key-value cache management in transformer models.
"""

__version__ = "0.1.0"

# Import main components
from .morph_cache import DynamicCache, OffloadedCache as MorphOffloadedCache
from .monkeypatch import patch_mistral
from .gen_utils import GenerationMixin as GenerationMixinMorph

# Import model patches
from .models.patch_mistral import (
    MistralAttention as MistralAttentionMorph, 
    MistralFlashAttention2 as MistralFlashAttention2Morph, 
    MistralModel as MistralModelMorph,
)

mistral_model_forward = MistralModelMorph.forward
morph_sample = GenerationMixinMorph._sample

# Define what gets imported with "from morphkv import *"
__all__ = [
    "DynamicCache",
    "MorphOffloadedCache", 
    "patch_mistral",
    "morph_sample",
    "MistralAttentionMorph",
    "MistralFlashAttention2Morph",
    "mistral_model_forward",
]
