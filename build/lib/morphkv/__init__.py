"""
MorphKV: A library for efficient key-value cache management in transformer models.
"""

__version__ = "0.1.0"

# Import main components
from .morph_cache import DynamicCache, MorphOffloadedCache
from .monkeypatch import patch_mistral
from .gen_utils import morph_sample

# Import model patches
from .models.patch_mistral import (
    MistralAttentionMorph, 
    MistralFlashAttention2Morph, 
    mistral_model_forward
)

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
