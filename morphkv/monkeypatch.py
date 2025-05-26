import transformers

from morphkv.models.patch_mistral import MistralAttentionMorph, MistralFlashAttention2Morph, mistral_model_forward
from morphkv.morph_cache import MorphOffloadedCache
from morphkv.gen_utils import morph_sample

import sys
sys.path.append("/home/rhg659/MorphKV")

def patch_mistral():
    # Patch the individual class references
    transformers.models.mistral.modeling_mistral.MistralAttention = MistralAttentionMorph
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2 = MistralFlashAttention2Morph
    transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward
    transformers.cache_utils.DynamicCache = MorphOffloadedCache
    transformers.generation.utils.GenerationMixin._sample = morph_sample
    
    # CRITICAL: Also patch the MISTRAL_ATTENTION_CLASSES dictionary
    # This is what actually gets used during model initialization
    transformers.models.mistral.modeling_mistral.MISTRAL_ATTENTION_CLASSES = {
        "eager": MistralAttentionMorph,
        "flash_attention_2": MistralFlashAttention2Morph,
        "sdpa": MistralAttentionMorph,  # You can use your custom class for SDPA too
    }
    
    # CRITICAL: Patch the cache mapping for generation
    # Add dynamic cache to the mapping so it uses MorphOffloadedCache
    from transformers.generation.configuration_utils import NEED_SETUP_CACHE_CLASSES_MAPPING, NEEDS_CACHE_CONFIG
    NEED_SETUP_CACHE_CLASSES_MAPPING["dynamic"] = MorphOffloadedCache
    
    # CRITICAL: Update ALL_CACHE_IMPLEMENTATIONS to include "dynamic"
    # This is needed for validation during generation
    import transformers.generation.configuration_utils as config_utils
    config_utils.ALL_CACHE_IMPLEMENTATIONS = list(NEED_SETUP_CACHE_CLASSES_MAPPING.keys()) + list(NEEDS_CACHE_CONFIG.keys())
    
    # CRITICAL: Patch the _get_cache method to always return MorphOffloadedCache
    def patched_get_cache(self, cache_implementation: str, batch_size: int, max_cache_len: int, device, model_kwargs):
        """Always return MorphOffloadedCache regardless of cache_implementation"""
        return MorphOffloadedCache()
    
    # Apply the patch
    transformers.generation.utils.GenerationMixin._get_cache = patched_get_cache
    
    # Also patch the DynamicCache import in generation utils
    transformers.generation.utils.DynamicCache = MorphOffloadedCache
    
    # Verify the patching worked
    print("MistralAttention patched to:", transformers.models.mistral.modeling_mistral.MistralAttention.__name__)
    print("MistralFlashAttention2 patched to:", transformers.models.mistral.modeling_mistral.MistralFlashAttention2.__name__)
    print("DynamicCache patched to:", transformers.cache_utils.DynamicCache.__name__)
    print("GenerationMixin patched to:", transformers.generation.utils.GenerationMixin.__name__)
    print("_get_cache method patched to always return MorphOffloadedCache")
    print("MISTRAL_ATTENTION_CLASSES updated:")
    for key, cls in transformers.models.mistral.modeling_mistral.MISTRAL_ATTENTION_CLASSES.items():
        print(f"  {key}: {cls.__name__}")
    print("NEED_SETUP_CACHE_CLASSES_MAPPING updated:")
    for key, cls in NEED_SETUP_CACHE_CLASSES_MAPPING.items():
        print(f"  {key}: {cls.__name__}")
    print("ALL_CACHE_IMPLEMENTATIONS updated:", config_utils.ALL_CACHE_IMPLEMENTATIONS)