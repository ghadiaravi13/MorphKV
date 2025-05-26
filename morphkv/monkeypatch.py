import transformers

from transformers.generation.configuration_utils import NEED_SETUP_CACHE_CLASSES_MAPPING, NEEDS_CACHE_CONFIG
import transformers.generation.configuration_utils as config_utils

from morphkv.models.patch_mistral import MistralAttentionMorph, MistralFlashAttention2Morph, mistral_model_forward
from morphkv.models.patch_llama import LlamaAttentionMorph, LlamaFlashAttention2Morph, llama_model_forward
from morphkv.models.patch_qwen import Qwen2AttentionMorph, Qwen2FlashAttention2Morph, qwen2_model_forward
from morphkv.models.patch_phi3 import Phi3AttentionMorph, Phi3FlashAttention2Morph, phi3_model_forward

from morphkv.morph_cache import MorphOffloadedCache
from morphkv.gen_utils import morph_sample



def patch_mistral():
    # Patch the individual class references
    transformers.models.mistral.modeling_mistral.MistralAttention = MistralAttentionMorph
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2 = MistralFlashAttention2Morph
    transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward
    # transformers.cache_utils.DynamicCache = MorphOffloadedCache
    # transformers.generation.utils.GenerationMixin._sample = morph_sample
    
    # CRITICAL: Also patch the MISTRAL_ATTENTION_CLASSES dictionary
    # This is what actually gets used during model initialization
    transformers.models.mistral.modeling_mistral.MISTRAL_ATTENTION_CLASSES = {
        "eager": MistralAttentionMorph,
        "flash_attention_2": MistralFlashAttention2Morph,
        "sdpa": MistralAttentionMorph,  # You can use your custom class for SDPA too
    }

def patch_llama():
    # Patch the individual class references
    transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttentionMorph
    transformers.models.llama.modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2Morph
    transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward
    
    transformers.models.llama.modeling_llama.LLAMA_ATTENTION_CLASSES = {
        "eager": LlamaAttentionMorph,
        "flash_attention_2": LlamaFlashAttention2Morph,
        "sdpa": LlamaAttentionMorph,  # You can use your custom class for SDPA too
    }

def patch_qwen2():
    # Patch the individual class references
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention = Qwen2AttentionMorph
    transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2 = Qwen2FlashAttention2Morph
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen2_model_forward
    
    transformers.models.qwen2.modeling_qwen2.QWEN2_ATTENTION_CLASSES = {
        "eager": Qwen2AttentionMorph,
        "flash_attention_2": Qwen2FlashAttention2Morph,
        "sdpa": Qwen2AttentionMorph,  # You can use your custom class for SDPA too
    }

def patch_phi3():
    # Patch the individual class references
    transformers.models.phi3.modeling_phi3.Phi3Attention = Phi3AttentionMorph
    transformers.models.phi3.modeling_phi3.Phi3FlashAttention2 = Phi3FlashAttention2Morph
    transformers.models.phi3.modeling_phi3.Phi3Model.forward = phi3_model_forward

    transformers.models.phi3.modeling_phi3.PHI3_ATTENTION_CLASSES = {
        "eager": Phi3AttentionMorph,
        "flash_attention_2": Phi3FlashAttention2Morph,
        "sdpa": Phi3AttentionMorph,  # You can use your custom class for SDPA too
    }

def patch_cache():

    transformers.cache_utils.DynamicCache = MorphOffloadedCache
    transformers.generation.utils.GenerationMixin._sample = morph_sample
    # CRITICAL: Patch the cache mapping for generation
    # Add dynamic cache to the mapping so it uses MorphOffloadedCache
    NEED_SETUP_CACHE_CLASSES_MAPPING["dynamic"] = MorphOffloadedCache
    
    # CRITICAL: Update ALL_CACHE_IMPLEMENTATIONS to include "dynamic"
    # This is needed for validation during generation
    config_utils.ALL_CACHE_IMPLEMENTATIONS = list(NEED_SETUP_CACHE_CLASSES_MAPPING.keys()) + list(NEEDS_CACHE_CONFIG.keys())
    
    # CRITICAL: Patch the _get_cache method to always return MorphOffloadedCache
    def patched_get_cache(self, cache_implementation: str, batch_size: int, max_cache_len: int, device, model_kwargs):
        """Always return MorphOffloadedCache regardless of cache_implementation"""
        return MorphOffloadedCache()
    
    # Apply the patch
    transformers.generation.utils.GenerationMixin._get_cache = patched_get_cache
    
    # Also patch the DynamicCache import in generation utils
    transformers.generation.utils.DynamicCache = MorphOffloadedCache

def patch_morphkv():

    patch_mistral()
    patch_llama()
    patch_qwen2()
    patch_phi3()
    patch_cache()

    # Verify the patching worked
    print("MistralAttention patched to:", transformers.models.mistral.modeling_mistral.MistralAttention.__name__)
    print("MistralFlashAttention2 patched to:", transformers.models.mistral.modeling_mistral.MistralFlashAttention2.__name__)
    print("LlamaAttention patched to:", transformers.models.llama.modeling_llama.LlamaAttention.__name__)
    print("LlamaFlashAttention2 patched to:", transformers.models.llama.modeling_llama.LlamaFlashAttention2.__name__)
    print("Qwen2Attention patched to:", transformers.models.qwen2.modeling_qwen2.Qwen2Attention.__name__)
    print("Qwen2FlashAttention2 patched to:", transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.__name__)
    print("Phi3Attention patched to:", transformers.models.phi3.modeling_phi3.Phi3Attention.__name__)   
    print("Phi3FlashAttention2 patched to:", transformers.models.phi3.modeling_phi3.Phi3FlashAttention2.__name__)
    
    print("DynamicCache patched to:", transformers.cache_utils.DynamicCache.__name__)
    print("GenerationMixin patched to:", transformers.generation.utils.GenerationMixin.__name__)
    
    
    print("NEED_SETUP_CACHE_CLASSES_MAPPING updated:")
    for key, cls in NEED_SETUP_CACHE_CLASSES_MAPPING.items():
        print(f"  {key}: {cls.__name__}")
    print("ALL_CACHE_IMPLEMENTATIONS updated:", config_utils.ALL_CACHE_IMPLEMENTATIONS)