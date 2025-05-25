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
    print("✓ MistralAttention patched to:", transformers.models.mistral.modeling_mistral.MistralAttention.__name__)
    print("✓ MistralFlashAttention2 patched to:", transformers.models.mistral.modeling_mistral.MistralFlashAttention2.__name__)
    print("✓ DynamicCache patched to:", transformers.cache_utils.DynamicCache.__name__)
    print("✓ GenerationMixin patched to:", transformers.generation.utils.GenerationMixin.__name__)
    print("✓ _get_cache method patched to always return MorphOffloadedCache")
    print("✓ MISTRAL_ATTENTION_CLASSES updated:")
    for key, cls in transformers.models.mistral.modeling_mistral.MISTRAL_ATTENTION_CLASSES.items():
        print(f"  {key}: {cls.__name__}")
    print("✓ NEED_SETUP_CACHE_CLASSES_MAPPING updated:")
    for key, cls in NEED_SETUP_CACHE_CLASSES_MAPPING.items():
        print(f"  {key}: {cls.__name__}")
    print("✓ ALL_CACHE_IMPLEMENTATIONS updated:", config_utils.ALL_CACHE_IMPLEMENTATIONS)

# import transformers
# from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig
# from transformers.models.mistral.modeling_mistral import MistralForCausalLM



# from morphkv.models.patch_mistral import MistralAttentionMorph, MistralFlashAttention2Morph
# from morphkv.morph_cache import OffloadedCache
# from morphkv.gen_utils import GenerationMixin

# def patch_mistral():
#     # Patch the classes BEFORE model initialization
#     transformers.models.llama.modeling_llama.LlamaAttention = MistralAttentionMorph
#     transformers.models.llama.modeling_llama.LlamaFlashAttention2 = MistralFlashAttention2Morph
    
#     # Also patch the modeling_mistral module itself to ensure imports work correctly
#     # transformers.models.mistral.modeling_mistral.MistralModel._keys_to_ignore_on_load_unexpected = []
    
#     # Now load the model with the patched classes
#     # from transformers import AutoModelForCausalLM, AutoConfig
#     model_path = "meta-llama/Llama-3.1-8B-Instruct"
#     config = AutoConfig.from_pretrained(
#         model_path,
#         cache_dir="/home/shared/model_chkpts/"
#     )
#     config._attn_implementation = "eager" #"flash_attention_2"

#     config.morphkv = "Yes"
    
#     # Force a reload of modules if necessary
#     # import importlib
#     # importlib.reload(transformers.models.mistral.modeling_mistral)
    
#     # model = AutoModelForCausalLM.from_pretrained(
#     #     "mistralai/Mistral-7B-Instruct-v0.2",
#     #     config=config,
#     #     cache_dir="/home/shared/model_chkpts/"
#     # )
#     model = MistralForCausalLM.from_pretrained(
#         model_path,
#         config=config,
#         cache_dir="/home/shared/model_chkpts/"
#     )
    
#     # # Verify all layers have the patched attention
#     # for i, layer in enumerate(model.model.layers):
#     #     assert isinstance(layer.self_attn, MistralFlashAttention2Morph), f"Layer {i} has wrong attention: {layer.self_attn.__class__.__name__}"
#     import pdb; pdb.set_trace()
    
#     return model