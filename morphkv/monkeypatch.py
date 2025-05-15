# import transformers

# from morphkv.models.patch_mistral import MistralAttentionMorph, MistralFlashAttention2Morph
# from morphkv.morph_cache import OffloadedCache
# from morphkv.gen_utils import GenerationMixin

# def patch_mistral():
#     transformers.models.mistral.modeling_mistral.MistralAttention = MistralAttentionMorph
#     transformers.models.mistral.modeling_mistral.MistralFlashAttention2 = MistralFlashAttention2Morph
#     from transformers import AutoModelForCausalLM, AutoConfig
#     config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",cache_dir = "/home/shared/model_chkpts/")
#     config._attn_implementation = "flash_attention_2"
#     model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",config=config,cache_dir = "/home/shared/model_chkpts/")
#     import pdb; pdb.set_trace()
#     # transformers.cache_utils.DynamicCache = OffloadedCache
#     # transformers.generation.utils.GenerationMixin = GenerationMixin

import transformers
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig

import sys
sys.path.append("/home/rhg659/MorphKV")

from morphkv.models.patch_mistral import MistralAttentionMorph, MistralFlashAttention2Morph
from morphkv.morph_cache import OffloadedCache
from morphkv.gen_utils import GenerationMixin

def patch_mistral():
    # Patch the classes BEFORE model initialization
    transformers.models.mistral.modeling_mistral.MistralAttention = MistralAttentionMorph
    # transformers.models.mistral.modeling_mistral.MistralFlashAttention2 = MistralFlashAttention2Morph
    
    # Also patch the modeling_mistral module itself to ensure imports work correctly
    # transformers.models.mistral.modeling_mistral.MistralModel._keys_to_ignore_on_load_unexpected = []
    
    # Now load the model with the patched classes
    # from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        cache_dir="/home/shared/model_chkpts/"
    )
    config._attn_implementation = "eager" #"flash_attention_2"

    config.morphkv = "Yes"
    
    # Force a reload of modules if necessary
    # import importlib
    # importlib.reload(transformers.models.mistral.modeling_mistral)
    
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        config=config,
        cache_dir="/home/shared/model_chkpts/"
    )
    
    # # Verify all layers have the patched attention
    # for i, layer in enumerate(model.model.layers):
    #     assert isinstance(layer.self_attn, MistralFlashAttention2Morph), f"Layer {i} has wrong attention: {layer.self_attn.__class__.__name__}"
    import pdb; pdb.set_trace()
    
    return model