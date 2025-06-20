from functools import partial
from typing import Callable, Optional, Tuple, Union

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

import transformers
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from morphkv.morph_cache import MorphOffloadedCache

# Try to import flash attention, but handle broken installations gracefully
_flash_attention_forward = None
if is_flash_attn_2_available():
    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
    except ImportError:
        # Flash attention is available but broken, disable it
        _flash_attention_forward = None

logger = logging.get_logger(__name__)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LlamaAttentionMorph(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        self.garbage = [True]*config.num_hidden_layers
        self.morph_type = ""
        self.WIN_SIZE = 1000000000
        self.MAX_CAPACITY = 1000000000

        if config.morphkv:
            self.WIN_SIZE = int(config.morphkv['window_size'])
            self.MAX_CAPACITY = int(config.morphkv['max_capacity'])
            self.morph_type = config.morphkv['morph_type'] 
            self.evict_after = config.morphkv['evict_after'] #for bursty eviction during generation, we evict only after cache is > max_capacity * evict_after (say, after every 10 tokens)
    

    def morphkv_mask(self, scores, past_key_value, key_heads, query_heads):
        # if self.layer_idx==0:
        #     import pdb; pdb.set_trace()
        if past_key_value.attn_cache[self.layer_idx]!=[]:
            if "h2o" in self.morph_type:
                soft_scores = nn.functional.softmax(scores,dim=-1)
                soft_scores[:,:,:,:past_key_value.attn_cache[self.layer_idx].shape[-1]] += past_key_value.attn_cache[self.layer_idx]
                past_key_value.attn_cache[self.layer_idx] = soft_scores
            else:
                scores = torch.cat([past_key_value.attn_cache[self.layer_idx],scores],dim=2) 
                #TODO: Unique head implementation next 1 line
                if self.morph_type=='indp': scores[:,:,-1,-1] = -torch.inf
        
        # we need to maintain two masks - one for KV and one for Attn weight caching
        if(key_heads!=query_heads):
            #TODO: For GQA, we reduce scores by summing over grouped heads
            if self.morph_type=='indp': 
                sim_tokens = torch.full_like(scores[:,:key_heads,-(self.WIN_SIZE+1):-1,:], -torch.inf) #work with last WIN_SIZE+1 tokens, exclude the current token    
                init_mask_kv = torch.where(torch.any(torch.eq(sim_tokens.scatter_(-1,torch.topk(scores.view(scores.shape[0],key_heads,-1,scores.shape[2],scores.shape[3]).sum(dim=2)[:, :, -(self.WIN_SIZE+1):-1], dim=-1, k=self.SIM_THRESH).indices,0.0),0),dim=2,keepdim=True),0.0,-torch.inf)
            elif "max" in self.morph_type or self.morph_type=='max_fused': 
                sim_tokens = torch.full_like(scores[:,:key_heads,-2:-1,:], -torch.inf) #work with last 1 tokens as we will fuse all window tokens into 1, exclude the current token
                init_mask_kv = sim_tokens[:,:,-1:].scatter_(-1,torch.topk(nn.functional.softmax(scores.view(scores.shape[0],key_heads,-1,scores.shape[2],scores.shape[3]).sum(dim=2)[:, :, -(self.WIN_SIZE+1):-1, :-(self.WIN_SIZE+1)],dim=-1).max(dim=2, keepdim=True)[0], dim=-1, k=self.MAX_CAPACITY-self.WIN_SIZE).indices,0.0)
            elif "sum" in self.morph_type or self.morph_type=='sum_fused': 
                sim_tokens = torch.full_like(scores[:,:key_heads,-2:-1,:], -torch.inf) #work with last 1 tokens as we will fuse all window tokens into 1, exclude the current token
                init_mask_kv = sim_tokens[:,:,-1:].scatter_(-1,torch.topk(nn.functional.softmax(scores.view(scores.shape[0],key_heads,-1,scores.shape[2],scores.shape[3]).sum(dim=2)[:, :, -(self.WIN_SIZE+1):-1, :-(self.WIN_SIZE+1)],dim=-1).sum(dim=2, keepdim=True), dim=-1, k=self.MAX_CAPACITY-self.WIN_SIZE).indices,0.0)
            init_mask_kv[:, :, -1, -(self.WIN_SIZE+1):] = 0.0  # attends to all window tokens and itself

        # we need to maintain two masks - one for KV and one for Attn weight caching
        if self.morph_type=='indp': 
            sim_tokens = torch.full_like(scores[:,:,-(self.WIN_SIZE+1):-1,:], -torch.inf) #work with last WIN_SIZE+1 tokens, exclude the current token
            init_mask_attn = torch.where(torch.any(torch.eq(sim_tokens.scatter_(-1,torch.topk(scores[:, :, -(self.WIN_SIZE+1):-1], dim=-1, k=self.SIM_THRESH).indices,0.0),0),dim=2,keepdim=True),0.0,-torch.inf)
        elif "max" in self.morph_type or self.morph_type=='max_fused': 
            sim_tokens = torch.full_like(scores[:,:,-(1+1):-1,:], -torch.inf) #work with last 1 tokens as we will fuse all window tokens into 1, exclude the current token
            init_mask_attn = sim_tokens[:,:,-1:].scatter_(-1,torch.topk(nn.functional.softmax(scores[:, :, -(self.WIN_SIZE+1):-1, :-(self.WIN_SIZE+1)],dim=-1).max(dim=2, keepdim=True)[0], dim=-1, k=self.MAX_CAPACITY-self.WIN_SIZE).indices,0.0)
        elif "sum" in self.morph_type or self.morph_type=='sum_fused': 
            sim_tokens = torch.full_like(scores[:,:,-(1+1):-1,:], -torch.inf) #work with last 1 tokens as we will fuse all window tokens into 1, exclude the current token
            init_mask_attn = sim_tokens[:,:,-1:].scatter_(-1,torch.topk(nn.functional.softmax(scores[:, :, -(self.WIN_SIZE+1):-1, :-(self.WIN_SIZE+1)],dim=-1).sum(dim=2, keepdim=True), dim=-1, k=self.MAX_CAPACITY-self.WIN_SIZE).indices,0.0)
        elif "h2o" in self.morph_type: 
            sim_tokens = torch.full_like(scores[:,:,-1:,:], -torch.inf) #work with last 1 tokens as we will fuse all window tokens into 1, exclude the current token
            init_mask_attn = sim_tokens[:,:,-1:].scatter_(-1,torch.topk(soft_scores[:, :, :, :-(self.WIN_SIZE+1)], dim=-1, k=self.MAX_CAPACITY-self.WIN_SIZE).indices,0.0)
        init_mask_attn[:, :, -1, -(self.WIN_SIZE+1):] = 0.0  # attends to all window tokens and itself

        if(key_heads!=query_heads):
            #TODO: For GQA, we reduce scores by summing over grouped heads
            past_key_value.cleanup(init_mask_kv,init_mask_attn,self.layer_idx) 
        else: past_key_value.cleanup(init_mask_attn,init_mask_attn,self.layer_idx) 
        
        return (init_mask_attn + scores[:,:,-1:,:]), init_mask_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_heads = query_states.shape[1]
        key_heads = key_states.shape[1]

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if "snapkv" in self.morph_type or "h2o" in self.morph_type: # or self.config.hopformer:
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
                query_heads = query_states.shape[1]
                key_heads = key_states.shape[1]
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if key_states.shape[1]!=query_states.shape[1]:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # use MorphKV only in generative phase, ie, when hidden states has 1 token (the newly generated)
        if self.config.morphkv and key_states.shape[2]>= (1 + self.MAX_CAPACITY) * self.evict_after:
            if hidden_states.shape[1]==1:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]] if attention_mask is not None else None
                attn_weights, init_mask = self.morphkv_mask(attn_weights, past_key_value, key_heads, query_heads)
                
                # morphkv call must have emptied KV Cache, so cleanup!
                if self.garbage[self.layer_idx]==True:
                    torch.cuda.empty_cache()
                    past_key_value.cleaned[self.layer_idx] = True
                    self.garbage[self.layer_idx] = False
            # seems like a new sequence, reset garbage variable to true        
            else: self.garbage[self.layer_idx] = True
            
        else:
            past_key_value.cleanup(None,None,self.layer_idx,dummy=True) ## just for the sake of profiling memory
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        if self.config.morphkv:
            #cache attn weights in past key value
            if past_key_value.attn_cache[self.layer_idx]==[]:
                # need to add another column to attn matrix to account for next token
                #TODO: Unique head implementation next 2 lines
                if self.morph_type=='indp': 
                    for i in range(1,min(attn_weights.shape[-1],self.WIN_SIZE)+1):
                        attn_weights[:,:,-i,-i] = -torch.inf
                if "h2o" in self.morph_type:
                    past_key_value.attn_cache[self.layer_idx] = nn.functional.softmax(attn_weights,dim=-1)
                else:
                    print("Error with below implementation, expected to use Flash attn for prefilling!!!")
                    # bs,n_heads,cache_len,cache_len = attn_weights.shape                
                    # past_key_value.attn_cache[self.layer_idx] = torch.cat([attn_weights[:,:,-self.WIN_SIZE:],torch.full((bs,n_heads,min(self.WIN_SIZE,cache_len),1),-torch.inf).to(attn_weights.device)],dim=-1)
            else:
                # import pdb;pdb.set_trace()
                if "h2o" in self.morph_type:
                    pass
                else:
                    if key_states.shape[2]>= (1 + self.MAX_CAPACITY) * self.evict_after: # only then the init mask would have been calculated
                        bs, n_heads, win_size, seq_len = past_key_value.attn_cache[self.layer_idx].shape
                        self.evict_after = self.config.morphkv['evict_after'] # after the first eviction, evict only after 'evict_after' factor times KV size is reached
                        
                        # remove the row corresponding to oldest window, and append the newly generated attn weight profile
                        past_key_value.attn_cache[self.layer_idx] = torch.roll(past_key_value.attn_cache[self.layer_idx],shifts=-1,dims=2)
                        past_key_value.attn_cache[self.layer_idx][:, :, -1:] = attn_weights.transpose(3,2)[init_mask.squeeze(2)==0].view(bs, n_heads, -1, 1).transpose(3,2)
                        
                    else:
                        if key_states.shape[2]<=self.WIN_SIZE: # attn cache has been padded till the max size, hence indexing the current attention row properly
                            past_key_value.attn_cache[self.layer_idx][:,:,key_states.shape[2]-1:key_states.shape[2],:] = attn_weights
                        else:
                            past_key_value.attn_cache[self.layer_idx] = torch.roll(past_key_value.attn_cache[self.layer_idx],shifts=-1,dims=2) #shift up by 1 position on the 2nd dim
                            #TODO: LEADS TO MEMORY CORRUPTION!!! # past_key_value.attn_cache[self.layer_idx][:, :, :(self.WIN_SIZE-1)] = past_key_value.attn_cache[self.layer_idx][:, :, -(self.WIN_SIZE-1):]
                            past_key_value.attn_cache[self.layer_idx][:, :, -1:] = attn_weights
                        bs, n_heads, win_size, seq_len = past_key_value.attn_cache[self.layer_idx].shape
                    past_key_value.attn_cache[self.layer_idx] = torch.cat([past_key_value.attn_cache[self.layer_idx],torch.full((bs,n_heads,win_size,1),-torch.inf).to(past_key_value.attn_cache[self.layer_idx].device)],dim=-1)
            

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2Morph(LlamaAttentionMorph):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )
        
        # import pdb; pdb.set_trace()
        if "snapkv" in self.morph_type:
            init_snapkv(self)
        
        if hidden_states.shape[1]==1 and self.layer_idx>-1:# and False:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel was using LlamaFlashAttention2 for prefilling, which does not support Hopformer KV eviction. Falling back to the eager attention implementation."
            )
            return super(LlamaFlashAttention2Morph,self).forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        
        output_attentions = False
        
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if "snapkv" in self.morph_type: 
                if hidden_states.shape[1]!=1: #SnapKV works only in prefill mode
                    key_states = repeat_kv(key_states, self.num_key_value_groups)
                    value_states = repeat_kv(value_states, self.num_key_value_groups)
                    key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attention_mask, self.num_key_value_groups)
                    past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
                else:
                    key_states = repeat_kv(key_states, self.num_key_value_groups)
                    value_states = repeat_kv(value_states, self.num_key_value_groups)
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                if "h2o" in self.morph_type:
                    key_states = repeat_kv(key_states, self.num_key_value_groups)
                    value_states = repeat_kv(value_states, self.num_key_value_groups)
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # repeat k/v heads if n_kv_heads < n_heads
        if key_states.shape[1]<query_states.shape[1]:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )
        if self.config.morphkv and self.layer_idx>-1:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            #key_states = repeat_kv(key_states, self.num_key_value_groups)
            if "h2o" in self.morph_type:
                attn_weights = nn.functional.softmax(torch.matmul(query_states[:,:,-self.MAX_CAPACITY:,:], key_states.transpose(2, 3)) / math.sqrt(self.head_dim),dim=-1)
                attn_weights = torch.tril(attn_weights,diagonal=max(attn_weights.shape[-1]-(self.MAX_CAPACITY),0))
                attn_weights = attn_weights.sum(dim=2, keepdim=True)
            else:    
                attn_weights = torch.matmul(query_states[:,:,-self.WIN_SIZE:,:], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # use hopformer only in generative phase, ie, when hidden states has 1 token (the newly generated)
        if self.config.morphkv and self.layer_idx>-1 and key_states.shape[2]>= 1 + self.MAX_CAPACITY:
            # import pdb; pdb.set_trace()
            if hidden_states.shape[1]==1:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights, init_mask = self.morphkv_mask(attn_weights, causal_mask, past_key_value, key_heads, query_heads)#>0)
                # hopformer call must have emptied KV Cache, so cleanup!
                if self.garbage[self.layer_idx]==True:
                    torch.cuda.empty_cache()
                    # gc.collect()
                    past_key_value.cleaned[self.layer_idx] = True
                    self.garbage[self.layer_idx] = False
            # seems like a new sequence, reset garbage variable to true        
            else: self.garbage[self.layer_idx] = True
        else:
            past_key_value.cleanup(None,None,self.layer_idx,dummy=True) ## just for the sake of profiling memory

        if self.config.morphkv and self.layer_idx>-1:# and key_states.shape[2]>=(self.WIN_SIZE*self.SIM_THRESH):
            #cache attn weights in past key value
            if past_key_value.attn_cache[self.layer_idx]==[]:
                # need to add another column to attn matrix to account for next token
                #TODO: Unique head implementation next 2 lines
                if self.morph_type=='indp': 
                    for i in range(1,min(attn_weights.shape[-1],self.WIN_SIZE)+1):
                        attn_weights[:,:,-i,-i] = -torch.inf
                if "h2o" in self.morph_type:
                    bs,n_heads,cache_len,cache_len = attn_weights.shape                
                    past_key_value.attn_cache[self.layer_idx] = attn_weights#torch.cat([nn.functional.softmax(attn_weights[:,:,-1:,:],dim=-1),torch.full((bs,n_heads,1,1),0).to(attn_weights.device)],dim=-1)
                else:
                    bs,n_heads,cache_len,cache_len = attn_weights.shape                
                    if cache_len<self.WIN_SIZE:
                        attn_weights = torch.cat([attn_weights,torch.full((bs,n_heads,self.WIN_SIZE-cache_len,cache_len),-torch.inf).to(attn_weights.device)],dim=2)
                    past_key_value.attn_cache[self.layer_idx] = torch.cat([attn_weights[:,:,-self.WIN_SIZE:,:],torch.full((bs,n_heads,self.WIN_SIZE,1),-torch.inf).to(attn_weights.device)],dim=-1)
            else:
                raise NotImplementedError("Attn cache update not expected for Flash Attention")
                # # import pdb;pdb.set_trace()
                # bs, n_heads, win_size, seq_len = past_key_value.attn_cache[self.layer_idx].shape
                # if key_states.shape[2]>= 1 + self.MAX_CAPACITY: # only then the init mask would have been calculated
                #     past_key_value.attn_cache[self.layer_idx] = torch.cat([past_key_value.attn_cache[self.layer_idx][:,:,1:], attn_weights.transpose(3,2)[init_mask.squeeze(2)==0].view(bs, n_heads, -1, 1).transpose(3,2)],dim=2)
                # else:
                #     past_key_value.attn_cache[self.layer_idx] = torch.cat([past_key_value.attn_cache[self.layer_idx][:,:,1:], attn_weights],dim=2)
                # # need to add another column to attn matrix to account for next token
                # # bs,n_heads,win_size,seq_len = past_key_value.attn_cache[self.layer_idx].shape
                # past_key_value.attn_cache[self.layer_idx] = torch.cat([past_key_value.attn_cache[self.layer_idx],torch.full((bs,n_heads,win_size,1),-torch.inf).to(past_key_value.attn_cache[self.layer_idx].device)],dim=-1)
            # print(f".........Layer: {self.layer_idx} attn cached........")

        # # upcast attention to fp32
        # if self.SOFTMAX_TYPE=='gumbel':
        #     attn_weights = nn.functional.gumbel_softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # else:
        #     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value

def llama_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # kept for BC (non `Cache` `past_key_values` inputs)
    return_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache):
        return_legacy_cache = True
        if past_key_values is None:
            past_key_values = MorphOffloadedCache(self.config.num_hidden_layers)
        else:
            past_key_values = MorphOffloadedCache.from_legacy_cache(past_key_values,self.config.num_hidden_layers)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
            )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )