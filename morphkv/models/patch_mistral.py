from functools import partial
from typing import Callable, Optional, Tuple, Union

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

import transformers
from transformers.cache_utils import Cache, StaticCache

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
from transformers.models.mistral.configuration_mistral import MistralConfig
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

class MistralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    # copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward
    # TODO(joao): add me back asap :)
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
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


class MistralAttentionMorph(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
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
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
        self.garbage = [True]*config.num_hidden_layers
        self.morph_type = ""
        self.WIN_SIZE = 1000000000
        self.MAX_CAPACITY = 1000000000

        if config.morphkv:
            self.WIN_SIZE = int(config.morphkv['window_size'])
            self.MAX_CAPACITY = int(config.morphkv['max_capacity'])
            self.morph_type = config.morphkv['morph_type'] 
            self.evict_after = config.morphkv['evict_after'] #for bursty eviction during generation, we evict only after cache is > max_capacity * evict_after (say, after every 10 tokens)
            self.window_queries = [None]*self.config.num_hidden_layers
            self.fuse_temperature = float(config.morphkv.get('fuse_temperature', 1.0))
            self.prefill_flag = True
            self.use_attn_offsets = config.morphkv.get('use_attn_offsets', False)
            self.imp_budget = config.morphkv.get('imp_budget', 0.5)
            self.pre_rope = config.morphkv.get('pre_rope', False)
            self.score_percentile = float(config.morphkv.get('score_percentile', 0.9))
    

    def morphkv_mask(self, scores, past_key_value, key_heads, query_heads):
        
        #softmax_scores = nn.functional.softmax(scores[:, :, -(self.WIN_SIZE+1):-1, :-(self.WIN_SIZE+1)],dim=-1)
        start_idx = 0
        # if not past_key_value.fusion_done:
        #     start_idx = 1 # if fusion is done, we need to attend to the fused token

        past_val_norms = past_key_value.value_cache[self.layer_idx][:, :, start_idx:-(self.WIN_SIZE+1), :].norm(dim=-1)
        past_val_norms = past_val_norms.unsqueeze(2)

        if(key_heads!=query_heads):
            #For GQA, we reduce scores by summing over grouped heads -> changed to taking max over grouped heads
            if "max" in self.morph_type or self.morph_type=='max_fused': 
                sim_tokens = torch.full_like(scores[:,:key_heads,-2:-1,:], -torch.inf) #work with last 1 tokens as we will fuse all window tokens into 1, exclude the current token
                init_mask_kv = sim_tokens[:,:,-1:].scatter_(-1,torch.topk(past_val_norms * nn.functional.softmax(scores.view(scores.shape[0],key_heads,-1,scores.shape[2],scores.shape[3]).sum(dim=2)[:, :, -(self.WIN_SIZE+1):-1, start_idx:-(self.WIN_SIZE+1)],dim=-1).max(dim=2, keepdim=True)[0], dim=-1, k=self.MAX_CAPACITY-self.WIN_SIZE-start_idx).indices+start_idx,0.0)
            elif "sum" in self.morph_type or self.morph_type=='sum_fused': 
                sim_tokens = torch.full_like(scores[:,:key_heads,-2:-1,:], -torch.inf) #work with last 1 tokens as we will fuse all window tokens into 1, exclude the current token
                init_mask_kv = sim_tokens[:,:,-1:].scatter_(-1,torch.topk(past_val_norms * nn.functional.softmax(scores.view(scores.shape[0],key_heads,-1,scores.shape[2],scores.shape[3]).sum(dim=2)[:, :, -(self.WIN_SIZE+1):-1, start_idx:-(self.WIN_SIZE+1)],dim=-1).sum(dim=2, keepdim=True), dim=-1, k=self.MAX_CAPACITY-self.WIN_SIZE-start_idx).indices+start_idx,0.0)
            
            init_mask_kv[:, :, -1, -(self.WIN_SIZE+1):] = 0.0  # attends to all window tokens and itself
            
            if not past_key_value.fusion_done:
                init_mask_kv[:, :, -1, :start_idx] = 0.0 # if fusion is done, we need to attend to the fused token
            

        # attn mask is deprecated, use None for now
        init_mask_attn = None
        
        if(key_heads!=query_heads):
            #For GQA, we have seperate masks for attention and KVs
            past_key_value.cleanup(init_mask_kv,init_mask_attn,self.layer_idx) 
        else: 
            raise ValueError("MHA not supported yet: key_heads should not be equal to query_heads")
            past_key_value.cleanup(init_mask_attn,init_mask_attn,self.layer_idx)
        
        # absolutely no reason to mask the current scores, let the first decoded token attend to full KV cache
        # return (init_mask_attn + scores[:,:,-1:,:]), init_mask_attn
        return scores[:,:,-1:,:], init_mask_attn

    def morphkv_plus_fuse_unimportant(self, scores, past_key_value, key_heads, query_heads):
        """KV cache compression that preserves important tokens and only fuses the rest.

        The top ``MAX_CAPACITY // 2`` most-important past tokens (per KV head)
        are kept as-is — treated identically to window tokens.  Only the
        remaining *unimportant* tokens undergo cumsum-based equal-mass
        partitioning and weighted-average fusion.

        Final cache layout (totals to MAX_CAPACITY):
            [fused_buckets | important_singletons | window + current]

        Args:
            scores:         Raw attention logits  [bs, query_heads, seq_len, seq_len].
            past_key_value: KV cache object with a ``fuse_kv_eff_split()`` method.
            key_heads:      Number of KV heads (may differ from *query_heads* under GQA).
            query_heads:    Number of query attention heads.

        Returns:
            attn_logit_offsets: Tensor of shape [bs, query_heads, 1, MAX_CAPACITY].
                Log-correction for bucket sizes (gated by config flag).
            zero_pad_offset:    Tensor of shape [bs, query_heads, 1, MAX_CAPACITY].
                ``-inf`` for empty (zero-padded) fused-bucket slots, ``0``
                elsewhere.  Always added to attention logits.
        """
        num_important = int(self.MAX_CAPACITY * self.imp_budget)
        num_fused_slots = int((self.MAX_CAPACITY * (1 - self.imp_budget))) - self.WIN_SIZE - 1

        # ---- 1. Per-KV-head importance scores --------------------------------
        scores_per_kv = (
            scores
            .view(scores.shape[0], key_heads, -1, scores.shape[2], scores.shape[3])
            .max(dim=2)[0]
        )  # [bs, key_heads, seq_len, seq_len]

        past_attn = scores_per_kv[:, :, -(self.WIN_SIZE + 1):-1, :-(self.WIN_SIZE + 1)]
        num_past = past_attn.shape[-1]
        importance = nn.functional.softmax(past_attn / self.fuse_temperature, dim=-1).mean(dim=2)
        # importance: [bs, key_heads, num_past]

        # Weight importance by value vector magnitudes
        past_val_norms = past_key_value.value_cache[self.layer_idx][:, :, :num_past, :].norm(dim=-1)
        importance = importance * past_val_norms

        # ---- 2. Identify top-K important tokens (kept as singletons) ---------
        num_important = min(num_important, num_past)  # guard against short sequences
        num_unimportant = num_past - num_important

        _, important_idx = importance.topk(num_important, dim=-1, sorted=False)
        important_idx, _ = important_idx.sort(dim=-1)  # preserve temporal order
        # important_idx: [bs, key_heads, num_important]

        # ---- 3. Extract unimportant tokens (sorted by position) --------------
        _, unimportant_idx = importance.topk(num_unimportant, dim=-1, largest=False, sorted=False)
        unimportant_idx, _ = unimportant_idx.sort(dim=-1)  # temporal order
        # unimportant_idx: [bs, key_heads, num_unimportant]

        unimp_importance = importance.gather(2, unimportant_idx)
        # unimp_importance: [bs, key_heads, num_unimportant]

        # ---- 4. Cumsum-based equal-mass partitioning on unimportant tokens ---
        cs = torch.cumsum(unimp_importance, dim=-1)
        total_mass = cs[..., -1:].clamp(min=1e-8)
        bucket_ids = (cs * num_fused_slots / total_mass).long().clamp(0, num_fused_slots - 1)
        # bucket_ids: [bs, key_heads, num_unimportant]

        # ---- 5. Fuse KV cache (unimportant → scatter-add, important → as-is)
        past_key_value.fuse_kv_eff_split(
            important_idx, unimportant_idx, bucket_ids,
            unimp_importance, num_fused_slots, self.layer_idx,
        )

        # ---- 6. Log-correction offsets: log(bucket_size) --------------------
        bs = scores.shape[0]
        device = scores.device

        # Fused bucket sizes
        bucket_sizes = torch.zeros(
            bs, key_heads, num_fused_slots, device=device, dtype=torch.float32,
        )
        bucket_sizes.scatter_add_(
            2, bucket_ids,
            torch.ones(bs, key_heads, num_unimportant, device=device, dtype=torch.float32),
        )

        # Identify zero-padded (empty) fused-bucket slots *before* clamping
        is_zero_pad = (bucket_sizes == 0)  # [bs, key_heads, num_fused_slots]

        bucket_sizes.clamp_(min=1.0)   # safe for log: log(1) = 0 for empties

        # Important singletons and window singletons all have size 1
        important_sizes = torch.ones(
            bs, key_heads, num_important, device=device, dtype=torch.float32,
        )
        window_sizes = torch.ones(
            bs, key_heads, self.WIN_SIZE + 1, device=device, dtype=torch.float32,
        )

        all_sizes = torch.cat([bucket_sizes, important_sizes, window_sizes], dim=-1)
        # all_sizes: [bs, key_heads, MAX_CAPACITY]

        # Replicate for GQA
        all_sizes = all_sizes.repeat_interleave(query_heads // key_heads, dim=1)
        # [bs, query_heads, MAX_CAPACITY]

        attn_logit_offsets = torch.log(all_sizes).unsqueeze(2)
        # [bs, query_heads, 1, MAX_CAPACITY]

        # ---- 7. Zero-pad offset: -inf for empty fused-bucket slots -----------
        zero_pad_fused = torch.where(
            is_zero_pad,
            torch.tensor(float('-inf'), device=device, dtype=torch.float32),
            torch.zeros(1, device=device, dtype=torch.float32),
        )  # [bs, key_heads, num_fused_slots]

        zero_pad_rest = torch.zeros(
            bs, key_heads, num_important + self.WIN_SIZE + 1,
            device=device, dtype=torch.float32,
        )
        zero_pad = torch.cat([zero_pad_fused, zero_pad_rest], dim=-1)
        # [bs, key_heads, MAX_CAPACITY]

        zero_pad = zero_pad.repeat_interleave(query_heads // key_heads, dim=1)
        zero_pad_offset = zero_pad.unsqueeze(2)
        # [bs, query_heads, 1, MAX_CAPACITY]

        return attn_logit_offsets, zero_pad_offset

    def morphkv_hierarchical_cache(self, scores, past_key_value, key_heads, query_heads):
        """Hierarchical KV cache compression with contiguous island fusion.

        After selecting important tokens, unimportant tokens form contiguous
        islands (spans between important token positions).  Each island is
        fused into a single representative via importance-weighted averaging
        of post-RoPE keys and values.  Islands are scored by mean importance,
        and the top-K islands are retained where K is the unimportant budget.

        Final cache layout (totals to <= MAX_CAPACITY):
            [selected_fused_islands | important_singletons | window + current]

        Pipeline:
          1. Compute per-KV-head importance (softmax attn × value norm).
          2. Select top-K important tokens as singletons.
          3. Identify contiguous islands of unimportant tokens via cumsum.
          4. Fuse each island via importance-weighted averaging (post-RoPE).
          5. Score islands by mean importance; retain top-K islands.
          6. Assemble compressed cache and compute zero-pad offsets.
        """
        num_important = int(self.MAX_CAPACITY * self.imp_budget)
        num_fused_slots = self.MAX_CAPACITY - self.WIN_SIZE - 1 - num_important

        # ---- 1. Per-KV-head importance scores --------------------------------
        scores_per_kv = (
            scores
            .view(scores.shape[0], key_heads, -1, scores.shape[2], scores.shape[3])
            .max(dim=2)[0]
        )

        past_attn = scores_per_kv[:, :, -(self.WIN_SIZE + 1):-1, :-(self.WIN_SIZE + 1)]
        num_past = past_attn.shape[-1]
        importance = nn.functional.softmax(past_attn / self.fuse_temperature, dim=-1).mean(dim=2)

        past_val_norms = past_key_value.value_cache[self.layer_idx][:, :, :num_past, :].norm(dim=-1)
        importance = importance * past_val_norms
        # importance: [bs, key_heads, num_past]

        # ---- 2. Select top-K important tokens --------------------------------
        num_important = min(num_important, num_past)
        num_unimportant = num_past - num_important

        _, important_idx = importance.topk(num_important, dim=-1, sorted=False)
        important_idx, _ = important_idx.sort(dim=-1)

        _, unimportant_idx = importance.topk(num_unimportant, dim=-1, largest=False, sorted=False)
        unimportant_idx, _ = unimportant_idx.sort(dim=-1)
        unimp_importance = importance.gather(2, unimportant_idx)

        # ---- 3. Identify contiguous islands of unimportant tokens ------------
        key_cache = past_key_value.key_cache[self.layer_idx]
        bs = key_cache.shape[0]
        head_dim = key_cache.shape[3]
        device = key_cache.device

        is_important = torch.zeros(bs, key_heads, num_past, device=device, dtype=torch.long)
        is_important.scatter_(2, important_idx, 1)

        island_ids_all = torch.cumsum(is_important, dim=-1)
        unimp_island_ids = island_ids_all.gather(2, unimportant_idx)
        max_islands = num_important + 1

        # ---- 4. Fuse each island (post-RoPE weighted averaging) --------------
        past_keys = key_cache[:, :, :num_past, :]
        past_vals = past_key_value.value_cache[self.layer_idx][:, :, :num_past, :]

        unimp_exp = unimportant_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        unimp_keys = past_keys.gather(2, unimp_exp)
        unimp_vals = past_vals.gather(2, unimp_exp)

        island_mass = torch.zeros(bs, key_heads, max_islands, device=device, dtype=importance.dtype)
        island_mass.scatter_add_(2, unimp_island_ids, unimp_importance)

        weights = unimp_importance / island_mass.gather(2, unimp_island_ids).clamp(min=1e-8)
        w = weights.unsqueeze(-1).to(past_keys.dtype)

        bkt_exp = unimp_island_ids.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        fused_keys = torch.zeros(bs, key_heads, max_islands, head_dim, device=device, dtype=past_keys.dtype)
        fused_vals = torch.zeros(bs, key_heads, max_islands, head_dim, device=device, dtype=past_vals.dtype)
        fused_keys.scatter_add_(2, bkt_exp, unimp_keys * w)
        fused_vals.scatter_add_(2, bkt_exp, unimp_vals * w)

        # ---- 5. Score islands by mean importance, select top-K ---------------
        island_counts = torch.zeros(bs, key_heads, max_islands, device=device, dtype=torch.float32)
        island_counts.scatter_add_(
            2, unimp_island_ids,
            torch.ones(bs, key_heads, num_unimportant, device=device, dtype=torch.float32),
        )

        island_scores = island_mass / island_counts.clamp(min=1)
        island_scores = island_scores.masked_fill(island_counts == 0, float('-inf'))

        K = min(num_fused_slots, max_islands)
        _, top_island_idx = island_scores.topk(K, dim=-1, sorted=False)
        top_island_idx, _ = top_island_idx.sort(dim=-1)

        sel_exp = top_island_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        selected_fused_keys = fused_keys.gather(2, sel_exp)
        selected_fused_vals = fused_vals.gather(2, sel_exp)
        selected_sizes = island_counts.gather(2, top_island_idx)

        if K < num_fused_slots:
            pad_size = num_fused_slots - K
            pad_kv = torch.zeros(bs, key_heads, pad_size, head_dim, device=device, dtype=past_keys.dtype)
            selected_fused_keys = torch.cat([selected_fused_keys, pad_kv], dim=2)
            selected_fused_vals = torch.cat([selected_fused_vals, pad_kv], dim=2)
            selected_sizes = torch.cat([
                selected_sizes,
                torch.zeros(bs, key_heads, pad_size, device=device, dtype=torch.float32),
            ], dim=2)

        # ---- 6. Assemble compressed cache ------------------------------------
        past_key_value.fuse_kv_hierarchical(
            selected_fused_keys, selected_fused_vals,
            important_idx, num_past, self.layer_idx,
        )

        # ---- 7. Zero-pad offset (no log-correction) -------------------------
        is_zero_pad = (selected_sizes == 0)

        attn_logit_offsets = torch.zeros(
            bs, key_heads, 1, num_fused_slots + num_important + self.WIN_SIZE + 1,
            device=device, dtype=torch.float32,
        )
        attn_logit_offsets = attn_logit_offsets.repeat_interleave(query_heads // key_heads, dim=1)

        zero_pad_fused = torch.where(
            is_zero_pad,
            torch.tensor(float('-inf'), device=device, dtype=torch.float32),
            torch.zeros(1, device=device, dtype=torch.float32),
        )
        zero_pad_rest = torch.zeros(
            bs, key_heads, num_important + self.WIN_SIZE + 1,
            device=device, dtype=torch.float32,
        )
        zero_pad = torch.cat([zero_pad_fused, zero_pad_rest], dim=-1)
        zero_pad = zero_pad.repeat_interleave(query_heads // key_heads, dim=1)
        zero_pad_offset = zero_pad.unsqueeze(2)

        return attn_logit_offsets, zero_pad_offset

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        query_cache: List = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        self.window_queries = query_cache
        
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_heads = query_states.shape[1]
        key_heads = key_states.shape[1]
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            

        if key_states.shape[1]!=query_states.shape[1]:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        if self.config.morphkv: # cache queries for MorphKV
            query_states = past_key_value.update_win_queries(query_states[:,:,-(self.WIN_SIZE+1):,:],self.layer_idx)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        
        # use MorphKV only in generative phase, ie, when hidden states has 1 token (the newly generated)
        if self.config.morphkv and key_states.shape[2]>= (1 + self.MAX_CAPACITY) * self.evict_after:
            if hidden_states.shape[1]==1:
                if 'unimp' not in self.morph_type: # use normal morphkv
                    attn_weights, init_mask = self.morphkv_mask(attn_weights, past_key_value, key_heads, query_heads)
                else:
                    if not past_key_value.fusion_done[self.layer_idx]:
                        if self.pre_rope:
                            attn_logit_offsets, zero_pad_offset = self.morphkv_hierarchical_cache(attn_weights, past_key_value, key_heads, query_heads)
                        else:
                            attn_logit_offsets, zero_pad_offset = self.morphkv_plus_fuse_unimportant(attn_weights, past_key_value, key_heads, query_heads)
                        key_states = past_key_value.key_cache[self.layer_idx]
                        value_states = past_key_value.value_cache[self.layer_idx]
                        if key_states.shape[1]!=query_states.shape[1]:
                            key_states = repeat_kv(key_states, self.num_key_value_groups)
                            value_states = repeat_kv(value_states, self.num_key_value_groups)
                        recent_query = query_states[:,:,-1:,...]
                        attn_weights = torch.matmul(recent_query, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                        attn_weights = attn_weights + zero_pad_offset  # always mask empty buckets
                        if self.use_attn_offsets:
                            attn_weights = attn_weights + attn_logit_offsets
                    else:
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
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


class MistralFlashAttention2Morph(MistralAttentionMorph):
    """
    Mistral flash attention module. This module inherits from `MistralAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
            )
        # print("Using Morph FA...\n")    
        if hidden_states.shape[1]==1 and self.config.morphkv and past_key_value.key_cache[self.layer_idx].shape[2]>self.MAX_CAPACITY:
            # logger.warning_once(
            #     "MistralModel was using MistralFlashAttention2 for prefilling, which does not support MorphKV eviction. Falling back to the eager attention implementation.\n\n"
            # )
            return super(MistralFlashAttention2Morph,self).forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                query_cache=self.window_queries if self.config.morphkv else None,
            )
        
        output_attentions = False
        
        # reset window queries for every new sequence
        if(hidden_states.shape[1]!=1):
            past_key_value.query_cache[self.layer_idx] = []

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += cache_position[0]

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        if key_states.shape[1]<query_states.shape[1]:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
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

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self.config, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        # cache win queries after attn output
        query_states = past_key_value.update_win_queries(query_states.transpose(1,2)[...,-(self.WIN_SIZE+1):,:],self.layer_idx)

        past_key_value.cleanup(None,None,self.layer_idx,dummy=True) ## just for the sake of profiling memory

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


def mistral_model_forward(
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

    # retrieve input_ids and inputs_embeds
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # past_key_values = MorphOffloadedCache(self.config.num_hidden_layers)
    # kept for BC (non `Cache` `past_key_values` inputs)
    # import pdb; pdb.set_trace()
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
        attention_mask, inputs_embeds, cache_position, past_key_values, use_cache, output_attentions
    )

    hidden_states = inputs_embeds

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