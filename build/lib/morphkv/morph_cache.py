import importlib.metadata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from packaging import version

from transformers.utils import is_hqq_available, is_quanto_available, logging


logger = logging.get_logger(__name__)

from transformers.cache_utils import Cache

class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
        ```
    """

    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        if num_hidden_layers is None:
            self.key_cache: List[torch.Tensor] = []
            self.value_cache: List[torch.Tensor] = []
        else:
            self.key_cache: List[torch.Tensor] = [[] for _ in range(num_hidden_layers)]
            self.value_cache: List[torch.Tensor] = [[] for _ in range(num_hidden_layers)]
            self.query_cache: List[torch.Tensor] = [[] for _ in range(num_hidden_layers)]
            self.attn_cache: List[torch.Tensor] = [[] for _ in range(num_hidden_layers)]
            self.attn_mask: List[torch.Tensor] = [[] for _ in range(num_hidden_layers)]
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        # self.cache_size = {'key':10000000,'value':10000000,'query':10000000, 'attn_wts':10000000, 'len':10000000} #attn_wts
        self.cache_size = {'key':0,'value':0, 'attn_wts':0, 'len':0} 
        self.input_size = torch.inf
        self.max_capacity = torch.inf
        self.win_size = torch.inf
        self.prefill = False
    
    def set_max_capacity(self, max_capacity:int, win_size:int):
        self.max_capacity = max_capacity
        self.win_size = win_size

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx], self.attn_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")
        
    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx], self.attn_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def cleanup(
            self,
            init_mask, #use for KV pruning
            init_mask_attn, #use for attn pruning
            layer_idx: int,
            dummy: bool = False
    ) -> None:
        """
        clears unnecessary KV cache
        """

        # import pdb; pdb.set_trace()
        BS,NH,LEN,DIM = self.key_cache[layer_idx].shape
        if not dummy:
            # import pdb; pdb.set_trace()
            _, init_mask_heads, _, _ = init_mask.shape
            if self.key_cache[layer_idx]!=[]: 
                BS,NH,LEN,DIM = self.key_cache[layer_idx].shape
                # For GQA, NH = 8, init_mask_heads = 32, hence, we view it as (1,8,4,len,dim) and then take logical OR along dim2
                #self.key_cache[layer_idx] = self.key_cache[layer_idx][(init_mask==0).squeeze(2)].view(BS,NH,-1,DIM) #pick only the relevant indices along the seq_len axis (BS,num_heads,seq_len,hidden_dim)

                ##Efficient
                assert BS==1, "Only supported for BS = 1 for now.\n"
                indices = (init_mask.squeeze(2).reshape(-1)==0).nonzero(as_tuple=True)[0]
                self.key_cache[layer_idx] = torch.index_select(self.key_cache[layer_idx].view(BS,-1,DIM), dim=1, index=indices).view(BS,NH,-1,DIM)
            if self.value_cache[layer_idx]!=[]: 
                BS,NH,LEN,DIM = self.value_cache[layer_idx].shape
                #self.value_cache[layer_idx] = self.value_cache[layer_idx][(init_mask==0).squeeze(2)].view(BS,NH,-1,DIM) #pick only the relevant indices along the seq_len axis (BS,num_heads,seq_len,hidden_dim)

                # Efficient
                indices = (init_mask.squeeze(2).reshape(-1)==0).nonzero(as_tuple=True)[0]
                self.value_cache[layer_idx] = torch.index_select(self.value_cache[layer_idx].view(BS,-1,DIM), dim=1, index=indices).view(BS,NH,-1,DIM)
            if self.attn_cache[layer_idx]!=[]: 
                BS,NH,WS,LEN = self.attn_cache[layer_idx].shape
                #self.attn_cache[layer_idx] = self.attn_cache[layer_idx].transpose(3,2)[(init_mask_attn==0).squeeze(2)].view(BS,NH,-1,WS).transpose(3,2) #pick only the relevant indices along the seq_len axis (BS,num_heads,win_size,seq_len)
                
                #Efficient
                init_mask_attn = init_mask_attn.expand(BS,NH,WS,LEN)
                indices_attn = (init_mask_attn.reshape(-1)==0).nonzero(as_tuple=True)[0]
                self.attn_cache[layer_idx] = torch.index_select(self.attn_cache[layer_idx].view(-1),dim=0,index=indices_attn).reshape(BS,NH,WS,-1)
            if layer_idx==0:
                if self.key_cache[layer_idx] != []: self.cache_size['key'] = max(self.cache_size['key'],self.key_cache[layer_idx].shape[2])#len(self.key_cache) * self.key_cache[0].shape[0] * self.key_cache[0].shape[1] * self.key_cache[0].shape[2] * self.key_cache[0].shape[3])
                if self.value_cache[layer_idx] != []: self.cache_size['value'] = max(self.cache_size['value'], self.value_cache[layer_idx].shape[2]) #len(self.value_cache) * self.value_cache[0].shape[0] * self.value_cache[0].shape[1] * self.value_cache[0].shape[2] * self.value_cache[0].shape[3])
                if self.attn_cache[layer_idx] != []: self.cache_size['attn_wts'] = max(self.cache_size['attn_wts'], self.attn_cache[layer_idx].shape[2]) #len(self.attn_cache) * self.attn_cache[0].shape[0] * self.attn_cache[0].shape[1] * self.attn_cache[0].shape[2] * self.attn_cache[0].shape[3])    
        # Profiling memory
        if layer_idx==0:
            # if self.key_cache[layer_idx] != []: self.cache_size['key'] = max(self.cache_size['key'],self.key_cache[layer_idx].shape[2])#len(self.key_cache) * self.key_cache[0].shape[0] * self.key_cache[0].shape[1] * self.key_cache[0].shape[2] * self.key_cache[0].shape[3])
            # if self.value_cache[layer_idx] != []: self.cache_size['value'] = max(self.cache_size['value'], self.value_cache[layer_idx].shape[2]) #len(self.value_cache) * self.value_cache[0].shape[0] * self.value_cache[0].shape[1] * self.value_cache[0].shape[2] * self.value_cache[0].shape[3])
            # if self.attn_cache[layer_idx] != []: self.cache_size['attn_wts'] = max(self.cache_size['attn_wts'], self.attn_cache[layer_idx].shape[2]) #len(self.attn_cache) * self.attn_cache[0].shape[0] * self.attn_cache[0].shape[1] * self.attn_cache[0].shape[2] * self.attn_cache[0].shape[3])
            if not self.prefill:
                self.cache_size['len'] = LEN
            else:
                self.cache_size['len']+=1
            self.prefill = True
        # new_size = sys.getsizeof(self.key_cache) + sys.getsizeof(self.value_cache) + sys.getsizeof(self.attn_cache)
        # print(f"cleared {100*(1-old_size/new_size)}% of KV Cache!\n")
    
    def update_win_queries(self, win_queries, layer_idx):
        if(self.query_cache[layer_idx]==[]):
            self.query_cache[layer_idx] = win_queries
        else:
            self.query_cache[layer_idx] = torch.roll(self.query_cache[layer_idx],shifts=-1, dims=2)
            self.query_cache[layer_idx][...,-1:,:] = win_queries
        return self.query_cache[layer_idx]
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        # content on layer cache can be a tensor and checking not tensor causes errors
        # so we explicitly check for the empty list
        elif self.key_cache[layer_idx] == []:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        if self.input_size==torch.inf:
            self.input_size = self.key_cache[layer_idx].shape[2]
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx or (len(self.key_cache) > layer_idx and self.key_cache[layer_idx] == []):
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
        backward compatibility."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, num_hidden_layers: int = None
    ) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
        backward compatibility."""
        cache = cls(num_hidden_layers)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states, _ = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx] != []:
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def batch_split(self, full_batch_size: int, split_size: int, num_hidden_layers: int) -> List["DynamicCache"]:
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = DynamicCache(num_hidden_layers)
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["DynamicCache"], num_hidden_layers: int) -> "DynamicCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        cache = cls(num_hidden_layers)
        for idx in range(len(splits[0])):
            key_cache = [current.key_cache[idx] for current in splits if current.key_cache[idx] != []]
            value_cache = [current.key_cache[idx] for current in splits if current.key_cache[idx] != []]
            if key_cache != []:
                layer_keys = torch.cat(key_cache, dim=0)
                layer_values = torch.cat(value_cache, dim=0)
                cache.update(layer_keys, layer_values, idx)
        return cache

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]


# This implementation has been adopted from HuggingFace's OffloadedCache
class MorphOffloadedCache(DynamicCache):
    """
    A drop-in replacement for DynamicCache that conserves GPU memory at the expense of more CPU memory.
    Useful for generating from models with very long context.

    In addition to the default CUDA stream, where all forward() computations happen,
    this class uses another stream, the prefetch stream, which it creates itself.
    Since scheduling of operations on separate streams happens independently, this class uses
    the prefetch stream to asynchronously prefetch the KV cache of layer k+1 when layer k is executing.
    The movement of the layer k-1 cache to the CPU is handled by the default stream as a simple way to
    ensure the eviction is scheduled after all computations on that cache are finished.
    """

    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("OffloadedCache can only be used with a GPU")
        super().__init__(num_hidden_layers)
        self.original_device = ["None"]*num_hidden_layers
        self.prefetch_stream = torch.cuda.Stream()
        self.beam_idx = None  # used to delay beam search operations
        self.cleaned = [False]*num_hidden_layers
        print("Offloaded Cache....")

    def prefetch_layer(self, layer_idx: int):
        "Starts prefetching the next layer cache"
        if layer_idx < len(self) and self.key_cache[layer_idx]!=[]:
            with torch.cuda.stream(self.prefetch_stream):
                # Prefetch next layer tensors to GPU
                device = self.original_device[layer_idx]
                self.key_cache[layer_idx] = self.key_cache[layer_idx].to(device, non_blocking=True)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].to(device, non_blocking=True)
                
                if self.query_cache[layer_idx]!=[]:
                    self.query_cache[layer_idx] = self.query_cache[layer_idx].to(device, non_blocking=True)
                if self.attn_cache[layer_idx]!=[]:
                    self.attn_cache[layer_idx] = self.attn_cache[layer_idx].to(device, non_blocking=True)
        

    def evict_previous_layer(self, layer_idx: int):
        "Moves the previous layer cache to the CPU"
        prev_layer_idx = (layer_idx - 1) % len(self)
        if self.key_cache[prev_layer_idx]!=[]:
            # We do it on the default stream so it occurs after all earlier computations on these tensors are done
            self.key_cache[prev_layer_idx] = self.key_cache[prev_layer_idx].to("cpu", non_blocking=True)
            self.value_cache[prev_layer_idx] = self.value_cache[prev_layer_idx].to("cpu", non_blocking=True)
        
        if self.attn_cache[prev_layer_idx]!=[]:    
            self.attn_cache[prev_layer_idx] = self.attn_cache[prev_layer_idx].to("cpu", non_blocking=True)
        if self.query_cache[prev_layer_idx]!=[]:    
            self.query_cache[prev_layer_idx] = self.query_cache[prev_layer_idx].to("cpu", non_blocking=True)
        

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        "Gets the cache for this layer to the device. Prefetches the next and evicts the previous layer."
        if layer_idx < len(self):
            # Evict the previous layer if necessary
            torch.cuda.current_stream().synchronize()
            self.evict_previous_layer(layer_idx)
            # Load current layer cache to its original device if not already there
            original_device = self.original_device[layer_idx]
            self.prefetch_stream.synchronize()
            key_tensor = self.key_cache[layer_idx]
            value_tensor = self.value_cache[layer_idx]
            query_tensor = self.query_cache[layer_idx]
            
            # Prefetch the next layer
            self.prefetch_layer((layer_idx + 1) % len(self))
            return (key_tensor, value_tensor, query_tensor)
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx], self.query_cache[layer_idx])

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Saves the beam indices and reorders the cache when the tensor is back to its device."""
        # We delay this operation until the tensors are back to their original
        # device because performing torch.index_select on the CPU is very slow
        del self.beam_idx
        self.beam_idx = beam_idx.clone()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `OffloadedCache`.
        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.original_device.append(key_states.device)
            # self.evict_previous_layer(layer_idx)
            if layer_idx==len(self)-1: # if prefilling last layer KV, prefetch 0th layer KV
                self.prefetch_layer((layer_idx + 1) % len(self))
        # content on layer cache can be a tensor and checking not tensor causes errors
        # so we explicitly check for the empty list
        elif self.key_cache[layer_idx] == []:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        
            self.original_device[layer_idx] = key_states.device
            self.evict_previous_layer(layer_idx)
            if layer_idx==len(self)-1: # if prefilling last layer KV, prefetch 0th layer KV
                self.prefetch_layer((layer_idx + 1) % len(self))
        else:
            key_tensor, value_tensor, query_tensor = self[layer_idx]
            self.key_cache[layer_idx] = torch.cat([key_tensor, key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([value_tensor, value_states], dim=-2)
            
        
        if self.input_size==torch.inf:
            self.input_size = self.key_cache[layer_idx].shape[2]
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

