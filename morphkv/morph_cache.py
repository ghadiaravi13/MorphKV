import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from transformers.cache_utils import Cache as TransformersCache
from transformers.utils import logging


logger = logging.get_logger(__name__)


class Cache(TransformersCache):
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def __init__(self):
        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_length(self) -> Optional[int]:
        raise NotImplementedError("Make sure to implement `get_max_length` in a subclass.")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] != []:
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            if self.value_cache[layer_idx] != []:
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    @property
    def seen_tokens(self):
        logger.warning_once(
            "The `seen_tokens` attribute is deprecated and will be removed in v4.41. Use the `cache_position` "
            "model input instead."
        )
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated, with support for
    MorphKV eviction via ``cleanup()``.
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
        self._seen_tokens = 0
        self.cache_size = {'key': 0, 'value': 0, 'attn_wts': 0, 'len': 0}
        self.input_size = torch.inf
        self.max_capacity = torch.inf
        self.win_size = torch.inf
        self.prefill = False

    def set_max_capacity(self, max_capacity: int, win_size: int):
        self.max_capacity = max_capacity
        self.win_size = win_size

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx], self.attn_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx], self.attn_cache[layer_idx])

    def __len__(self):
        return len(self.key_cache)

    def cleanup(
            self,
            eviction_mask,
            layer_idx: int,
            dummy: bool = False
    ) -> None:
        """Evicts KV cache entries indicated by eviction_mask (positions with -inf are evicted)."""
        BS, NH, LEN, DIM = self.key_cache[layer_idx].shape
        if not dummy:
            assert BS == 1, "Only supported for BS = 1 for now.\n"
            indices = (eviction_mask.squeeze(2).reshape(-1) == 0).nonzero(as_tuple=True)[0]
            if self.key_cache[layer_idx] != []:
                BS, NH, LEN, DIM = self.key_cache[layer_idx].shape
                self.key_cache[layer_idx] = torch.index_select(
                    self.key_cache[layer_idx].view(BS, -1, DIM), dim=1, index=indices
                ).view(BS, NH, -1, DIM)
            if self.value_cache[layer_idx] != []:
                BS, NH, LEN, DIM = self.value_cache[layer_idx].shape
                self.value_cache[layer_idx] = torch.index_select(
                    self.value_cache[layer_idx].view(BS, -1, DIM), dim=1, index=indices
                ).view(BS, NH, -1, DIM)
        if layer_idx == 0:
            if self.key_cache[layer_idx] != []:
                self.cache_size['key'] = max(self.cache_size['key'], self.key_cache[layer_idx].shape[2])
            if self.value_cache[layer_idx] != []:
                self.cache_size['value'] = max(self.cache_size['value'], self.value_cache[layer_idx].shape[2])
            if not self.prefill:
                self.cache_size['len'] = LEN
            else:
                self.cache_size['len'] += 1
            self.prefill = True

    def update_win_queries(self, win_queries, layer_idx):
        if self.query_cache[layer_idx] == []:
            self.query_cache[layer_idx] = win_queries
        else:
            self.query_cache[layer_idx] = torch.roll(self.query_cache[layer_idx], shifts=-1, dims=2)
            self.query_cache[layer_idx][..., -1:, :] = win_queries
        return self.query_cache[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif self.key_cache[layer_idx] == []:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        self.input_size = min(self.input_size, self.key_cache[layer_idx].shape[2])

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx or (len(self.key_cache) > layer_idx and self.key_cache[layer_idx] == []):
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, num_hidden_layers: int = None
    ) -> "DynamicCache":
        cache = cls(num_hidden_layers)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def crop(self, max_length: int):
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
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]


class OffloadedCache(DynamicCache):
    """
    A drop-in replacement for DynamicCache that conserves GPU memory at the expense of more CPU memory.
    Uses a prefetch stream to asynchronously move KV cache between CPU and GPU.
    """

    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("OffloadedCache can only be used with a GPU")
        super().__init__(num_hidden_layers)
        self.original_device = ["None"] * num_hidden_layers
        self.prefetch_stream = torch.cuda.Stream()
        self.beam_idx = None
        self.cleaned = [False] * num_hidden_layers

    def prefetch_layer(self, layer_idx: int):
        """Starts prefetching the next layer cache"""
        if layer_idx < len(self):
            with torch.cuda.stream(self.prefetch_stream):
                device = self.original_device[layer_idx]
                if self.key_cache[layer_idx] != []:
                    self.key_cache[layer_idx] = self.key_cache[layer_idx].to(device, non_blocking=True)
                if self.value_cache[layer_idx] != []:
                    self.value_cache[layer_idx] = self.value_cache[layer_idx].to(device, non_blocking=True)
                if self.attn_cache[layer_idx] != []:
                    self.attn_cache[layer_idx] = self.attn_cache[layer_idx].to(device, non_blocking=True)

    def evict_previous_layer(self, layer_idx: int):
        """Moves the previous layer cache to the CPU"""
        prev_layer_idx = (layer_idx - 1) % len(self)
        if self.key_cache[prev_layer_idx] != []:
            self.key_cache[prev_layer_idx] = self.key_cache[prev_layer_idx].to("cpu", non_blocking=True)
            self.value_cache[prev_layer_idx] = self.value_cache[prev_layer_idx].to("cpu", non_blocking=True)
        if self.attn_cache[prev_layer_idx] != []:
            self.attn_cache[prev_layer_idx] = self.attn_cache[prev_layer_idx].to("cpu", non_blocking=True)

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """Gets the cache for this layer. Prefetches the next and evicts the previous layer."""
        if layer_idx < len(self):
            torch.cuda.current_stream().synchronize()
            self.evict_previous_layer(layer_idx)
            self.prefetch_stream.synchronize()
            key_tensor = self.key_cache[layer_idx]
            value_tensor = self.value_cache[layer_idx]
            query_tensor = self.query_cache[layer_idx]
            self.prefetch_layer((layer_idx + 1) % len(self))
            return (key_tensor, value_tensor, query_tensor)
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx], self.query_cache[layer_idx])

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Saves the beam indices and reorders the cache when the tensor is back to its device."""
        del self.beam_idx
        self.beam_idx = beam_idx.clone()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        query_states: torch.Tensor = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        phase: str = "join"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.original_device.append(key_states.device)
            if layer_idx == len(self) - 1:
                self.prefetch_layer((layer_idx + 1) % len(self))
        elif self.key_cache[layer_idx] == []:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            if query_states is not None:
                self.query_cache[layer_idx] = query_states
            self.original_device[layer_idx] = key_states.device
            self.evict_previous_layer(layer_idx)
            if layer_idx == len(self) - 1:
                self.prefetch_layer((layer_idx + 1) % len(self))
        else:
            key_tensor, value_tensor, query_tensor = self[layer_idx]
            if phase == "write":
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            elif phase == "join":
                self.key_cache[layer_idx] = torch.cat([key_tensor, key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([value_tensor, value_states], dim=-2)

            if query_states is not None:
                if self.query_cache[layer_idx].shape[-2] == self.win_size:
                    self.query_cache[layer_idx] = torch.roll(self.query_cache[layer_idx], shifts=-1, dims=2)
                    self.query_cache[layer_idx][:, :, -1:, :] = query_states[..., -1:, :]
                else:
                    self.query_cache[layer_idx] = torch.cat([query_tensor, query_states], dim=-2)

        if layer_idx == 0 and query_states is not None:
            BS, NH, LEN, DIM = self.key_cache[layer_idx].shape
            if self.key_cache[layer_idx] != []:
                self.cache_size['key'] = min(self.cache_size['key'], self.key_cache[layer_idx].shape[2])
            if self.value_cache[layer_idx] != []:
                self.cache_size['value'] = min(self.cache_size['value'], self.value_cache[layer_idx].shape[2])
            if self.query_cache[layer_idx] != []:
                self.cache_size['query'] = min(self.cache_size['query'], self.query_cache[layer_idx].shape[2])
            if not self.prefill:
                self.cache_size['len'] = LEN
            elif self.prefill and phase == "join":
                self.cache_size['len'] += 1
            self.prefill = True

        self.input_size = min(self.input_size, self.key_cache[layer_idx].shape[2])

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    from_legacy_cache = None

    to_legacy_cache = None


MorphOffloadedCache = OffloadedCache
