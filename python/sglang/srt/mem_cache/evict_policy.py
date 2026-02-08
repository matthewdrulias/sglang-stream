from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: "TreeNode") -> Union[float, Tuple]:
        pass


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: "TreeNode") -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class SinkProtectedLRUStrategy(EvictionStrategy):
    """LRU eviction with sink token protection for StreamingLLM.

    Sink tokens (first N tokens of each sequence) are protected from eviction.
    This enables infinite context with constant memory by keeping "attention sink"
    tokens that stabilize the model's attention patterns.

    See: https://arxiv.org/abs/2309.17453 (Efficient Streaming Language Models with Attention Sinks)
    """

    def __init__(self, sink_token_count: int = 4):
        self.sink_token_count = sink_token_count

    def get_priority(self, node: "TreeNode") -> Tuple[int, float]:
        # Priority tuple: (is_protected, last_access_time)
        # Protected nodes (sink) get priority 1, others get 0
        # Lower priority = evicted first, so sinks (1) are evicted last
        is_sink = getattr(node, "seq_start_offset", float("inf")) < self.sink_token_count
        return (1 if is_sink else 0, node.last_access_time)
