# SPDX-License-Identifier: Apache-2.0
"""KV-Cache Utilities."""
from __future__ import annotations

import os
from collections import deque, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import GiB_bytes, sha256
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheSpec,
                                        KVCacheTensor, SlidingWindowSpec)
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashType(NamedTuple):
    """Hash value of a block (int), the token IDs in the block, and extra keys.
    We keep a tuple of token IDs and extra keys to reduce the likelihood of
    hash collisions when the hash value is the same. By using SHA256 however,
    hash collisions are practically impossible.
    """
    # Hash value of the block in an integer.
    hash_value: int
    # Token IDs in the block.
    token_ids: tuple[int, ...]
    # Extra keys for the block.
    extra_keys: Optional[Any] = None


# The hash seed for the first block of the prefix block sequence.
#
# Even if the hash function is the builtin hash(), we use sha256 to generate
# the initial hash to simplify the code. This is not performance critical
# as it is done one per process.
#
# We use a random value to avoid hash collisions or PYTHONHASHSEED environment
# variable if set such that processes can share the seed if needed.
# This aligns with the behavior of Python's hash() function, which also uses
# a random seed if PYTHONHASHSEED is not set.
NONE_HASH = int.from_bytes(os.urandom(32), byteorder="big") if os.getenv(
    'PYTHONHASHSEED') is None else sha256(os.getenv('PYTHONHASHSEED'))


class PrefixCachingMetrics:
    """Metrics for prefix caching with a hit rate of the max recent N requests.

    Args:
        max_recent_requests: The number of the max recent requests to aggregate.
            Defaults to 1000.
    """

    def __init__(self, max_recent_requests: int = 1000):
        self.max_recent_requests = max_recent_requests
        # The current aggregated values.
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0
        # A deque of (requests, queries, hits) for the most recent requests.
        self.query_queue: deque[tuple[int, int, int]] = deque()

    def observe(self, stats: PrefixCacheStats):
        """Observe the prefix caching for a set of requests.

        This function is called with information gathered when new requests
        are being scheduled and are looking for computed blocks.

        When there are more than `interval` requests, the oldest set of
        requests are removed from the metrics.

        Args:
            stats: The prefix cache stats.
        """
        # reset_prefix_cache was invoked before the current update.
        # Reset the metrics before aggregating the current stats.
        if stats.reset:
            self.reset()

        # Update the metrics.
        self.query_queue.append((stats.requests, stats.queries, stats.hits))
        self.aggregated_requests += stats.requests
        self.aggregated_query_total += stats.queries
        self.aggregated_query_hit += stats.hits

        # Remove the oldest stats if the number of requests exceeds.
        if self.aggregated_requests > self.max_recent_requests:
            old_requests, old_queries, old_hits = self.query_queue.popleft()
            self.aggregated_requests -= old_requests
            self.aggregated_query_total -= old_queries
            self.aggregated_query_hit -= old_hits

    def reset(self):
        """Reset the metrics."""
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0
        self.query_queue.clear()

    @property
    def hit_rate(self) -> float:
        """Calculate the hit rate for the past N requests."""
        if self.aggregated_query_total == 0:
            return 0.0
        return self.aggregated_query_hit / self.aggregated_query_total


@dataclass
class KVCacheBlock:
    """KV-cache block metadata."""
    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The hash of the block composed of (block hash, tuple of token IDs).
    # It is only available when the block is full.
    _block_hash: Optional[BlockHashType] = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: Optional["KVCacheBlock"] = None
    next_free_block: Optional["KVCacheBlock"] = None

    _block_depth: int = 0

    def incr_ref(self):
        self.ref_cnt += 1

    def decr_ref(self):
        self.ref_cnt -= 1

    @property
    def block_hash(self) -> Optional[BlockHashType]:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, block_hash: BlockHashType):
        assert self.block_hash is None, (
            "The block already has a hash. This should not happen.")
        self._block_hash = block_hash

    def reset_hash(self):
        """Reset the block hash when the block is evicted."""
        self._block_hash = None

    @property
    def block_depth(self) -> int:
        return self._block_depth

    @block_depth.setter
    def block_depth(self, block_depth: int):
        self._block_depth = block_depth

    def __repr__(self) -> str:
        # Use block_id instead of KVCacheBlock object to avoid calling __repr__
        # on KVCacheBlock object recursively.
        prev_block_id = self.prev_free_block.block_id \
            if self.prev_free_block else None
        next_block_id = self.next_free_block.block_id \
            if self.next_free_block else None
        return (f"KVCacheBlock(block_id={self.block_id}, "
                f"ref_cnt={self.ref_cnt}, "
                f"_block_hash={self._block_hash}, "
                f"prev_free_block={prev_block_id}, "
                f"next_free_block={next_block_id})")

@dataclass
class CommonPrefixGroups:
    """List of request ids for requests which we group together.
    We group together consecutive requests."""
    request_ids: list[str]

    """Group metadata composed of list of 
    (num_common_prefix_blocks, start_index, end_index). 
    The requests corresponding to request_ids[start_index: end_index + 1] 
    form a group with the specified num_common_prefix_blocks."""
    group_metadata: list[tuple[int, int, int]]

    def extend(self, groups: CommonPrefixGroups):
        num_old_reqs = len(self.request_ids)
        self.request_ids.extend(groups.request_ids)
        self.group_metadata.extend([(a, b + num_old_reqs, c + num_old_reqs)
                                    for a,b,c in groups.group_metadata])

def get_scheduled_requests(request_list: list[tuple[str, bool]],
                           groups_list: list[tuple[int, int, int]]) \
        -> CommonPrefixGroups:

    # Post-process request and group list to remove
    # non-scheduled running requests
    actual_request_list = []
    actual_groups_list = []

    for group_num_common_blocks, start_id, end_id in groups_list:
        scheduled = [req_id for req_id, is_scheduled in
                     request_list[start_id:end_id + 1] if
                     is_scheduled]

        actual_start_id = len(actual_request_list)
        actual_request_list.extend(scheduled)

        if scheduled:
            actual_groups_list.append((group_num_common_blocks, actual_start_id,
                                       len(actual_request_list)))

    return CommonPrefixGroups(actual_request_list, actual_groups_list)


class KVCacheBlockPrefixTrie:
    """Prefix Trie of KVCacheBlocks for Forested Cascade Attention"""

    def __init__(self, depth: int, absorption_threshold_ratio: float):
        """
        Args:
            depth: The depth of the trees level 1 KVCacheBlock nodes.
            absorption_threshold_ratio: Threshold parameter that specifies
            whether to group leaves under a parent node, or its child nodes.
        """
        self.depth = depth
        self.absorption_threshold_ratio = absorption_threshold_ratio
        # node_id == 0 means node is a sentinel
        self.sentinel = KVCacheBlockPrefixTrieNode()
        # Values in dict are of the form (req_id, is_scheduled)
        self.req_id_to_req_id_wrapper: dict[str, bool] = {}
        self.req_id_to_block_id: dict[str, int] = {}
        self.block_id_to_req_id: defaultdict[int, set[str]] = defaultdict(set)
        self.block_id_to_leaf_node: dict[int, KVCacheBlockPrefixTrieNode] = {}
        self.num_groups = 0

        self.ALLOC_METHODS = {
            "full_pass": self.alloc_full_pass,
            "leaf_pass": self.alloc_leaf_pass,
        }

    def insert(self, request: Request, blocks: list[KVCacheBlock]):
        """
        Inserts blocks corresponding to the given request into the trie.
        If request already exists in trie, update path to request.
        Args:
            request: Request whose blocks we parse.
            blocks: Blocks which we add to the trie.
        """
        if not blocks:
            return
        req_id = request.request_id

        # First check if request is already in trie. Start inserting from
        # leaf block corresponding to the request if so, else start
        # from sentinel.
        curr_block_id = self.req_id_to_block_id.get(req_id, 0)
        curr_block_node = self.sentinel
        if curr_block_id:
            req_ids = self.block_id_to_req_id[curr_block_id]
            req_ids.remove(req_id)
            if not req_id:
                del self.block_id_to_req_id[curr_block_id]
            curr_block_node = self.block_id_to_leaf_node[curr_block_id]
            del self.block_id_to_leaf_node[curr_block_id]

        for i in range(len(blocks)):
            curr_block_node.add_request(req_id)
            parent_block_node = curr_block_node
            curr_block = blocks[i]
            curr_block_node = parent_block_node.add_child(curr_block.block_id)

        leaf_block_id = curr_block_node.block_id
        self.req_id_to_req_id_wrapper[req_id] = True
        self.req_id_to_block_id[req_id] = leaf_block_id
        self.block_id_to_req_id[leaf_block_id].add(req_id)
        self.block_id_to_leaf_node[leaf_block_id] = curr_block_node

    def remove(self, req_id: str):
        """
        Calls remove_child on nodes corresponding to given request.
        Args:
            req_id: Request id of the request which we remove from trie.
        """

        if req_id not in self.req_id_to_block_id:
            return
        del self.req_id_to_req_id_wrapper[req_id]
        block_id = self.req_id_to_block_id.pop(req_id)
        self.block_id_to_req_id[block_id].remove(req_id)
        if not self.block_id_to_req_id[block_id]:
            del self.block_id_to_req_id[block_id]
        node = self.block_id_to_leaf_node[block_id]

        curr_node = node
        if curr_node.block_id not in self.block_id_to_req_id:
            del self.block_id_to_leaf_node[curr_node.block_id]

        while curr_node and curr_node != self.sentinel:
            curr_node.remove_request(req_id)
            parent_node = curr_node.parent
            parent_node.remove_child(curr_node.block_id)
            curr_node = parent_node

    def allocate_group(self, alloc_mode: str)-> CommonPrefixGroups:
        return self.ALLOC_METHODS[alloc_mode]()

    def alloc_full_pass(self) -> CommonPrefixGroups:
        """
        Allocates groups of requests based on the weight of each node.
        Traverses the entire trie to find best groupings.
        Returns:
            A tuple consisting of:
                request_list: List of requests which we group in model runner.
                groups_list: List of indices of the form (num_common_prefix_blocks, start, end)
                where all requests in request_list[start: end+1] form a group.
        """
        request_list: list[tuple[str, bool]] = []
        groups_list: list[tuple[int, int, int]] = []

        stack: list[tuple[KVCacheBlockPrefixTrieNode, bool]] = [(self.sentinel, False)]
        while stack:
            node, visited = stack.pop()
            if node is None:
                continue
            if visited:
                # Post-order logic
                node.min_accessible_leaf_id = min(node.min_accessible_leaf_id, len(request_list))
                child_group_weights = sum(child.groups_weight for child in node.child_list)
                num_groups = sum(child.num_groups for child in node.child_list)
                start_direct_leaf = len(request_list)
                start_of_next_group = node.min_accessible_leaf_id + node.leaf_cnt
                if node.weight >= self.absorption_threshold_ratio * child_group_weights:
                    node.groups_weight = node.weight
                    del groups_list[-1 * num_groups:]
                    groups_list.append((node.depth, node.min_accessible_leaf_id, start_of_next_group - 1))
                    node.num_groups = 1
                else:
                    node.groups_weight = child_group_weights
                    groups_list.extend([(node.depth, i,i) for i in
                                        range(start_direct_leaf, start_of_next_group)])
                    node.num_groups = num_groups + start_of_next_group - len(request_list)
                request_list.extend(list(map(lambda k: (k, self.req_id_to_req_id_wrapper[k]), node.node_req_ids)))
            else:
                # Push current node back as visited (for post-order)
                stack.append((node, True))
                for i in range(len(node.child_list) - 1, -1, -1):
                    child_node = node.child_list[i]
                    stack.append((child_node, False))

                # Pre-order logic
                node.min_accessible_leaf_id = float('inf')

        return get_scheduled_requests(request_list, groups_list)

    def alloc_leaf_pass(self) -> CommonPrefixGroups:
        """
        Allocates groups of requests based on the weight of each node.
        Traverses only trie leaves to find best groupings.
        Returns:
            A tuple consisting of:
                request_list: List of requests which we group in model runner.
                groups_list: List of indices of the form (num_common_prefix_blocks, start, end)
                where all requests in request_list[start: end+1] form a group.
        """
        request_list: list[tuple[str, bool]] = []
        groups_list: list[tuple[int, int, int]] = []

        for block_id in self.block_id_to_req_id:
            groups_list.append((self.block_id_to_leaf_node[block_id].depth,
                                len(request_list),
                                len(request_list) + len(self.block_id_to_req_id[block_id]) - 1))
            request_list.extend(list(map(lambda k: (k, self.req_id_to_req_id_wrapper[k]),
                                         self.block_id_to_req_id[block_id])))

        return get_scheduled_requests(request_list, groups_list)

    def unschedule_request(self, request_id):
        self.req_id_to_req_id_wrapper[request_id] = False

class KVCacheBlockPrefixTrieNode:
    """Node that stores each KVCacheBlock still in use by
       the GPUModelRunner. The root of the Trie is a sentinel node which
       points to KVCacheBlocks with distinct prefixes. A node is the child of
       another Trie node if it contains the parent node as a prefix."""

    def __init__(self,
                 depth: int = 0,
                 parent: "KVCacheBlockPrefixTrieNode" = None,
                 block_id: int = 0,
                 min_accessible_leaf_id: float = 0,
                 leaf_cnt: int = 0):
        """
        Args:
            depth: The depth of the KVCacheBlock encompassed by this node.
            parent: The parent of this node.
            block_id: The id of the KVCacheBlock encompassed by this node.
            min_accessible_leaf_id: The minimum node_id of a leaf node traversable from this node.
        """
        self.depth: int = depth
        self.parent: KVCacheBlockPrefixTrieNode = parent
        self.child_map: dict[int, KVCacheBlockPrefixTrieNode] = {}
        self.child_list: list[KVCacheBlockPrefixTrieNode] = []
        self.node_req_ids: set[str] = set()
        self.block_id = block_id
        self.min_accessible_leaf_id = min_accessible_leaf_id
        self.num_groups = 0
        self.groups_weight = 0
        self.leaf_cnt = leaf_cnt

    def add_child(self, block_id: int) -> KVCacheBlockPrefixTrieNode:
        """
        Responsible for adding a child node corresponding to block with given
        block_id. Manages the leaf_cnt of the child_node. Only parent nodes can edit
        the leaf_cnt of their child_node.

        Args:
            block_id: The id of the block corresponding to the node we want to make
            a child.
        Returns:
            The child node we added to children of self.
        """

        # Nodes cannot edit their own leaf_cnt. Only the leaf_cnt of their children.
        if block_id in self.child_map:
            child_node = self.child_map[block_id]
            child_node.leaf_cnt += 1
            return child_node
        child_node = KVCacheBlockPrefixTrieNode(
            depth=self.depth + 1,
            parent=self,
            block_id=block_id,
            min_accessible_leaf_id=float('inf'),
            leaf_cnt=1)
        self.child_map[block_id] = child_node
        self.child_list.append(child_node)
        return child_node

    def remove_child(self, block_id: int):
        """
        Called when we want to remove a request, and all nodes associated with that
        request. Decrements ref_cnt of node with given block_id, and removes from
        trie if ref_cnt becomes 0.

        Args:
            block_id: The id of the block corresponding to the node we want to make
            a child.
        """
        if block_id not in self.child_map:
            return
        child_node = self.child_map[block_id]
        child_node.leaf_cnt -= 1
        if child_node.leaf_cnt == 0:
            del self.child_map[block_id]
            self.child_list.remove(child_node)
            child_node.parent = None

    def add_request(self, req_id: str):
        self.node_req_ids.add(req_id)

    def remove_request(self, req_id: str):
        self.node_req_ids.remove(req_id)

    @property
    def weight(self) -> float:
        """
        Returns:
            The importance given to a given node when allocating groups.
        """
        return self.depth * (self.leaf_cnt - 1)

class FreeKVCacheBlockQueue:
    """This class organizes a list of KVCacheBlock objects to a doubly linked
    list of free blocks. We implement this class instead of using Python
    builtin deque to support removing a block in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
    manipulating the linked list. Instead, this class manipulates the
    prev_free_block and next_free_block attributes of the given blocks.

    The queue is ordered by block ID in the beginning. When a block is allocated
    and then freed, it will be appended back with the eviction order:
    1. The least recent used block is at the front (LRU).
    2. If two blocks have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.

    Args:
        blocks: A list of KVCacheBlock objects.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)

        # Initialize the doubly linked list of free blocks.
        self.free_list_head: Optional[KVCacheBlock] = blocks[0]
        self.free_list_tail: Optional[KVCacheBlock] = blocks[-1]
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

    def popleft(self) -> KVCacheBlock:
        """Pop the first free block and reduce num_free_blocks by 1.

        Returns:
            The first free block.
        """
        if not self.free_list_head:
            raise ValueError("No free blocks available")

        block = self.free_list_head
        self.remove(block)
        return block

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block in the free list and reduce num_free_blocks by 1.

        Args:
            block: The block to remove.
        """
        if block.prev_free_block is not None:
            # Link the previous block to the next block.
            block.prev_free_block.next_free_block = block.next_free_block
        if block.next_free_block is not None:
            # Link the next block to the previous block.
            block.next_free_block.prev_free_block = block.prev_free_block

        if block == self.free_list_head:
            # Update the head if the block is the head.
            self.free_list_head = block.next_free_block
        if block == self.free_list_tail:
            # Update the tail if the block is the tail.
            self.free_list_tail = block.prev_free_block

        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list and increase
        num_free_blocks by 1.

        Args:
            block: The block to append.
        """
        if self.free_list_tail is not None:
            # Link the last block to the new block.
            self.free_list_tail.next_free_block = block
            block.prev_free_block = self.free_list_tail
            self.free_list_tail = block
        else:
            # The free list is empty.
            assert self.free_list_head is None
            self.free_list_head = self.free_list_tail = block

        block.next_free_block = None
        self.num_free_blocks += 1

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Get all free blocks in the free list. Mainly used for testing.

        Returns:
            A list of free blocks.
        """
        ret = []
        curr_block = self.free_list_head
        while curr_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret


def need_extra_keys(request: Request) -> bool:
    """Check whether the blocks allocated to this request need extra hash keys.

    Args:
        request (Request): The request.

    Returns:
        bool: Whether blocks allocated to this request need extra hash keys.
    """

    # Multimodal requests need to include the MM hash.
    # LoRA requests need to include the LoRA ID.
    # Request with provided cache salt need to include the salt.
    return bool(request.mm_positions) or (request.lora_request
                                          is not None) or (request.cache_salt
                                                           is not None)


def _gen_mm_extra_hash_keys(request: Request, start_token_idx: int,
                            end_token_idx: int,
                            start_mm_idx: int) -> tuple[list[Any], int]:
    """Generate extra keys related to MultiModal request for block hash
    computation. For multi-modal inputs, the extra keys are
    (mm_hash, start_offset) that indicate a mm input contained in the
    block and its starting offset in the block tokens.

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    extra_keys: list[Any] = []

    mm_positions, mm_hashes = request.mm_positions, request.mm_hashes
    if not mm_positions:
        return extra_keys, start_mm_idx

    if mm_positions and len(mm_positions) != len(mm_hashes):
        raise ValueError(
            "The number of multi-modal positions and hashes must match. This "
            "is likely because you do not enable MM preprocessor hashing. "
            "Please set disable_mm_preprocessor_cache=False.")

    # Note that we assume mm_positions is sorted by offset.
    # We do not need to check all mm inputs if the start token index is out of
    # range. This usually happens in the late prefill phase and decoding phase.
    if mm_positions[-1].offset + mm_positions[-1].length < start_token_idx:
        return extra_keys, start_mm_idx

    # Support start_mm_idx == -1 to indicate the last mm input.
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_positions)
        start_mm_idx = len(mm_positions) + start_mm_idx

    curr_mm_idx = start_mm_idx
    while mm_positions and curr_mm_idx < len(mm_positions):
        assert mm_hashes[curr_mm_idx] is not None
        offset = mm_positions[curr_mm_idx].offset
        length = mm_positions[curr_mm_idx].length
        if end_token_idx > offset:
            if start_token_idx > offset + length:
                # This block has passed the current mm input.
                curr_mm_idx += 1
                continue

            # The block contains the current mm input.
            extra_keys.append(mm_hashes[curr_mm_idx])

            if end_token_idx >= offset + length:
                # If this block contains the end of the current mm input,
                # move to the next mm input as this block may also contain
                # the next mm input.
                curr_mm_idx += 1
            else:
                # Otherwise this block is done with mm inputs.
                break
        else:
            # This block has not reached the current mm input.
            break
    return extra_keys, curr_mm_idx


def _gen_lora_extra_hash_keys(request: Request) -> list[int]:
    """Generate extra keys related to LoRA for block hash computation.

    Args:
        request: The request object.

    Returns:
        Return LoRA id of the request if it is a LoRA request. Return empty
        list otherwise.
    """
    if not request.lora_request:
        return []
    return [request.lora_request.lora_int_id]


def generate_block_hash_extra_keys(
        request: Request, start_token_idx: int, end_token_idx: int,
        start_mm_idx: int) -> tuple[Optional[tuple[Any, ...]], int]:
    """Generate extra keys for the block hash. The extra keys can come from
    the multi-modal inputs and request specific metadata (e.g., LoRA ID).

    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    mm_extra_keys: list[Any]
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx)
    lora_extra_keys: list[int] = _gen_lora_extra_hash_keys(request)
    cache_salt_keys: list[str] = [request.cache_salt] if (
        start_token_idx == 0 and request.cache_salt) else []

    extra_keys: list[Any] = lora_extra_keys + mm_extra_keys + cache_salt_keys

    if not extra_keys:
        return None, new_start_mm_idx

    return tuple(extra_keys), new_start_mm_idx


def hash_block_tokens(
        hash_function: Callable,
        parent_block_hash: Optional[int],
        curr_block_token_ids: Sequence[int],
        extra_keys: Optional[tuple[Any, ...]] = None) -> BlockHashType:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.

    Args:
        parent_block_hash: The hash of the parent block. None
            if this is the first block.
        curr_block_token_ids: A list of token ids in the current
            block. The current block is assumed to be full.
        extra_keys: Extra keys for the block.

    Returns:
        The hash value of the block and the token ids in the block.
        The entire tuple is used as the hash key of the block.
    """
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHashType(
        hash_function(
            (parent_block_hash, curr_block_token_ids_tuple, extra_keys)),
        curr_block_token_ids_tuple, extra_keys)


def hash_request_tokens(hash_function: Any, block_size: int,
                        request: Request) -> list[BlockHashType]:
    """Computes hash values of a chain of blocks given a sequence of
    token IDs. The hash value is used for prefix caching.

    Args:
        block_size: The size of each block.
        request: The request object.

    Returns:
        The list of computed hash values.
    """
    token_ids = request.all_token_ids

    req_need_extra_keys = need_extra_keys(request)
    req_extra_keys = None
    curr_mm_idx = 0

    ret = []
    parent_block_hash_value = None
    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        # Do not hash the block if it is not full.
        if len(block_token_ids) < block_size:
            break

        if req_need_extra_keys:
            # MM and LoRA requests need extra keys for block-hash computation.
            req_extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start, end, curr_mm_idx)

        block_hash = hash_block_tokens(hash_function, parent_block_hash_value,
                                       block_token_ids, req_extra_keys)
        ret.append(block_hash)
        parent_block_hash_value = block_hash.hash_value
    return ret


def estimate_max_model_len(vllm_config: VllmConfig,
                           kv_cache_spec: dict[str, KVCacheSpec],
                           available_memory: int) -> int:
    """
    Estimates the maximum model length that can fit in the available memory
    using binary search.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The estimated maximum model length that can fit in the available memory.
    """

    # Define a function to check if a given model length fits in memory
    def fits_in_memory(model_len: int) -> bool:
        # Modify the max_model_len for this calculation
        vllm_config.model_config.max_model_len = model_len
        # Calculate memory needed for the given model length
        memory_needed = sum(
            (layer_spec.max_memory_usage_bytes(vllm_config)
             for layer_spec in kv_cache_spec.values()),
            start=0,
        )
        return memory_needed <= available_memory

    # Binary search for the maximum model length
    current_max = vllm_config.model_config.max_model_len
    left, right = 1, current_max

    # If even the smallest model length doesn't fit, return 0
    if not fits_in_memory(left):
        return 0

    # Binary search for the maximum model length that fits
    result = 1
    while left <= right:
        mid = (left + right) // 2
        if fits_in_memory(mid):
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    return result


def check_enough_kv_cache_memory(vllm_config: VllmConfig,
                                 kv_cache_spec: dict[str, KVCacheSpec],
                                 available_memory: int):
    """
    Checks whether `available_memory` is enough for the KV cache to hold at
    least one request with the model's max_model_len.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Raises:
        ValueError: If there is not enough memory available for the KV cache.
    """

    if available_memory <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")

    max_model_len = vllm_config.model_config.max_model_len
    needed_memory = 0
    for layer_spec in kv_cache_spec.values():
        needed_memory += layer_spec.max_memory_usage_bytes(vllm_config)

    if needed_memory > available_memory:
        # Estimate the maximum model length that can fit in the available memory
        estimated_max_len = estimate_max_model_len(vllm_config, kv_cache_spec,
                                                   available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = " Based on the available memory,"
            f" the estimated maximum model length is {estimated_max_len}."

        raise ValueError(
            f"To serve at least one request with the models's max seq len "
            f"({max_model_len}), ({needed_memory/GiB_bytes:.2f} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({available_memory/GiB_bytes:.2f} GiB)."
            f"{estimated_msg} "
            f" Try increasing `gpu_memory_utilization` or decreasing "
            f"`max_model_len` when initializing the engine.")


def create_kv_cache_group_specs(
        kv_cache_spec: dict[str, KVCacheSpec],
        grouped_layer_names: list[list[str]]) -> list[KVCacheGroupSpec]:
    """
     Create KVCacheGroupSpec object for each kv cache group layer.
     The layers in the same group should share the same
     KVCacheSpec.

     Args:
         kv_cache_spec:
             A mapping from each layer name to its corresponding KVCacheSpec.
         grouped_layer_names:
             A list of kv cache groups, where each element is a list of layer
             names that belong to the same group and should share the same
             KVCacheSpec.
     Returns:
         A list of KVCacheGroupSpec objects, one for each group.
     """
    kv_cache_groups = []
    for layer_names_one_group in grouped_layer_names:
        layer_specs = [
            kv_cache_spec[layer_name] for layer_name in layer_names_one_group
        ]
        merged_layer_spec = layer_specs[0].merge(layer_specs)
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names_one_group, merged_layer_spec))
    return kv_cache_groups


def is_kv_cache_type_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same type of KV cache.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model

    Returns:
        True if all layers have the same type, False otherwise.
    """

    layer_keys = set(layer.type_id for layer in kv_cache_spec.values())
    return len(layer_keys) == 1


def _get_kv_cache_config_uniform_type(vllm_config: VllmConfig,
                                      kv_cache_spec: dict[str, KVCacheSpec],
                                      available_memory: int) -> KVCacheConfig:
    """
    Generates the KV cache configuration for a model with one type of KV cache.
    Divide the available memory equally among all layers.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The generated KVCacheConfig
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    assert len(page_sizes) == 1
    page_size = page_sizes.pop()

    num_blocks = int(available_memory // page_size // len(kv_cache_spec))
    num_blocks = max(num_blocks, 0)

    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks_override = \
            vllm_config.cache_config.num_gpu_blocks_override
        logger.info(
            "Overriding num_gpu_blocks=%d with "
            "num_gpu_blocks_override=%d", num_blocks, num_gpu_blocks_override)
        num_blocks = num_gpu_blocks_override

    num_tokens = num_blocks * vllm_config.cache_config.block_size
    num_tokens_str = f"{num_tokens:,}"
    logger.info("GPU KV cache size: %s tokens", num_tokens_str)
    max_model_len_str = f"{vllm_config.model_config.max_model_len:,}"
    max_concurrency = num_tokens / vllm_config.model_config.max_model_len
    logger.info("Maximum concurrency for %s tokens per request: %.2fx",
                max_model_len_str, max_concurrency)

    per_layer_size = page_size * num_blocks
    # All layers have the same KV cache spec, so we create one kv cache group
    # for all layers.
    grouped_layer_names = [list(kv_cache_spec.keys())]

    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        tensors={
            layer_name: KVCacheTensor(size=per_layer_size)
            for layer_name in kv_cache_spec
        },
        kv_cache_groups=create_kv_cache_group_specs(kv_cache_spec,
                                                    grouped_layer_names),
        forested_cascade_config=vllm_config.model_config.forested_cascade_config
    )
    return kv_cache_config


def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
    """
    Only models with one type of KV cache are supported yet. This function tries
    to convert the KV cache specs to one type if the model is a hybrid model
    with multiple type of KV cache. It will convert all SlidingWindowSpec to
    FullAttentionSpec if both types are present.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model
    """

    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in kv_cache_spec.values())
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values())
    if has_full_attention and has_sliding_window:
        for layer_name, spec in kv_cache_spec.items():
            if isinstance(spec, SlidingWindowSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    use_mla=spec.use_mla,
                    sliding_window=spec.sliding_window,
                )


def get_kv_cache_config(vllm_config: VllmConfig,
                        kv_cache_spec: dict[str, KVCacheSpec],
                        available_memory: int) -> KVCacheConfig:
    """
    Generates the KV cache configuration for a model
    TODO: support hybrid models with more than one type of KV cache.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The generated KVCacheConfigs
    """
    check_enough_kv_cache_memory(vllm_config, kv_cache_spec, available_memory)
    unify_hybrid_kv_cache_specs(kv_cache_spec)
    if is_kv_cache_type_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_config_uniform_type(vllm_config, kv_cache_spec,
                                                 available_memory)

    raise NotImplementedError


def unify_kv_cache_configs(kv_cache_configs: list[KVCacheConfig]):
    """
    Make the KV cache configurations for each worker consistent, so that all
    workers can be controlled by the same KVCacheManager.
    This function verifies that the layer group of each worker are the same,
    and changes the num_blocks of each worker to the smallest among all workers.

    Args:
        kv_cache_configs: The KV cache configurations for each worker. Will be
            in-place modified to make them consistent.
    """

    # Sort the kv cache groups by the type_id of their KV cache spec.
    # This can avoid the inconsistency caused by the order of groups.
    for kv_cache_config in kv_cache_configs:
        kv_cache_config.kv_cache_groups.sort(
            key=lambda x: x.kv_cache_spec.type_id)

    # Verify that the groups of each rank are the same.
    for kv_cache_config in kv_cache_configs[1:]:
        for group_rank_0, group_rank_i in zip(
                kv_cache_configs[0].kv_cache_groups,
                kv_cache_config.kv_cache_groups):
            assert group_rank_0.kv_cache_spec == group_rank_i.kv_cache_spec

    # Change the num_blocks of each rank to the smallest among all ranks. We
    # do not need to shrink the tensor size because it is valid to only use the
    # first `num_blocks` blocks of the tensor.
    min_num_blocks = min(kv_cache_config.num_blocks
                         for kv_cache_config in kv_cache_configs)
    for kv_cache_config in kv_cache_configs:
        kv_cache_config.num_blocks = min_num_blocks

    return kv_cache_configs
