import asyncio
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

# TODO: rm
import cloudpickle
# TODO: rm
import msgspec

import vllm.envs as envs
from vllm.executor.executor_base import (
    DistributedExecutorBase)  # yapf: disable
from vllm.executor.msgspec_utils import encode_hook
from vllm.executor.ray_utils import (RayWorkerWrapper, initialize_ray_cluster,
                                     ray)
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.platforms import current_platform
from vllm.sequence import ExecuteModelRequest
from vllm.utils import (_run_task_with_lock, get_distributed_init_method,
                        get_ip, get_open_port, make_async)

if ray is not None:
    from ray.actor import ActorHandle
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
else:
    ActorHandle = None

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)