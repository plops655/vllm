import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

# TODO: rm
import msgspec

from vllm.config import ParallelConfig
from vllm.executor.msgspec_utils import decode_hook, encode_hook
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sequence import ExecuteModelRequest, IntermediateTensors
from vllm.utils import get_ip
from vllm.worker.worker_base import WorkerWrapperBase

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)
PG_WAIT_TIMEOUT = 1800

try:
    import torch
    import torch.distributed
    from torch.distributed import init_process_group, init_device_mesh, is_initialized

class TorchWorkerWrapper(WorkerWrapperBase):
    def __init__(self):
        pass
        # Understand functionality of ray utils and comment functions to implement

    # get_node_ip. Why?
    # get_node_and_gpu_ids. Why? Device mesh.
    # execute_model_spmd. How?

    def initialize_torch_(self):

