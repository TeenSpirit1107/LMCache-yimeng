# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import TYPE_CHECKING, Optional
import threading

# Third Party
from vllm.utils import make_zmq_socket
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
import torch
import vllm.envs as envs
import zmq

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

logger = init_logger(__name__)


def get_zmq_rpc_path_lmcache(
    role: "KVConnectorRole",
    is_tp: bool = False,
    vllm_config: Optional["VllmConfig"] = None,
) -> str:
    """Get the ZMQ RPC path for LMCache lookup communication."""
    base_url = envs.VLLM_RPC_BASE_PATH
    # Default to 0 if not configured
    rpc_port = 0
    if vllm_config is not None:
        rpc_port = vllm_config.kv_transfer_config.get_from_extra_config(
            "lmcache_rpc_port", 0
        )
    
    # In data parallel setup, all clients should connect to 
    # the lookup server on data_parallel_rank 0. We use a consistent socket
    # path that incorporates both the base rpc_port and ensures all DP workers
    # use the same path regardless of their individual rank.
    if vllm_config is not None and vllm_config.parallel_config.data_parallel_size > 1:
        # In DP setup, use a deterministic path that all workers can connect to
        # This ensures clients from all DP ranks connect to the server on rank 0
        socket_path = f"ipc://{base_url}/lmcache_rpc_port_{rpc_port}_dp_shared"
    else:
        # Single worker case - use original logic
        socket_path = f"ipc://{base_url}/lmcache_rpc_port_{rpc_port}"
    
    logger.debug("ZMQ Socket Path: %s, RPC Port: %s, DP Size: %s", 
                socket_path, rpc_port, 
                vllm_config.parallel_config.data_parallel_size if vllm_config else 1)
    return socket_path


class LMCacheLookupClient(LookupClientInterface):
    """ZMQ-based lookup client that communicates with a lookup server."""

    def __init__(self, role: "KVConnectorRole", is_tp: bool, vllm_config: "VllmConfig"):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lmcache(role, is_tp, vllm_config)
        
        # Store config for potential reconnection
        self.socket_path = socket_path
        self.vllm_config = vllm_config
        
        # Set socket timeout to prevent indefinite blocking
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )
        
        # Set receive timeout to 5 seconds to avoid hanging
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second send timeout
        
        # In DP setup, log which rank this client belongs to
        if vllm_config.parallel_config.data_parallel_size > 1:
            logger.info(
                f"[DP Setup] LookupClient on data_parallel_rank={vllm_config.parallel_config.data_parallel_rank} "
                f"connecting to shared lookup server at {socket_path}"
            )

    def lookup(self, token_ids: torch.Tensor) -> int:
        try:
            request = self.encoder.encode(token_ids)
            self.socket.send_multipart(request, copy=False)
            resp = self.socket.recv()
            result = int.from_bytes(resp, "big")
            return result
        except zmq.Again:  # Timeout occurred
            # In DP setup, rank 0's server might not be ready yet
            if (hasattr(self, 'vllm_config') and 
                self.vllm_config.parallel_config.data_parallel_size > 1 and
                self.vllm_config.parallel_config.data_parallel_rank != 0):
                logger.warning(
                    f"[DP Setup] Lookup timeout on data_parallel_rank={self.vllm_config.parallel_config.data_parallel_rank}. "
                    f"Lookup server on rank 0 may not be ready yet. Returning 0 tokens."
                )
                return 0  # No tokens found, safe fallback
            else:
                logger.error(f"ZMQ lookup timeout at {self.socket_path}")
                raise
        except Exception as e:
            logger.error(f"Error during lookup: {e}")
            return 0  # Safe fallback

    def close(self):
        self.socket.close(linger=0)


class LMCacheLookupServer:
    """ZMQ-based lookup server that handles lookup requests using LMCacheEngine."""

    def __init__(
        self,
        lmcache_engine: LMCacheEngine,
        role: "KVConnectorRole",
        is_tp: bool,
        vllm_config: "VllmConfig",
    ):
        self.decoder = MsgpackDecoder(torch.Tensor)
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lmcache(role, is_tp, vllm_config)
        
        # Enhanced logging for DP setup
        if vllm_config.parallel_config.data_parallel_size > 1:
            logger.info(
                f"[DP Setup] LookupServer starting on data_parallel_rank={vllm_config.parallel_config.data_parallel_rank} "
                f"(should only be rank 0) at socket {socket_path}"
            )
        
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

        self.lmcache_engine = lmcache_engine
        self.running = True

        def process_request():
            logger.info(f"[DP Setup] Lookup server request handler started at {socket_path}")
            while self.running:
                try:
                    frames = self.socket.recv_multipart(copy=False)
                    token_ids = self.decoder.decode(frames)
                    result = self.lmcache_engine.lookup(token_ids, pin=True)
                    response = result.to_bytes(4, "big")
                    self.socket.send(response)
                except Exception as e:
                    logger.error("Error in LMCache lookup server: %s", e)
                    # Continue running even if there's an error
                    continue

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()
        
        logger.info(f"[DP Setup] LookupServer successfully started at {socket_path}")

    def close(self):
        self.running = False
        self.socket.close(linger=0)
        # TODO: close the thread!
