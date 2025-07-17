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

# Timeout for ZMQ operations in seconds
ZMQ_TIMEOUT_SECONDS = 30


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
        
    logger.info("[DEBUG LOOKUP] Base URL: %s, RPC Port: %s, Role: %s, DP Rank: %s", 
                base_url, rpc_port, role, 
                vllm_config.parallel_config.data_parallel_rank if vllm_config else "N/A")
    path = f"ipc://{base_url}/lmcache_rpc_port_{rpc_port}"
    logger.info("[DEBUG LOOKUP] Generated ZMQ path: %s", path)
    return path


class LMCacheLookupClient(LookupClientInterface):
    """ZMQ-based lookup client that communicates with a lookup server."""

    def __init__(self, role: "KVConnectorRole", is_tp: bool, vllm_config: "VllmConfig"):
        logger.info("[DEBUG LOOKUP CLIENT] Initializing LMCacheLookupClient - role: %s, is_tp: %s", role, is_tp)
        
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lmcache(role, is_tp, vllm_config)
        
        logger.info("[DEBUG LOOKUP CLIENT] Creating ZMQ socket on path: %s", socket_path)
        try:
            self.socket = make_zmq_socket(
                self.ctx,
                socket_path,
                zmq.REQ,  # type: ignore[attr-defined]
                bind=False,
            )
            # Set timeout for send and receive operations to prevent deadlock
            self.socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT_SECONDS * 1000)  # milliseconds
            self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT_SECONDS * 1000)  # milliseconds
            logger.info("[DEBUG LOOKUP CLIENT] ZMQ socket created successfully with %ds timeout", ZMQ_TIMEOUT_SECONDS)
        except Exception as e:
            logger.error("[DEBUG LOOKUP CLIENT] Failed to create ZMQ socket: %s", e)
            raise

    def lookup(self, token_ids: torch.Tensor) -> int:
        logger.debug("[DEBUG LOOKUP CLIENT] Starting lookup for token_ids with shape: %s", token_ids.shape)
        try:
            request = self.encoder.encode(token_ids)
            logger.debug("[DEBUG LOOKUP CLIENT] Sending lookup request")
            
            # Send request with timeout
            self.socket.send_multipart(request, copy=False)
            logger.debug("[DEBUG LOOKUP CLIENT] Waiting for response")
            
            # Receive response with timeout
            resp = self.socket.recv()
            result = int.from_bytes(resp, "big")
            logger.debug("[DEBUG LOOKUP CLIENT] Lookup completed, result: %d", result)
            return result
        except zmq.Again:
            logger.warning("[DEBUG LOOKUP CLIENT] Lookup operation timed out after %ds, returning 0", ZMQ_TIMEOUT_SECONDS)
            # Return 0 to indicate no cache hit rather than blocking indefinitely
            return 0
        except Exception as e:
            logger.error("[DEBUG LOOKUP CLIENT] Lookup failed: %s", e)
            # Return 0 on any error to prevent blocking
            return 0

    def close(self):
        logger.info("[DEBUG LOOKUP CLIENT] Closing lookup client")
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
        logger.info("[DEBUG LOOKUP SERVER] Initializing LMCacheLookupServer - role: %s, is_tp: %s", role, is_tp)
        
        self.decoder = MsgpackDecoder(torch.Tensor)
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lmcache(role, is_tp, vllm_config)
        
        logger.info("[DEBUG LOOKUP SERVER] Creating ZMQ socket on path: %s", socket_path)
        try:
            self.socket = make_zmq_socket(
                self.ctx,
                socket_path,
                zmq.REP,  # type: ignore[attr-defined]
                bind=True,
            )
            # Set timeout for server operations as well
            self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT_SECONDS * 1000)  # milliseconds
            self.socket.setsockopt(zmq.SNDTIMEO, ZMQ_TIMEOUT_SECONDS * 1000)  # milliseconds
            logger.info("[DEBUG LOOKUP SERVER] ZMQ socket created successfully with %ds timeout", ZMQ_TIMEOUT_SECONDS)
        except Exception as e:
            logger.error("[DEBUG LOOKUP SERVER] Failed to create ZMQ socket: %s", e)
            raise

        self.lmcache_engine = lmcache_engine
        self.running = True

        def process_request():
            logger.info("[DEBUG LOOKUP SERVER] Starting request processing thread")
            while self.running:
                try:
                    logger.debug("[DEBUG LOOKUP SERVER] Waiting for request")
                    frames = self.socket.recv_multipart(copy=False)
                    logger.debug("[DEBUG LOOKUP SERVER] Received request, decoding")
                    token_ids = self.decoder.decode(frames)
                    logger.debug("[DEBUG LOOKUP SERVER] Decoded token_ids with shape: %s", token_ids.shape)
                    
                    result = self.lmcache_engine.lookup(token_ids, pin=True)
                    logger.debug("[DEBUG LOOKUP SERVER] Lookup result: %d", result)
                    
                    response = result.to_bytes(4, "big")
                    self.socket.send(response)
                    logger.debug("[DEBUG LOOKUP SERVER] Response sent")
                except zmq.Again:
                    logger.debug("[DEBUG LOOKUP SERVER] Receive operation timed out, continuing")
                    # Continue the loop on timeout rather than breaking
                    continue
                except Exception as e:
                    logger.error("[DEBUG LOOKUP SERVER] Error in request processing: %s", e)
                    if self.running:
                        # Send error response to avoid client hanging
                        try:
                            error_response = (0).to_bytes(4, "big")
                            self.socket.send(error_response)
                        except:
                            pass
                    break

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()
        logger.info("[DEBUG LOOKUP SERVER] Request processing thread started")

    def close(self):
        logger.info("[DEBUG LOOKUP SERVER] Closing lookup server")
        self.running = False
        self.socket.close(linger=0)
        # TODO: close the thread!
