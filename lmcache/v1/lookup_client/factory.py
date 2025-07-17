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

# First Party
from lmcache.integration.vllm.utils import lmcache_get_config
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.mooncake_lookup_client import MooncakeLookupClient

if TYPE_CHECKING:
    # Third Partyz
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    # First Party
    from lmcache.v1.lookup_client.lmcache_lookup_client import LMCacheLookupServer

logger = init_logger(__name__)


class LookupClientFactory:
    """Factory for creating lookup clients and servers based on configuration."""

    @staticmethod
    def create_lookup_client(
        role: "KVConnectorRole",
        is_tp: bool,
        vllm_config: "VllmConfig",
    ) -> LookupClientInterface:
        """
        Create a lookup client based on the configuration.

        Args:
            role: The KV connector role
            is_tp: Whether tensor parallelism is enabled
            vllm_config: The vLLM configuration

        Returns:
            A lookup client instance
        """
        logger.info(
            "[DEBUG FACTORY] Creating lookup client - role: %s, is_tp: %s, rank: %d, data_parallel_rank: %d, data_parallel_size: %d",
            role,
            is_tp,
            vllm_config.parallel_config.rank,
            vllm_config.parallel_config.data_parallel_rank,
            vllm_config.parallel_config.data_parallel_size,
        )
        
        config = lmcache_get_config()
        logger.info("[DEBUG FACTORY] LMCache config enable_p2p: %s", config.enable_p2p)

        # Check if external_lookup_client is configured
        if config.external_lookup_client is not None:
            logger.info("[DEBUG FACTORY] Using external lookup client: %s", config.external_lookup_client)
            return LookupClientFactory._create_external_lookup_client(
                config.external_lookup_client, role, is_tp, vllm_config
            )
        else:
            # First Party
            from lmcache.v1.lookup_client.lmcache_lookup_client import (
                LMCacheLookupClient,
            )

            logger.info("[DEBUG FACTORY] Creating LMCacheLookupClient")
            client = LMCacheLookupClient(role, is_tp, vllm_config)
            logger.info("[DEBUG FACTORY] LMCacheLookupClient created successfully")
            return client

    @staticmethod
    def create_lookup_server(
        lmcache_engine: LMCacheEngine,
        role: "KVConnectorRole",
        is_tp: bool,
        vllm_config: "VllmConfig",
    ) -> Optional["LMCacheLookupServer"]:
        """
        Create a lookup server based on the configuration.

        Args:
            lmcache_engine: The LMCache engine instance
            role: The KV connector role
            is_tp: Whether tensor parallelism is enabled
            vllm_config: The vLLM configuration

        Returns:
            A lookup server instance, or None if no server should be created
        """
        logger.info(
            "[DEBUG FACTORY] create_lookup_server called - role: %s, is_tp: %s, rank: %d, data_parallel_rank: %d, data_parallel_size: %d",
            role,
            is_tp,
            vllm_config.parallel_config.rank,
            vllm_config.parallel_config.data_parallel_rank,
            vllm_config.parallel_config.data_parallel_size,
        )
        
        config = lmcache_get_config()
        logger.info("[DEBUG FACTORY] LMCache config enable_p2p: %s, external_lookup_client: %s", config.enable_p2p, config.external_lookup_client)

        # Assert that when data parallelism is enabled, P2P search must also be enabled
        # assert not (
        #     vllm_config.parallel_config.data_parallel_size > 1 and not config.enable_p2p
        # ), "When data parallelism is enabled (data_parallel_size > 1), P2P search must also be enabled (enable_p2p = True)"

        # Only create the KV lookup API server on data parallel rank 0
        # when there are multiple workers and when not using external lookup client
        # Note: In data parallel setup, all workers have rank == 0, but data_parallel_rank 
        # correctly distinguishes between different DP workers
        should_create_server = (
            vllm_config.parallel_config.data_parallel_rank == 0
            and config.external_lookup_client is None
        )
        
        logger.info(
            "[DEBUG FACTORY] Should create lookup server: %s (data_parallel_rank == 0: %s, external_lookup_client is None: %s)",
            should_create_server,
            vllm_config.parallel_config.data_parallel_rank == 0,
            config.external_lookup_client is None,
        )
        
        if should_create_server:
            # First Party
            from lmcache.v1.lookup_client.lmcache_lookup_client import (
                LMCacheLookupServer,
            )

            logger.info("[DEBUG FACTORY] Creating LMCacheLookupServer")
            try:
                server = LMCacheLookupServer(lmcache_engine, role, is_tp, vllm_config)
                logger.info("[DEBUG FACTORY] LMCacheLookupServer created successfully")
                return server
            except Exception as e:
                logger.error("[DEBUG FACTORY] Failed to create LMCacheLookupServer: %s", e)
                raise

        logger.info("[DEBUG FACTORY] Not creating lookup server, returning None")
        return None

    @staticmethod
    def _create_external_lookup_client(
        external_lookup_uri: str,
        role: "KVConnectorRole",
        is_tp: bool,
        vllm_config: "VllmConfig",
    ) -> LookupClientInterface:
        """
        Create an external lookup client based on the URI format.

        Args:
            external_lookup_uri: URI in format <scheme>://<address>
            role: The KV connector role
            is_tp: Whether tensor parallelism is enabled
            vllm_config: The vLLM configuration

        Returns:
            A lookup client instance

        Raises:
            ValueError: If the URI format is unsupported
        """
        # Parse URI scheme and address
        if "://" not in external_lookup_uri:
            raise ValueError(
                f"Invalid external lookup client URI format: {external_lookup_uri}. "
                "Expected format: <scheme>://<address>"
            )

        scheme, address = external_lookup_uri.split("://", 1)

        # Route to appropriate client based on scheme
        if scheme == "mooncakestore":
            return LookupClientFactory._create_mooncake_lookup_client(
                address, role, is_tp, vllm_config
            )
        else:
            raise ValueError(
                f"Unsupported external lookup client scheme: {scheme}. "
                "Supported schemes: mooncakestore"
            )

    @staticmethod
    def _create_mooncake_lookup_client(
        master_address: str,
        role: "KVConnectorRole",
        is_tp: bool,
        vllm_config: "VllmConfig",
    ) -> "MooncakeLookupClient":
        """Create a MooncakeLookupClient instance."""
        # First Party
        from lmcache.v1.lookup_client.mooncake_lookup_client import (
            MooncakeLookupClient,
        )

        return MooncakeLookupClient(role, is_tp, vllm_config, master_address)
