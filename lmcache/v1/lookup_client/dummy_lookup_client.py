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

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

logger = init_logger(__name__)


class DummyLookupClient(LookupClientInterface):
    """
    Dummy lookup client that always returns 0 (no cache hit).
    Used for DP workers that don't need lookup functionality.
    """

    def __init__(self):
        logger.info("[DEBUG DUMMY CLIENT] Initializing DummyLookupClient")

    def lookup(self, token_ids: torch.Tensor) -> int:
        """
        Always return 0 indicating no cache hit.
        
        Args:
            token_ids: The token IDs to lookup (ignored)
            
        Returns:
            Always returns 0
        """
        logger.debug("[DEBUG DUMMY CLIENT] Dummy lookup called, returning 0")
        return 0

    def close(self) -> None:
        """No-op close method."""
        logger.info("[DEBUG DUMMY CLIENT] Closing DummyLookupClient (no-op)")
        pass

    def supports_producer_reuse(self) -> bool:
        """Return False as dummy client doesn't support any features."""
        return False 