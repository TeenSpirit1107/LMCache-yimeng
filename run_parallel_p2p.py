#!/usr/bin/env python3
"""
Test script with both data parallelism and p2p search enabled.
"""

import os
import time
import getpass
import hashlib

# Generate a unique RPC port 
username = getpass.getuser()
pid = os.getpid()
unique_id = hashlib.md5(f"{username}_{pid}".encode()).hexdigest()[:8]
rpc_port = int(unique_id, 16) % 10000

print(f"Using unique RPC port: {rpc_port} for user {username} (PID: {pid})")

# LMCache configuration
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
os.environ["LMCACHE_LOCAL_CPU"] = "True" 
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "1.0"  # Reduced memory
os.environ["LMCACHE_P2P_SEARCH"] = "True"  # Enable p2p search

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Configure KV cache transfer
ktc = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",
    kv_connector_extra_config={
        "lmcache_rpc_port": rpc_port,
        "lmcache_p2p_search": True  # Enable p2p search in connector config
    }
)

print("Creating LLM with data parallelism and p2p search...")
llm = LLM(model="facebook/opt-1.3b",
          kv_transfer_config=ktc,
          max_model_len=512,  # Much smaller to reduce load time
          gpu_memory_utilization=0.4,  # Reduced GPU usage
          data_parallel_size=2,
          trust_remote_code=True)

print("LLM created successfully, starting inference...")

# Very simple prompt to minimize processing
prompts = ["Hello, how are you?"]
sampling_params = SamplingParams(temperature=0, max_tokens=5)

print("Starting generation...")
try:
    outputs = llm.generate(prompts, sampling_params)
    print("Generation completed successfully!")
    for output in outputs:
        print(f"Generated: {output.outputs[0].text}")
except Exception as e:
    print(f"Error during generation: {e}")
    import traceback
    traceback.print_exc()

print("Cleaning up...")
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME
LMCacheEngineBuilder.destroy(ENGINE_NAME)
print("Test completed.") 