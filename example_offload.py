"""
https://docs.lmcache.ai/getting_started/quickstart/offload_kv_cache.html#offload-kv-cache
"""

import os

# >>> not included in the quickstart rst

import random
import time
import getpass

# Generate a unique RPC port to avoid conflicts with other users
username = getpass.getuser()
pid = os.getpid()
# Use a hash of username and PID to generate a unique port number
import hashlib
unique_id = hashlib.md5(f"{username}_{pid}".encode()).hexdigest()[:8]
rpc_port = int(unique_id, 16) % 10000  # Keep it within a reasonable range

print(f"Using unique RPC port: {rpc_port} for user {username} (PID: {pid})")

# Wait a moment for the system to release resources
time.sleep(1)

## <<< not included in the quickstart rst

# Set token chunk size to 256
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
# Enable CPU memory backend
os.environ["LMCACHE_LOCAL_CPU"] = "True"
# Set CPU memory limit to 5GB
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Configure KV cache transfer to use LMCache
ktc = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",
    kv_connector_extra_config={"lmcache_rpc_port": rpc_port}
)

# Initialize LLM with LMCache configuration and DATA PARALLELISM
# This enables data parallelism to reproduce the issue
llm = LLM(model="facebook/opt-1.3b",
          kv_transfer_config=ktc,
          max_model_len=2048,  # OPT-1.3B has a context length of 2048 tokens
          gpu_memory_utilization=0.6,
          # Enable data parallelism with 2 workers to reproduce the issue
          data_parallel_size=2)

# Create example prompts with shared prefix
shared_prompt = "Hello, how are you?" * 100
prompts = [
    shared_prompt + "Hello, my name is",
]

# Define sampling parameters
sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

# Run inference
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")

from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME

LMCacheEngineBuilder.destroy(ENGINE_NAME)