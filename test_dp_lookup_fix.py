#!/usr/bin/env python3
"""
Test script to verify the data parallel lookup server sharing fix.
This script should no longer hang at "process prompts" stage.
"""

import os
import getpass
import hashlib
import logging

# Generate unique RPC port to avoid conflicts
username = getpass.getuser()
pid = os.getpid()
unique_id = hashlib.md5(f"{username}_{pid}".encode()).hexdigest()[:8]
rpc_port = int(unique_id, 16) % 10000

print(f"🔧 Using unique RPC port: {rpc_port}")

# LMCache configuration
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
os.environ["LMCACHE_LOCAL_CPU"] = "True" 
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "1.0"

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Configure KV cache transfer
ktc = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",
    kv_connector_extra_config={"lmcache_rpc_port": rpc_port}
)

# Enable debug logging to track the fix
logging.basicConfig(level=logging.INFO)

print("🚀 Creating LLM with data parallelism (data_parallel_size=2)...")
print("📋 Expected behavior:")
print("   - Only data_parallel_rank=0 should create lookup server")
print("   - All ranks should create lookup clients")
print("   - All clients should connect to rank 0's server via shared socket")
print("   - No hanging at 'process prompts' stage")
print()

try:
    llm = LLM(
        model="facebook/opt-1.3b",
        kv_transfer_config=ktc,
        max_model_len=512,
        gpu_memory_utilization=0.4,
        data_parallel_size=2,  # This triggers the DP setup
        trust_remote_code=True
    )
    
    print("✅ LLM created successfully!")
    print("🔍 Check the logs above for [DP Setup] messages")
    
    # Test inference to ensure lookup communication works
    print("\n🧪 Testing inference...")
    prompts = ["Hello, how are you?"]
    sampling_params = SamplingParams(temperature=0, max_tokens=5)
    
    print("⏳ Generating (should not hang)...")
    outputs = llm.generate(prompts, sampling_params)
    
    print("✅ Inference completed successfully!")
    for output in outputs:
        print(f"Generated: {output.outputs[0].text}")
        
    print("\n🎉 SUCCESS: DP lookup server sharing is working correctly!")
        
except Exception as e:
    print(f"❌ Error during LLM creation/inference: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 If this still hangs, check:")
    print("   1. ZMQ socket paths in logs")
    print("   2. Whether lookup server was created on rank 0")
    print("   3. Whether clients are connecting to correct path")
    
finally:
    # Cleanup
    print("\n🧹 Cleaning up...")
    try:
        from lmcache.v1.cache_engine import LMCacheEngineBuilder
        from lmcache.integration.vllm.utils import ENGINE_NAME
        LMCacheEngineBuilder.destroy(ENGINE_NAME)
    except:
        pass
    print("✅ Test completed.") 