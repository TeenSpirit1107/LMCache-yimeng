#!/usr/bin/env python3
"""
Unified test script that supports different configurations for LMCache testing.
"""

import os
import time
import getpass
import hashlib
import argparse
from typing import Optional

from config import TEST_CONFIGS, COMMON_CONFIG

def generate_rpc_port(offset: int = 0) -> int:
    """Generate a unique RPC port based on username and process ID"""
    username = getpass.getuser()
    pid = os.getpid()
    unique_id = hashlib.md5(f"{username}_{pid}".encode()).hexdigest()[:8]
    return int(unique_id, 16) % 10000 + offset

def run_test(config_name: str, port_offset: int = 0):
    """Run test with specified configuration"""
    if config_name not in TEST_CONFIGS:
        raise ValueError(f"Unknown configuration: {config_name}. Available configs: {list(TEST_CONFIGS.keys())}")
    
    config = TEST_CONFIGS[config_name]
    print(f"\nRunning test with configuration: {config_name}")
    print(f"Description: {config['description']}")
    
    # Generate unique RPC port
    rpc_port = generate_rpc_port(port_offset)
    print(f"Using unique RPC port: {rpc_port} for user {getpass.getuser()} (PID: {os.getpid()})")
    
    redis_started = False  # Track if we started Redis
    
    # Set environment variables
    os.environ["LMCACHE_CHUNK_SIZE"] = COMMON_CONFIG["chunk_size"]
    os.environ["LMCACHE_LOCAL_CPU"] = str(COMMON_CONFIG["local_cpu"])
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = COMMON_CONFIG["max_local_cpu_size"]
    os.environ["LMCACHE_ENABLE_P2P"] = str(config["p2p_search"])
    
    # Use GPU 1 which is mostly free (GPU 0 is occupied by other processes)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print("DEBUG: Using GPU 1 to avoid memory conflicts")
    
    # Set P2P URLs when P2P is enabled
    if config["p2p_search"]:
        # Use the RPC port as base for P2P services
        lookup_port = rpc_port + 100
        distributed_port = rpc_port + 200
        os.environ["LMCACHE_LOOKUP_URL"] = f"localhost:{lookup_port}"
        os.environ["LMCACHE_DISTRIBUTED_URL"] = f"localhost:{distributed_port}"
        
        # Start Redis server for P2P lookup
        print(f"DEBUG: Starting Redis server on port {lookup_port} for P2P lookup...")
        import subprocess
        import time
        try:
            # Kill any existing Redis on this port
            subprocess.run(f"pkill -f 'redis-server.*{lookup_port}'", shell=True, capture_output=True)
            time.sleep(1)
            
            # Start Redis server in background
            redis_process = subprocess.Popen(
                ["redis-server", "--port", str(lookup_port), "--daemonize", "yes"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(2)  # Give Redis time to start
            
            # Test Redis connection
            test_result = subprocess.run(
                ["redis-cli", "-p", str(lookup_port), "ping"],
                capture_output=True,
                text=True
            )
            if test_result.returncode == 0 and "PONG" in test_result.stdout:
                print(f"DEBUG: Redis server successfully started on port {lookup_port}")
                redis_started = True
            else:
                print(f"WARNING: Redis server might not have started properly on port {lookup_port}")
                
        except Exception as e:
            print(f"WARNING: Failed to start Redis server: {e}")
            print("P2P mode may not work without Redis server")
        
        print(f"DEBUG: P2P enabled - lookup_url: {os.environ['LMCACHE_LOOKUP_URL']}, distributed_url: {os.environ['LMCACHE_DISTRIBUTED_URL']}")
    
    # Debug: Print key configuration settings
    print(f"DEBUG: LMCACHE_ENABLE_P2P = {os.environ['LMCACHE_ENABLE_P2P']}")
    print(f"DEBUG: Expected P2P setting = {config['p2p_search']}")
    print(f"DEBUG: Data parallel = {config['data_parallel']}")
    
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    
    # Configure KV cache transfer
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
        kv_connector_extra_config={
            "lmcache_rpc_port": rpc_port,
            "lmcache_p2p_search": config["p2p_search"]
        }
    )
    
    # Create LLM with appropriate configuration
    llm_config = {
        "model": COMMON_CONFIG["model"],
        "kv_transfer_config": ktc,
        "max_model_len": COMMON_CONFIG["max_model_len"],
        "gpu_memory_utilization": COMMON_CONFIG["gpu_memory_utilization"],
        "trust_remote_code": True
    }
    
    if config["data_parallel"]:
        llm_config["data_parallel_size"] = config["data_parallel_size"]
    
    print(f"Creating LLM with configuration: data_parallel={config['data_parallel']}, p2p_search={config['p2p_search']}")
    llm = LLM(**llm_config)
    
    print("LLM created successfully, starting inference...")
    
    # Simple test prompt
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
    
    # Clean up Redis server if P2P was used and we started it
    if config["p2p_search"] and redis_started:
        lookup_port = rpc_port + 100
        print(f"DEBUG: Stopping Redis server on port {lookup_port}...")
        try:
            import subprocess
            # Gracefully shutdown Redis
            subprocess.run(
                ["redis-cli", "-p", str(lookup_port), "shutdown"],
                capture_output=True,
                timeout=5
            )
            # Force kill if still running
            subprocess.run(f"pkill -f 'redis-server.*{lookup_port}'", shell=True, capture_output=True)
            print(f"DEBUG: Redis server on port {lookup_port} stopped")
        except Exception as e:
            print(f"WARNING: Failed to stop Redis server: {e}")
    
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.integration.vllm.utils import ENGINE_NAME
    LMCacheEngineBuilder.destroy(ENGINE_NAME)
    print("Test completed.")

def main():
    parser = argparse.ArgumentParser(description="Run LMCache tests with different configurations")
    parser.add_argument("config", choices=list(TEST_CONFIGS.keys()), help="Test configuration to use")
    parser.add_argument("--port-offset", type=int, default=0, help="Offset to add to RPC port (default: 0)")
    args = parser.parse_args()
    
    run_test(args.config, args.port_offset)

if __name__ == "__main__":
    main() 