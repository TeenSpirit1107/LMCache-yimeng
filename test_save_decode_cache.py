"""
Test script to verify save_decode_cache functionality
This script compares performance with save_decode_cache enabled vs disabled
"""

import os
import time
import getpass
import hashlib
import tempfile
import yaml
from typing import Dict, Any

def generate_unique_rpc_port():
    """Generate a unique RPC port to avoid conflicts"""
    username = getpass.getuser()
    pid = os.getpid()
    unique_id = hashlib.md5(f"{username}_{pid}".encode()).hexdigest()[:8]
    rpc_port = int(unique_id, 16) % 10000
    print(f"Using unique RPC port: {rpc_port} for user {username} (PID: {pid})")
    return rpc_port

def create_lmcache_config(save_decode_cache: bool, config_suffix: str) -> str:
    """Create LMCache configuration file"""
    config = {
        'chunk_size': 256,
        'local_cpu': True,
        'max_local_cpu_size': 5.0,
        'save_decode_cache': save_decode_cache,
        'remote_serde': 'cachegen'
    }
    
    # Create temporary config file
    config_file = f"lmcache_config_{config_suffix}.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created config file: {config_file}")
    print(f"save_decode_cache: {save_decode_cache}")
    return config_file

def setup_environment_variables():
    """Setup basic environment variables"""
    env_vars = {
        "LMCACHE_CHUNK_SIZE": "256",
        "LMCACHE_LOCAL_CPU": "True", 
        "LMCACHE_MAX_LOCAL_CPU_SIZE": "5.0"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    print("Environment variables set up")

def run_inference_test(config_file: str, test_name: str, rpc_port: int):
    """Run inference test with given configuration"""
    print(f"\n{'='*50}")
    print(f"Running test: {test_name}")
    print(f"Config file: {config_file}")
    print(f"{'='*50}")
    
    # Set config file environment variable
    os.environ["LMCACHE_CONFIG_FILE"] = config_file
    
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.integration.vllm.utils import ENGINE_NAME
    
    # Configure KV cache transfer
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
        kv_connector_extra_config={"lmcache_rpc_port": rpc_port + hash(test_name) % 100}
    )
    
    # Initialize LLM
    llm = LLM(
        model="facebook/opt-1.3b",
        kv_transfer_config=ktc,
        max_model_len=2048,
        gpu_memory_utilization=0.6
    )
    
    # Create prompts that will trigger decode phase
    # First a long prompt for prefill, then shorter prompts for decode
    base_prompt = "Hello, how are you?" * 200  # Base prompt for prefill
    
    # Test prompts - these will test decode caching
    test_prompts = [
        base_prompt + " Tell me about artificial intelligence.",
        base_prompt + " What is machine learning?", 
        base_prompt + " Explain deep learning.",
    ]
    
    sampling_params = SamplingParams(
        temperature=0, 
        top_p=0.95, 
        max_tokens=20  # Generate enough tokens to trigger decode caching
    )
    
    results = {}
    
    # First run - this should store KV cache (prefill phase)
    print(f"\n--- First run (prefill + decode) ---")
    start_time = time.time()
    outputs = llm.generate([test_prompts[0]], sampling_params)
    first_run_time = time.time() - start_time
    
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    
    print(f"First run time: {first_run_time:.3f} seconds")
    results['first_run'] = first_run_time
    
    # Wait a moment before second run
    time.sleep(1)
    
    # Second run with same base prompt - this should benefit from decode cache if enabled
    print(f"\n--- Second run (should hit decode cache if enabled) ---")
    start_time = time.time()
    outputs = llm.generate([test_prompts[1]], sampling_params)  # Same base, different ending
    second_run_time = time.time() - start_time
    
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    
    print(f"Second run time: {second_run_time:.3f} seconds")
    results['second_run'] = second_run_time
    
    # Third run - another test
    print(f"\n--- Third run (should hit decode cache if enabled) ---")
    start_time = time.time()
    outputs = llm.generate([test_prompts[2]], sampling_params)  # Same base, different ending
    third_run_time = time.time() - start_time
    
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    
    print(f"Third run time: {third_run_time:.3f} seconds")
    results['third_run'] = third_run_time
    
    # Calculate average time for subsequent runs
    avg_subsequent_time = (second_run_time + third_run_time) / 2
    speedup = first_run_time / avg_subsequent_time if avg_subsequent_time > 0 else 1.0
    
    print(f"\n--- Results for {test_name} ---")
    print(f"First run time: {first_run_time:.3f}s")
    print(f"Average subsequent runs: {avg_subsequent_time:.3f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    results['avg_subsequent'] = avg_subsequent_time
    results['speedup'] = speedup
    
    # Cleanup
    try:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)
    except:
        pass
    
    return results

def main():
    """Main test function"""
    print("Testing save_decode_cache functionality")
    print("This test will compare performance with save_decode_cache enabled vs disabled")
    
    # Setup
    rpc_port = generate_unique_rpc_port()
    setup_environment_variables()
    
    # Wait for system to be ready
    time.sleep(1)
    
    # Test 1: save_decode_cache = True
    config_enabled = create_lmcache_config(save_decode_cache=True, config_suffix="enabled")
    results_enabled = run_inference_test(config_enabled, "save_decode_cache=True", rpc_port)
    
    # Wait between tests
    time.sleep(2)
    
    # Test 2: save_decode_cache = False  
    config_disabled = create_lmcache_config(save_decode_cache=False, config_suffix="disabled")
    results_disabled = run_inference_test(config_disabled, "save_decode_cache=False", rpc_port + 1000)
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"save_decode_cache=True:")
    print(f"  First run: {results_enabled['first_run']:.3f}s")
    print(f"  Avg subsequent: {results_enabled['avg_subsequent']:.3f}s") 
    print(f"  Speedup: {results_enabled['speedup']:.2f}x")
    
    print(f"\nsave_decode_cache=False:")
    print(f"  First run: {results_disabled['first_run']:.3f}s")
    print(f"  Avg subsequent: {results_disabled['avg_subsequent']:.3f}s")
    print(f"  Speedup: {results_disabled['speedup']:.2f}x")
    
    # Analysis
    enabled_speedup = results_enabled['speedup']
    disabled_speedup = results_disabled['speedup']
    speedup_improvement = enabled_speedup - disabled_speedup
    
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    print(f"Speedup improvement with save_decode_cache: {speedup_improvement:.2f}x")
    
    if speedup_improvement > 0.1:  # Threshold for meaningful improvement
        print("✅ save_decode_cache appears to be working!")
        print("   Enabled configuration shows better speedup for subsequent runs.")
    else:
        print("❓ save_decode_cache effect is unclear.")
        print("   May need longer prompts or more decode tokens to see the effect.")
    
    # Cleanup config files
    try:
        os.remove(config_enabled)
        os.remove(config_disabled)
        print(f"\nCleaned up config files")
    except:
        pass

if __name__ == "__main__":
    main() 