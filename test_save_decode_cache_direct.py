"""
Direct test for save_decode_cache functionality
This script uses multiple detection methods:
1. Log message analysis
2. Cache statistics monitoring  
3. Storage backend inspection
4. Decode phase specific testing
"""

import os
import time
import getpass
import hashlib
import yaml
import logging
import sys
from io import StringIO

def setup_logging_capture():
    """Setup logging to capture LMCache messages"""
    # Create a string buffer to capture logs
    log_capture = StringIO()
    
    # Setup logging handler
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create handler for our string buffer
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return log_capture, handler, logger

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
    
    config_file = f"lmcache_config_{config_suffix}.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created config file: {config_file}")
    print(f"save_decode_cache: {save_decode_cache}")
    return config_file

def test_decode_cache_directly(config_file: str, test_name: str, rpc_port: int):
    """Test decode cache functionality with direct monitoring"""
    print(f"\n{'='*60}")
    print(f"DIRECT TEST: {test_name}")
    print(f"{'='*60}")
    
    # Setup log capture
    log_capture, handler, logger = setup_logging_capture()
    
    # Set environment
    os.environ["LMCACHE_CONFIG_FILE"] = config_file
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"
    
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.integration.vllm.utils import ENGINE_NAME
    
    # Configure KV cache transfer
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
        kv_connector_extra_config={"lmcache_rpc_port": rpc_port}
    )
    
    # Initialize LLM
    llm = LLM(
        model="facebook/opt-1.3b",
        kv_transfer_config=ktc,
        max_model_len=2048,
        gpu_memory_utilization=0.6
    )
    
    # Get cache engine for monitoring
    try:
        from lmcache.v1.cache_engine import LMCacheEngineBuilder
        cache_engine = LMCacheEngineBuilder.get(ENGINE_NAME)
        print(f"Cache engine obtained: {type(cache_engine)}")
        
        # Check initial configuration
        if hasattr(cache_engine, 'config'):
            print(f"save_decode_cache setting: {cache_engine.config.save_decode_cache}")
        
        # Check stats monitor
        if hasattr(cache_engine, 'stats_monitor'):
            stats_monitor = cache_engine.stats_monitor
            print(f"Stats monitor available: {type(stats_monitor)}")
        else:
            stats_monitor = None
            
    except Exception as e:
        print(f"Could not access cache engine: {e}")
        cache_engine = None
        stats_monitor = None
    
    # Test Phase 1: Long prefill to fill initial cache
    print(f"\n--- Phase 1: Initial prefill (should store cache) ---")
    base_prompt = "Hello, how are you? " * 100  # Long prompt for prefill
    
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        max_tokens=30  # Generate multiple tokens for decode phase
    )
    
    # Clear log buffer
    log_capture.seek(0)
    log_capture.truncate(0)
    
    # Get initial stats
    if stats_monitor:
        initial_stats = stats_monitor.get_stats_and_clear()
        print(f"Initial cache hit rate: {initial_stats.cache_hit_rate}")
    
    # Run first inference
    print("Running first inference...")
    start_time = time.time()
    outputs = llm.generate([base_prompt], sampling_params)
    first_time = time.time() - start_time
    
    print(f"First run completed in {first_time:.3f}s")
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated: {generated_text[:100]}...")
    
    # Check logs for storage messages
    log_content = log_capture.getvalue()
    print(f"\n--- Log Analysis (Phase 1) ---")
    if "Storing KV cache" in log_content:
        print("✅ Found 'Storing KV cache' message")
    else:
        print("❌ No 'Storing KV cache' message found")
    
    if "Store" in log_content and "tokens" in log_content:
        print("✅ Found cache store activity")
    else:
        print("❌ No cache store activity detected")
    
    # Get stats after first run
    if stats_monitor:
        after_first_stats = stats_monitor.get_stats_and_clear()
        print(f"Cache hit rate after first run: {after_first_stats.cache_hit_rate}")
        print(f"Store requests: {after_first_stats.interval_store_requests}")
        print(f"Hit tokens: {after_first_stats.interval_hit_tokens}")
    
    # Wait before next phase
    time.sleep(2)
    
    # Test Phase 2: Same base prompt + different ending (should trigger decode cache)
    print(f"\n--- Phase 2: Decode cache test (should hit decode cache if enabled) ---")
    
    # Use same base but different ending to test decode cache
    decode_test_prompt = base_prompt + " Tell me about AI."
    
    # Clear log buffer
    log_capture.seek(0)
    log_capture.truncate(0)
    
    print("Running decode cache test...")
    start_time = time.time()
    outputs = llm.generate([decode_test_prompt], sampling_params)
    second_time = time.time() - start_time
    
    print(f"Second run completed in {second_time:.3f}s")
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated: {generated_text[:100]}...")
    
    # Check logs for cache hits
    log_content = log_capture.getvalue()
    print(f"\n--- Log Analysis (Phase 2) ---")
    if "LMCache hit tokens" in log_content:
        print("✅ Found 'LMCache hit tokens' message")
    else:
        print("❌ No 'LMCache hit tokens' message found")
    
    if "need to load" in log_content:
        print("✅ Found cache loading activity")
    else:
        print("❌ No cache loading activity detected")
    
    # Get final stats
    if stats_monitor:
        final_stats = stats_monitor.get_stats_and_clear()
        print(f"Final cache hit rate: {final_stats.cache_hit_rate}")
        print(f"Retrieve requests: {final_stats.interval_retrieve_requests}")
        print(f"Hit tokens: {final_stats.interval_hit_tokens}")
        print(f"Requested tokens: {final_stats.interval_requested_tokens}")
    
    # Print all captured logs
    print(f"\n--- All Captured Logs ---")
    all_logs = log_capture.getvalue()
    if all_logs:
        print(all_logs)
    else:
        print("No logs captured")
    
    # Performance comparison
    speedup = first_time / second_time if second_time > 0 else 1.0
    print(f"\n--- Performance Results ---")
    print(f"First run: {first_time:.3f}s")
    print(f"Second run: {second_time:.3f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    # Cleanup
    logger.removeHandler(handler)
    try:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)
    except:
        pass
    
    return {
        'first_time': first_time,
        'second_time': second_time,
        'speedup': speedup,
        'logs': all_logs,
        'has_store_msg': "Storing KV cache" in all_logs,
        'has_hit_msg': "LMCache hit tokens" in all_logs,
        'stats': final_stats if stats_monitor else None
    }

def check_storage_backend():
    """Check if storage backend has any cached data"""
    print(f"\n--- Storage Backend Check ---")
    
    # Check common cache directories
    cache_dirs = [
        "/tmp/lmcache",
        "/tmp/gds/test-cache", 
        "/tmp/weka/test-cache",
        os.path.expanduser("~/.cache/lmcache")
    ]
    
    found_cache = False
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                files = os.listdir(cache_dir)
                if files:
                    print(f"✅ Found cache files in {cache_dir}: {len(files)} files")
                    found_cache = True
                    # Show some file names
                    for f in files[:5]:
                        print(f"   - {f}")
                    if len(files) > 5:
                        print(f"   ... and {len(files) - 5} more")
                else:
                    print(f"📁 Empty cache directory: {cache_dir}")
            except Exception as e:
                print(f"❌ Error accessing {cache_dir}: {e}")
        else:
            print(f"📁 Cache directory not found: {cache_dir}")
    
    if not found_cache:
        print("❌ No cache files found in any storage backend")
    
    return found_cache

def main():
    """Main test function with multiple detection methods"""
    print("COMPREHENSIVE save_decode_cache DETECTION TEST")
    print("=" * 60)
    
    # Setup
    rpc_port = generate_unique_rpc_port()
    
    # Check storage backend first
    check_storage_backend()
    
    # Test 1: save_decode_cache = True
    print(f"\n🧪 TESTING save_decode_cache = True")
    config_enabled = create_lmcache_config(save_decode_cache=True, config_suffix="enabled")
    results_enabled = test_decode_cache_directly(config_enabled, "save_decode_cache=True", rpc_port)
    
    time.sleep(3)
    
    # Test 2: save_decode_cache = False
    print(f"\n🧪 TESTING save_decode_cache = False")
    config_disabled = create_lmcache_config(save_decode_cache=False, config_suffix="disabled")
    results_disabled = test_decode_cache_directly(config_disabled, "save_decode_cache=False", rpc_port + 1000)
    
    # Final analysis
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\nsave_decode_cache=True Results:")
    print(f"  Performance: {results_enabled['first_time']:.3f}s → {results_enabled['second_time']:.3f}s (speedup: {results_enabled['speedup']:.2f}x)")
    print(f"  Store messages: {'✅' if results_enabled['has_store_msg'] else '❌'}")
    print(f"  Hit messages: {'✅' if results_enabled['has_hit_msg'] else '❌'}")
    
    print(f"\nsave_decode_cache=False Results:")
    print(f"  Performance: {results_disabled['first_time']:.3f}s → {results_disabled['second_time']:.3f}s (speedup: {results_disabled['speedup']:.2f}x)")
    print(f"  Store messages: {'✅' if results_disabled['has_store_msg'] else '❌'}")
    print(f"  Hit messages: {'✅' if results_disabled['has_hit_msg'] else '❌'}")
    
    # Final verdict
    print(f"\n{'='*80}")
    print("FINAL VERDICT")
    print(f"{'='*80}")
    
    evidence_count = 0
    evidence_list = []
    
    # Evidence 1: Log messages
    if results_enabled['has_store_msg'] and not results_disabled['has_store_msg']:
        evidence_count += 1
        evidence_list.append("Store messages only appear when save_decode_cache=True")
    
    # Evidence 2: Cache hit messages
    if results_enabled['has_hit_msg'] and not results_disabled['has_hit_msg']:
        evidence_count += 1
        evidence_list.append("Cache hit messages only appear when save_decode_cache=True")
    
    # Evidence 3: Performance difference
    speedup_diff = results_enabled['speedup'] - results_disabled['speedup']
    if speedup_diff > 0.2:
        evidence_count += 1
        evidence_list.append(f"Better speedup with save_decode_cache=True ({speedup_diff:.2f}x improvement)")
    
    # Evidence 4: Storage backend
    has_storage = check_storage_backend()
    if has_storage:
        evidence_count += 1
        evidence_list.append("Cache files found in storage backend")
    
    if evidence_count >= 2:
        print("✅ save_decode_cache APPEARS TO BE WORKING!")
        print(f"Evidence found ({evidence_count}/4):")
        for i, evidence in enumerate(evidence_list, 1):
            print(f"  {i}. {evidence}")
    elif evidence_count == 1:
        print("❓ save_decode_cache PARTIALLY WORKING")
        print(f"Limited evidence found ({evidence_count}/4):")
        for i, evidence in enumerate(evidence_list, 1):
            print(f"  {i}. {evidence}")
    else:
        print("❌ save_decode_cache NOT WORKING OR NOT DETECTABLE")
        print("No clear evidence found. Possible issues:")
        print("  - Feature not implemented")
        print("  - Test conditions not triggering decode cache")
        print("  - Logging not capturing the right messages")
    
    # Cleanup
    try:
        os.remove(config_enabled)
        os.remove(config_disabled)
    except:
        pass

if __name__ == "__main__":
    main() 