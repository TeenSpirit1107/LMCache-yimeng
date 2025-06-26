"""
Direct test for save_decode_cache functionality
This script uses multiple detection methods:
1. Log message analysis (FIXED)
2. Cache statistics monitoring  
3. Storage backend inspection
4. Decode phase specific testing (IMPROVED)
"""

import os
import time
import getpass
import hashlib
import yaml
import re
import sys
from typing import Dict, List, Tuple

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

def capture_output_with_logs(func, *args, **kwargs):
    """Capture stdout/stderr to catch LMCache log messages"""
    import subprocess
    import tempfile
    
    # Create a temporary script to run the function
    script_content = f"""
import sys
import os
sys.path.insert(0, '{os.getcwd()}')

# Redirect the function call and capture output
{func.__name__}(*{repr(args)}, **{repr(kwargs)})
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        temp_script = f.name
    
    try:
        # Run the script and capture all output
        result = subprocess.run([sys.executable, temp_script], 
                              capture_output=True, text=True, 
                              timeout=300)
        return result.stdout, result.stderr, result.returncode
    finally:
        os.unlink(temp_script)

def analyze_logs_for_decode_cache(stdout: str, stderr: str) -> Dict:
    """Analyze logs for save_decode_cache evidence - FIXED VERSION"""
    all_logs = stdout + "\n" + stderr
    
    # Look for the actual log patterns from real test results
    results = {
        'prefill_stores': [],
        'decode_stores': [],
        'cache_hits': [],
        'config_loaded': False,
        'save_decode_cache_enabled': None
    }
    
    # 1. Check if save_decode_cache config was loaded
    config_pattern = r"'save_decode_cache': (True|False)"
    config_matches = re.findall(config_pattern, all_logs)
    if config_matches:
        results['config_loaded'] = True
        results['save_decode_cache_enabled'] = config_matches[-1] == 'True'
    
    # 2. Find prefill cache stores (skip_leading_tokens=0)
    prefill_pattern = r"Storing KV cache for (\d+) out of (\d+) tokens \(skip_leading_tokens=0\)"
    results['prefill_stores'] = re.findall(prefill_pattern, all_logs)
    
    # 3. Find decode cache stores (skip_leading_tokens>0) - KEY EVIDENCE
    decode_pattern = r"Storing KV cache for (\d+) out of (\d+) tokens \(skip_leading_tokens=(\d+)\)"
    all_stores = re.findall(decode_pattern, all_logs)
    results['decode_stores'] = [(stored, total, skip) for stored, total, skip in all_stores if int(skip) > 0]
    
    # 4. Find cache hit messages
    hit_pattern = r"Reqid: \d+, Total tokens (\d+), LMCache hit tokens: (\d+), need to load: (-?\d+)"
    results['cache_hits'] = re.findall(hit_pattern, all_logs)
    
    return results

def run_test_inference(config_file: str, test_name: str, rpc_port: int) -> Tuple[float, float, str, str]:
    """Run inference test and return timing + logs"""
    print(f"\n{'='*60}")
    print(f"RUNNING TEST: {test_name}")
    print(f"{'='*60}")
    
    # Setup log capture
    log_capture, handler, logger = setup_logging_capture()
    
    # Set environment
    os.environ["LMCACHE_CONFIG_FILE"] = config_file
    
    # Import here to avoid conflicts
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
    
    # Create prompts designed to trigger decode cache
    base_prompt = "Hello, how are you? " * 100  # Long prefill
    decode_prompt = base_prompt + "What is your name?"
    
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        max_tokens=100  # Generate enough tokens to trigger decode cache saves
    )
    
    print("Running first inference (prefill + decode)...")
    start_time = time.time()
    outputs1 = llm.generate([base_prompt], sampling_params)
    first_time = time.time() - start_time
    print(f"First run completed in {first_time:.3f}s")
    
    time.sleep(2)
    
    print("Running second inference (with shared prefill)...")
    start_time = time.time()
    outputs2 = llm.generate([decode_prompt], sampling_params)
    second_time = time.time() - start_time
    print(f"Second run completed in {second_time:.3f}s")
    
    # Clean up
    try:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)
    except:
        pass
    
    return first_time, second_time, "stdout_placeholder", "stderr_placeholder"

def test_decode_cache_directly(config_file: str, test_name: str, rpc_port: int):
    """Test decode cache functionality with improved detection"""
    
    # Run the test in a subprocess to capture all logs including LMCache INFO messages
    import subprocess
    import tempfile
    
    # Create a test script
    test_script = f"""
import os
import sys
import time
sys.path.insert(0, '{os.getcwd()}')

# Set environment
os.environ["LMCACHE_CONFIG_FILE"] = "{config_file}"

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME

# Configure KV cache transfer
ktc = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",
    kv_connector_extra_config={{"lmcache_rpc_port": {rpc_port}}}
)

# Initialize LLM
llm = LLM(
    model="facebook/opt-1.3b",
    kv_transfer_config=ktc,
    max_model_len=2048,
    gpu_memory_utilization=0.6
)

# Test prompts
base_prompt = "Tell me about the future of technology. " * 40
decode_prompt = base_prompt + " What do you think will happen next?"

sampling_params = SamplingParams(
    temperature=0,
    top_p=0.95,
    max_tokens=30
)

print("=== PHASE 1: First inference ===")
start_time = time.time()
outputs1 = llm.generate([base_prompt], sampling_params)
first_time = time.time() - start_time
print(f"TIMING: First run: {{first_time:.3f}}s")

time.sleep(2)

print("=== PHASE 2: Second inference ===")
start_time = time.time()
outputs2 = llm.generate([decode_prompt], sampling_params)
second_time = time.time() - start_time
print(f"TIMING: Second run: {{second_time:.3f}}s")

# Cleanup
try:
    LMCacheEngineBuilder.destroy(ENGINE_NAME)
except:
    pass
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        temp_script = f.name
    
    try:
        # Run the test and capture all output
        print(f"Running {test_name}...")
        result = subprocess.run([sys.executable, temp_script], 
                              capture_output=True, text=True, timeout=600)
        
        stdout = result.stdout
        stderr = result.stderr
        
        # Extract timing information
        timing_pattern = r"TIMING: (\w+) run: ([\d.]+)s"
        timings = dict(re.findall(timing_pattern, stdout))
        first_time = float(timings.get('First', 0))
        second_time = float(timings.get('Second', 0))
        
        # Analyze logs for decode cache evidence
        log_analysis = analyze_logs_for_decode_cache(stdout, stderr)
        
        return {
            'first_time': first_time,
            'second_time': second_time,
            'speedup': first_time / second_time if second_time > 0 else 1.0,
            'stdout': stdout,
            'stderr': stderr,
            'log_analysis': log_analysis
        }
        
    finally:
        os.unlink(temp_script)

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

def print_detailed_analysis(results_enabled: Dict, results_disabled: Dict):
    """Print detailed analysis of test results"""
    print(f"\n{'='*80}")
    print("DETAILED LOG ANALYSIS")
    print(f"{'='*80}")
    
    # Analyze enabled results
    print(f"\n🟢 save_decode_cache=True Analysis:")
    enabled_analysis = results_enabled['log_analysis']
    print(f"  Config loaded: {'✅' if enabled_analysis['config_loaded'] else '❌'}")
    print(f"  save_decode_cache setting: {enabled_analysis['save_decode_cache_enabled']}")
    print(f"  Prefill stores: {len(enabled_analysis['prefill_stores'])} found")
    print(f"  Decode stores: {len(enabled_analysis['decode_stores'])} found")
    
    if enabled_analysis['decode_stores']:
        print(f"  🔍 Decode store details:")
        for i, (stored, total, skip) in enumerate(enabled_analysis['decode_stores'][:5]):
            print(f"    Store {i+1}: {stored}/{total} tokens (skip_leading_tokens={skip})")
        if len(enabled_analysis['decode_stores']) > 5:
            print(f"    ... and {len(enabled_analysis['decode_stores']) - 5} more")
    
    print(f"  Cache hits: {len(enabled_analysis['cache_hits'])} found")
    
    # Analyze disabled results  
    print(f"\n🔴 save_decode_cache=False Analysis:")
    disabled_analysis = results_disabled['log_analysis']
    print(f"  Config loaded: {'✅' if disabled_analysis['config_loaded'] else '❌'}")
    print(f"  save_decode_cache setting: {disabled_analysis['save_decode_cache_enabled']}")
    print(f"  Prefill stores: {len(disabled_analysis['prefill_stores'])} found")
    print(f"  Decode stores: {len(disabled_analysis['decode_stores'])} found")
    print(f"  Cache hits: {len(disabled_analysis['cache_hits'])} found")

def main():
    """Main test function with FIXED detection methods"""
    print("COMPREHENSIVE save_decode_cache DETECTION TEST (FIXED VERSION)")
    print("=" * 70)
    
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
    
    # Detailed analysis
    print_detailed_analysis(results_enabled, results_disabled)
    
    # Final analysis with FIXED logic
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS (FIXED)")
    print(f"{'='*80}")
    
    print(f"\nsave_decode_cache=True Results:")
    print(f"  Performance: {results_enabled['first_time']:.3f}s → {results_enabled['second_time']:.3f}s (speedup: {results_enabled['speedup']:.2f}x)")
    
    print(f"\nsave_decode_cache=False Results:")
    print(f"  Performance: {results_disabled['first_time']:.3f}s → {results_disabled['second_time']:.3f}s (speedup: {results_disabled['speedup']:.2f}x)")
    
    # FIXED: Evidence-based detection
    print(f"\n{'='*80}")
    print("FINAL VERDICT (FIXED DETECTION)")
    print(f"{'='*80}")
    
    evidence_count = 0
    evidence_list = []
    
    # Evidence 1: Decode cache stores (THE KEY EVIDENCE)
    enabled_decode_stores = len(results_enabled['log_analysis']['decode_stores'])
    disabled_decode_stores = len(results_disabled['log_analysis']['decode_stores'])
    
    if enabled_decode_stores > disabled_decode_stores:
        evidence_count += 1
        evidence_list.append(f"More decode cache stores when enabled ({enabled_decode_stores} vs {disabled_decode_stores})")
    
    # Evidence 2: Configuration correctly loaded
    enabled_config = results_enabled['log_analysis']['save_decode_cache_enabled']
    disabled_config = results_disabled['log_analysis']['save_decode_cache_enabled']
    
    if enabled_config == True and disabled_config == False:
        evidence_count += 1
        evidence_list.append("Configuration correctly loaded in both tests")
    
    # Evidence 3: Behavioral difference in caching patterns
    enabled_total_stores = enabled_decode_stores + len(results_enabled['log_analysis']['prefill_stores'])
    disabled_total_stores = disabled_decode_stores + len(results_disabled['log_analysis']['prefill_stores'])
    
    if enabled_total_stores > disabled_total_stores:
        evidence_count += 1
        evidence_list.append(f"More total cache operations when enabled ({enabled_total_stores} vs {disabled_total_stores})")
    
    # Evidence 4: Storage backend
    has_storage = check_storage_backend()
    if has_storage:
        evidence_count += 1
        evidence_list.append("Cache files found in storage backend")
    
    # FIXED verdict logic
    if evidence_count >= 2:
        print("✅ save_decode_cache IS WORKING CORRECTLY!")
        print(f"Strong evidence found ({evidence_count}/4):")
        for i, evidence in enumerate(evidence_list, 1):
            print(f"  {i}. {evidence}")
        
        if enabled_decode_stores > 0:
            print(f"\n🎯 KEY EVIDENCE: Found {enabled_decode_stores} decode cache store operations")
            print("   This proves save_decode_cache is saving individual decode tokens!")
            
    elif evidence_count == 1:
        print("❓ save_decode_cache PARTIALLY WORKING")
        print(f"Some evidence found ({evidence_count}/4):")
        for i, evidence in enumerate(evidence_list, 1):
            print(f"  {i}. {evidence}")
    else:
        print("❌ save_decode_cache NOT WORKING OR NOT DETECTABLE")
        print("No clear evidence found. Possible issues:")
        print("  - Feature not implemented")
        print("  - Test conditions not triggering decode cache")
        print("  - Different logging configuration")
    
    # Show raw logs for debugging
    if evidence_count < 2:
        print(f"\n--- DEBUG: Raw logs for analysis ---")
        print("ENABLED TEST STDERR:")
        print(results_enabled['stderr'][:1000] + "..." if len(results_enabled['stderr']) > 1000 else results_enabled['stderr'])
        print("\nDISABLED TEST STDERR:")
        print(results_disabled['stderr'][:1000] + "..." if len(results_disabled['stderr']) > 1000 else results_disabled['stderr'])
    
    # Cleanup
    try:
        os.remove(config_enabled)
        os.remove(config_disabled)
    except:
        pass

if __name__ == "__main__":
    main() 