#!/usr/bin/env python3
"""
Isolate Data Parallel Issue
Tests to determine if the problem is in vLLM data parallel itself or LMCache integration.
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path

def test_vllm_dp_without_lmcache():
    """Test pure vLLM data parallel without any LMCache integration"""
    
    print("🧪 TEST 1: Pure vLLM Data Parallel (NO LMCache)")
    print("=" * 60)
    
    test_script = '''
import os
import time
import sys

# Ensure no LMCache interference
os.environ.pop("LMCACHE_ENABLE_P2P", None)
os.environ.pop("LMCACHE_LOOKUP_URL", None)
os.environ.pop("LMCACHE_DISTRIBUTED_URL", None)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

try:
    print(f"[{time.time():.1f}] Importing vLLM...")
    from vllm import LLM, SamplingParams
    
    print(f"[{time.time():.1f}] Creating LLM with data_parallel_size=2 (NO KV transfer)...")
    llm = LLM(
        model="facebook/opt-1.3b",
        data_parallel_size=2,
        max_model_len=64,
        gpu_memory_utilization=0.3,
        trust_remote_code=True
        # NO kv_transfer_config - pure vLLM
    )
    
    print(f"[{time.time():.1f}] LLM created successfully!")
    print(f"[{time.time():.1f}] Starting generation...")
    
    prompts = ["Hello, how are you?"]
    sampling_params = SamplingParams(temperature=0, max_tokens=3)
    
    print(f"[{time.time():.1f}] Calling generate()...")
    outputs = llm.generate(prompts, sampling_params)
    
    print(f"[{time.time():.1f}] ✅ SUCCESS! Generation completed!")
    for i, output in enumerate(outputs):
        print(f"Result {i}: '{output.outputs[0].text.strip()}'")
        
except Exception as e:
    print(f"[{time.time():.1f}] ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    return run_test_with_timeout("pure_vllm_dp", test_script, timeout=120)

def test_vllm_dp_with_lmcache():
    """Test vLLM data parallel WITH LMCache integration"""
    
    print("🧪 TEST 2: vLLM Data Parallel + LMCache")
    print("=" * 60)
    
    test_script = '''
import os
import time
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# Minimal LMCache config (no P2P)
os.environ["LMCACHE_ENABLE_P2P"] = "False"
os.environ["LMCACHE_LOCAL_CPU"] = "True"

try:
    print(f"[{time.time():.1f}] Importing vLLM with LMCache integration...")
    from vllm import LLM, SamplingParams
    from vllm.v1.utils import KVTransferConfig
    
    print(f"[{time.time():.1f}] Creating KVTransferConfig...")
    kv_config = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_connector_extra_config={
            "lmcache_rpc_port": 12345,
            "lmcache_p2p_search": False
        }
    )
    
    print(f"[{time.time():.1f}] Creating LLM with data_parallel_size=2 + LMCache...")
    llm = LLM(
        model="facebook/opt-1.3b",
        data_parallel_size=2,
        max_model_len=64,
        gpu_memory_utilization=0.3,
        trust_remote_code=True,
        kv_transfer_config=kv_config  # LMCache integration
    )
    
    print(f"[{time.time():.1f}] LLM created successfully!")
    print(f"[{time.time():.1f}] Starting generation...")
    
    prompts = ["Hello, how are you?"]
    sampling_params = SamplingParams(temperature=0, max_tokens=3)
    
    print(f"[{time.time():.1f}] Calling generate()...")
    outputs = llm.generate(prompts, sampling_params)
    
    print(f"[{time.time():.1f}] ✅ SUCCESS! Generation completed!")
    for i, output in enumerate(outputs):
        print(f"Result {i}: '{output.outputs[0].text.strip()}'")
        
except Exception as e:
    print(f"[{time.time():.1f}] ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    return run_test_with_timeout("vllm_dp_lmcache", test_script, timeout=120)

def run_test_with_timeout(test_name: str, script_content: str, timeout: int):
    """Run a test script with timeout and monitor results"""
    
    script_path = f"/tmp/{test_name}.py"
    
    # Write test script
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"📝 Running {test_name} (timeout: {timeout}s)...")
    
    try:
        # Start process
        proc = subprocess.Popen(
            ['python', script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid  # Create new process group
        )
        
        start_time = time.time()
        output_lines = []
        
        # Monitor output with timeout
        while True:
            line = proc.stdout.readline()
            if line:
                output_lines.append(line.strip())
                print(f"  {line.strip()}")
            
            # Check if process finished
            if proc.poll() is not None:
                break
                
            # Check timeout
            if time.time() - start_time > timeout:
                print(f"⏰ TIMEOUT after {timeout}s - terminating...")
                
                # Kill entire process group
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    time.sleep(2)
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                
                return {
                    'success': False,
                    'timeout': True,
                    'duration': timeout,
                    'output': output_lines
                }
        
        # Process completed
        duration = time.time() - start_time
        return_code = proc.returncode
        
        result = {
            'success': return_code == 0,
            'timeout': False,
            'duration': duration,
            'return_code': return_code,
            'output': output_lines
        }
        
        if result['success']:
            print(f"✅ {test_name} PASSED in {duration:.1f}s")
        else:
            print(f"❌ {test_name} FAILED in {duration:.1f}s (code: {return_code})")
            
        return result
        
    except Exception as e:
        print(f"💥 {test_name} execution error: {e}")
        return {
            'success': False,
            'timeout': False,
            'error': str(e),
            'output': []
        }
    finally:
        # Cleanup
        if os.path.exists(script_path):
            os.remove(script_path)

def analyze_results(results):
    """Analyze test results to determine root cause"""
    
    print("\n🔍 ROOT CAUSE ANALYSIS")
    print("=" * 60)
    
    pure_vllm = results.get('pure_vllm_dp', {})
    vllm_lmcache = results.get('vllm_dp_lmcache', {})
    
    print(f"Pure vLLM DP:     {'✅ PASS' if pure_vllm.get('success') else '❌ FAIL'}")
    if pure_vllm.get('timeout'):
        print(f"                  ⏰ TIMEOUT after {pure_vllm.get('duration', 0)}s")
    elif 'duration' in pure_vllm:
        print(f"                  ⏱️  Completed in {pure_vllm['duration']:.1f}s")
    
    print(f"vLLM DP + LMCache: {'✅ PASS' if vllm_lmcache.get('success') else '❌ FAIL'}")
    if vllm_lmcache.get('timeout'):
        print(f"                  ⏰ TIMEOUT after {vllm_lmcache.get('duration', 0)}s")
    elif 'duration' in vllm_lmcache:
        print(f"                  ⏱️  Completed in {vllm_lmcache['duration']:.1f}s")
    
    print("\n🎯 CONCLUSION:")
    
    if pure_vllm.get('success') and not vllm_lmcache.get('success'):
        print("  ➤ 🎯 **LMCache integration breaks vLLM data parallel coordination**")
        print("  ➤ vLLM data parallel works fine by itself")
        print("  ➤ Problem is in LMCache connector or KV transfer logic")
        
    elif not pure_vllm.get('success') and not vllm_lmcache.get('success'):
        print("  ➤ 🎯 **vLLM data parallel itself has issues**")
        print("  ➤ Problem is in vLLM v1 data parallel implementation")
        print("  ➤ LMCache is not the root cause")
        
    elif pure_vllm.get('success') and vllm_lmcache.get('success'):
        print("  ➤ ❓ **Both tests passed - need to investigate other factors**")
        print("  ➤ The issue may be environment-specific or intermittent")
        
    else:
        print("  ➤ ❓ **Unexpected result pattern - need manual investigation**")
    
    print("\n📋 NEXT STEPS:")
    if pure_vllm.get('success') and not vllm_lmcache.get('success'):
        print("  1. Investigate LMCache KVConnectorBase_V1 initialization")
        print("  2. Check LMCache v1 adapter NCCL interference")
        print("  3. Look for LMCache blocking calls in data parallel context")
        print("  4. Consider disabling LMCache for data parallel mode")

if __name__ == "__main__":
    print("🚀 DATA PARALLEL ISOLATION TEST")
    print("Testing to isolate whether the issue is vLLM DP or LMCache integration")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Pure vLLM data parallel
    results['pure_vllm_dp'] = test_vllm_dp_without_lmcache()
    
    print("\n" + "="*80)
    
    # Test 2: vLLM data parallel + LMCache
    results['vllm_dp_lmcache'] = test_vllm_dp_with_lmcache()
    
    print("\n" + "="*80)
    
    # Analyze results
    analyze_results(results) 