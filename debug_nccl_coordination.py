#!/usr/bin/env python3
"""
NCCL Coordination Diagnostic Tool
Tests the specific issue where data parallel EngineCore processes fail to coordinate properly.
"""

import os
import time
import signal
import psutil
import subprocess
from typing import List, Dict

def monitor_process_tree(parent_pid: int, duration: int = 60) -> Dict:
    """Monitor parent process and all its children"""
    
    def get_process_info(pid):
        try:
            proc = psutil.Process(pid)
            return {
                'pid': pid,
                'name': proc.name(),
                'status': proc.status(),
                'cpu_percent': proc.cpu_percent(),
                'memory_percent': proc.memory_percent(),
                'cmdline': ' '.join(proc.cmdline()[:3]),  # First 3 args only
                'children': [child.pid for child in proc.children(recursive=False)]
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {'pid': pid, 'status': 'GONE'}
    
    results = {'samples': [], 'engine_cores': set()}
    
    for i in range(duration):
        sample = {
            'timestamp': time.time(),
            'parent': get_process_info(parent_pid),
            'children': {}
        }
        
        try:
            parent = psutil.Process(parent_pid)
            for child in parent.children(recursive=True):
                child_info = get_process_info(child.pid)
                sample['children'][child.pid] = child_info
                
                # Track EngineCore processes
                if 'EngineCore' in child_info.get('name', ''):
                    results['engine_cores'].add(child.pid)
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
            
        results['samples'].append(sample)
        time.sleep(1)
    
    return results

def test_data_parallel_minimal():
    """Run a minimal data parallel test to reproduce the hang"""
    
    print("🔍 MINIMAL DATA PARALLEL TEST")
    print("=" * 50)
    
    # Simple test script
    test_script = '''
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

try:
    from vllm import LLM, SamplingParams
    
    print(f"[{time.time():.1f}] Creating LLM with data_parallel_size=2...")
    llm = LLM(
        model="facebook/opt-1.3b",
        data_parallel_size=2,
        max_model_len=64,  # Smaller for faster test
        gpu_memory_utilization=0.3
    )
    
    print(f"[{time.time():.1f}] LLM created, starting generation...")
    prompts = ["Hello"]
    sampling_params = SamplingParams(temperature=0, max_tokens=3)
    
    print(f"[{time.time():.1f}] Calling generate()...")
    outputs = llm.generate(prompts, sampling_params)
    print(f"[{time.time():.1f}] Generation completed!")
    print(f"Result: {outputs[0].outputs[0].text}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
'''
    
    with open('/tmp/minimal_dp_test.py', 'w') as f:
        f.write(test_script)
    
    print("📝 Running minimal data parallel test...")
    print("   Expected: Hang at generation")
    print("   Monitoring for 120 seconds...")
    
    try:
        # Start the test process
        proc = subprocess.Popen(
            ['python', '/tmp/minimal_dp_test.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor the process tree
        monitor_data = monitor_process_tree(proc.pid, duration=120)
        
        # Try to terminate gracefully, then force kill
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        
        # Analyze results
        analyze_coordination_failure(monitor_data)
        
    except Exception as e:
        print(f"Test execution error: {e}")

def analyze_coordination_failure(monitor_data: Dict):
    """Analyze monitoring data to identify coordination issues"""
    
    print("\n🔍 COORDINATION ANALYSIS")
    print("=" * 50)
    
    engine_cores = monitor_data['engine_cores']
    print(f"EngineCore processes detected: {sorted(engine_cores)}")
    
    if len(engine_cores) != 2:
        print(f"❌ Expected 2 EngineCore processes, found {len(engine_cores)}")
        return
    
    # Analyze CPU activity patterns
    print("\n📊 CPU Activity Analysis:")
    for sample in monitor_data['samples'][-10:]:  # Last 10 samples
        timestamp = sample['timestamp']
        print(f"\nTime {timestamp:.1f}:")
        
        for pid in sorted(engine_cores):
            if pid in sample['children']:
                child = sample['children'][pid]
                cpu = child.get('cpu_percent', 0)
                status = child.get('status', 'unknown')
                print(f"  EngineCore {pid}: CPU={cpu:.1f}%, Status={status}")
    
    # Look for signs of deadlock
    print("\n🚨 DEADLOCK INDICATORS:")
    
    # Check if both processes are idle
    final_sample = monitor_data['samples'][-1] if monitor_data['samples'] else None
    if final_sample:
        idle_cores = []
        for pid in engine_cores:
            if pid in final_sample['children']:
                cpu = final_sample['children'][pid].get('cpu_percent', 0)
                if cpu < 1.0:  # Less than 1% CPU
                    idle_cores.append(pid)
        
        if len(idle_cores) == len(engine_cores):
            print("  ❌ ALL EngineCore processes are idle - likely deadlocked")
        else:
            print(f"  ⚠️  {len(idle_cores)}/{len(engine_cores)} EngineCore processes idle")

def analyze_log_patterns():
    """Analyze existing log files for patterns"""
    
    print("\n📋 LOG PATTERN ANALYSIS")
    print("=" * 50)
    
    patterns = {
        'success_single': ['no_para_no_p2p_succ', 'no_para_p2p_succ'],
        'failure_parallel': ['para_no_p2p_fail', 'para_p2p']
    }
    
    for category, files in patterns.items():
        print(f"\n{category.upper()}:")
        
        for pattern in files:
            matching_files = [f for f in os.listdir('.') if pattern in f and f.endswith('.log')]
            
            for logfile in matching_files:
                try:
                    with open(logfile, 'r') as f:
                        content = f.read()
                        
                    # Count EngineCore mentions
                    engine_0_count = content.count('EngineCore_0')
                    engine_1_count = content.count('EngineCore_1')
                    
                    # Count LMCache requests
                    lmcache_req_count = content.count('LMCache INFO] Reqid:')
                    
                    # Check final status
                    success = 'Generation completed successfully!' in content
                    timeout = 'Generation timed out' in content
                    
                    print(f"  📄 {logfile}:")
                    print(f"    EngineCore_0 mentions: {engine_0_count}")
                    print(f"    EngineCore_1 mentions: {engine_1_count}")
                    print(f"    LMCache requests: {lmcache_req_count}")
                    print(f"    Result: {'✅ SUCCESS' if success else '❌ TIMEOUT' if timeout else '❓ UNKNOWN'}")
                    
                except Exception as e:
                    print(f"  Error reading {logfile}: {e}")

if __name__ == "__main__":
    print("🚀 NCCL COORDINATION DIAGNOSTIC")
    print("Investigating data parallel hanging issue")
    print("=" * 60)
    
    # First analyze existing logs
    analyze_log_patterns()
    
    # Then run minimal test
    print("\n" + "=" * 60)
    test_data_parallel_minimal()
    
    print("\n🎯 DIAGNOSIS COMPLETE")
    print("Check output above for coordination failure analysis.") 