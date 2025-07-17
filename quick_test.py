#!/usr/bin/env python3
"""
Quick test script to rapidly identify working vs broken configurations.
Uses very short timeouts to quickly detect deadlocks.
"""

import subprocess
import time
import sys

def quick_test(config_name, timeout_seconds=30):
    """Run a quick test with short timeout"""
    print(f"\n🧪 QUICK TEST: {config_name} (timeout: {timeout_seconds}s)")
    print("-" * 50)
    
    cmd = ["python", "run_test.py", config_name]
    
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout_seconds,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ PASS: {config_name}")
            return True
        else:
            print(f"❌ FAIL: {config_name} (exit code: {result.returncode})")
            # Show last few lines of stderr for diagnosis
            if result.stderr:
                print("Last error output:")
                print(result.stderr.split('\n')[-5:])
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {config_name} - DEADLOCK DETECTED")
        return False
    except Exception as e:
        print(f"💥 ERROR: {config_name} - {e}")
        return False

def main():
    print("🚀 LMCache Quick Diagnosis Tool")
    print("Testing configurations with 30-second timeout to detect deadlocks quickly")
    print("=" * 70)
    
    # Test sequence - start with known working configurations
    configs = [
        "no_parallel_no_p2p",      # Should work (baseline)
        "parallel_no_p2p",         # Should work (data parallel only)  
        "no_parallel_p2p",         # Should work (P2P only)
        "parallel_p2p",            # Expected to hang (the problem case)
    ]
    
    results = {}
    total_start_time = time.time()
    
    for config in configs:
        start_time = time.time()
        success = quick_test(config, timeout_seconds=30)
        elapsed = time.time() - start_time
        results[config] = (success, elapsed)
        
        print(f"   Time taken: {elapsed:.1f}s")
        
        # If this config fails, we know the exact problem point
        if not success and "parallel_p2p" in config:
            print("🎯 CONFIRMED: Data Parallel + P2P causes deadlock!")
            break
    
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*70}")
    print(f"QUICK DIAGNOSIS COMPLETE ({total_time:.1f}s total)")
    print(f"{'='*70}")
    
    # Summary
    for config, (success, elapsed) in results.items():
        status = "✅ PASS" if success else "❌ FAIL/TIMEOUT"
        print(f"{status} {config:25} ({elapsed:.1f}s)")
    
    # Diagnosis
    print(f"\n💡 CONCLUSION:")
    baseline_works = results.get("no_parallel_no_p2p", (False, 0))[0]
    parallel_works = results.get("parallel_no_p2p", (False, 0))[0] 
    p2p_works = results.get("no_parallel_p2p", (False, 0))[0]
    combined_works = results.get("parallel_p2p", (False, 0))[0]
    
    if baseline_works and parallel_works and p2p_works and not combined_works:
        print("   🎯 Data Parallel + P2P incompatibility CONFIRMED")
        print("   📝 Recommendation: Use either Data Parallel OR P2P, not both")
    elif not parallel_works:
        print("   ⚠️  Data Parallel has issues")
    elif not p2p_works:
        print("   ⚠️  P2P has issues") 
    elif combined_works:
        print("   🤔 All configurations work - issue may be intermittent")
    else:
        print("   🔍 Need more investigation")
    
    print(f"\n⏱️  Total diagnosis time: {total_time:.1f}s")

if __name__ == "__main__":
    main() 