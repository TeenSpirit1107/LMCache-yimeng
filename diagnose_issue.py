#!/usr/bin/env python3
"""
Diagnostic script to identify the data_parallel + P2P hanging issue.
Runs multiple test configurations to isolate the problem.
"""

import subprocess
import time
import sys
import os

def run_test_with_timeout(config_name, timeout_minutes=5):
    """Run a test configuration with a timeout"""
    print(f"\n{'='*60}")
    print(f"TESTING: {config_name}")
    print(f"{'='*60}")
    
    cmd = ["python", "run_test.py", config_name]
    
    try:
        # Run with timeout
        result = subprocess.run(
            cmd,
            timeout=timeout_minutes * 60,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {config_name} completed successfully")
            return True
        else:
            print(f"❌ FAILED: {config_name} failed with return code {result.returncode}")
            print("STDERR:")
            print(result.stderr[-1000:])  # Last 1000 chars of stderr
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {config_name} timed out after {timeout_minutes} minutes")
        print("This indicates hanging/deadlock")
        return False
    except Exception as e:
        print(f"💥 ERROR: {config_name} failed with exception: {e}")
        return False

def main():
    print("LMCache Data Parallel + P2P Diagnostic Tool")
    print("=" * 50)
    
    # Test configurations in order of complexity
    test_sequence = [
        ("no_parallel_no_p2p", "Baseline: No parallel, No P2P"),
        ("parallel_no_p2p", "Data Parallel Only"),
        ("no_parallel_p2p", "P2P Only"),
        ("parallel_p2p_isolated", "DIAGNOSTIC: Data Parallel + P2P (Isolated)"),
        ("parallel_p2p", "PROBLEMATIC: Data Parallel + P2P (Shared)"),
    ]
    
    results = {}
    
    print("\nRunning diagnostic tests...")
    print("Each test has a 5-minute timeout to detect hanging.")
    
    for config_name, description in test_sequence:
        print(f"\n➡️  {description}")
        success = run_test_with_timeout(config_name, timeout_minutes=5)
        results[config_name] = success
        
        if not success and "parallel_p2p" in config_name:
            print(f"🔍 ANALYSIS: {config_name} failed - this confirms the issue")
            
        time.sleep(2)  # Brief pause between tests
    
    # Print summary
    print(f"\n{'='*60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    
    for config_name, description in test_sequence:
        status = "✅ PASS" if results[config_name] else "❌ FAIL"
        print(f"{status} {config_name}: {description}")
    
    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    parallel_only = results.get("parallel_no_p2p", False)
    p2p_only = results.get("no_parallel_p2p", False)
    parallel_p2p_shared = results.get("parallel_p2p", False)
    parallel_p2p_isolated = results.get("parallel_p2p_isolated", False)
    
    if parallel_only and p2p_only and not parallel_p2p_shared:
        print("🎯 CONFIRMED: Data Parallel + P2P conflict detected!")
        print("   - Data Parallel works independently ✅")
        print("   - P2P works independently ✅") 
        print("   - Combined they cause deadlock/hanging ❌")
        
        if parallel_p2p_isolated:
            print("   - Isolated instances fix works ✅")
            print("\n💡 SOLUTION: Use isolated LMCache instances for each DP rank")
        else:
            print("   - Isolated instances also fail ❌")
            print("\n💡 SOLUTION: Fundamental incompatibility - use only one feature")
            
    elif not parallel_only:
        print("⚠️  Data Parallel itself has issues")
        
    elif not p2p_only:
        print("⚠️  P2P itself has issues")
        
    else:
        print("🤔 Unexpected results - need further investigation")
    
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if parallel_only and p2p_only and not parallel_p2p_shared:
        if parallel_p2p_isolated:
            print("1. Use 'parallel_p2p_isolated' configuration for production")
            print("2. This uses separate LMCache instances per DP rank")
            print("3. Prevents resource conflicts between processes")
        else:
            print("1. Choose either Data Parallel OR P2P, not both")
            print("2. For large models: Use Data Parallel (parallel_no_p2p)")
            print("3. For cache sharing: Use P2P only (no_parallel_p2p)")
    else:
        print("1. Check individual component configurations")
        print("2. Review system resources and dependencies")
        print("3. Check CUDA and NCCL setup")

if __name__ == "__main__":
    main() 