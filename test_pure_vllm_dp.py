#!/usr/bin/env python3
"""
Pure vLLM Data Parallel Test
Tests vLLM data parallel functionality without any LMCache integration.
"""

import os
import time
import sys

def print_timestamp(message):
    """Print message with timestamp"""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}")

def main():
    print_timestamp("🚀 Starting Pure vLLM Data Parallel Test")
    print("=" * 60)
    
    # Environment setup - same as previous tests
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    # Ensure no LMCache interference
    lmcache_vars = [
        "LMCACHE_ENABLE_P2P", "LMCACHE_LOOKUP_URL", "LMCACHE_DISTRIBUTED_URL",
        "LMCACHE_CHUNK_SIZE", "LMCACHE_LOCAL_CPU", "LMCACHE_MAX_LOCAL_CPU_SIZE",
        "LMCACHE_LMCACHE_INSTANCE_ID"
    ]
    
    for var in lmcache_vars:
        if var in os.environ:
            del os.environ[var]
    
    print_timestamp("Environment cleaned - no LMCache variables")
    print_timestamp(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    try:
        print_timestamp("📦 Importing vLLM...")
        from vllm import LLM, SamplingParams
        
        print_timestamp("✅ vLLM imported successfully")
        
        print_timestamp("🔧 Creating LLM with data_parallel_size=2...")
        print("Configuration:")
        print("  - Model: facebook/opt-1.3b")
        print("  - Data Parallel Size: 2")
        print("  - Max Model Length: 128")
        print("  - GPU Memory Utilization: 0.4")
        print("  - NO KV Transfer Config (Pure vLLM)")
        
        llm = LLM(
            model="facebook/opt-1.3b",
            data_parallel_size=2,  # This is the key test parameter
            max_model_len=128,
            gpu_memory_utilization=0.4,
            trust_remote_code=True
            # NO kv_transfer_config - pure vLLM only
        )
        
        print_timestamp("✅ LLM created successfully!")
        
        # Test generation - same as previous tests
        print_timestamp("🎯 Preparing test generation...")
        prompts = ["Hello, how are you?"]
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=5,
            top_p=1.0
        )
        
        print_timestamp("📝 Starting generation...")
        print("This is where the hang typically occurs in data parallel mode...")
        
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        duration = time.time() - start_time
        
        print_timestamp(f"✅ SUCCESS! Generation completed in {duration:.2f} seconds")
        
        # Print results
        print("\n📋 Results:")
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            print(f"  Prompt {i}: '{prompts[i]}'")
            print(f"  Output {i}: '{generated_text}'")
        
        print_timestamp("🎉 Pure vLLM Data Parallel Test PASSED!")
        print("This means vLLM data parallel works fine by itself.")
        print("The issue is likely in LMCache integration.")
        
        return True
        
    except KeyboardInterrupt:
        print_timestamp("⚠️ Test interrupted by user")
        return False
        
    except Exception as e:
        print_timestamp(f"❌ ERROR: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        
        print_timestamp("💡 Analysis:")
        if "generate" in str(e).lower() or "timeout" in str(e).lower():
            print("- Generation phase failed - likely NCCL coordination issue")
            print("- Problem is in vLLM data parallel implementation")
            print("- LMCache is NOT the root cause")
        else:
            print("- Initialization phase failed")
            print("- Check CUDA/GPU setup or model loading")
        
        return False

if __name__ == "__main__":
    print("Pure vLLM Data Parallel Test")
    print("Tests if vLLM data parallel works without LMCache")
    print("Expected: Should work if LMCache is the problem")
    print("Expected: Should hang if vLLM data parallel has issues")
    print()
    
    success = main()
    
    print("\n" + "=" * 60)
    if success:
        print("🔍 CONCLUSION: vLLM data parallel works → LMCache integration issue")
    else:
        print("🔍 CONCLUSION: vLLM data parallel fails → vLLM implementation issue")
    
    sys.exit(0 if success else 1) 