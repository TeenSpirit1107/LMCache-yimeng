"""
Simple test to check if save_decode_cache configuration is being read correctly
"""

import os
import yaml
import tempfile

def test_config_loading():
    """Test if save_decode_cache configuration is loaded correctly"""
    print("Testing save_decode_cache configuration loading...")
    
    # Test 1: Environment variable
    print("\n1. Testing environment variable...")
    os.environ["LMCACHE_SAVE_DECODE_CACHE"] = "true"
    
    try:
        from lmcache.v1.config import LMCacheEngineConfig
        config = LMCacheEngineConfig.from_env()
        print(f"✅ save_decode_cache from env: {config.save_decode_cache}")
    except Exception as e:
        print(f"❌ Error loading config from env: {e}")
    
    # Test 2: Config file
    print("\n2. Testing config file...")
    config_data = {
        'chunk_size': 256,
        'local_cpu': True,
        'save_decode_cache': True
    }
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_file = f.name
    
    os.environ["LMCACHE_CONFIG_FILE"] = config_file
    
    try:
        config2 = LMCacheEngineConfig.from_env()
        print(f"✅ save_decode_cache from file: {config2.save_decode_cache}")
    except Exception as e:
        print(f"❌ Error loading config from file: {e}")
    
    # Cleanup
    os.unlink(config_file)
    
    # Test 3: Check vLLM adapter logic
    print("\n3. Testing vLLM adapter logic...")
    try:
        # Import the vLLM adapter where save_decode_cache logic is used
        from lmcache.integration.vllm.vllm_v1_adapter import ReqMeta
        print("✅ vLLM adapter imported successfully")
        
        # Check if the save_decode_cache logic exists in ReqMeta.from_request_tracker
        import inspect
        source = inspect.getsource(ReqMeta.from_request_tracker)
        if "save_decode_cache" in source:
            print("✅ save_decode_cache logic found in vLLM adapter")
        else:
            print("❌ save_decode_cache logic NOT found in vLLM adapter")
            
    except Exception as e:
        print(f"❌ Error checking vLLM adapter: {e}")

def test_cache_engine_initialization():
    """Test if cache engine properly initializes with save_decode_cache"""
    print("\n4. Testing cache engine initialization...")
    
    # Set up environment
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_SAVE_DECODE_CACHE"] = "True"
    
    try:
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.config import LMCacheEngineMetadata
        from lmcache.v1.cache_engine import LMCacheEngine
        
        # Create configuration
        config = LMCacheEngineConfig.from_env()
        metadata = LMCacheEngineMetadata(
            model_name="facebook/opt-1.3b",
            fmt="vllm",
            world_size=1,
            worker_id=0
        )
        
        # Initialize cache engine
        cache_engine = LMCacheEngine(config, metadata)
        
        print(f"✅ Cache engine initialized")
        print(f"   save_decode_cache setting: {cache_engine.config.save_decode_cache}")
        
        # Check storage manager
        if hasattr(cache_engine, 'storage_manager'):
            print(f"✅ Storage manager available: {type(cache_engine.storage_manager)}")
        else:
            print("❌ No storage manager found")
            
    except Exception as e:
        print(f"❌ Error initializing cache engine: {e}")
        import traceback
        traceback.print_exc()

def check_decode_logic():
    """Check if the decode cache logic exists in the codebase"""
    print("\n5. Checking decode cache logic in codebase...")
    
    try:
        # Check vLLM adapter for decode logic
        from lmcache.integration.vllm import vllm_adapter
        import inspect
        
        # Check lmcache_should_store function
        if hasattr(vllm_adapter, 'lmcache_should_store'):
            source = inspect.getsource(vllm_adapter.lmcache_should_store)
            if "save_decode_cache" in source:
                print("✅ save_decode_cache logic found in lmcache_should_store")
                
                # Check for DECODE handling
                if "DECODE" in source and "StoreStatus.DECODE" in source:
                    print("✅ DECODE status handling found")
                else:
                    print("❌ DECODE status handling NOT found")
            else:
                print("❌ save_decode_cache logic NOT found in lmcache_should_store")
        else:
            print("❌ lmcache_should_store function not found")
            
    except Exception as e:
        print(f"❌ Error checking decode logic: {e}")

def run_minimal_test():
    """Run a minimal test to see if save_decode_cache affects behavior"""
    print("\n6. Running minimal behavior test...")
    
    try:
        import os
        import tempfile
        import yaml
        
        # Create config with save_decode_cache=True
        config_true = {
            'chunk_size': 256,
            'local_cpu': True,
            'save_decode_cache': True
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_true, f)
            config_file_true = f.name
        
        # Create config with save_decode_cache=False
        config_false = {
            'chunk_size': 256,
            'local_cpu': True,
            'save_decode_cache': False
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_false, f)
            config_file_false = f.name
        
        # Test both configurations
        print("   Testing save_decode_cache=True...")
        os.environ["LMCACHE_CONFIG_FILE"] = config_file_true
        from lmcache.v1.config import LMCacheEngineConfig
        config1 = LMCacheEngineConfig.from_env()
        print(f"   ✅ Loaded config: save_decode_cache={config1.save_decode_cache}")
        
        print("   Testing save_decode_cache=False...")
        os.environ["LMCACHE_CONFIG_FILE"] = config_file_false
        # Need to reload the module to get new config
        import importlib
        import lmcache.v1.config
        importlib.reload(lmcache.v1.config)
        config2 = LMCacheEngineConfig.from_env()
        print(f"   ✅ Loaded config: save_decode_cache={config2.save_decode_cache}")
        
        if config1.save_decode_cache != config2.save_decode_cache:
            print("✅ Configuration difference detected - save_decode_cache setting is being read correctly")
        else:
            print("❌ No configuration difference - may be an issue with config loading")
        
        # Cleanup
        os.unlink(config_file_true)
        os.unlink(config_file_false)
        
    except Exception as e:
        print(f"❌ Error in minimal test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    print("SIMPLE save_decode_cache VERIFICATION")
    print("=" * 50)
    
    test_config_loading()
    test_cache_engine_initialization()
    check_decode_logic()
    run_minimal_test()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("This test checks if save_decode_cache configuration is properly loaded")
    print("and if the relevant code paths exist in the codebase.")
    print("For actual runtime behavior testing, use the comprehensive test script.")

if __name__ == "__main__":
    main() 