# save_decode_cache Testing
This example tests the `save_decode_cache` functionality in LMCache to verify it correctly saves decode phase KV cache tokens.

## Prerequisites
Your server should have at least 1 GPU with sufficient memory to run facebook/opt-1.3b model.

## Test Script
- `python test_save_decode_cache.py` - Automated test to detect save_decode_cache functionality

## What the test does

### Step 1: Setup
- Generates unique RPC ports to avoid conflicts
- Creates two LMCache configuration files:
  - `lmcache_config_enabled.yaml` with `save_decode_cache: True`
  - `lmcache_config_disabled.yaml` with `save_decode_cache: False`

### Step 2: System Warmup
- Runs a short inference to warm up GPU and initialize system state
- Uses minimal tokens to reduce overhead

### Step 3: Test save_decode_cache=True
- Loads configuration with `save_decode_cache: True`
- Runs first inference with long prompt to establish baseline
- Clears cache engine between inferences
- Runs second inference with extended prompt
- Records timing and captures logs

### Step 4: Test save_decode_cache=False
- Loads configuration with `save_decode_cache: False`
- Repeats the same inference pattern as Step 3
- Records timing and captures logs

### Step 5: Log Analysis
- Searches for configuration loading patterns
- Counts prefill cache store operations (skip_leading_tokens=0)
- Counts decode cache store operations (skip_leading_tokens>0)
- Extracts cache hit information

### Step 6: Results Comparison
- Compares decode cache operations between enabled/disabled tests
- Validates configuration was loaded correctly
- Reports performance metrics
- Provides verdict based on evidence

## Expected Output
The test should detect:
- More decode cache store operations when `save_decode_cache=True`
- Correct configuration loading for both tests
- Clear evidence of the feature working or failing