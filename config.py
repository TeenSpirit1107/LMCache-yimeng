"""
Configuration file for LMCache test scenarios
"""

TEST_CONFIGS = {
    "parallel_p2p": {
        "data_parallel": True,
        "data_parallel_size": 2,
        "p2p_search": True,
        "description": "Test with both data parallelism and p2p search enabled"
    },
    "parallel_simple": {
        "data_parallel": True,
        "data_parallel_size": 2,
        "p2p_search": False,
        "description": "SIMPLIFIED: Test with only data parallelism, no P2P (for debugging)"
    },
    "single_gpu_lmcache": {
        "data_parallel": False,
        "data_parallel_size": None,
        "p2p_search": True,
        "description": "FALLBACK: Single GPU with LMCache P2P (if data parallel fails)"
    },
    "no_parallel_p2p": {
        "data_parallel": False,
        "data_parallel_size": None,
        "p2p_search": True,
        "description": "Test without data parallelism but with p2p search enabled"
    },
    "parallel_no_p2p": {
        "data_parallel": True,
        "data_parallel_size": 2,
        "p2p_search": False,
        "description": "Test with data parallelism but without p2p search"
    },
    "no_parallel_no_p2p": {
        "data_parallel": False,
        "data_parallel_size": None,
        "p2p_search": False,
        "description": "Test without data parallelism and without p2p search"
    }
}

# Common configurations that don't change between tests
COMMON_CONFIG = {
    "model": "facebook/opt-1.3b",
    "max_model_len": 128,  # Further reduced from 256 to 128
    "gpu_memory_utilization": 0.4,  # Reduced from 0.8 to 0.6 for data parallel stability
    "chunk_size": "256",
    "local_cpu": True,
    "max_local_cpu_size": "1.0"
} 