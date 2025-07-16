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
    "max_model_len": 512,
    "gpu_memory_utilization": 0.4,
    "chunk_size": "256",
    "local_cpu": True,
    "max_local_cpu_size": "1.0"
} 