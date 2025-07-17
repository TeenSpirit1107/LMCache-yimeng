#!/usr/bin/env python3
"""
Unified test script that supports different configurations for LMCache testing.
"""

import os
import time
import getpass
import hashlib
import argparse
import threading
import psutil
from typing import Optional

from config import TEST_CONFIGS, COMMON_CONFIG

def print_debug(message: str, timestamp: bool = True):
    """Print debug message with timestamp"""
    if timestamp:
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"[DEBUG {current_time}] {message}")
    else:
        print(f"[DEBUG] {message}")

def print_resource_usage():
    """Print current system resource usage"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        # Memory usage
        memory = psutil.virtual_memory()
        # GPU processes (if nvidia-ml-py available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_info = [(gpu.id, gpu.memoryUtil, gpu.load) for gpu in gpus]
        except:
            gpu_info = "GPU info unavailable"
        
        print_debug(f"RESOURCE USAGE - CPU: {cpu_percent}%, Memory: {memory.percent}%, GPU: {gpu_info}")
    except Exception as e:
        print_debug(f"Failed to get resource usage: {e}")

def check_port_usage(port: int) -> bool:
    """Check if a port is in use"""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except Exception as e:
        print_debug(f"Error checking port {port}: {e}")
        return False

def log_environment_state():
    """Log important environment variables and system state"""
    important_vars = [
        "CUDA_VISIBLE_DEVICES", "NCCL_ASYNC_ERROR_HANDLING", "NCCL_BLOCKING_WAIT",
        "LMCACHE_CHUNK_SIZE", "LMCACHE_LOCAL_CPU", "LMCACHE_MAX_LOCAL_CPU_SIZE",
        "LMCACHE_ENABLE_P2P", "LMCACHE_LOOKUP_URL", "LMCACHE_DISTRIBUTED_URL",
        "LMCACHE_LMCACHE_INSTANCE_ID"
    ]
    
    print_debug("=== ENVIRONMENT STATE ===")
    for var in important_vars:
        value = os.environ.get(var, "NOT SET")
        print_debug(f"{var} = {value}")
    
    print_debug(f"Process ID: {os.getpid()}")
    print_debug(f"User: {getpass.getuser()}")
    print_resource_usage()

def monitor_redis_health(port: int, duration: int = 30):
    """Monitor Redis health in background thread"""
    def monitor():
        import subprocess
        for i in range(duration):
            try:
                result = subprocess.run(
                    ["redis-cli", "-p", str(port), "ping"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode != 0 or "PONG" not in result.stdout:
                    print_debug(f"REDIS HEALTH: Port {port} not responding at check {i+1}")
                else:
                    if i % 10 == 0:  # Log every 10 seconds
                        print_debug(f"REDIS HEALTH: Port {port} healthy at check {i+1}")
            except Exception as e:
                print_debug(f"REDIS HEALTH: Error checking port {port}: {e}")
            time.sleep(1)
    
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return thread

def time_function(func_name: str):
    """Decorator to time function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            print_debug(f"Starting {func_name}...")
            print_resource_usage()  # Log resources before operation
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                print_debug(f"Completed {func_name} in {end_time - start_time:.2f} seconds")
                print_resource_usage()  # Log resources after operation
                return result
            except Exception as e:
                end_time = time.time()
                print_debug(f"Failed {func_name} after {end_time - start_time:.2f} seconds: {e}")
                print_resource_usage()  # Log resources on failure
                raise
        return wrapper
    return decorator

def generate_rpc_port(offset: int = 0) -> int:
    """Generate a unique RPC port based on username and process ID"""
    username = getpass.getuser()
    pid = os.getpid()
    unique_id = hashlib.md5(f"{username}_{pid}".encode()).hexdigest()[:8]
    base_port = int(unique_id, 16) % 10000 + 20000  # Use higher port range
    final_port = base_port + offset
    print_debug(f"Generated RPC port: {final_port} (base: {base_port}, offset: {offset})")
    return final_port

@time_function("LLM Creation")
def create_llm(llm_config):
    """Create LLM with timing"""
    print_debug("=== LLM CREATION START ===")
    print_debug(f"LLM Config: {llm_config}")
    log_environment_state()
    
    from vllm import LLM, SamplingParams
    
    print_debug("About to instantiate LLM object...")
    llm = LLM(**llm_config)
    print_debug("LLM object created successfully")
    
    print_debug("=== LLM CREATION END ===")
    return llm

@time_function("LLM Generation")
def run_generation(llm, prompts, sampling_params):
    """Run generation with timing"""
    print_debug("=== GENERATION START ===")
    print_debug(f"Prompts: {prompts}")
    print_debug(f"Sampling params: {sampling_params}")
    print_resource_usage()
    
    # Add process monitoring for data parallel mode
    import multiprocessing
    active_children = multiprocessing.active_children()
    print_debug(f"Active child processes: {len(active_children)}")
    for i, child in enumerate(active_children):
        print_debug(f"Child {i}: PID={child.pid}, name={child.name}, alive={child.is_alive()}")
    
    print_debug("About to call llm.generate()...")
    
    # Add timeout mechanism and periodic checking
    import signal
    import time
    from threading import Thread, Event
    
    result_container = [None]
    exception_container = [None]
    completion_event = Event()
    
    def generation_worker():
        try:
            print_debug("WORKER: Starting llm.generate() in worker thread...")
            result = llm.generate(prompts, sampling_params)
            result_container[0] = result
            print_debug("WORKER: llm.generate() completed successfully")
        except Exception as e:
            print_debug(f"WORKER: Exception in llm.generate(): {e}")
            exception_container[0] = e
        finally:
            completion_event.set()
    
    # Start generation in a separate thread
    worker_thread = Thread(target=generation_worker, daemon=True)
    worker_thread.start()
    
    # Monitor progress with timeout
    timeout = 180  # Reduced from 300 to 30 seconds - deadlock detection, not slow operation
    check_interval = 10  # Reduced from 10 to 5 seconds for faster feedback
    elapsed = 0
    
    while elapsed < timeout:
        if completion_event.wait(check_interval):
            print_debug("Generation completed within timeout")
            break
        
        elapsed += check_interval
        print_debug(f"TIMEOUT CHECK: {elapsed}s elapsed, still waiting for generation...")
        print_resource_usage()
        
        # Check if worker thread is still alive
        if not worker_thread.is_alive():
            print_debug("TIMEOUT CHECK: Worker thread died unexpectedly!")
            break
        
        # Check child processes
        current_children = multiprocessing.active_children()
        print_debug(f"TIMEOUT CHECK: Active children: {len(current_children)}")
        for i, child in enumerate(current_children):
            print_debug(f"TIMEOUT CHECK: Child {i}: PID={child.pid}, alive={child.is_alive()}")
            if not child.is_alive():
                print_debug(f"TIMEOUT CHECK: Child process {child.pid} died!")
    else:
        print_debug(f"TIMEOUT: Generation timed out after {timeout}s")
        print_debug("This indicates a deadlock or infinite wait condition")
        print_resource_usage()
        raise TimeoutError(f"Generation timed out after {timeout}s")
    
    # Check results
    if exception_container[0]:
        print_debug(f"Generation failed with exception: {exception_container[0]}")
        raise exception_container[0]
    
    if result_container[0] is None:
        print_debug("Generation completed but no result returned")
        raise RuntimeError("Generation completed but no result returned")
    
    print_debug("llm.generate() completed successfully")
    print_debug("=== GENERATION END ===")
    return result_container[0]

def run_test(config_name: str, port_offset: int = 0):
    """Run test with specified configuration"""
    if config_name not in TEST_CONFIGS:
        raise ValueError(f"Unknown configuration: {config_name}. Available configs: {list(TEST_CONFIGS.keys())}")
    
    config = TEST_CONFIGS[config_name]
    print_debug(f"=== TEST START: {config_name} ===")
    print_debug(f"Configuration: {config}")
    print_debug(f"Description: {config['description']}")
    
    # Generate unique RPC port
    rpc_port = generate_rpc_port(port_offset)
    print_debug(f"Using unique RPC port: {rpc_port} for user {getpass.getuser()} (PID: {os.getpid()})")
    
    # Check if ports are already in use
    ports_to_check = [rpc_port]
    if config["p2p_search"]:
        lookup_port = rpc_port + 100
        distributed_port = rpc_port + 200
        ports_to_check.extend([lookup_port, distributed_port])
    
    print_debug("=== PORT CONFLICT CHECK ===")
    for port in ports_to_check:
        in_use = check_port_usage(port)
        print_debug(f"Port {port} in use: {in_use}")
        if in_use:
            print_debug(f"WARNING: Port {port} is already in use!")
    
    redis_started = False  # Track if we started Redis
    redis_monitor_thread = None
    
    # Set environment variables
    print_debug("=== ENVIRONMENT SETUP ===")
    os.environ["LMCACHE_CHUNK_SIZE"] = COMMON_CONFIG["chunk_size"]
    os.environ["LMCACHE_LOCAL_CPU"] = str(COMMON_CONFIG["local_cpu"])
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = COMMON_CONFIG["max_local_cpu_size"]
    os.environ["LMCACHE_ENABLE_P2P"] = str(config["p2p_search"])
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Enable async error handling
    os.environ["NCCL_BLOCKING_WAIT"] = "1"  # Use blocking wait to improve error handling

    # Use GPU 0,1 for data parallel
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    print_debug("Using GPU 0,1 for data parallel")
    
    # Log potential conflicts between data parallel and P2P
    if config["data_parallel"] and config["p2p_search"]:
        print_debug("=== POTENTIAL CONFLICT WARNING ===")
        print_debug("Both data_parallel and p2p_search are enabled!")
        print_debug("This combination may cause resource conflicts or communication issues")
        print_debug("Monitor for: port conflicts, GPU memory conflicts, NCCL issues")
    
    # Set P2P URLs when P2P is enabled
    if config["p2p_search"]:
        print_debug("=== P2P SETUP ===")
        
        # Check if we need isolated instances for data parallel ranks
        isolated_instances = config.get("isolated_instances", False)
        if isolated_instances and config["data_parallel"]:
            print_debug("DIAGNOSTIC MODE: Using isolated LMCache instances per DP rank")
            # Use different port ranges for different ranks to avoid conflicts
            # Note: In real deployment, rank would be determined at runtime
            # For now, we'll set up the first rank's ports
            lookup_port = rpc_port + 100  # Rank 0 uses base ports
            distributed_port = rpc_port + 200
            
            # Set unique instance ID to avoid sharing
            os.environ["LMCACHE_LMCACHE_INSTANCE_ID"] = f"lmcache_rank_0_instance"
        else:
            # Standard P2P setup (shared instances)
            lookup_port = rpc_port + 100
            distributed_port = rpc_port + 200
        
        os.environ["LMCACHE_LOOKUP_URL"] = f"localhost:{lookup_port}"
        os.environ["LMCACHE_DISTRIBUTED_URL"] = f"localhost:{distributed_port}"
        
        print_debug(f"P2P lookup port: {lookup_port}")
        print_debug(f"P2P distributed port: {distributed_port}")
        if isolated_instances and config["data_parallel"]:
            print_debug(f"LMCache instance ID: {os.environ.get('LMCACHE_LMCACHE_INSTANCE_ID')}")
        
        # Start Redis server for P2P lookup
        print_debug(f"Starting Redis server on port {lookup_port} for P2P lookup...")
        import subprocess
        import time
        try:
            # Kill any existing Redis on this port
            kill_cmd = f"pkill -f 'redis-server.*{lookup_port}'"
            print_debug(f"Killing existing Redis: {kill_cmd}")
            subprocess.run(kill_cmd, shell=True, capture_output=True)
            time.sleep(2)  # Give more time for cleanup
            
            # Verify port is free
            if check_port_usage(lookup_port):
                print_debug(f"WARNING: Port {lookup_port} still in use after cleanup!")
            
            # Start Redis server in background
            redis_cmd = ["redis-server", "--port", str(lookup_port), "--daemonize", "yes"]
            if isolated_instances and config["data_parallel"]:
                # Use different database numbers for isolation
                redis_cmd.extend(["--databases", "16"])
                print_debug(f"DIAGNOSTIC: Starting Redis with multiple databases for isolation")
            
            print_debug(f"Starting Redis: {' '.join(redis_cmd)}")
            redis_process = subprocess.Popen(
                redis_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            time.sleep(3)  # Give Redis more time to start
            
            # Test Redis connection
            ping_cmd = ["redis-cli", "-p", str(lookup_port), "ping"]
            print_debug(f"Testing Redis: {' '.join(ping_cmd)}")
            test_result = subprocess.run(
                ping_cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            print_debug(f"Redis ping result: return_code={test_result.returncode}, stdout='{test_result.stdout.strip()}', stderr='{test_result.stderr.strip()}'")
            
            if test_result.returncode == 0 and "PONG" in test_result.stdout:
                print_debug(f"Redis server successfully started on port {lookup_port}")
                redis_started = True
                # Start monitoring Redis health
                redis_monitor_thread = monitor_redis_health(lookup_port)
            else:
                print_debug(f"ERROR: Redis server failed to start properly on port {lookup_port}")
                print_debug(f"This may cause P2P functionality to fail")
                
        except Exception as e:
            print_debug(f"EXCEPTION: Failed to start Redis server: {e}")
            import traceback
            traceback.print_exc()
            print_debug("P2P mode may not work without Redis server")
        
        print_debug(f"P2P URLs - lookup: {os.environ['LMCACHE_LOOKUP_URL']}, distributed: {os.environ['LMCACHE_DISTRIBUTED_URL']}")
    
    # Log final environment state
    log_environment_state()
    
    print_debug("=== VLLM IMPORT ===")
    print_debug("Importing vLLM modules...")
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    print_debug("vLLM modules imported successfully")
    
    # Configure KV cache transfer
    print_debug("=== KV TRANSFER CONFIG ===")
    ktc_config = {
        "lmcache_rpc_port": rpc_port,
        "lmcache_p2p_search": config["p2p_search"]
    }
    
    # Add isolated instance configuration if needed
    if config.get("isolated_instances", False) and config["data_parallel"]:
        print_debug("DIAGNOSTIC: Configuring KV transfer for isolated instances")
        # This would be handled by LMCache internally, but we can log it
    
    print_debug(f"KVTransferConfig extra config: {ktc_config}")
    
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
        kv_connector_extra_config=ktc_config
    )
    print_debug("KVTransferConfig created successfully")
    
    # Create LLM with appropriate configuration
    print_debug("=== LLM CONFIG PREPARATION ===")
    llm_config = {
        "model": COMMON_CONFIG["model"],
        "kv_transfer_config": ktc,
        "max_model_len": COMMON_CONFIG["max_model_len"],
        "gpu_memory_utilization": COMMON_CONFIG["gpu_memory_utilization"],
        "trust_remote_code": True
    }
    
    if config["data_parallel"]:
        llm_config["data_parallel_size"] = config["data_parallel_size"]
        print_debug(f"Data parallel size set to: {config['data_parallel_size']}")
        print_debug("WARNING: Data parallel + P2P may cause synchronization issues")
    
    print_debug(f"Final LLM config: {llm_config}")
    
    # Create LLM with timing
    llm = create_llm(llm_config)
    
    print_debug("=== INFERENCE PREPARATION ===")
    print_debug("LLM created successfully, preparing inference...")
    
    # Simple test prompt
    prompts = ["Hello, how are you?"]
    sampling_params = SamplingParams(temperature=0, max_tokens=5)
    
    print_debug("=== GENERATION PHASE ===")
    print_debug("Starting generation...")
    print_debug("NOTE: If hanging occurs, it's likely during this phase")
    try:
        # Run generation with timing
        outputs = run_generation(llm, prompts, sampling_params)
        print_debug("Generation completed successfully!")
        for i, output in enumerate(outputs):
            print_debug(f"Generated {i}: {output.outputs[0].text}")
    except Exception as e:
        print_debug(f"ERROR during generation: {e}")
        print_debug("This is likely where the hanging occurs!")
        import traceback
        traceback.print_exc()
        print_resource_usage()
    
    print_debug("=== CLEANUP ===")
    
    # Clean up Redis server if P2P was used and we started it
    if config["p2p_search"] and redis_started:
        lookup_port = rpc_port + 100
        print_debug(f"Stopping Redis server on port {lookup_port}...")
        try:
            import subprocess
            # Gracefully shutdown Redis
            shutdown_result = subprocess.run(
                ["redis-cli", "-p", str(lookup_port), "shutdown"],
                capture_output=True,
                timeout=10
            )
            print_debug(f"Redis shutdown result: {shutdown_result.returncode}")
            
            time.sleep(2)
            
            # Force kill if still running
            kill_result = subprocess.run(f"pkill -f 'redis-server.*{lookup_port}'", shell=True, capture_output=True)
            print_debug(f"Redis force kill result: {kill_result.returncode}")
            
            # Verify port is free
            if not check_port_usage(lookup_port):
                print_debug(f"Redis server on port {lookup_port} stopped successfully")
            else:
                print_debug(f"WARNING: Port {lookup_port} still in use after cleanup")
                
        except Exception as e:
            print_debug(f"WARNING: Failed to stop Redis server: {e}")
    
    print_debug("Destroying LMCache engine...")
    try:
        from lmcache.v1.cache_engine import LMCacheEngineBuilder
        from lmcache.integration.vllm.utils import ENGINE_NAME
        LMCacheEngineBuilder.destroy(ENGINE_NAME)
        print_debug("LMCache engine destroyed successfully")
    except Exception as e:
        print_debug(f"Error destroying LMCache engine: {e}")
    
    print_debug(f"=== TEST COMPLETED: {config_name} ===")

def main():
    parser = argparse.ArgumentParser(description="Run LMCache tests with different configurations")
    parser.add_argument("config", choices=list(TEST_CONFIGS.keys()), help="Test configuration to use")
    parser.add_argument("--port-offset", type=int, default=0, help="Offset to add to RPC port (default: 0)")
    args = parser.parse_args()
    
    # Print overall test start time
    start_time = time.time()
    print_debug(f"Starting test run for config: {args.config}")
    
    try:
        run_test(args.config, args.port_offset)
        end_time = time.time()
        print_debug(f"Total test time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        end_time = time.time()
        print_debug(f"Test failed after {end_time - start_time:.2f} seconds: {e}")
        raise

if __name__ == "__main__":
    main() 