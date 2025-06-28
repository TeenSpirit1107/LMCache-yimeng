#!/usr/bin/env python3
"""
脚本用于自动选择可用端口，清理socket文件冲突，并更新LMCache配置
"""

import os
import sys
import yaml
import socket
import hashlib
import getpass
import subprocess
import glob
from pathlib import Path

def find_available_port(start_port=65000, max_port=65535):
    """查找可用的网络端口"""
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError("No available ports found")

def generate_unique_rpc_port():
    """生成唯一的RPC端口号以避免冲突"""
    username = getpass.getuser()
    pid = os.getpid()
    unique_id = hashlib.md5(f"{username}_{pid}".encode()).hexdigest()[:8]
    rpc_port = int(unique_id, 16) % 10000
    return rpc_port

def cleanup_socket_files(rpc_port=None):
    """清理可能存在的socket文件"""
    patterns = [
        "/tmp/lmcache_rpc_port_*",
        "/tmp/vllm_*",
    ]
    
    if rpc_port is not None:
        patterns.append(f"/tmp/lmcache_rpc_port_{rpc_port}")
    
    cleaned_files = []
    for pattern in patterns:
        for file_path in glob.glob(pattern):
            try:
                os.unlink(file_path)
                cleaned_files.append(file_path)
                print(f"已清理socket文件: {file_path}")
            except (OSError, PermissionError) as e:
                print(f"无法清理文件 {file_path}: {e}")
    
    return cleaned_files

def update_yaml_config(config_file="lmcache.yaml"):
    """更新YAML配置文件中的端口设置"""
    if not os.path.exists(config_file):
        print(f"配置文件 {config_file} 不存在")
        return None
        
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # 生成新的端口号用于fs URL
    new_fs_port = find_available_port()
    
    # 更新fs URL中的端口
    if 'remote_url' in config and config['remote_url'].startswith('fs://'):
        # 解析现有的fs URL并更新端口
        parts = config['remote_url'].split('/')
        if len(parts) >= 3:
            host_port = parts[2]
            if ':' in host_port:
                host = host_port.split(':')[0]
                path = '/'.join(parts[3:])
                config['remote_url'] = f"fs://{host}:{new_fs_port}/{path}"
            else:
                # 如果没有端口，添加一个
                path = '/'.join(parts[2:])
                config['remote_url'] = f"fs://localhost:{new_fs_port}/{path}"
    
    # 备份原配置
    backup_file = f"{config_file}.backup"
    if os.path.exists(config_file):
        os.rename(config_file, backup_file)
        print(f"原配置已备份到: {backup_file}")
    
    # 写入新配置
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"已更新配置文件 {config_file}")
    print(f"新的remote_url: {config['remote_url']}")
    
    return new_fs_port

def create_run_script_with_unique_rpc():
    """创建带有唯一RPC端口的运行脚本"""
    rpc_port = generate_unique_rpc_port()
    
    # 清理可能的socket冲突
    cleanup_socket_files(rpc_port)
    
    chat_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\\n' }}{% endif %}{% endfor %}Assistant:"
    
    script_content = f'''#!/bin/bash
# 自动生成的运行脚本，使用唯一RPC端口: {rpc_port}

export CUDA_VISIBLE_DEVICES=0

# 确保目录存在
mkdir -p /home/ymteng/tmp/lmcache_data

echo "使用LMCache RPC端口: {rpc_port}"
echo "启动vLLM服务器..."

vllm serve facebook/opt-1.3b \\
    --port 8000 \\
    --max-model-len 2048 \\
    --gpu-memory-utilization 0.6 \\
    --chat-template "{chat_template}" \\
    --kv-transfer-config \\
    '{{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config": {{"lmcache_rpc_port": "{rpc_port}"}}}}'
'''
    
    script_file = "run_server_fixed.sh"
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # 给脚本执行权限
    os.chmod(script_file, 0o755)
    
    print(f"已创建运行脚本: {script_file}")
    print(f"使用的RPC端口: {rpc_port}")
    
    return script_file, rpc_port

def main():
    print("=== LMCache端口冲突修复工具 ===")
    
    # 1. 更新YAML配置文件
    print("\\n1. 更新YAML配置文件...")
    fs_port = update_yaml_config()
    
    # 2. 创建新的运行脚本
    print("\\n2. 创建运行脚本...")
    script_file, rpc_port = create_run_script_with_unique_rpc()
    
    # 3. 清理现有的socket文件
    print("\\n3. 清理socket文件...")
    cleanup_socket_files()
    
    print("\\n=== 修复完成 ===")
    print(f"配置文件: lmcache.yaml (已更新)")
    print(f"运行脚本: {script_file}")
    print(f"FS端口: {fs_port}")
    print(f"RPC端口: {rpc_port}")
    print(f"\\n运行服务器: ./{script_file}")

if __name__ == "__main__":
    main() 