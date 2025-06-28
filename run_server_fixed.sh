#!/bin/bash
# 自动生成的运行脚本，使用唯一RPC端口: 8783

export CUDA_VISIBLE_DEVICES=0

# 确保目录存在
mkdir -p /home/ymteng/tmp/lmcache_data

echo "使用LMCache RPC端口: 8783"
echo "启动vLLM服务器..."

vllm serve facebook/opt-1.3b \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.6 \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config": {"lmcache_rpc_port": "8783"}}'
