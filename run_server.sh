LMCACHE_CONFIG_FILE=lmcache.yaml \
LMCACHE_USE_EXPERIMENTAL=True \
vllm serve facebook/opt-1.3b \
  --gpu-memory-utilization 0.6 \
  --max-model-len 2048 \
  --port 8000 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'