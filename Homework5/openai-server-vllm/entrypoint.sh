#!/bin/bash

# Default values
MODEL=${MODEL:-"microsoft/Phi-3-mini-4k-instruct"}
DTYPE=${DTYPE:-"half"}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-512}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-512}
ENFORCE_EAGER=${ENFORCE_EAGER:-true}

# Check and set permissions for directories
for dir in /tmp/huggingface /tmp/cache /tmp/numba_cache /tmp/outlines_cache /tmp/config; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
    fi
    chmod -R 777 "$dir"
    echo "Permissions for $dir:"
    ls -la "$dir"
done

# Construct the command
CMD="vllm serve $MODEL \
--host 0.0.0.0 \
--port 8000 \
--dtype $DTYPE \
--max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
--max-num-seqs $MAX_NUM_SEQS \
--gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
--max-model-len $MAX_MODEL_LEN"

# Add enforce-eager only if it's set to true
if [ "$ENFORCE_EAGER" = "true" ]; then
    CMD="$CMD --enforce-eager"
fi

# Execute the command
echo "Running command: $CMD"
exec $CMD