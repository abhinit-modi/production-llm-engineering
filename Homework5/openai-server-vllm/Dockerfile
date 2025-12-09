# Use the official vLLM image as the base image
FROM vllm/vllm-openai:latest

# Set environment variables
ENV HF_HOME="/tmp/huggingface"
ENV XDG_CACHE_HOME="/tmp/cache"
ENV NUMBA_CACHE_DIR="/tmp/numba_cache"
ENV OUTLINES_CACHE_DIR="/tmp/outlines_cache"
ENV VLLM_USE_MODELSCOPE="false"
ENV VLLM_DISABLE_USAGE_STATS="true"
ENV XDG_CONFIG_HOME="/tmp/config"

# Ensure PATH includes common Python locations
ENV PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin:$PATH"

# Set the working directory
WORKDIR /app

# Copy your entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose the port the app runs on
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]