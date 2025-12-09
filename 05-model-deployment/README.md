# Model Deployment

Production LLM serving with vLLM — implementing high-performance inference with streaming, continuous batching, and OpenAI-compatible APIs.

## Overview

Three deployment approaches for LLM inference:

1. **HuggingFace Inference Endpoints** — Fully managed, dedicated endpoints
2. **Custom vLLM + FastAPI** — Streaming inference with Server-Sent Events
3. **OpenAI-Compatible Server** — Drop-in replacement API with vLLM

## Deployment Options

### 1. HuggingFace Inference Endpoints

Easiest deployment — managed infrastructure with auto-scaling:

```python
from openai import OpenAI

client = OpenAI(
    base_url="[YOUR_ENDPOINT_URL]",
    api_key="[YOUR_HF_TOKEN]"
)

response = client.chat.completions.create(
    model="microsoft/Phi-3-mini-4k-instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
```

### 2. Custom vLLM + FastAPI

Full control with streaming support:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from vllm import AsyncLLMEngine, SamplingParams

app = FastAPI()

engine = AsyncLLMEngine.from_engine_args(
    AsyncEngineArgs(
        model='microsoft/Phi-3-mini-4k-instruct',
        max_num_batched_tokens=512,
        max_model_len=512,
        gpu_memory_utilization=0.85,
        max_num_seqs=16,
        dtype='half',
    )
)

@app.post("/generate-stream")
async def generate_text(request: GenerationRequest):
    return StreamingResponse(
        generate_stream(request.prompt, request.max_tokens, request.temperature),
        media_type="text/event-stream"
    )
```

### 3. OpenAI-Compatible Server

Use vLLM's built-in OpenAI API server:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-3-mini-4k-instruct \
    --port 8000
```

Then use standard OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(...)
```

## vLLM Configuration

Key parameters for production deployment:

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `max_num_batched_tokens` | Max tokens per batch | Balance throughput vs latency |
| `max_num_seqs` | Concurrent sequences | Memory dependent |
| `gpu_memory_utilization` | Target GPU memory | 0.85-0.95 |
| `max_model_len` | Max sequence length | Model context limit |
| `enforce_eager` | Disable CUDA graphs | True for debugging |
| `dtype` | Model precision | 'half' for FP16 |

## Streaming with SSE

Server-Sent Events for real-time token streaming:

```python
async def generate_stream(prompt: str, max_tokens: int, temperature: float):
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens
    )
    request_id = str(uuid.uuid4())
    
    async for output in engine.generate(prompt, sampling_params, request_id=request_id):
        yield f"data: {json.dumps({'text': output.outputs[0].text})}\n\n"
    
    yield "data: [DONE]\n\n"
```

Client-side consumption:

```python
import requests

response = requests.post(url, json=payload, stream=True)
for line in response.iter_lines():
    if line.startswith(b'data: '):
        data = json.loads(line[6:])
        print(data['text'], end='', flush=True)
```

## Project Structure

```
├── custom-vllm-fastapi/
│   ├── Dockerfile           # Container configuration
│   ├── app.py              # FastAPI + vLLM streaming server
│   ├── client.py           # Example client code
│   └── requirements.txt
└── openai-server-vllm/
    ├── Dockerfile          # vLLM OpenAI server
    ├── entrypoint.sh       # Server startup script
    └── client.py           # OpenAI SDK client example
```

## Docker Deployment

### Custom FastAPI Server

```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### HuggingFace Spaces

Deploy on HuggingFace Spaces with GPU:

1. Create Space with Docker SDK
2. Select GPU hardware (e.g., T4, A10G)
3. Push Dockerfile + app code
4. Access via Direct URL

## Sampling Parameters

Control generation behavior:

```python
SamplingParams(
    temperature=0.7,      # Randomness (0 = deterministic)
    top_k=50,            # Sample from top K tokens
    top_p=0.9,           # Nucleus sampling threshold
    max_tokens=100,      # Maximum generation length
    presence_penalty=0,   # Penalize repeated tokens
    frequency_penalty=0,  # Penalize frequent tokens
)
```

## Production Considerations

### Memory Management
- Use `gpu_memory_utilization` to reserve headroom
- Monitor with `nvidia-smi`
- Consider model quantization (AWQ, GPTQ)

### Throughput Optimization
- Increase `max_num_seqs` for better batching
- Use continuous batching (vLLM default)
- Enable CUDA graphs (`enforce_eager=False`)

### Latency Optimization
- Reduce `max_num_batched_tokens` for faster first token
- Use tensor parallelism for large models
- Consider speculative decoding

## Client Libraries

### OpenAI SDK
```python
from openai import OpenAI
client = OpenAI(base_url=url, api_key=token)
```

### LangChain
```python
from langchain_openai import ChatOpenAI
chat = ChatOpenAI(
    model="microsoft/Phi-3-mini-4k-instruct",
    openai_api_base=f"{url}/v1",
    openai_api_key=token
)
```

### Raw Requests
```python
import requests
response = requests.post(url, headers=headers, json=payload, stream=True)
```

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [HuggingFace Inference Endpoints](https://huggingface.co/docs/inference-endpoints)
- [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [FastAPI StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)

