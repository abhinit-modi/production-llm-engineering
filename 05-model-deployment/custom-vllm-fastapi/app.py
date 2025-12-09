from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
import json
import uuid

app = FastAPI()

# TODO: In the AsyncEngineArgs select the additional parameters 
# to make this deployment efficient. Specifically, consider:
# - max_num_batched_tokens: Sets the maximum number of tokens that can be processed 
# in a single batch. Make sure to accommodate for the memory constraints of GPU hosting the application.
# - max_num_seqs: Limits the maximum number of sequences that can 
# be processed concurrently. Smaller numbers will reduce the memory pressure on the GPU.
# - gpu_memory_utilization: Sets the target GPU memory utilization. 
# Adjust to make more efficient use of available GPU memory.
# - max_model_len: Specifies the maximum sequence length the model can handle.
# - enforce_eager: Disables or enables CUDA graph optimization. This can be useful 
# for debugging or when CUDA graph optimization causes issues.
# - dtype='half': Sets the data type for model parameters to half-precision 
# (float16). This reduces memory usage and can speed up computations, especially on GPUs with good half-precision performance.
engine = AsyncLLMEngine.from_engine_args(
    AsyncEngineArgs(
        model='microsoft/Phi-3-mini-4k-instruct',
        max_num_batched_tokens=512,
        max_model_len=512, # model context length
        gpu_memory_utilization=0.85,
        max_num_seqs=16,
        enforce_eager=True,
        dtype='half',
    )
)

class GenerationRequest(BaseModel):
    # FastAPI uses classes like GenerationRequest for several important reasons:
    # - Automatic Request Parsing
    # - Data Validation
    # - Default Values
    # - Self-Documenting APIs
    # - Type Safety in Your Code
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7


async def generate_stream(prompt: str, max_tokens: int, temperature: float):
    """
    The function generate_stream is an asynchronous generator that produces a stream of 
    text from a language model. Asynchronous functions can pause their execution, 
    allowing other code to run while waiting for operations to complete.

    prompt: The initial text to start the generation.
    max_tokens: The maximum number of tokens (words or word pieces) to generate.
    temperature: Controls the randomness of the generation. Higher values (e.g., 1.0) 
    make output more random, while lower values (e.g., 0.1) make it more deterministic.
    """

    #  SamplingParams configures how the text generation will behave. 
    # It uses the temperature and max_tokens values passed to the function.
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens
    )

    # The request_id is used by vLLM to track different generation requests, 
    # especially useful in scenarios with multiple concurrent requests.
    # Using a UUID ensures that each request has a unique identifier, 
    # preventing conflicts between different generation tasks.
    request_id = str(uuid.uuid4())
    
    # async for is an asynchronous loop that works with asynchronous generators.
    # engine.generate() is an instance of the language model that generates text 
    # based on the given prompt and parameters. The loop will receive chunks of 
    # generated text one at a time rather than waiting for the entire text to be generated.
    # The generate function requires a request_id, which I set to 1
    async for output in engine.generate(prompt, sampling_params, request_id=request_id):  
        # yield is used in generator functions to produce a series of values 
        # over time rather than computing them all at once. The yielded string 
        # follows the Server-Sent Events (SSE) format:
        # - It starts with "data: ".
        # - The content is a JSON string containing the generated text.
        # - It ends with two newlines (\n\n) to signal the end of an SSE message.
        yield f"data: {json.dumps({'text': output.outputs[0].text})}\n\n"
    
    # After the generation is complete, we yield a special "DONE" signal, 
    # also in SSE format, to indicate that the stream has ended.
    yield "data: [DONE]\n\n"


# This line tells FastAPI that this function should handle POST requests 
# to the "/generate-stream" endpoint.
@app.post("/generate-stream")
async def generate_text(request: GenerationRequest):
    """
    The function generate_text is a FastAPI route that handles POST requests to "/generate-stream". 
    It's designed to stream generated text back to the client as it's being produced 
    rather than waiting for all the text to be generated before sending a response.
    """
    try:        
        # StreamingResponse is used to send a streaming response back to the client.
        # generate_stream() is called with the parameters from the request. This function is expected to be a generator that yields chunks of text.
        # media_type="text/event-stream" indicates that this is a Server-Sent Events (SSE) stream, a format for sending real-time updates from server to client.
        return StreamingResponse(
            generate_stream(request.prompt, request.max_tokens, request.temperature),
            media_type="text/event-stream"
        )
    except Exception as e:
        # If an exception occurs, it returns a streaming response with an error message,
        #  maintaining the SSE format.
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': str(e)})}\n\n"]),
            media_type="text/event-stream"
        )
    
@app.get("/")
def greet_json():
    return {"Hello": "World!"}