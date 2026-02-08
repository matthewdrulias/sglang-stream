"""
RunPod Serverless Handler for SGLang with StreamingLLM

Provides OpenAI-compatible API for LLM inference with infinite context support.
"""

import os
import subprocess
import time
import requests
import runpod

# Configuration from environment
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B")
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
QUANTIZATION = os.environ.get("QUANTIZATION", "")
SINK_TOKEN_COUNT = int(os.environ.get("SINK_TOKEN_COUNT", "4"))
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", "32768"))

SGLANG_HOST = "127.0.0.1"
SGLANG_PORT = 8000
SGLANG_URL = f"http://{SGLANG_HOST}:{SGLANG_PORT}"

server_process = None


def start_sglang_server():
    """Start the SGLang server in the background."""
    global server_process

    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--tp", str(TP_SIZE),
        "--host", SGLANG_HOST,
        "--port", str(SGLANG_PORT),
        "--context-length", str(CONTEXT_LENGTH),
    ]

    # Add StreamingLLM sink tokens
    if SINK_TOKEN_COUNT > 0:
        cmd.extend(["--sink-token-count", str(SINK_TOKEN_COUNT)])

    # Add quantization if specified
    if QUANTIZATION:
        cmd.extend(["--quantization", QUANTIZATION])

    print(f"Starting SGLang server: {' '.join(cmd)}")
    server_process = subprocess.Popen(cmd)

    # Wait for server to be ready
    max_retries = 120  # 10 minutes max
    for i in range(max_retries):
        try:
            response = requests.get(f"{SGLANG_URL}/health", timeout=5)
            if response.status_code == 200:
                print(f"SGLang server ready after {i * 5} seconds")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)

    print("SGLang server failed to start")
    return False


def handler(event):
    """
    RunPod serverless handler.

    Supports OpenAI-compatible chat completion format:
    {
        "input": {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."}
            ],
            "max_tokens": 500,
            "temperature": 0.7,
            "stream": false
        }
    }
    """
    try:
        input_data = event.get("input", {})

        # Forward to SGLang server
        response = requests.post(
            f"{SGLANG_URL}/v1/chat/completions",
            json={
                "model": MODEL_PATH,
                "messages": input_data.get("messages", []),
                "max_tokens": input_data.get("max_tokens", 500),
                "temperature": input_data.get("temperature", 0.7),
                "top_p": input_data.get("top_p", 0.9),
                "stop": input_data.get("stop"),
            },
            timeout=300,
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"SGLang error: {response.text}"}

    except Exception as e:
        return {"error": str(e)}


# Start server on cold start
if __name__ == "__main__":
    if start_sglang_server():
        runpod.serverless.start({"handler": handler})
    else:
        print("Failed to start SGLang server")
        exit(1)
