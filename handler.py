import runpod
import os
import subprocess
import time
import requests
import warnings

# Suppress Python warnings that clutter logs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

# Configuration from environment
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B")
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
QUANTIZATION = os.environ.get("QUANTIZATION", "")
SINK_TOKEN_COUNT = int(os.environ.get("SINK_TOKEN_COUNT", "4"))
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", "32768"))
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "32768"))

SGLANG_HOST = "127.0.0.1"
SGLANG_PORT = 8000
SGLANG_URL = f"http://{SGLANG_HOST}:{SGLANG_PORT}"

server_started = False


def start_sglang_server():
    global server_started
    if server_started:
        return True

    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--tp", str(TP_SIZE),
        "--host", SGLANG_HOST,
        "--port", str(SGLANG_PORT),
        "--context-length", str(CONTEXT_LENGTH),
    ]

    if SINK_TOKEN_COUNT > 0:
        cmd.extend(["--sink-token-count", str(SINK_TOKEN_COUNT)])

    if QUANTIZATION:
        cmd.extend(["--quantization", QUANTIZATION])

    print(f"Starting SGLang server: {' '.join(cmd)}")
    # Redirect stderr to suppress warnings that RunPod logs as errors
    subprocess.Popen(cmd, stderr=subprocess.DEVNULL)

    for i in range(120):
        try:
            response = requests.get(f"{SGLANG_URL}/health", timeout=5)
            if response.status_code == 200:
                print(f"SGLang server ready after {i * 5} seconds")
                server_started = True
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)

    return False


def handler(event):
    input_data = event["input"]

    # Simple health check - no model needed
    if input_data.get("ping"):
        return {"status": "ok", "message": "Handler is running"}

    if not start_sglang_server():
        return {"error": "Failed to start SGLang server"}

    # Resolve max_tokens: accept max_tokens, max_new_tokens, or env default
    max_tokens = input_data.get("max_tokens",
                 input_data.get("max_new_tokens", DEFAULT_MAX_TOKENS))

    # Build SGLang request — pass through all supported params
    sglang_payload = {
        "model": input_data.get("model", MODEL_PATH),
        "messages": input_data.get("messages", []),
        "max_tokens": max_tokens,
        "temperature": input_data.get("temperature", 0.7),
        "top_p": input_data.get("top_p", 0.9),
    }

    # Optional params — only include if provided
    if input_data.get("stop"):
        sglang_payload["stop"] = input_data["stop"]
    if "top_k" in input_data:
        sglang_payload["top_k"] = input_data["top_k"]
    if "frequency_penalty" in input_data:
        sglang_payload["frequency_penalty"] = input_data["frequency_penalty"]
    if "presence_penalty" in input_data:
        sglang_payload["presence_penalty"] = input_data["presence_penalty"]

    # Scale timeout with max_tokens — large generations need more time
    timeout = max(300, max_tokens // 10)

    try:
        response = requests.post(
            f"{SGLANG_URL}/v1/chat/completions",
            json=sglang_payload,
            timeout=timeout,
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"SGLang error ({response.status_code}): {response.text[:500]}"}
    except requests.exceptions.Timeout:
        return {"error": f"SGLang request timed out after {timeout}s"}
    except requests.exceptions.RequestException as e:
        return {"error": f"SGLang connection error: {str(e)}"}


runpod.serverless.start({"handler": handler})
