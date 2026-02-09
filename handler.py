import runpod
import os
import subprocess
import time
import requests

# Configuration from environment
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B")
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
QUANTIZATION = os.environ.get("QUANTIZATION", "")
SINK_TOKEN_COUNT = int(os.environ.get("SINK_TOKEN_COUNT", "4"))
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", "32768"))

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
    subprocess.Popen(cmd)

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

    if not start_sglang_server():
        return {"error": "Failed to start SGLang server"}

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


runpod.serverless.start({"handler": handler})
