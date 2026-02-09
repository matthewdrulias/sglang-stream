FROM lmsysorg/sglang:v0.4.5.post2-cu124

WORKDIR /app

# Install the StreamingLLM fork
COPY . /app/sglang-streamingllm
RUN pip install -e /app/sglang-streamingllm/python --no-deps

# Install runpod and missing/updated deps
RUN pip install runpod pybase64 --upgrade transformers

# Copy handler
COPY handler.py /app/handler.py

# Environment defaults
ENV MODEL_PATH="Qwen/Qwen3-8B"
ENV TP_SIZE="1"
ENV QUANTIZATION=""
ENV SINK_TOKEN_COUNT="4"
ENV CONTEXT_LENGTH="32768"
ENV HF_HOME="/runpod-volume/huggingface"

CMD ["python3", "/app/handler.py"]
