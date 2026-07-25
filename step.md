# DeepSeek V4 Flash NVFP4 on 2× DGX Spark

Hardware-validated deployment and troubleshooting steps for running:

```text
Model         : nvidia/DeepSeek-V4-Flash-NVFP4
Served name   : deepseek-v4-flash
Runtime image : ghcr.io/anemll/dspark-vllm-gx10:0.1.1
Topology      : 2× NVIDIA DGX Spark
Master        : 10.100.152.1
Worker        : 10.100.152.2
Backend       : vLLM multiprocessing (mp)
Tensor parallel size : 2
Nodes         : 2
API port      : 8000
Context       : 1,048,576 tokens
```

Last hardware validation: **2026-07-22**.

## 1. Controller Script

Use the corrected v8.2 controller:

```text
deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh
```

Make it executable and validate Bash syntax:

```bash
chmod +x deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh
bash -n deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh
```

Do **not** run the controller with `sudo`. Run it as the normal user that owns the Hugging Face and vLLM cache directories.

## 2. Important Defaults

The validated controller uses conservative settings for DGX Spark:

```bash
MOE_BACKEND=marlin
CUDAGRAPH_MODE=PIECEWISE
ENABLE_DSPARK_SPECULATION=false
SPECULATIVE_TOKENS=0
MAX_MODEL_LEN=1048576
GPU_MEMORY_UTILIZATION=0.85
MAX_NUM_SEQS=6
MAX_NUM_BATCHED_TOKENS=8192
```

Why:

- DSpark speculative decoding is disabled because the tested model snapshot does not expose the required parallel-drafting token marker.
- `marlin` is the conservative MoE backend for the tested image and avoids previously observed FlashInfer/CUTLASS compatibility problems.
- FlashInfer JIT artifacts are isolated by runtime image ID to avoid stale generated-module signature mismatches.

## 3. First Deployment

Run all commands from the master node.

### 3.1 Pull and lock the runtime image

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh prepare-runtime
```

This pulls the same container image on both nodes and verifies that the immutable image IDs match.

### 3.2 Download the model on the master

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh download
```

Expected model size is approximately 168 GB with 46 safetensors files.

### 3.3 Verify master model files

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh verify-files
```

### 3.4 Sync the model cache to the worker

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh sync-worker
```

### 3.5 Verify worker files

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh verify-worker
```

### 3.6 Inspect runtime compatibility

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh doctor
```

### 3.7 Configure the 200 Gb/s RoCE interface

Set the actual interface and HCA names found on the systems:

```bash
export NCCL_SOCKET_IFNAME=<200G_INTERFACE_NAME>
export NCCL_IB_HCA=<ROCE_HCA_NAME>
export NCCL_IB_GID_INDEX=3
```

Inspect available interfaces when necessary:

```bash
ip -br link
ibdev2netdev 2>/dev/null || true
rdma link 2>/dev/null || true
```

Without `NCCL_IB_HCA`, the controller deliberately falls back to TCP. TCP may function but can reduce multi-node throughput.

## 4. Start the Server

For general API testing with reasoning enabled by default:

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh stop
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh start
```

For Cline or another OpenAI-compatible agent, initially disable thinking to simplify streaming compatibility:

```bash
ENABLE_THINKING=false \
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh restart
```

The worker is intentionally started first as rank 1. The master is then started as rank 0 and exposes the API.

## 5. Confirm Server Status

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh status
```

Validated status on 2026-07-22:

```text
Head container   : Up
Worker container : Up
API health       : /v1/models returns the served model
Model ID         : deepseek-v4-flash
Max model length : 1048576
```

Direct health check:

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

Passing `/v1/models` proves only that the API is ready. Run the functional tests below before declaring the deployment healthy.

## 6. Required Functional Validation

### 6.1 Basic text generation

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh test-text
```

Hardware-validated result:

```text
PASS: Paris is a timeless city of light, where art, romance, and history intertwine along the Seine.
```

### 6.2 Required tool call

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh test-tools required
```

Hardware-validated result:

```text
PASS: get_weather({"location": "Bangkok"})
```

### 6.3 Two-turn tool-result loop

This is critical for IDE agents. It verifies:

1. The model emits a tool call.
2. The client sends the tool result back.
3. The model produces a final answer.

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh test-tool-loop
```

Hardware-validated result:

```text
PASS (turn 2): The weather in Chiang Mai is currently sunny with a temperature of 32°C.
```

### 6.4 Concurrent request test

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh stress 4
```

Hardware-validated result:

```text
4 OK, 0 FAIL (6.8s)
```

### 6.5 Optional reasoning test

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh test-reasoning
```

The expected answer contains `1591` for `37 × 43`.

## 7. v8.2 Test-Command Fix

Earlier test functions used a single-quoted heredoc:

```bash
<<'PYEOF'
```

Inside that heredoc, Python received the literal string `${API_PORT}` instead of the numeric port. The symptom was:

```text
http.client.InvalidURL: nonnumeric port: '${API_PORT}'
```

The v8.2 controller fixes `test-tools`, `test-tool-loop`, and `stress` by passing values as Python arguments, for example:

```bash
python3 - "$API_PORT" "$SERVED_MODEL_NAME" "$mode" <<'PYEOF'
```

Do not diagnose that old error as a vLLM, network, GPU, or tool-parser failure. It was a shell test-harness bug.

## 8. Validate Plain Streaming

```bash
curl -N --http1.1 \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer none' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {
        "role": "user",
        "content": "Reply exactly with: STREAM_OK"
      }
    ],
    "temperature": 0,
    "max_tokens": 32,
    "stream": true,
    "stream_options": {
      "include_usage": true
    },
    "chat_template_kwargs": {
      "thinking": false
    }
  }'
```

A successful stream must:

- emit content chunks that concatenate to `STREAM_OK`;
- emit `finish_reason: "stop"`;
- optionally emit a final usage chunk with `"choices": []`;
- end with exactly:

```text
data: [DONE]
```

The final empty-choices usage chunk is valid when `stream_options.include_usage=true`.

Hardware validation completed successfully on 2026-07-22.

## 9. Validate Tool-Call Streaming

```bash
curl -N --http1.1 \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer none' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {
        "role": "user",
        "content": "Read the file README.md"
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "read_file",
          "description": "Read a file",
          "parameters": {
            "type": "object",
            "properties": {
              "path": {
                "type": "string"
              }
            },
            "required": ["path"]
          }
        }
      }
    ],
    "tool_choice": "required",
    "temperature": 0,
    "max_tokens": 256,
    "stream": true,
    "stream_options": {
      "include_usage": true
    },
    "chat_template_kwargs": {
      "thinking": false
    }
  }'
```

A successful response must contain:

```text
function.name = read_file
```

The streamed argument fragments must concatenate to:

```json
{"path": "README.md"}
```

It must then emit:

```text
finish_reason = tool_calls
data: [DONE]
```

Hardware validation completed successfully on 2026-07-22.

## 10. Cline Configuration

Use the OpenAI-compatible provider configuration:

```text
Provider          : OpenAI Compatible
Base URL          : http://<MASTER_REACHABLE_IP>:8000/v1
API key           : none
Model ID          : deepseek-v4-flash
Context window    : start with 131072 for client testing
Max output tokens : 8192
Tool use          : enabled
Image support     : disabled unless separately validated
```

Important:

- The Base URL must stop at `/v1`.
- Do not enter `/v1/chat/completions`; Cline appends the endpoint itself.
- Use a master-node IP that the VS Code host can reach. `127.0.0.1` works only when VS Code/Cline runs on the master itself or uses a suitable tunnel.
- Start a new Cline task after changing provider settings or restarting vLLM.
- Update Cline to a current release before diagnosing protocol incompatibility.

Recommended first Cline task:

```text
Create a file named cline_test.txt containing exactly HELLO_FROM_DEEPSEEK
```

Expected sequence:

1. Cline sends the prompt and tool definitions.
2. The model emits a file-writing tool call.
3. Cline executes or requests approval for that tool.
4. Cline sends the tool result back to the model.
5. The model returns a final completion and the task stops.

## 11. If Cline Appears to Answer and Then Hangs

The server has been validated independently when all of the following pass:

```text
test-text
required tool call
two-turn tool loop
stress 4
plain SSE stream ending in [DONE]
tool-call SSE stream ending in [DONE]
```

When all six pass, first investigate the client rather than changing NCCL, GPU, model, or vLLM settings.

Use this order:

1. Restart with `ENABLE_THINKING=false`.
2. Update Cline and reload the VS Code window.
3. Create a new Cline task rather than resuming a corrupted task.
4. Confirm Base URL ends in `/v1`.
5. Confirm Model ID is exactly `deepseek-v4-flash`.
6. Test a one-tool file operation.
7. Watch the vLLM head log while reproducing the issue.
8. Inspect Cline's Output/Developer Tools logs for a client-side exception after the usage chunk or `[DONE]`.

The server may legitimately send this final usage chunk before `[DONE]`:

```json
{
  "choices": [],
  "usage": {
    "prompt_tokens": 11,
    "completion_tokens": 4,
    "total_tokens": 15
  }
}
```

A client must tolerate `choices: []` in this usage-only chunk.

## 12. Live Log Collection

Head:

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh logs head 1000
```

Worker:

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh logs worker 1000
```

Container status:

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh status
```

Shared memory:

```bash
df -h /dev/shm
ls -lah /dev/shm | tail -50
```

Network and RoCE state:

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh network-info
ip -s link show "$NCCL_SOCKET_IFNAME"
```

## 13. Interpreting Common Logs

### `/models` or `//models` returns 404

Some clients probe multiple paths. This is harmless when `/v1/models` subsequently returns `200 OK`.

### Triton kernel JIT compilation warning

Example:

```text
Triton kernel JIT compilation during inference ... latency spike
```

This can make the first request for a new shape slower. Warm up representative prompt and tool-schema sizes before measuring latency.

### Shared-memory broadcast warning

Example:

```text
No available shared memory broadcast block found in 60 seconds
```

A single warning during compilation or another long operation is not enough to prove a deadlock. Investigate further only when it repeats and requests do not reach `[DONE]`.

If it repeats during a genuine hang, collect synchronized head and worker logs and inspect `/dev/shm` before changing runtime flags.

### GPU memory displays as `[N/A]`

On DGX Spark, the `nvidia-smi --query-gpu=memory.used,memory.total` fields may report `[N/A]` because of the unified-memory architecture. This does not by itself indicate that the GPU is unavailable.

## 14. Troubleshooting Decision Tree

### `/v1/models` fails

Check:

```bash
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh status
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh logs head 300
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh logs worker 300
```

### `/v1/models` passes but `test-text` fails

Suspect model generation, runtime, worker synchronization, or distributed execution.

### `test-text` passes but `test-tools required` fails

Suspect the chat template, `deepseek_v4` tool parser, tool schema, or output parsing.

### Tool call passes but `test-tool-loop` fails

Suspect handling of assistant tool-call messages, `tool_call_id`, tool-result message format, or the second model turn.

### Non-streaming tests pass but curl streaming fails to end in `[DONE]`

Suspect the vLLM SSE path, proxy buffering, HTTP transport, or an engine hang.

### All server tests pass but Cline hangs

Suspect Cline settings, client stream parsing, reasoning-stream compatibility, task state, or extension bugs.

Do not rebuild the distributed server until the failing layer has been identified.

## 15. Conservative Tuning Order

After all functional tests pass:

1. Run `stress 4`.
2. Run a representative Cline task.
3. Run `bench`.
4. Observe both node logs and GPU utilization.
5. Change only one parameter at a time.
6. Re-run all functional tests after every runtime change.

For memory pressure, reduce context before changing kernels:

```bash
MAX_MODEL_LEN=524288 \
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh restart
```

Then reduce batching if necessary:

```bash
MAX_NUM_SEQS=2 MAX_NUM_BATCHED_TOKENS=4096 \
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh restart
```

Do not increase `GPU_MEMORY_UTILIZATION` as the first response to an out-of-memory condition.

## 16. Current Validation Matrix

| Capability | Result | Evidence date |
|---|---:|---:|
| Both containers remain running | PASS | 2026-07-22 |
| `/v1/models` | PASS | 2026-07-22 |
| Basic non-streaming text | PASS | 2026-07-22 |
| Required tool call | PASS | 2026-07-22 |
| Two-turn tool-result continuation | PASS | 2026-07-22 |
| Four concurrent requests | PASS | 2026-07-22 |
| Plain SSE streaming | PASS | 2026-07-22 |
| Tool-call SSE streaming | PASS | 2026-07-22 |
| Stream terminates with `[DONE]` | PASS | 2026-07-22 |
| Cline end-to-end file operation | CLIENT VALIDATION PENDING | — |
| Thinking-enabled Cline workflow | NOT YET VALIDATED | — |
| One-million-token request | NOT YET VALIDATED | — |
| Production soak test | NOT YET VALIDATED | — |

## 17. Security Note

The model only emits tool calls. Cline or another agent executes them.

Use:

- workspace restrictions;
- command allowlists;
- confirmation for destructive operations;
- filesystem and network sandboxing;
- timeouts and output limits;
- a non-root account;
- no unrestricted secret access.

A successful tool parser does not make arbitrary tool execution safe.

## 18. Minimal Handoff Prompt for Another Model

Use the following context when asking another model to continue troubleshooting:

```text
We are running nvidia/DeepSeek-V4-Flash-NVFP4 on two DGX Spark nodes using
vLLM mp backend, TP=2, nnodes=2, runtime image
`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, served as `deepseek-v4-flash`
on port 8000 with max_model_len=1048576.

Controller: deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh

Hardware-validated passes on 2026-07-22:
- /v1/models
- test-text
- test-tools required
- test-tool-loop
- stress 4: 4 OK, 0 FAIL
- plain SSE stream ends with data: [DONE]
- tool-call SSE emits read_file({"path":"README.md"}), finish_reason=tool_calls,
  and ends with data: [DONE]

The v8.2 script fixed a test-harness heredoc bug where `${API_PORT}` was passed
literally to Python. Do not treat that old InvalidURL error as a vLLM failure.

If Cline still hangs after displaying an answer, focus on the client layer:
disable default thinking, verify Base URL ends in /v1, update/reload Cline,
start a new task, and inspect client logs. The server's final usage-only SSE
chunk may validly contain choices: [] before [DONE].
```
