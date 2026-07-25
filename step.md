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

Controller version: **3.1.0**. Last hardware validation: **2026-07-22**.

## 1. Controller Script

Use the canonical stacked controller:

```text
deepseek-v4-flash-nvfp4-stacked.sh
```

Make it executable and validate Bash syntax:

```bash
chmod +x deepseek-v4-flash-nvfp4-stacked.sh
bash -n deepseek-v4-flash-nvfp4-stacked.sh
```

Do **not** run the controller with `sudo`. Run it as the normal user that owns the Hugging Face and vLLM cache directories. Caches resolve under `$HOME` (`$HOME/.cache/huggingface`, `$HOME/.cache/vllm`, `$HOME/.cache/flashinfer`), and `SSH_USER` defaults to the current user (`${USER:-$(id -un)}`) — no username is hard-coded, so nothing has to be edited before a teammate runs it.

## 2. Identify What Is Running (`info`)

The fastest way to see which model a controller serves and whether it is up is the `info` command (`banner` is an alias):

```bash
./deepseek-v4-flash-nvfp4-stacked.sh info
```

Sample output:

```text
   ____   ____ __  __    ____                   _
  |  _ \ / ___|\ \/ /   / ___| _ __   __ _ _ __| | __
  | | | | |  _  \  /    \___ \| '_ \ / _` | '__| |/ /
  | |_| | |_| | /  \     ___) | |_) | (_| | |  |   <
  |____/ \____|/_/\_\   |____/| .__/ \__,_|_|  |_|\_\
                              |_|
       =[ DGX Spark Controller · v3.1.0 ]
+ -- --=[ DeepSeek-V4-Flash (NVFP4) · 2-node ]
+ -- --=[ vLLM (Docker, stacked) · reasoning · tools · tool-loop · 1M ctx ]
+ -- --=[ Designed by neronain · fb.com/neronain.minidev ]

  Model     : DeepSeek-V4-Flash (NVFP4) · 2-node
  Model ID  : nvidia/DeepSeek-V4-Flash-NVFP4
  Runtime   : vLLM (Docker, stacked)
  Features  : reasoning · tools · tool-loop · 1M ctx
  Context   : 1048576 tokens
  API (v1)  : http://10.100.152.1:8000/v1
  State     : RUNNING  (port 8000)
```

Notes:

- `State` is derived from `http://127.0.0.1:${API_PORT}/health`. It shows `RUNNING` when that health probe succeeds and `stopped` otherwise.
- `API (v1)` shows the advertised URL, so it can be pasted straight into a client such as Cline.
- `info` is read-only, needs no arguments, and never prompts. It is safe to run at any time, including while the server is serving traffic.
- Every controller in this repository declares `SCRIPT_VERSION="${SCRIPT_VERSION:-3.1.0}"` and supports `info`, so the same command identifies any of them.

`info` reports intent and liveness only. It does not replace `status` or the functional tests in section 7.

## 3. Important Defaults

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

### 3.1 Option validation

Command-line overrides are validated before anything starts:

```bash
--context TOKENS    must be an integer greater than 0
--port PORT         must be an integer in 1..65535
```

Invalid values abort immediately instead of producing a half-started cluster:

```text
Invalid --port: 99999 (use 1..65535)
Invalid --context: 0
```

## 4. First Deployment

Run all commands from the master node.

### 4.1 Pull and lock the runtime image

```bash
./deepseek-v4-flash-nvfp4-stacked.sh prepare-runtime
```

This pulls the same container image on both nodes and verifies that the immutable image IDs match.

### 4.2 Download the model on the master

```bash
./deepseek-v4-flash-nvfp4-stacked.sh download
```

Expected model size is approximately 168 GB with 46 safetensors files.

### 4.3 Verify master model files

```bash
./deepseek-v4-flash-nvfp4-stacked.sh verify-files
```

### 4.4 Sync the model cache to the worker

```bash
./deepseek-v4-flash-nvfp4-stacked.sh sync-worker
```

### 4.5 Verify worker files

```bash
./deepseek-v4-flash-nvfp4-stacked.sh verify-worker
```

### 4.6 Inspect runtime compatibility

```bash
./deepseek-v4-flash-nvfp4-stacked.sh doctor
```

### 4.7 Configure the 200 Gb/s RoCE interface

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

## 5. Start the Server

For general API testing with reasoning enabled by default:

```bash
./deepseek-v4-flash-nvfp4-stacked.sh stop
./deepseek-v4-flash-nvfp4-stacked.sh start
```

For Cline or another OpenAI-compatible agent, initially disable thinking to simplify streaming compatibility:

```bash
ENABLE_THINKING=false \
./deepseek-v4-flash-nvfp4-stacked.sh restart
```

The worker is intentionally started first as rank 1. The master is then started as rank 0 and exposes the API.

### 5.1 Interactive cluster configuration

On `start` and `restart` only, and only when stdin is a TTY, the controller asks for the cluster addresses before doing any work (`prompt_cluster_config()`):

```text
== Cluster configuration (press Enter to keep the current value) ==
  Head (master) node IP [10.100.152.1]:
  Worker node IP        [10.100.152.2]:
  SSH user for nodes    [dgxuser]:
```

- Pressing Enter at a prompt keeps the value shown in brackets.
- The bracketed default is the current value: the built-in default, or whatever was supplied through the environment.
- The SSH-user default is the current login user, not a hard-coded account.
- No other command prompts. `info`, `status`, `logs`, `doctor`, `download`, `sync-worker`, and the test commands run unattended.

This prompt exists so that teammates or customers whose cluster IPs or Linux username differ from the reference topology can use the script directly, without editing it.

### 5.2 Staying non-interactive (automation, cron, CI)

Environment overrides still work and are the supported non-interactive path. Redirect stdin from `/dev/null` (or pipe into the script) so it is not a TTY and the prompt is skipped entirely:

```bash
MASTER_IP=10.100.152.1 WORKER_IP=10.100.152.2 SSH_USER=dgxuser \
./deepseek-v4-flash-nvfp4-stacked.sh restart </dev/null
```

In a crontab, where stdin is already not a TTY, the values simply come from the environment:

```bash
@reboot MASTER_IP=10.100.152.1 WORKER_IP=10.100.152.2 SSH_USER=dgxuser \
  /path/to/deepseek-v4-flash-nvfp4-stacked.sh start </dev/null >>/var/log/ds4flash.log 2>&1
```

If an automated start reaches the wrong node, verify the environment variables first. A prompt that appears to hang in automation means stdin was a TTY when it was not expected to be.

## 6. Confirm Server Status

```bash
./deepseek-v4-flash-nvfp4-stacked.sh status
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

## 7. Required Functional Validation

### 7.1 Basic text generation

```bash
./deepseek-v4-flash-nvfp4-stacked.sh test-text
```

Hardware-validated result:

```text
PASS: Paris is a timeless city of light, where art, romance, and history intertwine along the Seine.
```

### 7.2 Required tool call

```bash
./deepseek-v4-flash-nvfp4-stacked.sh test-tools required
```

Hardware-validated result:

```text
PASS: get_weather({"location": "Bangkok"})
```

### 7.3 Two-turn tool-result loop

This is critical for IDE agents. It verifies:

1. The model emits a tool call.
2. The client sends the tool result back.
3. The model produces a final answer.

```bash
./deepseek-v4-flash-nvfp4-stacked.sh test-tool-loop
```

Hardware-validated result:

```text
PASS (turn 2): The weather in Chiang Mai is currently sunny with a temperature of 32°C.
```

### 7.4 Concurrent request test

```bash
./deepseek-v4-flash-nvfp4-stacked.sh stress 4
```

Hardware-validated result:

```text
4 OK, 0 FAIL (6.8s)
```

### 7.5 Optional reasoning test

```bash
./deepseek-v4-flash-nvfp4-stacked.sh test-reasoning
```

The expected answer contains `1591` for `37 × 43`.

## 8. Historical test-harness heredoc bug (already fixed)

Keep this note for diagnosis only. The fix is included in the current controller; the error below cannot occur with `deepseek-v4-flash-nvfp4-stacked.sh` as shipped.

Earlier test functions used a single-quoted heredoc:

```bash
<<'PYEOF'
```

Inside that heredoc, Python received the literal string `${API_PORT}` instead of the numeric port. The symptom was:

```text
http.client.InvalidURL: nonnumeric port: '${API_PORT}'
```

`test-tools`, `test-tool-loop`, and `stress` now pass values as Python arguments instead:

```bash
python3 - "$API_PORT" "$SERVED_MODEL_NAME" "$mode" <<'PYEOF'
```

Do not diagnose that old error as a vLLM, network, GPU, or tool-parser failure. It was a shell test-harness bug. If it ever reappears, the script being executed is an outdated copy — check it with `info` and compare the reported version.

## 9. Validate Plain Streaming

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

## 10. Validate Tool-Call Streaming

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

## 11. Cline Configuration

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

- The Base URL must stop at `/v1`. The `API (v1)` line printed by `info` is already in that form.
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

## 12. If Cline Appears to Answer and Then Hangs

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

## 13. Live Log Collection

Head:

```bash
./deepseek-v4-flash-nvfp4-stacked.sh logs head 1000
```

Worker:

```bash
./deepseek-v4-flash-nvfp4-stacked.sh logs worker 1000
```

Container status:

```bash
./deepseek-v4-flash-nvfp4-stacked.sh status
```

Shared memory:

```bash
df -h /dev/shm
ls -lah /dev/shm | tail -50
```

Network and RoCE state:

```bash
./deepseek-v4-flash-nvfp4-stacked.sh network-info
ip -s link show "$NCCL_SOCKET_IFNAME"
```

## 14. Interpreting Common Logs

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

## 15. Troubleshooting Decision Tree

### `/v1/models` fails

Check:

```bash
./deepseek-v4-flash-nvfp4-stacked.sh info
./deepseek-v4-flash-nvfp4-stacked.sh status
./deepseek-v4-flash-nvfp4-stacked.sh logs head 300
./deepseek-v4-flash-nvfp4-stacked.sh logs worker 300
```

### `/v1/models` passes but `test-text` fails

Suspect model generation, runtime, worker synchronization, or distributed execution.

### `test-text` passes but `test-tools required` fails

Suspect the chat template, `deepseek_v4` tool parser, tool schema, or output parsing.

### Tool call passes but `test-tool-loop` fails

Suspect handling of assistant tool-call messages, `tool_call_id`, tool-result message format, or the second model turn.

### Non-streaming tests pass but curl streaming fails to end in `[DONE]`

Suspect the vLLM SSE path, proxy buffering, HTTP transport, or an engine hang.

### `start` reached the wrong node

Suspect the cluster addresses, not the runtime. Re-run `start` interactively and read the bracketed defaults in the prompt, or set `MASTER_IP`, `WORKER_IP`, and `SSH_USER` explicitly as shown in section 5.2.

### All server tests pass but Cline hangs

Suspect Cline settings, client stream parsing, reasoning-stream compatibility, task state, or extension bugs.

Do not rebuild the distributed server until the failing layer has been identified.

## 16. Conservative Tuning Order

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
./deepseek-v4-flash-nvfp4-stacked.sh restart
```

Then reduce batching if necessary:

```bash
MAX_NUM_SEQS=2 MAX_NUM_BATCHED_TOKENS=4096 \
./deepseek-v4-flash-nvfp4-stacked.sh restart
```

Do not increase `GPU_MEMORY_UTILIZATION` as the first response to an out-of-memory condition.

## 17. Current Validation Matrix

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

## 18. Security Note

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

## 19. Minimal Handoff Prompt for Another Model

Use the following context when asking another model to continue troubleshooting:

```text
We are running nvidia/DeepSeek-V4-Flash-NVFP4 on two DGX Spark nodes using
vLLM mp backend, TP=2, nnodes=2, runtime image
`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, served as `deepseek-v4-flash`
on port 8000 with max_model_len=1048576.

Controller: deepseek-v4-flash-nvfp4-stacked.sh (SCRIPT_VERSION 3.1.0)

Run `./deepseek-v4-flash-nvfp4-stacked.sh info` first: it prints the model,
model ID, runtime, features, context, the /v1 URL, and whether the server is
RUNNING or stopped on its port.

Hardware-validated passes on 2026-07-22:
- /v1/models
- test-text
- test-tools required
- test-tool-loop
- stress 4: 4 OK, 0 FAIL
- plain SSE stream ends with data: [DONE]
- tool-call SSE emits read_file({"path":"README.md"}), finish_reason=tool_calls,
  and ends with data: [DONE]

The current controller already includes the fix for an old test-harness heredoc
bug where `${API_PORT}` was passed literally to Python. Do not treat that old
InvalidURL error as a vLLM failure.

`start` and `restart` prompt interactively for head IP, worker IP, and SSH user
when stdin is a TTY; Enter keeps the shown default. For automation, set
MASTER_IP / WORKER_IP / SSH_USER and redirect stdin from /dev/null.

If Cline still hangs after displaying an answer, focus on the client layer:
disable default thinking, verify Base URL ends in /v1, update/reload Cline,
start a new task, and inspect client logs. The server's final usage-only SSE
chunk may validly contain choices: [] before [DONE].
```

---

Designed by neronain — https://www.facebook.com/neronain.minidev
