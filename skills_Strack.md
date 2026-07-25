---
name: stacked-large-model-deployment
description: >-
  Design, deploy, validate, troubleshoot, and optimize large language models
  across multiple GPU nodes using Docker, vLLM, NCCL, Tensor Parallelism,
  versioned runtime caches, health checks, and repeatable controller scripts.
version: 1.0
reference_implementation: deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh
language: th
---

# Skill: Stacked Large-Model Deployment

> ชื่อไฟล์ใช้ `skills_Strack.md` ตามที่กำหนด แต่คำศัพท์ทางเทคนิคที่ถูกต้องในเอกสารนี้คือ **Stacked** หรือ **Multi-node distributed inference**

## 1. วัตถุประสงค์

Skill นี้ใช้สำหรับออกแบบและควบคุมระบบรันโมเดลภาษาขนาดใหญ่ที่ไม่สามารถใส่ใน GPU เครื่องเดียวได้ โดยแบ่งโมเดลและงานคำนวณไปยังหลาย GPU หรือหลายเครื่องอย่างเป็นระบบ

เป้าหมายหลักคือทำให้ระบบ:

- โหลดโมเดลขนาดใหญ่ได้สำเร็จ
- ใช้หลายโหนดเป็นระบบ inference เดียว
- ตรวจจับความไม่เข้ากันของ runtime และ kernel ก่อนเริ่มงาน
- เปิดและหยุดระบบได้อย่างทำซ้ำได้
- มีคำสั่งตรวจสุขภาพ ดู log ทดสอบ และ benchmark
- แยกปัญหา Model, Runtime, GPU, Network, Cache และ API ออกจากกัน
- นำโครงสร้างเดิมไปใช้กับโมเดลอื่นได้โดยเปลี่ยน configuration

---

## 2. ใช้ Skill นี้เมื่อใด

ใช้ Skill นี้เมื่อพบสถานการณ์ต่อไปนี้:

- โมเดลมีขนาดใหญ่กว่า VRAM หรือ unified memory ของ GPU หนึ่งตัว
- ต้องการรัน vLLM แบบหลาย GPU หรือหลาย node
- ต้องใช้ Tensor Parallel, Pipeline Parallel หรือ Expert Parallel
- ต้องวางระบบ head/worker ผ่าน Docker และ SSH
- ต้องตั้งค่า NCCL ผ่าน Ethernet, InfiniBand หรือ RoCE
- ต้องจัดการโมเดล quantized เช่น FP8, FP4, NVFP4, AWQ หรือ GPTQ
- เกิดข้อผิดพลาดใน CUDA kernel, FlashInfer, DeepGEMM, Marlin หรือ compiled extension
- ต้องสร้าง controller script ที่มี `start`, `stop`, `status`, `logs`, `test` และ `bench`
- ต้องการย้าย deployment จากโมเดลหนึ่งไปอีกโมเดลหนึ่งอย่างปลอดภัย

ไม่ควรใช้วิธีหลาย node หากโมเดลใส่ GPU เดียวได้สบายและไม่มีเหตุผลด้าน throughput เพราะระบบ distributed เพิ่ม latency, dependency และจุดผิดพลาด

---

## 3. แนวคิดหลัก

ระบบรันโมเดลใหญ่ต้องทำให้สัญญา 6 ชั้นตรงกัน:

```text
Model contract
    ↓
Runtime contract
    ↓
Hardware contract
    ↓
Distributed topology
    ↓
Operations contract
    ↓
Validation contract
```

หากชั้นใดชั้นหนึ่งไม่ตรงกัน ระบบอาจล้มได้แม้ Bash syntax ถูกต้องทั้งหมด

### 3.1 Model contract

ตรวจให้ชัดเจนว่า:

- Architecture คืออะไร
- Dense หรือ Mixture of Experts
- Quantization algorithm คืออะไร
- Weight dtype, activation dtype และ KV-cache dtype คืออะไร
- Tokenizer mode และ chat template คืออะไร
- Reasoning parser และ tool-call parser ที่รองรับ
- Context length สูงสุด
- จำนวน checkpoint shards
- ต้องใช้ speculative decoding หรือไม่
- ต้องใช้ custom code หรือ custom kernels หรือไม่

ไฟล์ที่ควรตรวจ:

```text
config.json
generation_config.json
tokenizer_config.json
tokenizer.json
quantization config
model.safetensors.index.json
model-*.safetensors
```

### 3.2 Runtime contract

ล็อกหรือบันทึกข้อมูลต่อไปนี้:

```text
Docker image digest
vLLM version
Python version
PyTorch version
CUDA runtime version
NVIDIA driver version
NCCL version
FlashInfer version
Marlin/DeepGEMM/TileLang versions
GPU compute capability
```

อย่าเชื่อ Docker tag เพียงอย่างเดียว ให้เปรียบเทียบ immutable image ID หรือ digest ทั้ง head และ worker

### 3.3 Hardware contract

บันทึก:

- จำนวน node
- จำนวน GPU ต่อ node
- GPU architecture และ compute capability
- VRAM หรือ unified memory ต่อ GPU
- CPU RAM
- ความเร็ว storage
- ความเร็ว interconnect
- ชื่อ network interface
- HCA หรือ RDMA device
- GID index สำหรับ RoCEv2

### 3.4 Distributed topology

กำหนดให้ชัดเจน:

```text
World size
Tensor Parallel size
Pipeline Parallel size
Data Parallel size
Expert Parallel size
Node rank
Local rank
Master address
Master port
```

### 3.5 Operations contract

ระบบต้องตอบได้ว่า:

- เตรียม runtime อย่างไร
- ดาวน์โหลดโมเดลอย่างไร
- ตรวจไฟล์อย่างไร
- sync worker อย่างไร
- start/stop/restart อย่างไร
- ดูสถานะและ log อย่างไร
- ล้าง cache อย่างไร
- rollback อย่างไร
- ทดสอบ API และ model behavior อย่างไร

### 3.6 Validation contract

กำหนดเกณฑ์ผ่านก่อนใช้งานจริง:

```text
L1  Containers running
L2  Distributed ranks joined
L3  All checkpoint shards loaded
L4  KV cache created
L5  CUDA/kernel warm-up completed
L6  /health and /v1/models return success
L7  Text generation works
L8  Reasoning parser works
L9  Tool calling works
L10 Concurrent requests pass
L11 Throughput and latency are acceptable
L12 Long-running stability test passes
```

---

## 4. สถาปัตยกรรมอ้างอิง

ตัวอย่างระบบ 2 nodes และ Tensor Parallel 2:

```text
                         Client
                           │
                           ▼
                  OpenAI-compatible API
                   http://HEAD:8000/v1
                           │
                           ▼
             Head node: node-rank 0, TP rank 0
              ┌─────────────────────────────┐
              │ API server                  │
              │ Scheduler / tokenizer       │
              │ GPU: model partition 0      │
              └──────────────┬──────────────┘
                             │
                       NCCL / RoCE
                             │
              ┌──────────────▼──────────────┐
              │ Worker: node-rank 1         │
              │ Headless vLLM process       │
              │ GPU: model partition 1      │
              └─────────────────────────────┘
```

Client มองเห็นเป็นโมเดลเดียว แต่แต่ละ layer ถูกแบ่งคำนวณบน GPU ทั้งสองและแลกเปลี่ยนผลลัพธ์ผ่าน NCCL collective operations

---

## 5. การเลือก Parallelism

| วิธี | ใช้เมื่อ | ลักษณะ |
|---|---|---|
| Tensor Parallel | layer หรือ matrix ใหญ่เกิน GPU เดียว | แบ่ง matrix ภายในแต่ละ layer |
| Pipeline Parallel | โมเดลมีหลาย layer และแบ่งเป็นช่วงได้ | GPU แต่ละกลุ่มรับผิดชอบคนละช่วง layer |
| Data Parallel | ต้องการ throughput จากหลาย replica | แต่ละ replica มีโมเดลครบ |
| Expert Parallel | โมเดล MoE มี expert จำนวนมาก | แบ่ง expert ไปยัง GPU |
| Context Parallel | sequence ยาวมาก | แบ่ง context หรือ sequence |

หลักเลือกเบื้องต้น:

1. ถ้าน้ำหนักโมเดลใส่ GPU เดียวไม่ได้ ให้เริ่มจาก Tensor Parallel
2. ถ้า TP ข้าม node มี communication สูงเกินไป ให้พิจารณา PP หรือ topology แบบผสม
3. ถ้าโมเดลเป็น MoE ให้ตรวจว่า runtime รองรับ EP หรือ distributed expert routing หรือไม่
4. ถ้าต้องการเพิ่ม throughput หลังโมเดลใส่ได้แล้ว ค่อยเพิ่ม Data Parallel

---

## 6. การวางแผนหน่วยความจำ

ประมาณหน่วยความจำต่อ GPU ด้วยสูตร:

```text
Memory per GPU
≈ model weights ÷ tensor_parallel_size
+ KV cache
+ activations/workspace
+ CUDA Graph pool
+ NCCL buffers
+ temporary quantization buffers
+ fragmentation reserve
```

ตัวอย่างโมเดล checkpoint 156.72 GiB และ TP=2:

```text
156.72 ÷ 2 ≈ 78.36 GiB ต่อ GPU สำหรับ weights
```

อย่ากำหนด `gpu-memory-utilization=1.0` เพราะต้องเผื่อ:

- CUDA context
- NCCL communication buffers
- kernel workspace
- graph capture
- JIT compilation
- memory fragmentation

แนวทางเริ่มต้น:

```text
0.80–0.85  เน้นเสถียรภาพ
0.86–0.90  ปรับหลัง stress test
>0.90      ใช้เมื่อวัด memory จริงและมี safety margin
```

เมื่อใช้ context ยาวมาก ให้คำนึงว่า KV cache อาจเป็นตัวใช้ memory หลัก

---

## 7. หลักการออกแบบ Controller Script

สคริปต์ควรใช้:

```bash
set -Eeuo pipefail
```

เพื่อให้:

- หยุดเมื่อคำสั่งล้ม
- หยุดเมื่อใช้ตัวแปรที่ยังไม่กำหนด
- ตรวจจับความล้มเหลวใน pipeline

แบ่งสคริปต์เป็นส่วนดังนี้:

```text
1. Configuration
2. Helper functions
3. Runtime preparation
4. Model download and verification
5. Worker synchronization
6. Distributed start/stop
7. Status and diagnostics
8. Functional tests
9. Benchmark and stress
10. Command dispatch
```

### 7.1 Configuration pattern

ใช้ environment override:

```bash
MODEL_ID="${MODEL_ID:-vendor/model-name}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
```

ทำให้ปรับค่าโดยไม่แก้ source:

```bash
MAX_MODEL_LEN=262144 \
GPU_MEMORY_UTILIZATION=0.82 \
./controller.sh start
```

### 7.2 Fail fast

ตรวจ preconditions ก่อนโหลดโมเดล:

- ห้ามรัน controller ด้วย root
- Docker ใช้งานได้
- SSH worker ได้
- GPU มองเห็น
- image มีอยู่ทั้งสอง node
- image IDs ตรงกัน
- cache เขียนได้
- model snapshot มีอยู่
- shard count ถูกต้อง
- API port ว่าง
- master port ไม่ชน
- network interface มีจริง

หลักการ:

> ให้ล้มภายในไม่กี่วินาทีพร้อมข้อความที่แก้ได้ แทนการโหลดโมเดล 15 นาทีแล้วค่อยล้ม

---

## 8. Permission และการใช้ sudo

ห้ามรัน controller ทั้งไฟล์ด้วย:

```bash
sudo ./controller.sh start
```

เพราะ `sudo` อาจเปลี่ยน:

```text
HOME=/root
USER=root
SSH_USER=root
PATH
SSH key
Docker context
cache location
file ownership
```

ให้รันด้วยผู้ใช้ปกติ แล้วใช้สิทธิ์ root เฉพาะจุด เช่น:

```bash
sudo chown -R "$(id -u):$(id -g)" "$HOME/.cache/flashinfer"
```

Controller ที่ดีควรตรวจ:

```bash
if (( EUID == 0 )); then
  die "Do not run this controller with sudo/root"
fi
```

เมื่อ container สร้าง cache เป็น root ให้มีฟังก์ชันซ่อม ownership ทั้ง head และ worker

---

## 9. การจัดการ Runtime Image

### 9.1 Pull และ lock image

ทำบนทั้งสอง node:

```bash
docker pull "$VLLM_IMAGE"
docker inspect "$VLLM_IMAGE" --format '{{.Id}}'
```

บันทึก image ID ลง lock file เช่น:

```text
~/.cache/huggingface/.vllm-stacked-image-id
```

ก่อน start ต้องยืนยัน:

```text
head image ID == worker image ID == locked image ID
```

ถ้าไม่ตรง ให้หยุดทันทีและ re-pull ทั้งสอง node

### 9.2 เหตุผลที่ tag ไม่เพียงพอ

Tag เดียวกันอาจชี้ไปคนละ digest หาก image ถูกอัปเดตต่างเวลา จึงต้องใช้ digest หรือ image ID เป็น source of truth

---

## 10. การจัดการ Model Artifact

แยก lifecycle ออกจาก `start`:

```text
download
verify-files
sync-worker
verify-worker
start
```

### 10.1 ดาวน์โหลดแบบ resume-safe

ใช้ Hugging Face `snapshot_download` ภายใน runtime image เพื่อให้ Python dependency ตรงกัน

### 10.2 Resolve snapshot path

รองรับโครงสร้าง:

```text
models--ORG--MODEL/
├── refs/main
├── blobs/
└── snapshots/<commit>/
```

ลำดับ resolve:

1. ตรวจ `snapshots/$MODEL_REVISION`
2. อ่าน `refs/$MODEL_REVISION`
3. ใช้ commit ที่อ้างถึง
4. fallback ไป snapshot ที่มีอยู่เมื่อจำเป็น

### 10.3 ตรวจไฟล์

ตรวจอย่างน้อย:

- config files ครบ
- shard count ตรงค่าที่คาด
- shard ทุกไฟล์มีขนาดสมเหตุผล
- symlink ไม่ขาด
- worker มี shard ครบ
- ใช้ SHA-256 เมื่อจำเป็นต้องยืนยันความเหมือน

อย่าตรวจเพียงว่า “พบ shard อย่างน้อยหนึ่งไฟล์” ต้องบังคับจำนวนให้ตรง

---

## 11. Cache Versioning

Compiled cache อาจทำให้ container ใหม่ล้มแม้ image ถูกต้อง เพราะ bind-mounted cache เก่ายังอยู่บน host

Cache ที่มี compiled artifacts ได้แก่:

```text
FlashInfer JIT cache
vLLM compile cache
Torch extensions
Triton cache
TileLang cache
DeepGEMM cache
```

แยก cache ตาม immutable runtime identity:

```text
~/.cache/flashinfer/<image-id-prefix>/
```

หรือใช้ key ที่รวม:

```text
image digest + package version + CUDA architecture
```

ตัวอย่างแนวคิด:

```bash
IMAGE_ID=$(docker inspect "$VLLM_IMAGE" --format '{{.Id}}')
CACHE_KEY=${IMAGE_ID#sha256:}
CACHE_KEY=${CACHE_KEY:0:16}
FLASHINFER_RUNTIME_CACHE="$FLASHINFER_CACHE/$CACHE_KEY"
```

เมื่อเกิด ABI mismatch เช่น wrapper ส่งจำนวน argument ไม่ตรง compiled module ให้สงสัย cache/version mismatch ก่อน

---

## 12. Network และ NCCL

กำหนด IP สองชุดได้:

```text
Management IP  ใช้ SSH และควบคุม node
Transport IP   ใช้ NCCL และ distributed traffic
```

Environment สำคัญ:

```bash
NCCL_SOCKET_IFNAME=<fabric-interface>
GLOO_SOCKET_IFNAME=<fabric-interface>
TP_SOCKET_IFNAME=<fabric-interface>
NCCL_IB_HCA=<rdma-hca>
NCCL_IB_GID_INDEX=3
NCCL_IB_DISABLE=0
```

ถ้าไม่มี RDMA/HCA ให้ fallback TCP:

```bash
NCCL_IB_DISABLE=1
```

แต่ throughput อาจลดลงมาก

ตรวจสอบ:

```bash
ip link show
ip -4 addr show
rdma link
ibv_devinfo
nvidia-smi
ss -tlnp
```

หลักสำคัญ:

- Interface ต้องมีชื่อเหมือนจริงในแต่ละ node
- Master address ต้องเข้าถึงจาก worker ได้
- Master port ต้องเปิดและไม่ชน
- MTU, routing และ firewall ต้องสอดคล้องกัน
- ตรวจ latency และ bandwidth ก่อนโทษโมเดล

---

## 13. ลำดับการเปิดระบบ

ใช้ worker-first startup:

```text
1. ตรวจ runtime และ model artifacts
2. ลบ stale containers
3. เตรียม cache และ permissions
4. สร้าง common vLLM flags
5. เปิด worker node-rank 1 แบบ headless
6. ตรวจว่า worker container ยัง running
7. เปิด head node-rank 0 และ API server
8. Poll health endpoint จนพร้อม
9. ถ้า timeout ให้เก็บ logs ทั้งสอง node
```

เหตุผลที่เปิด worker ก่อน:

- worker พร้อม join distributed group
- ลดโอกาส head รอ rank ที่ยังไม่ถูกเปิด
- ตรวจ worker failure ได้ก่อนเสียเวลาโหลด head

Common flags ต้องใช้ชุดเดียวกันทั้ง head และ worker เช่น:

```bash
--tensor-parallel-size 2
--nnodes 2
--master-addr "$TRANSPORT_IP_MASTER"
--master-port "$MASTER_PORT"
--max-model-len "$MAX_MODEL_LEN"
--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
--kv-cache-dtype "$KV_CACHE_DTYPE"
--moe-backend "$MOE_BACKEND"
```

ต่างกันเฉพาะ:

```text
--node-rank
--host
--port
--headless
VLLM_HOST_IP
```

---

## 14. Backend Selection และ Stability-first Strategy

อย่าเลือก backend ที่เร็วที่สุดก่อนพิสูจน์ความเสถียร

ลำดับที่แนะนำ:

```text
1. เลือก backend ที่รองรับ checkpoint และ hardware แน่นอน
2. เปิดระบบให้ผ่าน
3. ทดสอบ text/reasoning/tools
4. stress test
5. benchmark baseline
6. ทดลอง backend เร็วกว่าเพียงตัวเดียว
7. benchmark และ stress ซ้ำ
8. rollback ทันทีเมื่อเกิด regression
```

สำหรับ NVFP4 MoE ตัวเลือกอาจประกอบด้วย:

```text
MARLIN
FLASHINFER_CUTLASS
FLASHINFER_TRTLLM
```

Backend ไม่สามารถสลับได้โดยดูชื่ออย่างเดียว ต้องตรวจ:

- checkpoint quantization format
- GPU compute capability
- vLLM implementation
- FlashInfer/CUTLASS build
- JIT cache version
- runtime image compatibility

Error ลักษณะนี้:

```text
Expected 7 but got 8 arguments
```

มักชี้ไปที่ API/ABI mismatch ของ Python wrapper กับ compiled extension ไม่ใช่ model shard เสียหรือ VRAM ไม่พอ

---

## 15. CUDA Graph และ Compilation

CUDA Graph ช่วยลด kernel launch overhead แต่ใช้ memory เพิ่มและอาจกระทบความเสถียรบน runtime ใหม่

แนวทาง:

- เริ่มจาก `PIECEWISE`
- จำกัด capture size ให้สอดคล้องกับ `MAX_NUM_SEQS`
- ดู memory estimate และ actual usage
- stress หลาย batch sizes
- ใช้ eager mode เฉพาะ diagnostic เมื่อจำเป็น

ตัวแปรที่เกี่ยวข้อง:

```text
CUDAGRAPH_MODE
MAX_CUDAGRAPH_CAPTURE_SIZE
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
```

หากปิด graph memory profiler ต้องเผื่อ memory เพิ่ม เพราะ KV cache allocation อาจไม่ได้นับ graph pool

---

## 16. Startup Timeout

อย่ากำหนด timeout แบบเดา

คำนวณจาก:

```text
Startup timeout
> weight loading time
+ KV cache profiling
+ kernel compilation
+ autotuning
+ CUDA graph capture
+ network synchronization
+ safety margin
```

โมเดลใหญ่บน storage ปกติอาจใช้เวลาโหลด weights มากกว่า 10 นาที และ warm-up เพิ่มอีกหลาย นาที

ใช้ health polling แทน `sleep` คงที่:

```bash
while before_deadline; do
  if curl -fsS http://127.0.0.1:8000/v1/models; then
    ready
  fi
  sleep 10
done
```

---

## 17. Observability Commands

Controller ควรมีคำสั่งอย่างน้อย:

```text
runtime-info
network-info
doctor
status
logs head
logs worker
props
```

### 17.1 doctor

แสดง:

- image ID ทั้งสอง node
- GPU name และ memory
- driver/CUDA
- vLLM version
- PyTorch version
- FlashInfer version
- compute capability
- cache paths
- selected backend

### 17.2 status

แสดง:

- container state
- GPU utilization
- memory usage
- API health
- model listing

### 17.3 logs

ต้องดึงได้ทั้ง:

```bash
./controller.sh logs head 300
./controller.sh logs worker 300
```

---

## 18. การอ่าน Log แบบ Root-cause

ใช้หลัก:

> หา error ตัวแรกที่เฉพาะเจาะจงที่สุด ไม่ใช่ error สุดท้ายที่เป็นผลตามมา

ตัวอย่างลำดับ:

```text
TypeError: Expected 7 but got 8
→ Worker failed
→ EngineCore failed
→ API server exited
```

Root cause คือ TypeError แรก ไม่ใช่ `EngineCore failed`

แบ่ง log ตาม phase:

```text
Phase 1  Runtime/import
Phase 2  Distributed/NCCL initialization
Phase 3  Model config resolution
Phase 4  Weight loading
Phase 5  Quantization/backend initialization
Phase 6  KV cache profiling
Phase 7  JIT/autotune/warm-up
Phase 8  CUDA graph capture
Phase 9  API startup
Phase 10 Request execution
```

ตำแหน่งที่ล้มช่วยลดขอบเขตการตรวจสอบได้มาก

---

## 19. Troubleshooting Matrix

| อาการ | สาเหตุที่ควรตรวจ | แนวทางแรก |
|---|---|---|
| SSH worker ไม่ได้ | key, user, route, firewall | ทดสอบ SSH แบบ BatchMode |
| Image not present on worker | ใช้ sudo ทำให้ SSH_USER=root หรือ image ไม่มีจริง | รัน controller แบบ non-root และ inspect image |
| Permission denied ใน cache | cache เป็นเจ้าของ root | chown เฉพาะ cache หรือใช้ repair function |
| Image IDs ไม่ตรง | pull คนละเวลา/tag เปลี่ยน | pull ใหม่และ lock digest |
| NCCL timeout | interface, port, firewall, route | ตรวจ transport IP และ NCCL env |
| Shard missing | sync ไม่ครบ, symlink ขาด | verify count และ SHA-256 |
| OOM ตอนโหลด weights | TP ต่ำเกินไป, backend workspace สูง | เพิ่ม TP หรือลด memory settings |
| OOM หลัง warm-up | KV cache หรือ CUDA Graph มากเกินไป | ลด utilization/context/capture size |
| ABI argument mismatch | wrapper กับ compiled cache คนละ version | clear/version cache หรือเปลี่ยน backend |
| API พร้อมแต่ generate ไม่ได้ | tokenizer/parser/template/backend | รัน test-text และดู request log |
| Tool call ไม่ออก | parser/chat template/request payload | test required mode และตรวจ JSON |
| ระบบค้างหลังหลาย request | CUDA graph/kernel/backend instability | PIECEWISE/eager test และ stress ซ้ำ |
| Start script timeout แต่ process ยังโหลด | timeout สั้นเกินจริง | เพิ่ม timeout และใช้ health polling |

---

## 20. Functional Test Ladder

หลัง API พร้อม ให้ทดสอบตามลำดับ:

```bash
./controller.sh status
./controller.sh props
./controller.sh test-text
./controller.sh test-reasoning
./controller.sh test-tools required
./controller.sh test-tool-loop
./controller.sh stress 4
./controller.sh stress 6
./controller.sh stress 8
./controller.sh bench
```

### 20.1 สิ่งที่แต่ละ test พิสูจน์

| Test | สิ่งที่ตรวจ |
|---|---|
| props | API และ model registration |
| test-text | tokenizer, prefill, decode, sampler |
| test-reasoning | reasoning parser และ chat template kwargs |
| test-tools required | tool-call parser แบบบังคับ |
| test-tool-loop | multi-turn continuation หลัง tool result |
| stress | scheduler, KV cache, concurrency, stability |
| bench | throughput และ baseline performance |

REST JSON ที่ยิงตรงต้องส่ง field ตาม API จริง ไม่ควรใช้ wrapper-only field เช่น `extra_body` เป็น key ระดับบนโดยไม่ตรวจ schema

---

## 21. Security

หาก bind:

```bash
--host 0.0.0.0
```

API จะรับจากทุก interface ที่ firewall อนุญาต

ต้องพิจารณา:

- bind เฉพาะ private IP
- host firewall
- reverse proxy
- TLS
- API key/authentication
- rate limiting
- request size limit
- audit log
- ห้ามเปิด master/NCCL port สู่ public network

ค่า `api_key="none"` เหมาะกับเครือข่ายปิดเท่านั้น

---

## 22. Reference Workflow

### Phase A: Discovery

1. อ่าน model config และ quantization config
2. คำนวณ weight memory ต่อ GPU
3. เลือก topology
4. ตรวจ hardware และ network
5. เลือก runtime image ที่รองรับ GPU architecture

### Phase B: Runtime preparation

1. Pull image บนทุก node
2. เปรียบเทียบ image IDs
3. บันทึก lock ID
4. รัน doctor
5. เตรียม writable caches

### Phase C: Artifact preparation

1. Download model บน head
2. Verify local shards
3. Sync ไป worker
4. Verify worker count/hash

### Phase D: Deployment

1. Clear stale containers
2. เปิด worker headless
3. ตรวจ worker running
4. เปิด head/API
5. Poll `/v1/models`

### Phase E: Validation

1. Basic generation
2. Reasoning
3. Tool calling
4. Multi-turn tool loop
5. Concurrent stress
6. Benchmark

### Phase F: Optimization

ปรับครั้งละหนึ่งตัว:

```text
MOE backend
GPU memory utilization
max model length
max num sequences
max batched tokens
CUDA graph mode
capture sizes
NCCL interface/HCA
```

บันทึก baseline ก่อนทุกการเปลี่ยน

---

## 23. Template สำหรับนำไปใช้กับโมเดลอื่น

เปลี่ยนค่าต่อไปนี้ก่อน:

```bash
MODEL_ID="vendor/new-model"
MODEL_REVISION="main"
SERVED_MODEL_NAME="new-model"
SHARD_COUNT=<expected>
TOTAL_SIZE_APPROX_GB=<expected>
MAX_MODEL_LEN=<supported>
KV_CACHE_DTYPE=<supported>
MOE_BACKEND=<supported>
TOOL_CALL_PARSER=<supported-or-empty>
REASONING_PARSER=<supported-or-empty>
```

จากนั้นตรวจว่า flags เฉพาะ DeepSeek ไม่ถูกนำไปใช้กับ architecture อื่นโดยอัตโนมัติ

สร้าง capability matrix:

| Capability | New model value |
|---|---|
| Architecture | |
| Dense/MoE | |
| Weight quantization | |
| Activation dtype | |
| KV cache dtype | |
| TP support | |
| PP support | |
| EP support | |
| Tool parser | |
| Reasoning parser | |
| Chat template kwargs | |
| Speculative decoding | |
| Max context | |
| Preferred backend | |
| Fallback backend | |

---

## 24. Controller Script Skeleton

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

# 1. Configuration
MODEL_ID="${MODEL_ID:-vendor/model}"
VLLM_IMAGE="${VLLM_IMAGE:-vendor/vllm:version}"
MASTER_IP="${MASTER_IP:-10.0.0.1}"
WORKER_IP="${WORKER_IP:-10.0.0.2}"
TP_SIZE="${TP_SIZE:-2}"

# 2. Helpers
die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date '+%H:%M:%S')] $*"; }
ssh_worker() { ssh -o BatchMode=yes "$SSH_USER@$WORKER_IP" "$@"; }

require_non_root() {
  (( EUID != 0 )) || die "Run as a normal user"
}

# 3. Preconditions
check_runtime() { :; }
check_network() { :; }
check_model() { :; }
check_cache_permissions() { :; }

# 4. Lifecycle
prepare_runtime() { :; }
download_model() { :; }
verify_model() { :; }
sync_worker() { :; }

# 5. Orchestration
start_worker() { :; }
start_head() { :; }
wait_for_health() { :; }
start() {
  require_non_root
  check_runtime
  check_network
  check_model
  check_cache_permissions
  start_worker
  start_head
  wait_for_health
}

stop() { :; }
status() { :; }
logs() { :; }
doctor() { :; }

# 6. Validation
test_text() { :; }
stress() { :; }
bench() { :; }

# 7. Dispatch
case "${1:-help}" in
  prepare-runtime) prepare_runtime ;;
  download) download_model ;;
  verify) verify_model ;;
  sync-worker) sync_worker ;;
  doctor) doctor ;;
  start) start ;;
  stop) stop ;;
  status) status ;;
  logs) logs ;;
  test-text) test_text ;;
  stress) stress ;;
  bench) bench ;;
  *) echo "Usage: $0 {prepare-runtime|download|verify|sync-worker|doctor|start|stop|status|logs|test-text|stress|bench}" ;;
esac
```

---

## 25. Acceptance Criteria

ถือว่า deployment พร้อมใช้งานเมื่อ:

- [ ] ผู้ใช้ปกติสามารถรัน controller ได้โดยไม่ใช้ sudo
- [ ] Head และ worker ใช้ image ID เดียวกัน
- [ ] Runtime version ถูกบันทึก
- [ ] SSH แบบ non-interactive ผ่าน
- [ ] Model shards ครบทั้งสอง node
- [ ] Cache ทุกชุดเขียนได้และ versioned
- [ ] Distributed world size ตรงค่าที่ออกแบบ
- [ ] NCCL เลือก interface ที่ถูกต้อง
- [ ] Model weights โหลดครบ
- [ ] KV cache ถูกสร้างโดยไม่ OOM
- [ ] CUDA Graph/JIT warm-up ผ่าน
- [ ] API health ผ่าน
- [ ] Text generation ผ่าน
- [ ] Reasoning และ tool calling ผ่านเมื่อรองรับ
- [ ] Stress test ผ่านหลายระดับ concurrency
- [ ] ไม่มี memory leak หรือ hang ใน soak test
- [ ] Firewall และ authentication ได้รับการพิจารณา
- [ ] มีวิธี rollback image และ configuration

---

## 26. Optimization Log Template

บันทึกทุกการทดลอง:

```markdown
### Experiment: <name>

- Date:
- Runtime image ID:
- Model revision/commit:
- Hardware/topology:
- Changed variable:
- Previous value:
- New value:
- Prompt/context profile:
- Concurrency:
- TTFT:
- Decode tok/s:
- Total throughput:
- Peak GPU memory:
- Errors/warnings:
- Stability duration:
- Decision: keep / rollback / investigate
```

เปลี่ยนครั้งละหนึ่งตัวเพื่อให้ระบุสาเหตุของผลลัพธ์ได้

---

## 27. หลักคิดสำคัญที่ต้องจำ

1. **เสถียรก่อน เร็วทีหลัง**
2. **Pin runtime ด้วย digest ไม่ใช่ tag อย่างเดียว**
3. **Compiled cache ต้อง version ตาม runtime**
4. **อย่ารัน controller ทั้งชุดด้วย sudo**
5. **Worker-first และตรวจ worker ก่อนเปิด head**
6. **Preflight checks ต้องล้มเร็วและบอกวิธีแก้**
7. **จำนวน shards ต้องตรวจตรง ไม่ใช่แค่พบไฟล์**
8. **Health check ดีกว่า sleep คงที่**
9. **API เปิดได้ยังไม่เท่ากับ inference ถูกต้อง**
10. **อ่าน error แรกที่เฉพาะเจาะจงที่สุด**
11. **อย่าปรับหลายค่าในครั้งเดียว**
12. **เก็บ baseline และ rollback path เสมอ**

---

## 28. Mental Model สำหรับแก้ปัญหา

เมื่อระบบล้ม ให้ถามตามลำดับ:

```text
1. Model artifact ถูกต้องหรือไม่
2. Runtime image เหมือนกันหรือไม่
3. GPU/driver/CUDA รองรับหรือไม่
4. Cache เก่าปะปนหรือไม่
5. Network และ NCCL เชื่อม rank ได้หรือไม่
6. Memory พอใน phase ใด
7. Backend/quantization kernel เข้ากันหรือไม่
8. Warm-up/CUDA graph ผ่านหรือไม่
9. API ขึ้นหรือไม่
10. Request schema และ parser ถูกหรือไม่
```

อย่ากระโดดไปแก้ performance ก่อนที่ functional correctness จะผ่าน

---

## 29. Reference Commands สำหรับ Implementation ปัจจุบัน

```bash
# เตรียมและล็อก runtime image
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh prepare-runtime

# ตรวจ environment และ compatibility
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh doctor

# ดาวน์โหลดและตรวจโมเดล
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh download
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh verify-files

# ส่งโมเดลและตรวจ worker
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh sync-worker
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh verify-worker

# เปิดระบบ
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh start

# ตรวจสอบ
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh status
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh logs head 300
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh logs worker 300

# ทดสอบ
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh test-text
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh test-reasoning
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh test-tools required
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh test-tool-loop
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh stress 4
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh bench

# หยุดระบบ
./deepseek-v4-flash-nvfp4-stacked-fixed-v8.1-permission-fixed.sh stop
```

---

## 30. ผลลัพธ์การเรียนรู้จาก Skill นี้

เมื่อเข้าใจและใช้ Skill นี้ได้ ผู้ปฏิบัติงานควรสามารถ:

- ประเมินว่าโมเดลต้องใช้ GPU กี่ตัว
- เลือก TP/PP/DP/EP ได้อย่างมีเหตุผล
- วางแผน memory และ context length
- สร้าง Docker-based multi-node controller
- ตั้งค่า NCCL/RoCE
- จัดการ Hugging Face snapshots และ shards
- ป้องกัน runtime drift และ stale JIT cache
- วิเคราะห์ stack trace ระดับ CUDA kernel
- สร้าง test ladder และ benchmark baseline
- ย้าย deployment ไปโมเดลใหม่โดยใช้ capability matrix
- ทำระบบให้ reproducible, diagnosable และ rollback ได้

---

## 31. สรุป

การรันโมเดลขนาดใหญ่แบบ Stacked ไม่ใช่เพียงการเปิด Docker สองเครื่อง แต่เป็นการประสาน:

```text
Model
+ Quantization
+ Runtime
+ GPU architecture
+ Kernel backend
+ Cache version
+ Distributed topology
+ Network
+ Memory allocation
+ Orchestration
+ Validation
```

Controller script ที่ดีจึงทำหน้าที่เป็น **deployment control plane ขนาดย่อม** ซึ่งต้องตรวจสอบ เตรียม เปิด เฝ้าดู ทดสอบ และหยุดระบบได้อย่างปลอดภัยและทำซ้ำได้
