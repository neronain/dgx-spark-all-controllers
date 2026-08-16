# DGX Spark Controller Collection v3.1.0

ชุด Controller สำหรับรันโมเดลบน NVIDIA DGX Spark จำนวน **21 ตัว** — เอาไปใช้งานได้ทันที ไม่ใช่ตัว generate script

ออกแบบโดย **neronain** · <https://www.facebook.com/neronain.minidev>

ทุกตัวใช้มาตรฐานเดียวกัน: ชุดคำสั่งเหมือนกัน, override context/port/bind/IP ได้, มี ASCII banner + คำสั่ง `info`, ไม่ผูกกับชื่อผู้ใช้หรือ IP ของเครื่องผู้พัฒนา

## เริ่มเร็วที่สุด

```bash
chmod +x ./*.sh audit-controllers.py

# โมเดลอะไร · พอร์ตอะไร · รองรับฟีเจอร์ไหน · รันอยู่หรือยัง
./qwen3-coder-next-single.sh info
```

ตัวอย่างผลลัพธ์:

```text
   ____   ____ __  __    ____                   _
  |  _ \ / ___|\ \/ /   / ___| _ __   __ _ _ __| | __
  | | | | |  _  \  /    \___ \| '_ \ / _` | '__| |/ /
  | |_| | |_| | /  \     ___) | |_) | (_| | |  |   <
  |____/ \____|/_/\_\   |____/| .__/ \__,_|_|  |_|\_\
                              |_|
       =[ DGX Spark Controller · v3.1.0 ]
+ -- --=[ Qwen3-Coder-Next (NVFP4-GB10) ]
+ -- --=[ vLLM (Docker) · code · tools (qwen3_coder) · 256K ctx ]
+ -- --=[ Designed by neronain · fb.com/neronain.minidev ]

  Model     : Qwen3-Coder-Next (NVFP4-GB10)
  Model ID  : ucbye/Qwen3-Coder-Next-NVFP4-GB10
  Runtime   : vLLM (Docker)
  Features  : code · tools (qwen3_coder) · 256K ctx
  Context   : 131072 tokens
  API (v1)  : http://192.168.101.127:8000/v1
  State     : RUNNING  (port 8000)
```

`State` ตรวจจากการเรียก `http://127.0.0.1:<port>/health` จริง — ไม่ใช่การเดา

ใช้ `info` หรือ `banner` ก็ได้ (ความหมายเดียวกัน)

## ไฟล์ Controller

### Single-node · vLLM (Docker)

```text
gemma-4-31b-it-uncensored-single.sh                                  Gemma-4-31B Uncensored (Iambackup)
gemma4-31b-single.sh                                                 Gemma-4-31B-IT NVFP4
llama33-70b-nvfp4-single.sh                                          Llama-3.3-70B-Instruct NVFP4
nemotron-3-super-single.sh                                           Nemotron-3-Super-120B-A12B NVFP4
nemotron-omni-aeon-single.sh                                         Nemotron-3-Nano-Omni AEON NVFP4
ornith-1.0-35b-uncensored-heretic-nvfp4-fp8dense-gb10-dgx-spark.sh   Ornith-1.0-35B Uncensored NVFP4/FP8
qwen3-coder-next-nvfp4-gb10-dgx-spark.sh                             Qwen3-Coder-Next NVFP4 (alt-source)
qwen3-coder-next-single.sh                                           Qwen3-Coder-Next NVFP4-GB10
```

### Single-node · llama.cpp (GGUF)

ตัวเหล่านี้ใช้ `--jinja` ในการเสิร์ฟ — เพื่อให้ llama.cpp ใช้ tool/function-calling template ของโมเดลได้ถูกต้อง (วิธีหนึ่ง modern llama.cpp ใช้)

```text
gemma-4-26b-a4b-it-gguf-single.sh                    Gemma-4-26B-A4B-it GGUF (text, alt-source)
gemma-4-31b-it-uncensored-heretic-q8_0-dgx-spark.sh  Gemma-4-31B Uncensored Heretic Q8_0
gemma4-26-a4b-q8xl-single.sh                         Gemma-4-26B-A4B-it Q8_K_XL (vision)
gpt-oss-120b-f16-single.sh                           GPT-OSS-120B F16
ornith-1.0-35b-bf16-dgx-spark.sh                     Ornith-1.0-35B BF16 (vision)
qwen3-vl-32b-instruct-1m-bf16-dgx-spark.sh           Qwen3-VL-32B-Instruct-1M BF16 (vision)
qwen3-vl-32b-thinking-single.sh                      Qwen3-VL-32B-Thinking (vision + thinking)
qwen36-hauhau-q6kp-single.sh                         Qwen3.6-35B-A3B Uncensored HauhauCS Q6_K_XL
redteam-modelctl.sh                                  GLM-4.7-Flash Uncensored Heretic NEO-CODE
```

### Stacked / multi-node (2× DGX Spark)

```text
deepseek-v4-flash-nvfp4-stacked.sh   DeepSeek-V4-Flash NVFP4 · 1M ctx
gemma4-31b-stacked.sh                Gemma-4-31B-IT NVFP4
minimax-m27-luke-stacked.sh          MiniMax-M2.7 NVFP4
vllm-stackctl.sh                     Llama-3.3-70B-Instruct (generic stacked)
```

## คำสั่งมาตรฐาน

```text
info | banner    โมเดล/พอร์ต/ฟีเจอร์/สถานะ (เริ่มที่นี่)
start            เปิดเซิร์ฟเวอร์
stop | restart   หยุด / เปิดใหม่
status           สถานะ process หรือ container + API
logs [N]         log ล่าสุด
network-info     bind address และ endpoint ที่ประกาศออกไป
client-config    ค่าตั้งฝั่ง client พร้อม token budget
download         ดาวน์โหลดน้ำหนักโมเดล (resume ได้)
help             วิธีใช้ทั้งหมดของตัวนั้น
```

หลายตัวมีคำสั่งเสริมเฉพาะรุ่น เช่น `verify-files`, `prepare-runtime`, `props`, `bench`, `stress`, `test-text`, `test-reasoning`, `test-image`, `test-tools`, `test-tool-loop`, `doctor`, `sync-worker` — ดูด้วย `help` ของไฟล์นั้น

## Context และ Port

```bash
./controller.sh start \
  --context 65536 \
  --port 8001
```

ตัวเลือกทั้งหมด:

```text
--context TOKENS
--port PORT
--bind ADDRESS
--advertise-ip ADDRESS
--interface NAME
--client-input TOKENS|auto
--client-output TOKENS
```

ใช้ environment variable ได้เช่นกัน (vLLM ใช้ `MAX_MODEL_LEN`, llama.cpp ใช้ `CTX_SIZE`):

```bash
MAX_MODEL_LEN=65536 \
API_PORT=8001 \
ADVERTISE_INTERFACE=enp1s0 \
./qwen3-coder-next-single.sh start
```

```bash
CTX_SIZE=65536 \
API_PORT=8001 \
ADVERTISE_INTERFACE=enp1s0 \
./qwen36-hauhau-q6kp-single.sh start
```

ทุกตัวปฏิเสธค่าที่ใช้ไม่ได้: port ต้องอยู่ใน 1–65535 และ context ต้องมากกว่า 0

## เอาไปใช้บนเครื่องของคุณเอง

สคริปต์ชุดนี้ไม่ผูกกับเครื่องผู้พัฒนา:

- **ชื่อผู้ใช้** — `SSH_USER` ใช้ค่าเริ่มต้นเป็นผู้ใช้ปัจจุบัน (`${USER:-$(id -un)}`) ไม่ใช่ชื่อที่ hard-code ไว้ · path ต่าง ๆ อ้างจาก `$HOME` (override ได้ด้วย `USER_HOME`)
- **IP ของ cluster** — Stacked controller จะ **ถามตอน `start` / `restart`** ถ้ารันบน terminal จริง
- **พอร์ตและ context** — override ได้ทุกตัวตามหัวข้อด้านบน

### Stacked: ถาม Head / Worker / SSH user

```text
== Cluster configuration (press Enter to keep the current value) ==
  Head (master) node IP [10.100.152.1]: 192.168.101.127
  Worker node IP        [10.100.152.2]: 192.168.101.128
  SSH user for nodes    [dgxuser]:
```

- กด Enter = ใช้ค่าเดิม
- ถามเฉพาะ `start` และ `restart` เท่านั้น — คำสั่งอื่นเช่น `info`, `status`, `logs` ไม่ถาม
- ถามเฉพาะเมื่อ stdin เป็น TTY ดังนั้น cron/automation จะไม่ค้าง

รันแบบไม่ถาม (สำหรับ automation) ให้กำหนดค่าผ่าน env และปิด stdin:

```bash
MASTER_IP=192.168.101.127 \
WORKER_IP=192.168.101.128 \
SSH_USER=dgxuser \
./deepseek-v4-flash-nvfp4-stacked.sh start </dev/null
```

ค่าจาก env ชนะค่าที่ถามเสมอเมื่อรันแบบไม่มี TTY

## การเลือก IP ที่ประกาศให้ client

ระบบไม่ใช้ IP ตายตัว และไม่ใช้ IP ตัวแรกจาก `hostname -I` เป็นวิธีหลัก

ลำดับการเลือก:

1. `--advertise-ip`
2. `--interface`
3. source address จาก `ip route get 1.1.1.1`
4. global IPv4 ที่ไม่ใช่ Docker/CNI/cluster-like interface
5. `hostname -I` เฉพาะ fallback สุดท้าย

ตรวจก่อนรัน:

```bash
./qwen36-hauhau-q6kp-single.sh network-info
```

บังคับ interface หรือ IP:

```bash
./qwen36-hauhau-q6kp-single.sh start --context 131072 --port 8001 --interface enp1s0
./qwen36-hauhau-q6kp-single.sh start --advertise-ip 192.168.101.127 --port 8001
```

## Bind กับ Advertise แยกกัน

`--bind` = address ที่เซิร์ฟเวอร์ฟังจริง

`--advertise-ip` / `--interface` = URL ที่แสดงให้ Hermes, OpenClaw, Cline และเครื่องอื่นใช้

ค่าที่แนะนำ:

```bash
./controller.sh start --bind 0.0.0.0 --interface enp1s0 --port 8000
```

## DGX หลายเครื่อง

DGX สองเครื่องใช้ port เดียวกันได้เพราะเป็นคนละ IP:

```text
DGX-1  192.168.101.127:8000
DGX-2  192.168.101.128:8000
```

สำหรับ Stacked Controller ค่า `MASTER_IP` / `WORKER_IP` ใช้สื่อสารภายใน cluster ส่วน public API URL ใช้ advertised address แยกกัน

```bash
./minimax-m27-luke-stacked.sh start --context 65536 --port 8000 --interface enp1s0
```

## Client token budget

เมื่อใช้ `--client-input auto` สคริปต์คำนวณ:

```text
client input = server context - max output - 8192 overhead
```

ตัวอย่าง:

```text
context      65536
max output    8192
overhead      8192
client input 49152
```

สคริปต์ปฏิเสธค่าที่ input + output มากกว่า server context

## ปัญหา Bash numeric separator

Bash arithmetic ไม่รองรับ underscore ในตัวเลข:

```bash
(( model_size > 25_000_000_000 ))   # error: value too great for base
```

ทั้ง 21 ตัวไม่มี pure numeric literal ที่ใช้ underscore เหลืออยู่ · ค่าขนาดไฟล์ใช้ decimal ปกติ:

```bash
MODEL_SIZE_BYTES="30649317504"
MMPROJ_SIZE_BYTES="899283072"
```

รุ่น GGUF ที่มี exact metadata จะตรวจ exact byte size, GGUF magic header และ SHA-256

## ตรวจทั้งหมด

```bash
./verify-all.sh
```

ผลที่ต้องได้:

```text
Audited 21 scripts: errors=0, warnings=0
All 21 DGX controllers passed static validation.
```

`verify-all.sh` ตรวจทุกตัว: `bash -n`, `help`, `info` (ต้องมี banner + Model/Runtime/Features/State), `network-info`, `client-config`, ปฏิเสธ port 70000, ปฏิเสธ context 0, Stacked ต้องมี `prompt_cluster_config` และต้องไม่ถามตอนสั่ง `info` และต้องไม่มีชื่อผู้ใช้ hard-code เหลือ

ตรวจ Controller ในเครื่อง:

```bash
python3 audit-controllers.py "$HOME"
python3 audit-controllers.py "$HOME" --json
```

Audit ครอบคลุม:

```text
Bash syntax
numeric separators
pipefail + grep -q
hard-coded single-node MASTER_IP
context/port ที่ override ไม่ได้
missing --context / --port
missing network-info
การเลือก IP จาก hostname -I โดยตรง
missing SCRIPT_VERSION
missing banner()/info() หรือ dispatch info|banner)
hard-coded ชื่อผู้ใช้ของผู้พัฒนา
stacked ที่ไม่มี prompt_cluster_config()
```

## ติดตั้งไปยังเครื่องเป้าหมาย

```bash
./install-canonical.sh "$HOME"
```

สำรองไฟล์ชื่อเดียวกันไว้ที่ `<target>/controller-backups/<timestamp>/` ก่อนเขียนทับ และ**ไม่**หยุดหรือ restart โมเดลที่กำลังทำงาน

## Validation scope

ผ่านแล้ว (static, ทำบนเครื่องพัฒนา):

```text
bash -n ทั้ง 21 Controller
help routing
info / banner ทั้ง 21 ตัว
network-info และ client-config option parsing
invalid port rejection (70000)
zero context rejection
stacked cluster prompt (ทดสอบผ่าน pty จริง — พิมพ์ค่าใหม่แล้วมีผล, Enter คงค่าเดิม, ไม่ถามเมื่อไม่มี TTY)
numeric separator audit
pipefail/grep-q audit
audit rules ใหม่ทั้งหมด: 21 scripts, errors=0, warnings=0
```

สถานะ **hardware-tested** ยังคงมีเฉพาะโมเดลที่ผู้ใช้ทดสอบและยืนยันเองบน DGX Spark จริง — การอัปเดตรุ่นนี้ไม่ได้เปิดโมเดลทั้ง 21 ตัวใหม่

## ระบบทั้งหมด — 4 repo ทำงานร่วมกัน

ชุด controller นี้เป็นส่วนหนึ่งของ ecosystem ที่ปรึกษาคำสั่งจากโมเดลหลาย ๆ ตัวทั้งเครื่องหนึ่งและ stacked พร้อม management ที่รวมศูนย์:

| Repo | บทบาท |
|---|---|
| [AutoDeployDGXProject](https://github.com/neronain/AutoDeployDGXProject) | **LMDS** — โหลด weight, สร้าง controller, deploy+รันโมเดลทั้งฟลีต |
| [AiGatewayLocal](https://github.com/neronain/AiGatewayLocal) | **LiteGate** — endpoint OpenAI/Anthropic เดียว พร้อม key/quota/ตรวจ capability |
| [dgx-spark-all-controllers](https://github.com/neronain/dgx-spark-all-controllers) | **controller กลาง (canonical)** — สคริปต์ที่ curate/ตรวจแล้ว (repo นี้) |
| [script-update](https://github.com/neronain/script-update) | **controller candidate** — ตัวที่เพิ่ง publish รอ review ก่อน promote |

**ลำดับการไหลเวียน:** LMDS deploy → ตัวที่พิสูจน์แล้ว publish ไป script-update (candidates) → promote ขึ้น repo นี้ (canonical) → ทุกเครื่อง `lmds recipes --sync` จากที่นี่ · LiteGate เสิร์ฟ+วัดความสามารถจริง

ทุก controller ที่นี่ (vLLM และ llama.cpp) ใช้มาตรฐาน v3.1.0 เหมือนกัน — มี `--jinja` สำหรับ tool-calling template ของ llama.cpp, `info` command, standard override option ทั้งหมด

## เอกสารอื่น

```text
AUDIT_REPORT.md    ผลตรวจและสิ่งที่แก้ในรุ่นนี้
CHANGELOG.md       ประวัติการเปลี่ยนแปลง
step.md            runbook DeepSeek-V4-Flash 2 โหนด (hardware-validated)
skills_Strack.md   skill/playbook การทำ stacked controller
MANIFEST.txt       รายการไฟล์ในแพ็ก
PACKAGE_SHA256SUMS ค่า SHA-256 ของทุกไฟล์
```
