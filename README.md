# DGX Spark Controller Collection v3.0.0

ชุดปรับปรุงรวม Controller สำหรับ NVIDIA DGX Spark จำนวน 12 ตัว โดยใช้มาตรฐานเดียวกับ 4 สคริปต์ที่ทดสอบใช้งานจริงผ่านแล้ว:

```text
gemma4-26-a4b-q8xl-single.sh
gpt-oss-120b-f16-single.sh
nemotron-omni-aeon-single.sh
qwen3-coder-next-single.sh
```

## ไฟล์ Controller

### Single-node

```text
gemma4-26-a4b-q8xl-single.sh
gemma4-31b-single.sh
gpt-oss-120b-f16-single.sh
llama33-70b-nvfp4-single.sh
nemotron-3-super-single.sh
nemotron-omni-aeon-single.sh
qwen3-coder-next-single.sh
qwen36-hauhau-q6kp-single.sh
redteam-modelctl.sh
```

### Stacked / multi-node

```text
gemma4-31b-stacked.sh
minimax-m27-luke-stacked.sh
vllm-stackctl.sh
```

## ปัญหา Bash numeric separator

Bash arithmetic ไม่รองรับตัวเลขรูปแบบนี้:

```bash
(( model_size > 25_000_000_000 ))
(( file_size > 80_000_000 ))
```

อาจเกิด error:

```text
value too great for base
```

ชุด v3.0.0 ไม่มี pure numeric literal ที่ใช้ underscore เหลืออยู่ใน Controller ทั้ง 12 ตัว

ค่าขนาดไฟล์ใช้ตัวเลข decimal ปกติ:

```bash
MODEL_SIZE_BYTES="30649317504"
MMPROJ_SIZE_BYTES="899283072"
```

และสำหรับ GGUF รุ่นที่มี exact metadata จะตรวจ:

```text
exact byte size
GGUF magic header
SHA-256
```

## Context และ Port

ทุก Controller รองรับรูปแบบเดียวกัน:

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

ใช้ environment variables ได้เช่นกัน:

```bash
MAX_MODEL_LEN=65536 \
API_PORT=8001 \
ADVERTISE_INTERFACE=enp1s0 \
./qwen3-coder-next-single.sh start
```

สำหรับ llama.cpp ที่ใช้ `CTX_SIZE`:

```bash
CTX_SIZE=65536 \
API_PORT=8001 \
ADVERTISE_INTERFACE=enp1s0 \
./qwen36-hauhau-q6kp-single.sh start
```

## การเลือก IP

ระบบไม่ใช้ IP ตายตัวสำหรับ public client URL และไม่ใช้ IP ตัวแรกจาก `hostname -I` เป็นวิธีหลักอีกต่อไป

ลำดับการเลือก:

1. `--advertise-ip`
2. `--interface`
3. source address จาก `ip route get 1.1.1.1`
4. global IPv4 ที่ไม่ใช่ Docker/CNI/cluster-like interface
5. `hostname -I` เฉพาะ fallback สุดท้าย

ตรวจค่าก่อนรัน:

```bash
./qwen36-hauhau-q6kp-single.sh network-info
```

บังคับ LAN interface:

```bash
./qwen36-hauhau-q6kp-single.sh start \
  --context 131072 \
  --port 8001 \
  --interface enp1s0
```

บังคับ IP:

```bash
./qwen36-hauhau-q6kp-single.sh start \
  --advertise-ip 192.168.101.127 \
  --port 8001
```

## Bind กับ Advertise แยกกัน

```text
--bind
```

กำหนด address ที่ server ฟังจริง

```text
--advertise-ip
--interface
```

กำหนด URL ที่แสดงให้ Hermes, OpenClaw, Cline และเครื่องอื่น

ค่าที่แนะนำ:

```bash
./controller.sh start \
  --bind 0.0.0.0 \
  --interface enp1s0 \
  --port 8000
```

## DGX หลายเครื่อง

DGX สองเครื่องสามารถใช้ port เดียวกันได้:

```text
DGX-1  192.168.101.127:8000
DGX-2  192.168.101.128:8000
```

เพราะเป็นคนละ IP

สำหรับ Stacked Controller ค่า `MASTER_IP` และ `WORKER_IP` ยังคงใช้สำหรับการสื่อสารภายใน cluster แต่ public API URL ใช้ advertised address แยกต่างหาก

ตัวอย่าง:

```bash
./minimax-m27-luke-stacked.sh start \
  --context 65536 \
  --port 8000 \
  --interface enp1s0
```

## Client token budget

เมื่อ `--client-input auto` สคริปต์คำนวณ:

```text
client input =
server context - max output - 8192 overhead
```

ตัวอย่าง:

```text
context      65536
max output    8192
overhead      8192
client input 49152
```

สคริปต์ปฏิเสธ configuration ที่ input + output มากกว่า server context

## ตรวจทั้งหมด

```bash
chmod +x ./*.sh audit-controllers.py
./verify-all.sh
```

ผลที่ต้องได้:

```text
Audited 12 scripts: errors=0, warnings=0
All canonical DGX controllers passed static validation.
```

## ตรวจ Controller เพิ่มเติมในเครื่อง

```bash
python3 audit-controllers.py /home/neronain
```

ตรวจเป็น JSON:

```bash
python3 audit-controllers.py /home/neronain --json
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
```

## ติดตั้งแทนไฟล์เดิม

คำสั่งนี้จะสำรองไฟล์ชื่อเดียวกันก่อน:

```bash
./install-canonical.sh /home/neronain
```

ไฟล์เดิมจะถูกเก็บที่:

```text
/home/neronain/controller-backups/<timestamp>/
```

Installer ไม่หยุดหรือ restart โมเดลที่กำลังทำงาน

## Validation scope

ผ่านแล้ว:

```text
bash -n ทั้ง 12 Controller
help routing
network-info option parsing
client-config option parsing
invalid port rejection
zero context rejection
numeric separator audit
pipefail/grep-q audit
ZIP integrity
```

ยังไม่ได้เปิดโมเดลทั้ง 12 ตัวบน DGX Spark ในการสร้างแพ็กนี้ สถานะ hardware-tested ยังคงมีเฉพาะโมเดลที่ผู้ใช้ทดสอบและยืนยันเอง
