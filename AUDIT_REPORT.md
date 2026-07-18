# Audit Report v3.0.0

## ขอบเขต

ตรวจ Controller ที่ไม่ซ้ำกัน 12 ตัวจากไฟล์ที่เคยสร้างและอัปโหลดไว้

ไฟล์สำเนาที่มีชื่อ `(1)`, `(2)`, `(3)` ไม่ถูกนำมาซ้ำในแพ็ก โดยเลือก revision ล่าสุดที่เกี่ยวข้องเป็น canonical file

## Bug ที่ยืนยัน

สคริปต์ Qwen36 รุ่นแรกมี:

```bash
(( model_size > 25_000_000_000 ))
(( mmproj_size > 800_000_000 ))
```

Bash ตีความ underscore ใน arithmetic literal ไม่ถูกต้องและแจ้ง:

```text
value too great for base
```

รุ่น canonical ใช้:

```bash
MODEL_SIZE_BYTES="30649317504"
MMPROJ_SIZE_BYTES="899283072"
```

พร้อม exact-size และ SHA-256 validation

## การแก้ไขร่วม

Controller ทั้ง 12 ตัวได้รับมาตรฐานดังนี้:

```text
SCRIPT_VERSION=3.0.0
context override
port override
bind override
advertised IP/interface override
route-based IP detection
network-info command
automatic client input budget
safe current-user fallback
no pure numeric underscore literal
bash -n validation
```

## Pipefail hardening

แก้รูปแบบที่เสี่ยง false failure:

```bash
producer | grep -q pattern
```

เมื่อใช้ร่วมกับ:

```bash
set -o pipefail
```

โดยเปลี่ยนเป็นการเก็บ output ก่อน หรือใช้ direct inspect/test แทน

จุดที่ปรับรวมถึง:

```text
model cache existence
Docker container existence/running
llama-server feature detection
remote worker cache/IP checks
```

## Stacked Controller

รักษา:

```text
MASTER_IP
WORKER_IP
cluster transport
Ray/NCCL/runtime-specific settings
```

แต่แยก public API endpoint ออกจาก cluster IP

## ผลตรวจ

```text
12/12 bash syntax passed
12/12 context option passed
12/12 port option passed
12/12 network-info passed
12/12 client configuration passed
0 numeric-separator findings
0 hard-coded single-node MASTER_IP findings
0 pipefail/grep-q findings
```
