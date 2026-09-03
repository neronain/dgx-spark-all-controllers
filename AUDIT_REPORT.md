# Audit Report v3.4.0

ตรวจเมื่อ 2026-09-03 · ออกแบบโดย neronain · <https://www.facebook.com/neronain.minidev>

## ขอบเขต

ไฟล์ `.sh` ทั้ง 44 ไฟล์ในรีโป (42 controller + 2 เครื่องมือ) ด้วย `bash -n`, `verify-all.sh` (help / info /
network-info / client-config / ปฏิเสธ port 70000 และ context 0 / stacked ต้องมี `prompt_cluster_config`),
`audit-controllers.py`, และสแกน IP/username ของผู้พัฒนา

## ผล

```text
bash -n            : 44/44 ผ่าน
verify-all.sh      : 42/42 ผ่าน (37 single-node, 5 stacked)
audit-controllers  : Audited 42 scripts: errors=0, warnings=0
--jinja (llama.cpp): 26/27 — ขาดเฉพาะ qwen3-coder-30b-a3b-instruct-gguf-single.sh (LMDS 0.3.0, ยังไม่ได้ rebuild)
IP ผู้พัฒนา         : 0 ไฟล์ (single-node) · stacked ใช้ 10.100.152.1/2 เป็นค่าตั้งต้นที่เอกสารระบุไว้
PACKAGE_SHA256SUMS : สร้างใหม่ ครอบทุกไฟล์ยกเว้นตัวเอง
```

รุ่นของ controller: ที่เขียนมือ 21 ตัวประกาศ `SCRIPT_VERSION=3.1.0` · ที่ LMDS สร้าง 21 ตัวประกาศรุ่นของ LMDS
(20 ตัว 0.5.1 หลัง rebuild 2026-09-03 · `qwen3-coder-30b-a3b-instruct-gguf` ยัง 0.3.0) — banner จึงต่างกันสองแบบ
("DGX Spark Controller" กับ "LMDS controller") และ `verify-all.sh` รับทั้งคู่

---

# ประวัติ — Audit Report v3.1.0

ตรวจเมื่อ 2026-07-25 · ออกแบบโดย neronain · <https://www.facebook.com/neronain.minidev>

## ขอบเขต

ตรวจไฟล์ `.sh` ทั้งหมดที่มีอยู่ใน repository (25 ไฟล์ก่อนเริ่ม) ด้วย `bash -n`, `audit-controllers.py` และการอ่านโค้ดทีละไฟล์

ผลก่อนแก้:

```text
bash -n            : ผ่านทั้งหมด (ไม่มี syntax error)
audit-controllers  : errors=0, warnings=23
canonical 12 ตัว   : สะอาด 0 findings
ไฟล์ส่วนเกิน 4 ตัว : รวม 23 warnings
```

## ไฟล์ซ้ำที่ลบออก

```text
gemma4-26b-a4b-q8xl-single.sh   SCRIPT_VERSION 1.0.2 · โมเดลและ quant เดียวกับ
                                gemma4-26-a4b-q8xl-single.sh (3.x) แต่ไม่มี network-info
                                และ override ไม่ได้ → 6 warnings
vllm-stackctl(1).sh             ไฟล์ดาวน์โหลดซ้ำของ vllm-stackctl.sh
                                diff ยืนยันเป็น subset แท้ (0 บรรทัดที่ไม่มีในตัวใหม่)
```

## ไฟล์ที่เปลี่ยนชื่อ

```text
deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh
  → deepseek-v4-flash-nvfp4-stacked.sh

qwen3-vl-32b-thinking-single-fixed-v5.sh
  → qwen3-vl-32b-thinking-single.sh
```

เปลี่ยนด้วย `git mv` เพื่อรักษาประวัติไฟล์ · เนื้อหาเดิมไม่ถูกแก้จากการเปลี่ยนชื่อ

## ไฟล์ที่โมเดลซ้ำแต่เก็บไว้เป็น alt-source

สองไฟล์นี้เป็นโมเดลเดียวกับ canonical แต่มาจาก uploader อื่น — เก็บไว้เป็นแหล่งสำรอง และยกให้เข้ามาตรฐานเดียวกันแล้ว

```text
gemma-4-26b-a4b-it-gguf-single.sh          revision อื่น (text-only) ของ
                                           unsloth/gemma-4-26B-A4B-it-GGUF
qwen3-coder-next-nvfp4-gb10-dgx-spark.sh   saricles/Qwen3-Coder-Next-NVFP4-GB10
                                           (canonical ใช้ ucbye/…)
```

## pipefail + grep -q : ผลตรวจจริง

warning `pipefail-grep-q` 12 จุดที่ตรวจพบใน 2 ไฟล์ **ไม่ใช่บั๊ก** — ทุกจุดอยู่ในบริบทเงื่อนไข (`if`, `if !`) ซึ่ง Bash ระงับ errexit อยู่แล้ว สถานะที่ไม่ใช่ 0 จึงเป็นแค่การเลือก branch ไม่ทำให้สคริปต์ตาย

แต่แก้ให้เข้ามาตรฐานเดียวกับ canonical (เก็บ output ก่อนแล้วค่อยเทียบ) เพื่อให้ audit ได้ 0 และอ่านง่ายขึ้น:

```bash
# เดิม
if docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then

# ใหม่
_container_running() {
  local n
  n="$(docker ps --format '{{.Names}}' 2>/dev/null || true)"
  grep -Fxq "$1" <<<"$n"
}
if _container_running "$CONTAINER_NAME"; then
```

จุดที่ปรับ: container running/exists, NVIDIA runtime detection, port-in-use check, model-file readiness

## บั๊กจริงที่พบ

พบบั๊กจริง 1 จุดจากการอ่านโค้ดทั้งหมด:

```text
gemma-4-31b-it-uncensored-heretic-q8_0-dgx-spark.sh
  local rc=$? วางหลัง if wait_for_health; then … fi ที่ไม่มี else
  → $? เป็นสถานะของ if-construct (0 เสมอ) ไม่ใช่ของ wait_for_health
  → branch timeout (( rc == 2 )) เป็น dead code
  ผลกระทบ: ข้อความวินิจฉัยผิดเมื่อ startup timeout (แจ้งว่า exited แทน timeout)
  ระดับ: ต่ำ — ไม่กระทบ control flow อื่น
```

ไม่พบ: heredoc พัง, unbound variable ภายใต้ `set -u`, arithmetic ผิด, unquoted expansion ที่ทำให้พัง, numeric separator หรือ pure-numeric underscore literal

## การยกมาตรฐาน v3.1.0

Controller ทั้ง 21 ตัวได้รับสิ่งเหล่านี้:

```text
SCRIPT_VERSION="${SCRIPT_VERSION:-3.1.0}"
ASCII banner แบบ Metasploit พร้อมเครดิตผู้ออกแบบ
คำสั่ง info | banner  (โมเดล, model ID, runtime, ฟีเจอร์, context, API URL, สถานะ+พอร์ต)
port validation 1..65535
context validation > 0
```

เพิ่มเฉพาะที่ยังขาด:

```text
--advertise-ip และ --interface        2 ไฟล์ (ornith uncensored, qwen3-coder alt-source)
detect_advertise_ip() รองรับ 2 ตัวเลือกข้างต้น
```

## Portability : เลิกผูกกับเครื่องผู้พัฒนา

```text
ก่อน: SSH_USER="${SSH_USER:-neronain}"        3 ไฟล์ (vllm-stackctl, gemma4-31b-stacked,
                                              minimax-m27-luke-stacked)
หลัง: SSH_USER="${SSH_USER:-${USER:-$(id -un)}}"   ทุกไฟล์
```

ตรวจแล้วไม่มีชื่อผู้ใช้ของผู้พัฒนา hard-code เหลืออยู่ (ยกเว้นบรรทัดเครดิตใน banner ซึ่งเป็นเจตนา)

## Stacked : ถามค่า cluster ตอนใช้งาน

Stacked ทั้ง 4 ตัวถาม Head IP, Worker IP และ SSH user ตอน `start` / `restart`

จุดสำคัญของการวางโค้ด: เรียก `prompt_cluster_config` **ทันทีหลัง** การประกาศ `MASTER_IP` / `WORKER_IP` / `SSH_USER` และ **ก่อน** ตัวแปรที่ derive จากค่าเหล่านั้น (`MASTER_HOME`, `SSH_TARGET`, `TRANSPORT_IP_*`) มิฉะนั้นค่า derive จะค้างที่ค่าเดิม

พฤติกรรม:

```text
กด Enter                     คงค่าเดิม
คำสั่งอื่นที่ไม่ใช่ start/restart  ไม่ถาม
stdin ไม่ใช่ TTY (cron, pipe)   ไม่ถาม ใช้ค่า env
env MASTER_IP/WORKER_IP/SSH_USER  ยังใช้ได้ตามปกติ
```

## กฎ audit ที่เพิ่มใหม่

`audit-controllers.py` ตรวจเพิ่ม:

```text
missing-script-version        ไม่มี SCRIPT_VERSION ที่ override ได้
missing-banner-info           ไม่มี banner() / info() หรือ dispatch info|banner)
hard-coded-author-username    hard-code ชื่อผู้ใช้ (ยกเว้นบรรทัดเครดิต)
missing-cluster-prompt        stacked ที่ไม่มี prompt_cluster_config()
```

ยืนยันว่ากฎใหม่จับของจริงได้ ด้วยการทดสอบกับไฟล์ตัวอย่างที่จงใจละเมิด

## ผลตรวจหลังแก้

```text
21/21 bash syntax passed
21/21 info / banner passed
21/21 context option passed
21/21 port option passed
21/21 network-info passed
21/21 client configuration passed
21/21 invalid port (70000) rejected
21/21 zero context rejected
 4/4  stacked cluster prompt passed (ทดสอบผ่าน pty จริง)
 0    numeric-separator findings
 0    hard-coded single-node MASTER_IP findings
 0    pipefail/grep-q findings
 0    hard-coded author username findings

audit-controllers.py : Audited 21 scripts: errors=0, warnings=0
```

## สิ่งที่ยังไม่ได้ทำ

```text
ไม่ได้เปิดโมเดลทั้ง 21 ตัวบน DGX Spark ในรอบนี้
  → สถานะ hardware-tested คงเดิมตามที่ผู้ใช้ยืนยันเอง
  → DeepSeek-V4-Flash 2 โหนดยังอ้าง hardware validation 2026-07-22 ตาม step.md
ไม่ได้แก้ค่า tuning ของโมเดลใด (GPU util, batching, backend) — ไม่อยู่ในขอบเขตการตรวจ
```
