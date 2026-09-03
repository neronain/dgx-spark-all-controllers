# Changelog

## 3.4.0 — 2026-09-03

ฟลีตจริง 13 เครื่อง rebuild ทุก controller ด้วย LMDS 0.5.1 แล้ว publish กลับเข้าคลัง · ออกแบบโดย neronain ·
<https://www.facebook.com/neronain.minidev>

### Added — controller ใหม่ 8 ตัว (ทุกตัวเสิร์ฟทราฟฟิกจริงแล้ว)

| controller | เครื่องที่รัน | หมายเหตุ |
|---|---|---|
| `qwen3-6-35b-a3b-nvfp4-single.sh` | dgx-spark04 | vLLM v0.28.0 สูตร DGX Spark ของ NVIDIA ครบ (fp8 KV · flashinfer · marlin MoE · MTP 3) · ผ่าน 9 ฟีเจอร์ · 65 tok/s เดี่ยว, 159 tok/s @6 |
| `qwen3-5-122b-a10b-nvfp4-single.sh` | spark-worker | Sehyo · MTP 2 · parser qwen3_xml/qwen3 · ผ่าน 9 ฟีเจอร์ · 30 tok/s |
| `qwen3-coder-next-nvfp4-gb10-single.sh` | spark-head | แทน `qwen3-coder-next-nvfp4-gb10-dgx-spark.sh` (alt-source ที่เขียนมือ) — ตัวนี้มี `ENGINE_ENV` marlin ที่ทำให้ NVFP4 MoE รันบน sm_121 ได้ · 63 tok/s |
| `qwen3-6-35b-a3b-mtp-gguf-single.sh` | dgx-spark02 | Q8 + MTP draft (`--spec-type draft-mtp`) · vision · tools · 128K |
| `qwen3-6-35b-a3b-uncensored-hauhaucs-aggressive-single.sh` | dgx-spark02 | Q8 · vision · tools · 128K |
| `gemma-4-31b-it-abliterated-gguf-single.sh` | msi-5 | |
| `gemma4-26b-a4b-qat-uncensored-hauhaucs-balanced-mtp-single.sh` | spark-head | MTP · vision |
| `qwen3-6-35b-a3b-uncensored-heretic-native-mtp-preserved-apex-gguf-single.sh` | spark-worker | MTP · vision |

### Changed — 12 ตัวที่ LMDS สร้าง รีเฟรชจากเครื่องที่รันจริง (LMDS 0.5.1)

`gemma-4-12b-it-gguf` · `huihui-ai-qwen3-coder-next-abliterated-gguf` · `huihui-gpt-oss-120b-…` ·
`huihui-qwen3-coder-30b-…` · `muse-glimmer-30b-gguf` · `ornith-1-5-35b-a3b-abliterated-gguf` ·
`qwen3-6-35b-a3b-gguf` · `qwen3-6-40b-…-imatrix-max-gguf` · `qwen3-8-27b-gguf` ·
`qwen3-8-27b-heretic-abliterated-uncensored-gguf` · `qwen3-8-27b-uncensored-gguf` · `qwen3-coder-next-gguf`

- **ทุกตัวมี `--jinja` แล้ว** — รุ่น 0.3.0 ที่อยู่ในคลัง 8 ตัวไม่มี ทำให้ tool calling เงียบทั้งที่ template รองรับ
  (README เดิมอ้างว่ามีทุกตัว)
- `--image-min-tokens 1024` สำหรับ vision · `stop` รอ process จบจริงแล้ว SIGKILL ถ้าไม่ยอม · บล็อก `bundle.env`
  อยู่เหนือค่าตั้งต้นทุกตัว · `EXTRA_SERVE_ARGS_DEFAULT` สำหรับแฟล็กเพิ่ม · `explain_crash()` ใน vLLM
- header ของตัว vLLM พกค่าที่พิสูจน์แล้วจาก `lmds set` (image digest, parser, ENGINE_ENV, extra args) —
  LMDS 0.5.1 พับให้ตอน publish · ค่าของเครื่อง (port, context, gpu-util) ยังเป็นค่าตั้งต้น override ได้เหมือนเดิม

### Removed

- `qwen3-coder-next-nvfp4-gb10-dgx-spark.sh` — alt-source ที่เขียนมือ ไม่มี env ที่จำเป็นบน GB10 · แทนด้วยตัวที่พิสูจน์แล้วข้างบน

### Fixed — เอกสารและเครื่องมือ

- `install-canonical.sh` ค้นหา controller จากไดเรกทอรีเหมือน `verify-all.sh` — รายชื่อที่เขียนมือติดตั้งได้ 21 จาก 35
  ตัวโดยรายงานว่าสำเร็จ
- 4 ตัวที่เขียนมือ (`gemma4-26-a4b-q8xl`, `gpt-oss-120b-f16`, `nemotron-omni-aeon`, `qwen3-coder-next-single`)
  มี IP ของเครื่องผู้พัฒนาในตัวอย่าง `client-config` → เปลี่ยนเป็น `192.0.2.10` (TEST-NET)
- `README.en.md` บอกว่ามี 22 ตัวมาตั้งแต่ 3.2.0 และไม่มีรายชื่อโมเดล → 42 พร้อมรายชื่อครบ · "Fabric discovery"
  ระบุรุ่นผิด (3.3.0 → 3.2.0)
- README ไทย: ตัวอย่าง banner บอก v3.3.1 ทั้งที่ไม่มี controller ตัวไหนพิมพ์ค่านั้น (ที่เขียนมือพิมพ์ 3.1.0, ที่ LMDS
  สร้างพิมพ์รุ่นของ LMDS) · ข้ออ้างเรื่อง `--jinja` ระบุข้อยกเว้นแล้ว
- `ADDING-GENERATED-CONTROLLERS.md` ไม่บอกว่า "เขียนมือทุกตัว" อีก และอธิบาย header ที่ฟลีตอ่าน
- `AUDIT_REPORT.md` มีผลตรวจรุ่นนี้ · `SKILL_UPDATE.md` / `skills_Strack.md` / `step.md` ไม่อ้าง "21 ตัว / 3.1.0 ทุกตัว" แล้ว

ยังค้าง: `qwen3-coder-30b-a3b-instruct-gguf-single.sh` เป็น LMDS 0.3.0 ไม่มี `--jinja` — ไม่มีเครื่องไหนรันอยู่ให้
rebuild จึงคงไว้พร้อมหมายเหตุ

ตรวจแล้ว: `verify-all.sh` 42/42 ผ่าน · `audit-controllers.py` errors=0 warnings=0 · ไม่มี IP/username ของผู้พัฒนาใน single-node ทุกตัว

## 3.3.1 — 2026-08-28

ดาวน์โหลดที่ถูกตัดกลางคันไม่ถูกนับว่าสำเร็จอีก · ออกแบบโดย neronain ·
<https://www.facebook.com/neronain.minidev>

### แก้ไข

**controller ทั้ง 13 ตัวที่โหลด GGUF จาก Hugging Face เอง**

- เคสจริง 2026-08-28 บน msi-5: ไฟล์ 20.3GB หลุดที่ 3,967MB ด้วย
  `curl: (92) HTTP/2 stream 1 was not closed cleanly: CANCEL (err 8)` — CDN ของ HF
  ตัดสตรีมกลางคันเมื่อโหลดยาว ๆ แล้ว `--retry` ของ curl **ไม่ยิงซ้ำให้** เพราะ curl
  นับ transient error แค่ timeout / 408 / 429 / 5xx เท่านั้น error 92 ไม่อยู่ในชุดนั้น
  curl จึงจบทันที เหลือไฟล์ที่ถูกตัดครึ่งไว้เฉย ๆ โดยไม่มีอะไรฟ้อง
- `fetch_one` วน resume ต่อเองจนขนาดตรงแล้ว (`-C -` ต่อจากของเดิม ไม่เริ่มใหม่)
- **ขนาดไฟล์คือเงื่อนไขจบ ไม่ใช่ exit code** — proxy ที่ส่ง body สั้นแต่ปิดสตรีม
  เรียบร้อยได้ exit 0 พร้อมไฟล์ไม่ครบ
- เพิ่ม `--retry-all-errors` โดยถาม `curl --help` ก่อนใช้ · curl < 7.71 (Ubuntu 20.04)
  ไม่รู้จัก flag นี้แล้วจะตายทันทีแทนที่จะโหลดได้
- resume แล้วไม่ได้ไบต์เพิ่มเลย = หยุดพร้อมบอกเหตุ ไม่วนไม่รู้จบ · ของเดิมไม่ถูกลบ
  รอบหน้ายัง resume ต่อได้ · ปรับจำนวนรอบได้ด้วย `FETCH_MAX_ATTEMPTS` (ค่าเริ่มต้น 20)
- `file_size` ถาม `stat` ทั้งแบบ GNU และ BSD

ตรงกับ LMDS commit `bd9538f` — ตัวเจนเนอเรเตอร์กับสำเนาที่นี่ใช้โค้ดชุดเดียวกัน

ตรวจแล้ว: `verify-all.sh` 35/35 ผ่าน errors=0 warnings=0

## 3.3.0 — 2026-08-28

โมเดลใหม่ 5 ตัวจาก LMDS ที่เสิร์ฟทราฟฟิกจริงมาแล้ว และการปิดช่องที่ทำให้ของใหม่
หลุดการตรวจไปเงียบ ๆ · ออกแบบโดย neronain · <https://www.facebook.com/neronain.minidev>

### Added — controller ใหม่ 5 ตัว (llama.cpp / GGUF)

ทุกตัวมาจาก bundle ที่ LMDS สร้าง แล้ว **ถูกใช้งานจริงบน DGX Spark (GB10)** —
จำนวนคำขอนับจาก `server.log` ของเครื่องนั้น ไม่ใช่จากการที่มัน start ขึ้นได้

| controller | เครื่องที่รัน | คำขอที่เสิร์ฟจริง |
|---|---|---|
| `qwen3-6-40b-claude-…-imatrix-max-gguf-single.sh` | spark-head | 886 · prompt เดียว 118,710 token |
| `huihui-ai-qwen3-coder-next-abliterated-gguf-single.sh` | msi-6 | 848 |
| `qwen3-8-27b-uncensored-gguf-single.sh` | msi-5 | 627 |
| `qwen3-8-27b-heretic-abliterated-uncensored-gguf-single.sh` | spark-worker | 603 |
| `ornith-1-5-35b-a3b-abliterated-gguf-single.sh` | spark-head | 83 |

อีก 2 ตัวที่ generate ไว้แล้ว (`gemma4-26b-a4b-qat-…-mtp`,
`qwen3-6-35b-a3b-uncensored-heretic-…-apex`) **ไม่ได้เข้าคลังนี้** — โหลดโมเดลขึ้นได้
แต่ยังไม่เคยรับคำขอจริง จึงไปพักที่ [script-update](https://github.com/neronain/script-update)
ตามเกณฑ์ใน `ADDING-GENERATED-CONTROLLERS.md` ที่ว่าด่านของรีโปนี้คือ "มีคนเสิร์ฟ
ทราฟฟิกจริงด้วยมัน" ไม่ใช่ "ผ่าน audit"

### Fixed — ของใหม่ 13 ตัวไม่เคยถูกตรวจเลย

- `verify-all.sh` เคยถือ **รายชื่อไฟล์ที่พิมพ์ด้วยมือ** · วันที่ตรวจพบ โฟลเดอร์มี 35
  controller แต่ในไฟล์เขียนไว้ 22 — ทุกตัวที่เพิ่มเข้ามาจาก bundle ของ LMDS จึงไม่เคย
  ผ่านสคริปต์ที่มีหน้าที่ตรวจมันเลย · ตอนนี้มัน **ไล่จากไฟล์จริงในโฟลเดอร์** และแยก
  stacked ด้วยการดูว่ามี `prompt_cluster_config` ไหม ไม่ใช่เดาจากชื่อไฟล์
- ด่าน banner บังคับข้อความ `DGX Spark Controller` ตัวเดียว ทำให้ controller ที่ LMDS
  สร้าง (ซึ่งพิมพ์ `LMDS controller · vX.Y`) ตกทุกตัว · รีโปนี้รับตัว generated เป็น
  พลเมืองชั้นหนึ่งอยู่แล้ว ด่านจึงควรตรวจ *สิ่งที่ผู้ใช้อ่าน* (มี banner, มีบรรทัด
  Model/Runtime/Features/State) ไม่ใช่ชื่อผลิตภัณฑ์

### Fixed — เอกสารตามของจริงไม่ทัน

- `MANIFEST.txt` นับ 22 ตัวขณะที่มี 35 · `README.md` ค้างที่ v3.1.0 เขียนว่า "21 ตัว"
  และไม่มี 14 ตัวที่เพิ่มมาหลังจากนั้น · ทั้งสองไฟล์ถูก **สร้างจากไฟล์จริง** แล้ว
  ไม่ใช่พิมพ์ตามกันมา
- README เพิ่มหัวข้อ `Stacked / multi-node · SGLang (Docker)` ซึ่งมี controller อยู่
  ตั้งแต่ 3.2.0 แต่ไม่เคยถูกลงรายการ

## 3.2.0 — 2026-08-05

Fabric auto-discovery, one new runtime, and a documented path for controllers
generated by LMDS. Designed by neronain · <https://www.facebook.com/neronain.minidev>

### Added

- **`minimax-m3-v0-nvfp4-reap50-stacked.sh`** — MiniMax-M3-v0 NVFP4-REAP50 on two
  DGX Sparks, and the collection's first **SGLang** controller (all others are vLLM
  or llama.cpp). Brought up to the v3.1.0 contract: `SCRIPT_VERSION`, `banner`/`info`,
  and `prompt_cluster_config` on `start`/`restart`.
- **Fabric interface auto-discovery** in `deepseek-v4-flash-nvfp4-stacked.sh` —
  the NCCL interface is derived from the transport IP instead of being typed in.
  Port names on DGX Spark are long and differ per fabric (`enp1s0f1np1` and
  `enP2p1s0f1np1` are two separate 200G fabrics on the same box); getting one wrong
  does not fail loudly, NCCL just falls back to the management NIC. The worker is
  asked for its own name over SSH, because it does not have to match the head's.
  An explicit `NCCL_SOCKET_IFNAME` still wins.
- **RoCE HCA auto-discovery** — `NCCL_IB_HCA` is resolved from the chosen interface
  via `/sys/class/infiniband/*/device/net/`. Without it NCCL runs over TCP and a
  200G fabric performs like ordinary ethernet while appearing to work fine.
- **`check_running_on_master`** in the DeepSeek controller — running the head script
  on the wrong machine now fails immediately with the reason, instead of dying inside
  NCCL initialisation with an opaque error.
- `UCX_NET_DEVICES` and `OMPI_MCA_btl_tcp_if_include` alongside the NCCL socket
  variables. These two choose their own transport and can otherwise stay on the
  management NIC while NCCL is correctly on the fabric.
- **`README.en.md`** — English documentation for the collection.
- **`ADDING-GENERATED-CONTROLLERS.md`** — how to add a controller produced by
  [LMDS](https://github.com/neronain/AutoDeployDGXProject) to this collection so that
  students and customers can use it without generating or testing anything themselves.

### Fixed

- `minimax-m3-v0-nvfp4-reap50-stacked.sh`: `100_000_000` numeric separator, and a
  `... | grep -Eq` pipeline that could fail randomly under `set -o pipefail` when
  `grep` closed the pipe early.
- `network-info` and `doctor` on the DeepSeek controller reported the fabric as
  `<NOT SET>` even when it was discoverable. They now resolve it first and print
  what will actually be used.

### Verified

- `verify-all.sh` on a real DGX Spark: 22 scripts, errors=0, warnings=0.
- Interface and HCA discovery confirmed on hardware: `10.100.152.1` →
  `enp1s0f1np1` → `rocep1s0f1`.

## 3.1.0 — 2026-07-25

Repository-wide cleanup and standardization. Designed by neronain · <https://www.facebook.com/neronain.minidev>

### Added

- ASCII banner (Metasploit style) in all 21 controllers, with a designer credit line.
- `info` command (alias `banner`) in all 21 controllers: model label, model ID, runtime, supported features, context, advertised API v1 URL, and live state with the port. State comes from a real `/health` probe.
- Interactive cluster configuration for all 4 stacked controllers: `start` and `restart` ask for the head node IP, worker node IP, and SSH user, each defaulting to the current value. Only prompts on a TTY, so automation and cron stay unattended.
- `--advertise-ip` and `--interface` options for the two controllers that lacked them.
- Audit rules: `missing-script-version`, `missing-banner-info`, `hard-coded-author-username`, `missing-cluster-prompt`.
- `verify-all.sh` now covers all 21 controllers, exercises `info`, asserts stacked controllers do not prompt outside `start`/`restart`, and fails if any controller hard-codes the author's username.

### Changed

- `SCRIPT_VERSION` is now 3.1.0 across every controller.
- `SSH_USER` defaults to `${USER:-$(id -un)}` everywhere; no controller hard-codes a Linux username.
- Port validation (1..65535) and context validation (> 0) added wherever it was missing or weaker.
- Container, runtime, port, and model-file checks rewritten to store command output before matching, replacing `producer | grep -q` pipelines.
- `install-canonical.sh` installs all 21 controllers.

### Fixed

- `gemma-4-31b-it-uncensored-heretic-q8_0-dgx-spark.sh`: `local rc=$?` read the status of the enclosing `if` construct instead of `wait_for_health`, leaving the startup-timeout branch unreachable and misreporting a timeout as a crash.

### Removed

- `gemma4-26b-a4b-q8xl-single.sh` — superseded 1.0.2 duplicate of `gemma4-26-a4b-q8xl-single.sh`.
- `vllm-stackctl(1).sh` — duplicate download of `vllm-stackctl.sh`, confirmed a strict subset.

### Renamed

- `deepseek-v4-flash-nvfp4-stacked-fixed-v8.2-test-heredoc-fixed.sh` → `deepseek-v4-flash-nvfp4-stacked.sh`
- `qwen3-vl-32b-thinking-single-fixed-v5.sh` → `qwen3-vl-32b-thinking-single.sh`

### Documentation

- `README.md`, `AUDIT_REPORT.md`, `MANIFEST.txt`, `SKILL_UPDATE.md`, `step.md`, and `skills_Strack.md` rewritten for the 21-controller v3.1.0 layout and the new commands.
- `PACKAGE_SHA256SUMS` regenerated.

## 3.0.0

- Audited 12 canonical DGX Spark controllers.
- Removed Bash pure-numeric underscore literals.
- Standardized `--context`, `--port`, `--bind`, `--advertise-ip`, and `--interface`.
- Added environment overrides for context, port, bind address, advertise address, API key, and client budgets.
- Added route-based advertised IP selection.
- Removed fixed `MASTER_IP` from single-node controllers.
- Kept cluster addresses separate from public API addresses in stacked controllers.
- Added `network-info`.
- Added automatic client input-budget calculation.
- Fixed pipefail-sensitive `grep -q` pipelines.
- Added audit, verification, and safe installation helpers.
- Updated the DGX Spark Model Deployer skill rules and templates.
