<div align="center">

# DGX Spark Controller Collection

**คลัง controller มาตรฐานสำหรับรันโมเดลภาษาบน NVIDIA DGX Spark — เอาไปรันได้ทันที ไม่ใช่ตัว generate**

controller หนึ่งตัวต่อหนึ่งโมเดล · ชุดคำสั่งเดียวกันทุกตัว · override port/context/bind ได้ ·
ไม่ผูกกับชื่อผู้ใช้หรือ IP ของเครื่องผู้พัฒนา — ก๊อปไปวางเครื่องไหนก็รันได้

[![version](https://img.shields.io/badge/version-3.2.0-1f5fbf)](CHANGELOG.md)
[![controllers](https://img.shields.io/badge/controllers-28-17703f)](#แคตตาล็อก)
[![tier](https://img.shields.io/badge/tier-canonical-8a5300)](#ที่ทางในระบบ-lmds)
[![engines](https://img.shields.io/badge/engines-llama.cpp%20·%20vLLM%20·%20SGLang-555)](#แคตตาล็อก)
[![platform](https://img.shields.io/badge/platform-DGX%20Spark%20·%20GB10-76b900)](#)

**[คู่มือ](USAGE.md)** · **[เพิ่ม controller ใหม่](ADDING-GENERATED-CONTROLLERS.md)** · **[รายงานตรวจสอบ](AUDIT_REPORT.md)** · **[English](README.en.md)**

</div>

---

## ที่ทางในระบบ LMDS

รีโปนี้คือ **ชั้น canonical** — controller ที่ผ่านการ review แล้วของทั้งกอง ·
[**LMDS**](https://github.com/neronain/AutoDeployDGXProject) `lmds recipes --sync` ดึงจากที่นี่
ไปเป็น "สูตรที่รันผ่านจริง" ให้ `deploy --no-llm` หยิบ image/parser/mmproj/quant ที่ถูกต้องไปใช้
แทนการเดา — เครื่องที่ไม่มี API key ก็ deploy ได้ตรงรุ่น

```
เครื่องที่เทสต์ผ่าน ──publish──▶  script-update (candidates)  ──review──▶  dgx-spark-all-controllers (รีโปนี้ · canonical)
                                                                                   │
       เครื่องทั้งกอง  ◀──────────────────── lmds recipes --sync ────────────────────┘
```

| รีโป | บทบาท |
|---|---|
| **dgx-spark-all-controllers** (รีโปนี้) | canonical — ผ่าน review แล้ว ทุกเครื่อง `--sync` ไปใช้ |
| [`script-update`](https://github.com/neronain/script-update) | candidates — controller ที่ `--publish` ส่งขึ้นมา รอ review |

controller ในรีโปนี้ถือ **เฉพาะค่าของโมเดล** — ค่าเฉพาะเครื่อง (port/context จริง) override ตอนรัน

## เริ่มเร็วที่สุด

```bash
chmod +x ./*.sh

# โมเดลอะไร · พอร์ตอะไร · รองรับฟีเจอร์ไหน · รันอยู่หรือยัง
./qwen3-coder-next-gguf-single.sh info

# วงจรเต็ม: โหลด weight → ตรวจไฟล์ → เตรียม runtime → รัน → ทดสอบ
./qwen3-coder-next-gguf-single.sh download && \
./qwen3-coder-next-gguf-single.sh start && \
./qwen3-coder-next-gguf-single.sh test-text
```

controller รุ่นล่าสุดโหลด weight ด้วย **aria2c หลาย connection ขนาน** (เลี่ยง CDN ที่ throttle
ต่อ connection · ถอยไป curl อัตโนมัติถ้าไม่มี aria2c) และ **ตรวจขนาดทุกไฟล์** หลังโหลด — ไฟล์
ที่โหลดไม่ครบถูกจับได้ก่อนจะเอาไป start

## แคตตาล็อก

28 controller · แต่ละตัวใช้ชุดคำสั่งเดียวกัน

### llama.cpp · GGUF (single-node)

| controller | โมเดล |
|---|---|
| `qwen3-8-27b-gguf` | Qwen3.8 27B — vision · tools · reasoning |
| `qwen3-6-35b-a3b-gguf` | Qwen3.6 35B-A3B (MoE เร็ว) — vision · tools |
| `qwen3-coder-next-gguf` | Qwen3-Coder-Next 80B-A3B — agentic coding |
| `qwen3-coder-30b-a3b-instruct-gguf` | Qwen3-Coder 30B-A3B Instruct |
| `muse-glimmer-30b-gguf` | Muse Glimmer 30B — creative · vision |
| `gemma-4-12b-it-gguf` · `gemma-4-26b-a4b-it-gguf` | Gemma 4 (12B · 26B-A4B) |
| `gpt-oss-120b-f16-single` | GPT-OSS 120B (F16) |
| `qwen3-vl-32b-thinking` · `qwen36-hauhau-q6kp` | Qwen3-VL 32B · Qwen3.6 Hauhau |

### llama.cpp · uncensored / abliterated

| controller | โมเดล |
|---|---|
| `huihui-gpt-oss-120b-abliterated-mxfp4-moe-gguf` | GPT-OSS 120B abliterated (MXFP4) |
| `huihui-qwen3-coder-30b-a3b-instruct-abliterated-gguf` | Qwen3-Coder 30B abliterated |
| `gemma-4-31b-it-uncensored-*` · `ornith-1.0-35b-*heretic*` | Gemma 4 31B · Ornith 35B (heretic) |

### vLLM / SGLang · NVFP4 (single-node · DGX Spark)

| controller | โมเดล |
|---|---|
| `qwen3-coder-next-nvfp4-gb10-dgx-spark` | Qwen3-Coder-Next NVFP4 (GB10 kernel) |
| `nemotron-3-super-single` · `nemotron-omni-aeon-single` | NVIDIA Nemotron 3 Super · Omni |
| `llama33-70b-nvfp4-single` | Llama 3.3 70B NVFP4 |
| `qwen3-vl-32b-instruct-1m-bf16-dgx-spark` · `ornith-1.0-35b-bf16-dgx-spark` | Qwen3-VL 32B (1M) · Ornith 35B |

### Stacked · หลายเครื่อง (2× DGX Spark)

| controller | โมเดล |
|---|---|
| `deepseek-v4-flash-nvfp4-stacked` | DeepSeek-V4-Flash NVFP4 |
| `minimax-m27-luke-stacked` · `minimax-m3-v0-nvfp4-reap50-stacked` | MiniMax M2.7 · M3 |
| `gemma4-31b-stacked` | Gemma 4 31B |

> ดูรายการเต็ม + ฟีเจอร์ที่วัดได้จริง: `<controller> info` หรือ [MANIFEST.txt](MANIFEST.txt)

## คำสั่งมาตรฐาน (ทุก controller เหมือนกัน)

```
info          โมเดล · port · ฟีเจอร์ · สถานะ
download      โหลด weight (aria2c ขนาน → curl · resume ได้ · ตรวจขนาด)
verify-files  ตรวจไฟล์ครบและขนาดตรง
start / stop / restart / status
test-text · test-tools · test-vision      พิสูจน์ว่าโมเดลทำได้จริง
client-config  พ่นค่าตั้งต้นให้ OpenAI / Anthropic client
```

override ได้ทุกตัว: `./<controller> start --port 8010 --context 131072 --bind 0.0.0.0`

## ตรวจทั้งชุด

```bash
./verify-all.sh              # bash -n + โครงสร้าง + secret scan ทุก controller
python3 audit-controllers.py # เทียบกับมาตรฐาน · ออกรายงาน
```

## License

controller ในรีโปนี้เป็นเครื่องมือ deploy · โมเดล / image / runtime ของบุคคลที่สามอยู่ใต้ license
ของเจ้าของนั้น ๆ

---

<div align="center">

ส่วนหนึ่งของ **LMDS · Local Model Deploy Studio** · ออกแบบโดย **neronain**

[facebook.com/neronain.minidev](https://www.facebook.com/neronain.minidev)

</div>
