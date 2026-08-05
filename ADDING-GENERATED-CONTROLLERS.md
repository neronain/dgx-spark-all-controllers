# Adding an LMDS-generated controller to this collection

Every controller here was written by hand and run on real hardware. That is the whole point of the
collection: students, teammates and customers take a file and run it, without generating or testing
anything themselves.

[LMDS](https://github.com/neronain/AutoDeployDGXProject) (Local Model Deploy Studio) produces
controllers from a Hugging Face link that follow **the same contract as this collection** —
`audit-controllers.py` reports 0 errors and 0 warnings on its output for both single-node and
stacked bundles. So a generated controller can join this collection, as long as it earns its place
the same way the hand-written ones did: by being run.

> **The bar for this repository is not "it passes the audit". It is "someone served real traffic
> with it on a DGX Spark."** The audit is necessary, never sufficient. A file that only passes the
> audit belongs in a bundle directory, not here.

## The flow

```bash
# 1. Generate on the machine that will run it
lmds deploy https://huggingface.co/<org>/<model> --target dgx-spark-single

# 2. Actually run it, end to end
cd bundles/<slug>
./<slug>-single.sh download && ./<slug>-single.sh verify-files
./<slug>-single.sh start && ./<slug>-single.sh test-text
./<slug>-single.sh status && ./<slug>-single.sh stop
```

For stacked bundles, also run `sync-worker`, `verify-worker` and a real two-node `start`.
If the machines are registered with the LMDS hub, write the cluster addresses first so the
controller does not have to ask:

```bash
lmds node cluster                       # which machines can be stacked, on which fabric
lmds node cluster --write <slug>        # writes cluster.env into the bundle
```

```bash
# 3. Check it against this collection's contract
python3 audit-controllers.py path/to/bundle
```

```bash
# 4. Copy it in, run the full suite, then commit
cp path/to/bundle/<slug>-single.sh ./<model-name>-single.sh
chmod +x ./<model-name>-single.sh
./verify-all.sh
```

## Naming

Match what is already here — the model, the quantisation if it matters, and the topology:

```text
qwen3-coder-next-single.sh
gemma-4-26b-a4b-it-gguf-single.sh
deepseek-v4-flash-nvfp4-stacked.sh
minimax-m3-v0-nvfp4-reap50-stacked.sh
```

## What to check by hand before committing

`verify-all.sh` catches contract violations, not judgement calls. Read the file for these:

| Check | Why |
|---|---|
| `MODEL_LABEL`, `RUNTIME_LABEL`, `MODEL_FEATURES` read correctly in `info` | This is the first thing a student sees |
| Context default is one the hardware can actually hold | A generated default may assume more memory than the target has |
| No IP, username or path from the machine you generated on | Everything must come from `$HOME`, `${USER}` or a prompt |
| Stacked: `prompt_cluster_config` still fires on `start`/`restart` | Someone else's cluster will not use your addresses |
| The commands listed in `help` all exist | Older bundles genuinely lack newer commands |

## Updating the collection files

After adding a controller:

1. `MANIFEST.txt` — add the file under the right runtime section and bump the totals
2. `CHANGELOG.md` — a new version entry saying what was added and **what hardware it ran on**
3. `README.md` and `README.en.md` — add it to the model list and update the counts
4. `PACKAGE_SHA256SUMS` — regenerate
5. `./verify-all.sh` — must end with `errors=0, warnings=0`

## Keeping the two projects in sync

Improvements found here get backported into the LMDS templates, so every future generated
controller inherits them. v3.2.0's fabric and RoCE HCA discovery went the other way — it was built
in LMDS, proven on hardware, then brought back into `deepseek-v4-flash-nvfp4-stacked.sh`.

When you fix something in a controller in this repository, ask whether the generator should learn
it too. A fix that lives only here has to be re-discovered by hand for every new model.
