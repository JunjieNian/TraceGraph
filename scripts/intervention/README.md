# Trap-Aware Recovery: Intervention Scripts

These scripts implement the **TraceGraph-guided trap-aware recovery pipeline**
(Section 5 / RQ4 in the paper). They use graph-derived trap states as runtime
triggers to inject lightweight continuation policies at detected trap states.

## Prerequisites

In addition to the base `tracegraph` library, SWE intervention scripts require:

### System dependencies
- **Docker**: SWE-bench uses containerized environments for each instance.
- **A local LLM server** (e.g., vLLM) or API credentials for a remote provider.

### Python packages
```bash
pip install ".[intervention]"
# Or manually:
pip install openai datasets
```

### Runtime and evaluation
- **SWE agent runtime**: Bundled as `tracegraph.sweagent`; no external agent
  framework or private repository is required.
- **swebench** (official harness): Install separately when evaluating patches.
- **Harmony**: Not used in the paper experiments. The released runner uses
  plain chat-completion `THOUGHT` / `ACTION` prompting.

### Environment variables
```bash
# For remote LLM providers (set as needed)
export DEEPSEEK_API_KEY="your-key"
export GLM_API_KEY="your-key"

# For SWE-bench harness (increase from default 120s)
export HARNESS_TIMEOUT=900
```

## Scripts

| Script | Purpose |
|--------|---------|
| `swe_runner.py` | SWE-bench prefix-fork recovery with multiple arm designs |
| `eval_patches.py` | Evaluate generated patches via official SWE-bench harness |
| `pool_analysis.py` | Difference-in-differences analysis across providers |
| `detector_sweep.py` | Replay-based trigger predicate sweep (no GPU needed) |

## Experiment Design

### Prefix-Fork (SWE-bench)

From an identical trigger state, three continuation arms are forked:

| Arm | Temperature | Note |
|-----|-------------|------|
| **Baseline** (`tg_baseline`) | T=0.6 | None |
| **Hot** (`tg_hot`) | T=0.9, top_p=0.95 | None |
| **Note** (`tg_repair_cool`) | T=0.6 | Evidence-grounded diagnosis |

The diagnosis note contains only information from the agent's own log:
last command, last error signal, top file paths, detector confidence,
and a family-level diagnosis. No oracle information is provided.

### Detector Library

Bundled SWE detector libraries are provided under `resources/swebench_detector/`. To rebuild them from graph artifacts, run:

```bash
python scripts/intervention/build_swe_trap_library.py
python scripts/intervention/build_swe_trap_diagnosis.py
```

Trap key sets are extracted from graph-derived trap-side blocks during
the offline pipeline. At runtime, each step's key set is scored via
IDF-weighted Jaccard against the trap library and a non-trap (core) library.
A trigger fires when:

```
sim_trap(s) >= threshold  AND  sim_trap(s) - sim_core(s) >= margin
```

Additional gates: warmup steps, cooldown between triggers, action intent
filter (edit/submit only), file-locality requirement.

## Example

```bash
# Run prefix-fork on a single SWE-bench instance
python scripts/intervention/swe_runner.py \
    --design prefix-fork \
    --arms tg_baseline tg_hot tg_repair_cool \
    --instances django__django-11066 \
    --seed 11 \
    --model-base-url http://localhost:8192/v1 \
    --model-name "Qwen/Qwen3.6-35B-A3B" \
    --output results/intervention/example.jsonl

# Evaluate patches
python scripts/intervention/eval_patches.py \
    --pilot-glob "results/intervention/example.jsonl" \
    --arms tg_baseline tg_hot tg_repair_cool
```
