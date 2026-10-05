# TraceGraph-Guided Recovery: Intervention Scripts

These scripts implement the SWE-bench recovery study of the paper. Graph-derived
trap regions supply runtime triggers, and matched continuations are forked from
the triggered state.

## Prerequisites

In addition to the base `tracegraph` library, the SWE scripts require:

### System dependencies
- **Docker** with the SWE-bench Verified instance images
  (`sweb.eval.x86_64.<instance_id>:latest`).
- **A local OpenAI-compatible LLM server** (e.g., vLLM) or API credentials for
  a provider.

### Python packages
```bash
pip install ".[intervention]"
# Or manually:
pip install openai datasets
```

- **Agent runtime**: bundled as `tracegraph.sweagent` (plain chat-completion
  `THOUGHT` / `ACTION` prompting, one bash command per turn).
- **swebench** (official harness): install separately to evaluate patches.

### Environment variables
```bash
# For provider APIs (set as needed)
export DEEPSEEK_API_KEY="your-key"
export GLM_API_KEY="your-key"
```

## Scripts

| Script | Purpose |
|--------|---------|
| `build_swe_trap_library.py` | Trap and reference libraries + IDF corpus from SWE-bench graph artifacts |
| `build_swe_trap_diagnosis.py` | Six-family diagnosis sidecar used by the Note arm |
| `swe_runner.py` | Prefix-fork runner with the trap detector |
| `eval_patches.py` | Evaluate generated patches with the official SWE-bench harness |

## Experiment Design

### Prefix fork

A probe rollout runs at the base temperature with the detector active. When the
detector first fires, the Docker workspace and the message prefix are
snapshotted, and each arm continues from that identical state with the same
total step budget and top-p 0.95:

| Arm | Paper name | Temperature | Note |
|-----|------------|-------------|------|
| `tg_baseline` | Baseline | 0.6 | None |
| `tg_hot` | Hot | 0.9 | None |
| `tg_repair_cool` | Note | 0.6 | Evidence-grounded diagnosis note |

The note fills four slots from the agent's own log (last command, last
error/test signal, files touched, detector similarity) and a fifth with the
pre-specified family diagnosis stored with the best-matching historical trap.
No gold patch, verified-test list, or grader signal is used.

### Detector

Trap key sets are harvested from graph-derived trap blocks (bottom quartile of
the negative outcome field) of the SWE-bench split. At runtime each step is
encoded into the same key alphabet, generic phase keys are stripped, and the
step is scored by IDF-weighted Jaccard similarity against the trap library. A
trigger fires when

```
sim_trap(s) >= 0.35
```

together with a warmup (2 steps), cooldown (4 steps), an edit or submit intent,
and a `FILE_PATH:*` cue, with at most one trigger per rollout. The similarity to
the core-side reference library and the trap-versus-reference margin (0.03) are
recorded per step for bookkeeping only and never gate a trigger.

## Example

```bash
# Prefix fork on a single SWE-bench instance (paper defaults)
python scripts/intervention/swe_runner.py \
    --instances django__django-11066 \
    --api-provider vllm \
    --model-base-url http://localhost:8000/v1 \
    --model-name <served-model-name> \
    --output results/cxcmu/intervention/example.jsonl

# Provider APIs
python scripts/intervention/swe_runner.py --api-provider glm --instances django__django-11066 \
    --output results/cxcmu/intervention/example_glm.jsonl

# Evaluate patches
python scripts/intervention/eval_patches.py \
    --rollout-glob "results/cxcmu/intervention/example.jsonl" \
    --arms tg_baseline tg_hot tg_repair_cool
```

The runner writes one row per (instance, arm) to `--output` and the saved
trigger prefixes to `<output stem>_prefixes.jsonl`; instances on which the
detector never fires produce a prefix row with `found_prefix = false` and no
continuation rows.
