# TraceGraph

**TraceGraph: Shared Decision Landscapes for Diagnosing and Improving Agent Trajectories**

Junjie Nian\*, Kang Chen\*, Ge Zhang, Yixin Cao, Yugang Jiang

Fudan University, ByteDance, Shanghai Innovation Institute

Current release: **v0.1.0**.

<p align="center">
  <img src="paper/figures/pipeline.png" width="90%" />
</p>

---

## Core Idea

Agent benchmarks reduce rich trajectories to single scalars (pass/fail, reward),
hiding the diverse *process* patterns underneath. Two models can score identically
yet fail for completely different reasons: one never inspects the right file, the
other finds it but applies a broken patch.

TraceGraph addresses this by:

1. **Building shared decision landscapes** from pooled multi-model rollouts
   (model identity is excluded during construction, added only afterward)
2. **Overlaying outcome information** to identify productive cores and trap regions
3. **Defining three rollout events** (Access, Trap exposure, Repair) that decompose
   model behavior into interpretable supply and benchmark demands
4. **Triggering recovery at runtime** when the live agent enters a graph-derived
   trap state

---

## Method Overview

### Stage 1: Shared Landscape Construction

Each action-observation step is encoded as a sparse **key set** from 7 symbolic
channels (tool name, action type, command class, observation pattern, file cues,
temporal phase, search-specific keys). Steps are compared via **IDF-weighted
Jaccard similarity** and connected into a **mutual-k-NN graph**. The graph is
decomposed into **Biconnected Components (BCCs)** that serve as coarse shared
agent states, with articulation points marking strategy transitions.

### Stage 2: Outcome Overlays

Each BCC block receives a Laplace-smoothed reward seed based on visiting
rollouts' outcomes. Seeds are propagated via **personalized PageRank diffusion**
over the block quotient graph. High-field blocks form **productive cores**;
low-field blocks form **trap regions**. Failure basins (connected trap clusters
with low escape probability) and recovery gates (blocks connecting basins to
cores with success uplift) are identified.

### Stage 3: Process Profiles

Three binary rollout events are extracted per trajectory:

| Event | Definition |
|-------|-----------|
| **Access** (A) | Did the rollout visit any productive core block? |
| **Trap** (E) | Did the rollout visit any trap block? |
| **Repair** (R) | Did the rollout escape a trap block to reach a core? |

These events are aggregated into:
- **Model supply vectors**: task-centered residuals showing each model's event
  propensity relative to peers on the same tasks
- **Benchmark demand vectors**: reward-weighted contrast showing which events
  separate successful from failed traffic on each benchmark

### Stage 4: Trap-Aware Recovery (Intervention)

Graph-derived trap key sets are stored in a **detector library**. At runtime,
each agent step's key set is compared against the library via IDF-weighted
Jaccard. When trap similarity exceeds a threshold (with a margin over core
similarity), a **prefix-fork** is triggered: the agent state is checkpointed,
and continuation proceeds with either a temperature bump or an evidence-grounded
diagnosis note.

---

## Repository Structure

```
tracegraph/                          # Core library
  constants.py                       #   All hyperparameters (k, sigma, alpha, ...)
  signature.py                       #   Key-set extraction + IDF-weighted Jaccard
  graph_construction.py              #   Mutual-kNN graph + BCC decomposition
  reward_field.py                    #   Reward diffusion + core/trap masks
  failure_basins.py                  #   Basin/gate/loop detection
  typed_state_mdp.py                 #   Typed-state kernels + mechanism metrics
  dataset.py                         #   Path helpers
  sweagent/                          #   Bundled MiniSWEAgent-style SWE runtime

scripts/
  pipeline/                          # Build shared landscapes
    extract_signatures.py            #   Parse -> key-sets + IDF + kNN
    build_graphs.py                  #   kNN -> mutual-kNN + BCC analysis
    compute_reward_field.py          #   Reward diffusion + core/trap overlay
    detect_failure_basins.py         #   Basin + gate + loop motif detection
    extract_typed_dynamics.py        #   Per-model metrics (committor, MFPT, ...)
    rollout_events.py                #   Access / Trap / Repair events + supply/demand

  analysis/                          # Process profile analysis
    cross_benchmark.py               #   ANOVA, rank consistency, Spearman rho
    annotation_sampling.py           #   BCC pair + articulation sampling for validation
    signature_ablation.py            #   Leave-one-key-type-out ablation
    enhanced_separability.py         #   Logistic regression + 4 capability axes
    supply_demand_decomposition.py   #   Bilinear supply x demand factorization
    counterfactual.py                #   MDP kernel interventions + matched controls
    shared_vs_permodel.py            #   Shared vs per-model graph robustness
    sensitivity.py                   #   Bootstrap CI + hyperparameter sweeps

  intervention/                      # Trap-aware recovery (RQ4)
    swe_runner.py                    #   SWE-bench prefix-fork intervention runner
    eval_patches.py                  #   SWE-bench harness evaluation
    pool_analysis.py                 #   Difference-in-differences analysis
    detector_sweep.py                #   Trigger predicate replay sweep

paper/                               # LaTeX source
```

---

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

### Additional dependencies for intervention experiments

The intervention scripts (`scripts/intervention/`) additionally require:
- Docker (for SWE-bench environment isolation)
- A local LLM server (e.g., vLLM) or API access (DeepSeek, GLM)
- The bundled `tracegraph.sweagent` runtime; no external agent framework is required
- `datasets` for loading SWE-bench Verified metadata
- `swebench` only when evaluating patches with the official harness

See `scripts/intervention/README.md` for setup details.

---

## Data Preparation

```bash
python scripts/data/download_cxcmu.py
python scripts/data/parse_cxcmu.py
```

The parser writes `data/cxcmu/parsed/{benchmark}/{task_id}.jsonl`, the input expected by the pipeline. Set `HF_TOKEN` if HuggingFace requires gated access to the trajectory release.

---

## Usage

### Pipeline: Building Shared Landscapes

All scripts are run from the repository root. They read from `data/` and write
to `results/`.

```bash
# 1. Extract symbolic signatures + IDF + kNN arrays
python scripts/pipeline/extract_signatures.py --benchmark swebench

# 2. Build mutual-kNN graphs + BCC decomposition
python scripts/pipeline/build_graphs.py --benchmark swebench

# 3. Compute reward field (diffusion + core/trap masks)
python scripts/pipeline/compute_reward_field.py --benchmark swebench

# 4. Detect failure basins, recovery gates, loop motifs
python scripts/pipeline/detect_failure_basins.py --benchmark swebench

# 5. Extract per-model typed-state dynamics
python scripts/pipeline/extract_typed_dynamics.py --benchmark swebench

# 6. Compute rollout events + supply/demand profiles
python scripts/pipeline/rollout_events.py --benchmark swebench
```

### Analysis: Process Profiles

```bash
# Cross-benchmark analysis (ANOVA, Kendall tau, Spearman rho)
python scripts/analysis/cross_benchmark.py

# Capability axes (4 task-centered composite dimensions)
python scripts/analysis/enhanced_separability.py

# Supply x demand bilinear decomposition
python scripts/analysis/supply_demand_decomposition.py

# Counterfactual MDP stress tests
python scripts/analysis/counterfactual.py

# Bootstrap CI + hyperparameter sensitivity
python scripts/analysis/sensitivity.py
```

### Intervention: Trap-Aware Recovery

The SWE detector ships with bundled libraries in `resources/swebench_detector/`. To rebuild them from local graph artifacts, run:

```bash
python scripts/intervention/build_swe_trap_library.py
python scripts/intervention/build_swe_trap_diagnosis.py
```

```bash
# SWE-bench prefix-fork recovery (requires docker + LLM server)
python scripts/intervention/swe_runner.py \
    --design prefix-fork \
    --arms tg_baseline tg_hot tg_repair_cool \
    --instances django__django-11066 \
    --seed 11

# Evaluate patches via official SWE-bench harness
python scripts/intervention/eval_patches.py \
    --pilot-glob "results/cxcmu/intervention/*.jsonl"

# Difference-in-differences analysis
python scripts/intervention/pool_analysis.py
```

Note: the paper experiments used the default chat-completion runner with
plain `THOUGHT` / `ACTION` text prompting. Harmony/native tool prompting was
not used in the reported experiments and is not required for reproduction.

---

## Key Hyperparameters

All hyperparameters are centralized in `tracegraph/constants.py`:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `NEIGHBOR_K` | 6 | Mutual-kNN neighborhood size |
| `DIST_SCALE` (sigma) | 0.35 | RBF bandwidth for edge weights |
| `PROPAGATION_ALPHA` | 0.65 | Teleport weight in reward diffusion |
| `PROPAGATION_STEPS` | 24 | Number of diffusion iterations |
| `CORE_POS_Q` | 0.75 | Positive quantile for core mask |
| `FAILURE_BASIN_MIN_FAIL_RATE` | 0.70 | Minimum fail rate for basin seeds |
| `FAILURE_BASIN_MAX_ESCAPE_3STEP` | 0.30 | Maximum 3-step escape probability |
| `MAX_NODES` | 3000 | Node cap per task |

---

## Citation

```bibtex
@article{nian2026tracegraph,
  title   = {TraceGraph: Shared Decision Landscapes for Diagnosing
             and Improving Agent Trajectories},
  author  = {Nian, Junjie and Chen, Kang and Zhang, Ge
             and Cao, Yixin and Jiang, Yugang},
  journal = {arXiv preprint arXiv:2605.31308},
  year    = {2026},
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
