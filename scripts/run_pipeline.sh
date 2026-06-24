#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# TraceGraph: end-to-end pipeline runner
#
# Builds shared decision landscapes from parsed agent trajectories,
# then runs analysis scripts to produce process profiles.
#
# Usage:
#   bash scripts/run_pipeline.sh [--benchmark BENCH] [--max-tasks N]
#
# Prerequisites:
#   - Parsed trajectories in data/cxcmu/parsed/{benchmark}/{task_id}.jsonl
#   - pip install -e .
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

BENCHMARK="${1:---benchmark swebench}"
MAX_TASKS="${2:-}"

ARGS=""
if [[ "$BENCHMARK" == --benchmark* ]]; then
    ARGS="$BENCHMARK"
else
    ARGS="--benchmark $BENCHMARK"
fi
if [[ -n "$MAX_TASKS" ]]; then
    ARGS="$ARGS --max-tasks $MAX_TASKS"
fi

echo "═══════════════════════════════════════════════════"
echo "  TraceGraph Pipeline"
echo "  Args: $ARGS"
echo "═══════════════════════════════════════════════════"

echo ""
echo "── Stage 1: Extract signatures + IDF + kNN ──"
python scripts/pipeline/extract_signatures.py $ARGS

echo ""
echo "── Stage 2: Build mutual-kNN graphs + BCC ──"
python scripts/pipeline/build_graphs.py $ARGS

echo ""
echo "── Stage 3: Compute reward field (diffusion) ──"
python scripts/pipeline/compute_reward_field.py $ARGS

echo ""
echo "── Stage 4: Detect failure basins + gates ──"
python scripts/pipeline/detect_failure_basins.py $ARGS

echo ""
echo "── Stage 5: Extract typed-state dynamics ──"
python scripts/pipeline/extract_typed_dynamics.py $ARGS

echo ""
echo "── Stage 6: Compute rollout events + profiles ──"
python scripts/pipeline/rollout_events.py $ARGS

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Pipeline complete. Results in results/cxcmu/"
echo "═══════════════════════════════════════════════════"

echo ""
echo "── Analysis: Cross-benchmark ──"
python scripts/analysis/cross_benchmark.py

echo ""
echo "── Analysis: Capability axes ──"
python scripts/analysis/enhanced_separability.py

echo ""
echo "── Analysis: Supply x Demand decomposition ──"
python scripts/analysis/supply_demand_decomposition.py

echo ""
echo "Done. See results/cxcmu/ for outputs."
