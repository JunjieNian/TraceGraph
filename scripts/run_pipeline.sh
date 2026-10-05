#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# TraceGraph: end-to-end pipeline runner
#
# Builds shared decision landscapes from parsed agent trajectories,
# then computes rollout events, supply/demand profiles, and their
# bootstrap CIs and threshold sweeps.
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
echo "── Stage 1: Build mutual-kNN graphs + BCC ──"
python scripts/pipeline/build_graphs.py $ARGS

echo ""
echo "── Stage 2: Compute reward field (diffusion + core mask) ──"
python scripts/pipeline/compute_reward_field.py $ARGS

echo ""
echo "── Stage 3: Compute rollout events + supply/demand ──"
python scripts/pipeline/rollout_events.py $ARGS

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Pipeline complete. Results in results/cxcmu/"
echo "═══════════════════════════════════════════════════"

echo ""
echo "── Analysis: bootstrap CIs + quantile sweeps ──"
python scripts/analysis/sensitivity.py

echo ""
echo "Done. See results/cxcmu/ for outputs."
