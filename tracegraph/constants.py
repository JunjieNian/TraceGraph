"""Canonical hyperparameters for the TraceGraph pipeline.

All values are held fixed across every experiment unless an explicit
sensitivity sweep is noted.
"""

# ── Graph construction ────────────────────────────────────────────
NEIGHBOR_K = 6               # mutual-kNN neighbourhood size
MAX_NODES = 3000             # node cap per task graph

# ── Reward field ─────────────────────────────────────────────────
PROPAGATION_ALPHA = 0.65     # seed (teleport) weight α in field diffusion
PROPAGATION_STEPS = 24       # number of diffusion iterations
SUPPORT_SHRINK_EXP = 0.5     # support-shrinkage exponent β of the block seed
CORE_POS_Q = 0.75            # positive-quantile threshold for core mask
MIN_RUN_SUPPORT = 3          # minimum visiting runs for a block to receive a seed
# Splits whose recorded reward is continuous rather than a pass/fail flag.
# For these the block seed averages the per-task max-normalised reward of
# the visiting runs, matching the reward-weighted demand contrast; using a
# binary seed here does NOT reproduce the published MCPBench demand row.
CONTINUOUS_REWARD_BENCHMARKS = ("mcpbench",)

# ── Typed-state kernel (signature-ablation statistic) ─────────────
LAPLACE_ALPHA = 0.5          # Laplace smoothing for kernel estimation

# ── EOS absorbing states ────────────────────────────────────────
EOS_RESOLVED = "EOS_resolved"
EOS_FAILED = "EOS_failed"
