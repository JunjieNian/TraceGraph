"""Canonical hyperparameters for the TraceGraph pipeline.

Adapted from SliceGraph constants for SWE agent trajectory analysis.
All values match the definitions in the method section and are held fixed
across every experiment unless an explicit sensitivity sweep is noted.
"""

# ── Signature extraction ──────────────────────────────────────────
WINDOW_SIZE = 1              # action-observation pairs per slice (1, 2, or 3)
WINDOW_STRIDE = 1            # stride between windows

# ── Graph construction (adapted from SliceGraph §3.1) ─────────────
NEIGHBOR_K = 6               # mutual-kNN neighbourhood size
DIST_SCALE = 0.35            # RBF bandwidth for graph-edge weights
MAX_NODES = 3000             # node cap per issue cell

# ── Reward field (adapted from SliceGraph §3.4) ──────────────────
PROPAGATION_ALPHA = 0.65     # teleport weight α in field diffusion
PROPAGATION_STEPS = 24       # number of diffusion iterations T
SUPPORT_SHRINK_EXP = 0.5     # support-shrinkage exponent β
CORE_POS_Q = 0.75            # positive-quantile threshold for core mask
MIN_RUN_SUPPORT = 3          # minimum visiting runs for a block to receive a seed

# ── Process families (adapted from SliceGraph §3.3) ──────────────
JACCARD_THRESHOLD = 0.05     # minimum weighted-Jaccard for an edge
LOUVAIN_RESOLUTION = 1.0     # Louvain resolution parameter γ
LOUVAIN_SEED = 42            # community-detection seed

# ── Typed-state semi-MDP (§3.5) ─────────────────────────────────
LAPLACE_ALPHA = 0.5          # Laplace smoothing α_L for kernel estimation
MIN_FAMILY_SIZE = 3          # minimum family size for TV computation
SUCCESSOR_GAMMAS = (0.9, 0.95, 0.99)  # discount factors for M = (I − γP)⁻¹

# ── Block roles and phases (agent-adapted) ───────────────────────
ROLES = ("common_setup", "decision_point", "intermediate",
         "weak_basin", "success_outcome")
PHASES = ("early", "mid", "late")
CORE_TAGS = ("core", "outer")

# ── EOS absorbing states ────────────────────────────────────────
EOS_RESOLVED = "EOS_resolved"
EOS_FAILED = "EOS_failed"

# ── Dynamic-metric definitions ──────────────────────────────────
DYNAMIC_METRICS = ("committor_dp", "mfpt_dp",
                   "escape_hazard_3step", "return_prob_3step")
DYNAMIC_SIGN = {
    "committor_dp":        +1.0,   # higher → more likely to reach resolved EOS
    "mfpt_dp":             -1.0,   # lower  → faster arrival at core
    "escape_hazard_3step": -1.0,   # lower  → more stable core occupancy
    "return_prob_3step":   +1.0,   # higher → better error correction
}

# ── Failure basin thresholds (NEW) ────────────────────────────────
FAILURE_BASIN_MIN_FAIL_RATE = 0.7    # τ_f: minimum fail rate to qualify as basin
FAILURE_BASIN_MAX_ESCAPE_3STEP = 0.3 # τ_e: maximum 3-step escape probability
RECOVERY_GATE_MIN_UPLIFT = 0.1       # minimum Δ(g) for a recovery gate

# ── Data filtering ───────────────────────────────────────────────
MIN_TRAJECTORIES_PER_ISSUE = 6
MIN_RESOLVED = 1
MIN_FAILED = 1
MIN_TRAJECTORY_STEPS = 5
