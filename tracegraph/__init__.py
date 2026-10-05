"""TraceGraph — shared decision-landscape analysis of agent trajectories.

Core modules for building shared decision landscapes from pooled,
observable agent traces (cx-cmu/agent_trajectories dataset).

Modules:
    signature           — runtime observation keys, IDF weighting, Jaccard distance
    graph_construction  — mutual-kNN graph and BCC decomposition
    reward_field        — outcome-seeded diffusion over the block quotient graph
    typed_state_mdp     — typed-state kernel behind the signature-ablation statistic
    constants           — all canonical hyperparameters
    dataset             — parsed-outcome loader
    sweagent            — bundled MiniSWEAgent-style SWE runtime
"""

__version__ = "0.1.0"
