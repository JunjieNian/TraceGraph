"""TraceGraph — shared decision-landscape analysis of agent trajectories.

Core modules for building graph-theoretic process atlases from
observable agent traces (cx-cmu/agent_trajectories dataset).

Modules:
    signature           — key-set extraction, IDF weighting, Jaccard distance
    graph_construction  — mutual-kNN graph, BCC decomposition, role assignment
    reward_field        — label-seeded propagation + high-value core mask
    typed_state_mdp     — typed-state encoding, Laplace kernels, TV
    failure_basins      — failure basin detection + recovery gates
    constants           — all canonical hyperparameters
    dataset             — multi-dataset path helpers + registry
    sweagent            — bundled MiniSWEAgent-style SWE runtime
"""

__version__ = "0.1.0"
