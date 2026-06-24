#!/usr/bin/env python3
"""Pool 3 Qwen/DeepSeek intervention experiments and produce paper-ready
artifacts: (A) bootstrap CIs on pooled specificity, (B) concrete case
studies, (C) trigger-fire-rate vs specificity figure.

Inputs:
  - results/cxcmu/intervention/_paired/swe_deepseek_n60.tsv   columns: instance_id, placebo, tracegraph, shuffled
  - results/cxcmu/intervention/_paired/swe_qwen_n60.tsv       same columns
  - results/cxcmu/intervention/_paired/hotpot_qwen_n60.tsv    columns: instance_id, placebo, gate_hint, shuffled, generic_note
  - results/cxcmu/intervention/hotpot_4arm/hotpot_qwen_*_n60.jsonl   full rollouts for case studies

Outputs:
  - results/cxcmu/intervention/_paired/pooled_specificity.json
  - results/cxcmu/intervention/_paired/case_studies.json
  - results/cxcmu/intervention/_paired/figure_fire_vs_spec.csv
  - figs/intervention_fire_vs_spec.pdf (matplotlib)
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

PAIRED_DIR = Path("results/cxcmu/intervention/_paired")
FIG_DIR = Path("figs")
FIG_DIR.mkdir(parents=True, exist_ok=True)
PAIRED_DIR.mkdir(parents=True, exist_ok=True)


def load_tsv(fp: Path) -> Tuple[List[str], List[Dict[str, int]]]:
    with open(fp) as fh:
        header = fh.readline().strip().split("\t")
        arms = header[1:]
        rows = []
        for line in fh:
            parts = line.strip().split("\t")
            iid = parts[0]
            d = {"instance_id": iid}
            for a, v in zip(arms, parts[1:]):
                d[a] = int(v)
            rows.append(d)
    return arms, rows


def paired_sign(rows: List[Dict[str, int]], arm_a: str, arm_b: str) -> Tuple[int, int, int, float]:
    w = sum(1 for r in rows if r[arm_a] > r[arm_b])
    l = sum(1 for r in rows if r[arm_a] < r[arm_b])
    t = sum(1 for r in rows if r[arm_a] == r[arm_b])
    n = w + l
    if n == 0:
        return w, l, t, 1.0
    k = min(w, l)
    p = 2 * sum(math.comb(n, i) * 0.5**n for i in range(k + 1))
    return w, l, t, min(p, 1.0)


def bootstrap_paired_delta(rows: List[Dict[str, int]], arm_a: str, arm_b: str,
                           n_iter: int = 10000, seed: int = 20260521) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(rows)
    deltas = []
    for _ in range(n_iter):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        da = sum(rows[i][arm_a] for i in idx) / n
        db = sum(rows[i][arm_b] for i in idx) / n
        deltas.append(da - db)
    deltas.sort()
    return (deltas[int(0.025 * n_iter)], deltas[int(0.5 * n_iter)], deltas[int(0.975 * n_iter)])


# ----------------------------------------------------------------------
# 2x2 FACTORIAL DiD (added 2026-05-21)
# Tests Δ_map-specific = (real,specific − real,generic) − (shuf,specific − shuf,generic)
# i.e. whether the specific repair note is more effective on the real TraceGraph
# state than on a content-matched non-target state, holding note content equal.

FACT_ARMS = ("tg_repair", "tg_generic", "shuf_repair", "shuf_generic")


def load_factorial_resolved(merged_jsonl: Path) -> List[Dict[str, int]]:
    """Load the merged 4-arm prefix-fork JSONL and return one row per
    instance_id present in *all four* arms. Each row has fields
    {instance_id, tg_repair, tg_generic, shuf_repair, shuf_generic} ∈ {0,1}.
    Also passes through prefix_obs_has_hard_error and prefix_recent_hard_error_context
    (same for all 4 arms by prefix-fork construction; we take the tg_repair value).
    """
    by_arm: Dict[str, Dict[str, Dict]] = {a: {} for a in FACT_ARMS}
    with open(merged_jsonl) as fh:
        for line in fh:
            r = json.loads(line)
            arm = r.get("arm")
            iid = r.get("instance_id")
            if arm not in by_arm or not iid:
                continue
            by_arm[arm][iid] = {
                "resolved": int(bool(r.get("resolved"))),
                "prefix_obs_has_hard_error": bool(r.get("prefix_obs_has_hard_error")),
                "prefix_recent_hard_error_context": bool(
                    r.get("prefix_recent_hard_error_context")
                ),
                "prefix_trigger_step": r.get("prefix_trigger_step"),
                "n_triggers": int(r.get("n_triggers") or 0),
            }
    common = set.intersection(*[set(d.keys()) for d in by_arm.values()])
    rows: List[Dict[str, int]] = []
    for iid in sorted(common):
        row = {"instance_id": iid}
        for a in FACT_ARMS:
            row[a] = by_arm[a][iid]["resolved"]
        # prefix conditions are shared across arms by construction;
        # take tg_repair's value (canonical).
        meta = by_arm["tg_repair"][iid]
        row["_prefix_obs_has_hard_error"] = int(meta["prefix_obs_has_hard_error"])
        row["_prefix_recent_hard_error"] = int(meta["prefix_recent_hard_error_context"])
        row["_prefix_trigger_step"] = meta["prefix_trigger_step"]
        rows.append(row)
    return rows


def did_point(rows: List[Dict[str, int]]) -> Dict[str, float]:
    n = len(rows)
    if n == 0:
        return {}
    tg_r = sum(r["tg_repair"]    for r in rows) / n
    tg_g = sum(r["tg_generic"]   for r in rows) / n
    sh_r = sum(r["shuf_repair"]  for r in rows) / n
    sh_g = sum(r["shuf_generic"] for r in rows) / n
    return {
        "n": n,
        "tg_repair": tg_r,
        "tg_generic": tg_g,
        "shuf_repair": sh_r,
        "shuf_generic": sh_g,
        "delta_specific": tg_r - sh_r,                          # current "specificity"
        "delta_instructive_real": tg_r - tg_g,                  # repair-content effect on real
        "delta_instructive_shuf": sh_r - sh_g,                  # repair-content effect on shuf
        "did": (tg_r - tg_g) - (sh_r - sh_g),                   # map-specific DiD
    }


def did_paired_bootstrap(rows: List[Dict[str, int]], *,
                         n_iter: int = 10000, seed: int = 20260521
                         ) -> Tuple[float, float, float]:
    """Paired-by-instance bootstrap of the 2×2 DiD estimator."""
    if not rows:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(rows)
    dids: List[float] = []
    for _ in range(n_iter):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        tg_r = sum(rows[i]["tg_repair"]    for i in idx) / n
        tg_g = sum(rows[i]["tg_generic"]   for i in idx) / n
        sh_r = sum(rows[i]["shuf_repair"]  for i in idx) / n
        sh_g = sum(rows[i]["shuf_generic"] for i in idx) / n
        dids.append((tg_r - tg_g) - (sh_r - sh_g))
    dids.sort()
    return (dids[int(0.025 * n_iter)],
            dids[int(0.5   * n_iter)],
            dids[int(0.975 * n_iter)])


def did_paired_bootstrap_clustered(
    rows: List[Dict[str, int]],
    *,
    cluster_key: str = "instance_id",
    n_iter: int = 10000,
    seed: int = 20260601,
) -> Tuple[float, float, float]:
    """Cluster bootstrap of the 2×2 DiD, resampling at the cluster (iid)
    level rather than at the row level.

    Rationale: when each (iid) contributes multiple rows (e.g. multiple
    seeds, or repeated rollouts), row-level bootstrap underestimates
    variance because it treats correlated rows as independent. Cluster
    bootstrap resamples whole iids and averages within-cluster across the
    four arms before differencing.

    Each row must have keys: cluster_key, tg_repair, tg_generic,
    shuf_repair, shuf_generic ∈ {0,1}.
    """
    if not rows:
        return (0.0, 0.0, 0.0)
    # Group rows by cluster key.
    clusters: Dict[str, List[Dict[str, int]]] = {}
    for r in rows:
        clusters.setdefault(str(r[cluster_key]), []).append(r)
    cluster_ids = list(clusters.keys())
    n_c = len(cluster_ids)
    if n_c == 0:
        return (0.0, 0.0, 0.0)

    # Pre-compute per-cluster within-cluster arm means (avg over seeds /
    # repeats inside each iid).
    per_cluster_means: Dict[str, Dict[str, float]] = {}
    for cid, crows in clusters.items():
        m = {a: 0.0 for a in FACT_ARMS}
        k = len(crows)
        for cr in crows:
            for a in FACT_ARMS:
                m[a] += float(cr[a])
        per_cluster_means[cid] = {a: m[a] / k for a in FACT_ARMS}

    rng = random.Random(seed)
    dids: List[float] = []
    for _ in range(n_iter):
        idx = [rng.randint(0, n_c - 1) for _ in range(n_c)]
        tg_r = sum(per_cluster_means[cluster_ids[i]]["tg_repair"]    for i in idx) / n_c
        tg_g = sum(per_cluster_means[cluster_ids[i]]["tg_generic"]   for i in idx) / n_c
        sh_r = sum(per_cluster_means[cluster_ids[i]]["shuf_repair"]  for i in idx) / n_c
        sh_g = sum(per_cluster_means[cluster_ids[i]]["shuf_generic"] for i in idx) / n_c
        dids.append((tg_r - tg_g) - (sh_r - sh_g))
    dids.sort()
    return (dids[int(0.025 * n_iter)],
            dids[int(0.5   * n_iter)],
            dids[int(0.975 * n_iter)])


def paired_sign_did_pvalue(
    rows: List[Dict[str, int]],
    *,
    cluster_key: str = "instance_id",
    n_iter: int = 10000,
    seed: int = 20260601,
) -> float:
    """One-sided paired-sign permutation p-value for the cluster-level DiD.

    For each iid we compute its within-cluster DiD contribution
    d_i = (tg_repair_i − tg_generic_i) − (shuf_repair_i − shuf_generic_i).
    Under the null E[d_i]=0, the joint sign pattern is exchangeable up to
    a ±1 flip per cluster. The permutation distribution is built by
    independently flipping each d_i's sign, and the one-sided p is the
    fraction of permuted means at least as large as the observed mean.
    """
    if not rows:
        return 1.0
    clusters: Dict[str, List[Dict[str, int]]] = {}
    for r in rows:
        clusters.setdefault(str(r[cluster_key]), []).append(r)
    cluster_ids = list(clusters.keys())

    contribs: List[float] = []
    for cid in cluster_ids:
        crows = clusters[cid]
        k = len(crows)
        tg_r = sum(float(cr["tg_repair"])    for cr in crows) / k
        tg_g = sum(float(cr["tg_generic"])   for cr in crows) / k
        sh_r = sum(float(cr["shuf_repair"])  for cr in crows) / k
        sh_g = sum(float(cr["shuf_generic"]) for cr in crows) / k
        contribs.append((tg_r - tg_g) - (sh_r - sh_g))
    if not contribs:
        return 1.0
    observed = sum(contribs) / len(contribs)
    rng = random.Random(seed)
    n_ge = 0
    for _ in range(n_iter):
        s = 0.0
        for c in contribs:
            if rng.random() < 0.5:
                s += c
            else:
                s -= c
        if s / len(contribs) >= observed - 1e-12:
            n_ge += 1
    # +1 numerator/denominator for the observed sample (one-sided).
    return (n_ge + 1) / (n_iter + 1)


def report_factorial(name: str, rows: List[Dict[str, int]],
                     pooled: Dict[str, Dict[str, float]]) -> None:
    if not rows:
        print(f"  {name:<48s}  (no rows)")
        return
    pt = did_point(rows)
    lo, med, hi = did_paired_bootstrap(rows)
    print(f"  {name}  n={pt['n']}")
    print(f"    tg_repair    = {pt['tg_repair']:.3f}    tg_generic   = {pt['tg_generic']:.3f}")
    print(f"    shuf_repair  = {pt['shuf_repair']:.3f}  shuf_generic = {pt['shuf_generic']:.3f}")
    print(f"    δ_specific   (tg_r − sh_r)                = {pt['delta_specific']:+.3f}")
    print(f"    δ_instr_real (tg_r − tg_g)                = {pt['delta_instructive_real']:+.3f}")
    print(f"    δ_instr_shuf (sh_r − sh_g)                = {pt['delta_instructive_shuf']:+.3f}")
    print(f"    DiD          (δ_instr_real − δ_instr_shuf)= {pt['did']:+.3f}  "
          f"boot95=[{lo:+.3f}, {hi:+.3f}]")
    pooled[name] = {**pt, "boot_lo": lo, "boot_med": med, "boot_hi": hi}


def main() -> None:
    # --- Load paired tables ---
    swe_ds_arms, swe_ds = load_tsv(PAIRED_DIR / "swe_deepseek_n60.tsv")
    swe_q_arms, swe_q  = load_tsv(PAIRED_DIR / "swe_qwen_n60.tsv")
    hp_arms, hp        = load_tsv(PAIRED_DIR / "hotpot_qwen_n60.tsv")

    # Normalize arm names for cross-experiment pooling: gate_hint == tracegraph
    for r in hp:
        r["tracegraph"] = r["gate_hint"]

    # ----------------------- A. POOLED BOOTSTRAP CIs -----------------------
    print("=" * 60)
    print("A. POOLED BOOTSTRAP 95% CIs (paired Δ resolved, tracegraph vs shuffled)")
    print("=" * 60)
    pooled: Dict[str, Dict[str, float]] = {}

    def report(name: str, rows: List[Dict[str, int]]) -> None:
        w, l, t, p = paired_sign(rows, "tracegraph", "shuffled")
        lo, med, hi = bootstrap_paired_delta(rows, "tracegraph", "shuffled")
        n = len(rows)
        delta = (w - l) / n
        print(f"  {name:<32s}  n={n:3d}  W/L/T={w:3d}/{l:3d}/{t:3d}  "
              f"Δ={delta:+.3f}  sign-p={p:.3f}  boot95=[{lo:+.3f}, {hi:+.3f}]")
        pooled[name] = dict(n=n, w=w, l=l, t=t, delta=delta, sign_p=p,
                            boot_med=med, boot_lo=lo, boot_hi=hi)

    report("SWE-DeepSeekV4", swe_ds)
    report("SWE-Qwen36B", swe_q)
    report("HotpotQA-Qwen36B (gate v shuf)", hp)
    print()

    # Pool Qwen-only (SWE-Qwen + HotpotQA-Qwen) at n=120
    qwen_pooled = swe_q + hp
    report("POOLED Qwen (SWE+Hotpot, n=120)", qwen_pooled)

    # Pool all 3 at n=180
    all_pooled = swe_ds + swe_q + hp
    report("POOLED ALL 3 (n=180)", all_pooled)

    # ----------------------- A2. 2×2 FACTORIAL DiD ------------------------
    print()
    print("=" * 60)
    print("A2. 2×2 FACTORIAL DiD (expand60 SWE-Qwen prefix-fork)")
    print("    Δ_map-specific = (tg_r − tg_g) − (sh_r − sh_g)")
    print("=" * 60)
    merged_fp = Path(
        "results/cxcmu/intervention/swe_eval_expand60_swe_qwen36_factorial60"
        "/pilot_swe_with_resolved.jsonl"
    )
    if merged_fp.exists():
        fac_rows = load_factorial_resolved(merged_fp)
        report_factorial("expand60_all", fac_rows, pooled)
        # fire-stratified (prefix had a hard-error context at fork point)
        fac_fired_lookback = [r for r in fac_rows if r["_prefix_recent_hard_error"]]
        if fac_fired_lookback:
            print()
            report_factorial(
                "expand60_prefix_recent_hard_error",
                fac_fired_lookback,
                pooled,
            )
        fac_fired_at_step = [r for r in fac_rows if r["_prefix_obs_has_hard_error"]]
        if fac_fired_at_step:
            print()
            report_factorial(
                "expand60_prefix_hard_error_at_fork_step",
                fac_fired_at_step,
                pooled,
            )
    else:
        print(f"  (skipped — {merged_fp} not found)")

    with open(PAIRED_DIR / "pooled_specificity.json", "w") as fh:
        json.dump(pooled, fh, indent=2)
    print(f"\nwrote {PAIRED_DIR}/pooled_specificity.json")

    # ----------------------- B. CASE STUDIES -----------------------------
    print()
    print("=" * 60)
    print("B. CASE STUDIES (paired flips where tracegraph wins and shuffled loses)")
    print("=" * 60)
    cases: Dict[str, List[Dict]] = {"swe_qwen": [], "hotpot_qwen": []}

    # SWE-Qwen: instance where placebo=0, tracegraph=1, shuffled=0
    for r in swe_q:
        if r["placebo"] == 0 and r["tracegraph"] == 1 and r["shuffled"] == 0:
            cases["swe_qwen"].append({
                "instance_id": r["instance_id"],
                "placebo": r["placebo"], "tracegraph": r["tracegraph"], "shuffled": r["shuffled"],
            })
    # HotpotQA-Qwen: same condition + we can also grab predicted answers
    hp_qid_to_pred: Dict[str, Dict[str, Dict]] = {}
    for arm in ["placebo", "gate_hint", "shuffled", "generic_note"]:
        fp = f"results/cxcmu/intervention/hotpot_4arm/hotpot_qwen_{arm}_n60.jsonl"
        for line in open(fp):
            rec = json.loads(line)
            qid = rec["qid"]
            hp_qid_to_pred.setdefault(qid, {})[arm] = {
                "predicted_answer": rec.get("predicted_answer"),
                "gold_answer": rec.get("gold_answer"),
                "em": rec.get("em"),
                "f1": rec.get("f1"),
                "triggers_fired": rec.get("triggers_fired", 0),
                "n_steps": rec.get("n_steps"),
                "question": rec.get("question"),
            }
    for r in hp:
        if r["placebo"] == 0 and r["gate_hint"] == 1 and r["shuffled"] == 0:
            qid = r["instance_id"]
            cases["hotpot_qwen"].append({
                "qid": qid,
                "question": hp_qid_to_pred[qid]["placebo"]["question"],
                "gold_answer": hp_qid_to_pred[qid]["placebo"]["gold_answer"],
                "placebo_pred": hp_qid_to_pred[qid]["placebo"]["predicted_answer"],
                "gate_hint_pred": hp_qid_to_pred[qid]["gate_hint"]["predicted_answer"],
                "shuffled_pred": hp_qid_to_pred[qid]["shuffled"]["predicted_answer"],
                "gate_hint_triggered": hp_qid_to_pred[qid]["gate_hint"]["triggers_fired"] > 0,
                "shuffled_triggered": hp_qid_to_pred[qid]["shuffled"]["triggers_fired"] > 0,
            })

    print(f"  SWE-Qwen flips (placebo=0, tg=1, shuf=0): {len(cases['swe_qwen'])}")
    for c in cases["swe_qwen"][:5]:
        print(f"    - {c['instance_id']}")
    print(f"\n  HotpotQA-Qwen flips (placebo=0, gate=1, shuf=0): {len(cases['hotpot_qwen'])}")
    for c in cases["hotpot_qwen"][:5]:
        print(f"    - qid={c['qid'][:8]}  gate_pred={c['gate_hint_pred']!r:<40s}  shuf_pred={c['shuffled_pred']!r}")
        print(f"        Q: {c['question'][:100]}")
        print(f"        gold: {c['gold_answer'][:50]!r}  gate_triggered={c['gate_hint_triggered']}  shuf_triggered={c['shuffled_triggered']}")

    with open(PAIRED_DIR / "case_studies.json", "w") as fh:
        json.dump(cases, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {PAIRED_DIR}/case_studies.json")

    # ----------------------- C. TRIGGER FIRE RATE vs SPECIFICITY -----------
    print()
    print("=" * 60)
    print("C. TRIGGER-FIRE RATE vs SPECIFICITY")
    print("=" * 60)

    # We need a fire rate per experiment. SWE Qwen full: 95%; SWE DS full: 95%;
    # HotpotQA gate_hint fired 14/60 = 23.3%; HotpotQA shuffled 33/60 = 55%
    fig_rows: List[Dict] = []

    # Fire rates we computed earlier (or take from per-arm n_triggers when JSONL exists)
    # Hotpot Qwen: count triggers_fired from gate_hint arm
    hp_gate_fire = sum(1 for q, d in hp_qid_to_pred.items() if d.get("gate_hint", {}).get("triggers_fired", 0) > 0)
    hp_gate_rate = hp_gate_fire / len(hp_qid_to_pred)
    fig_rows.append({"experiment": "HotpotQA-Qwen", "fire_rate": hp_gate_rate, "spec_delta": pooled["HotpotQA-Qwen36B (gate v shuf)"]["delta"], "spec_lo": pooled["HotpotQA-Qwen36B (gate v shuf)"]["boot_lo"], "spec_hi": pooled["HotpotQA-Qwen36B (gate v shuf)"]["boot_hi"]})

    # SWE-Qwen / SWE-DS: count triggers from raw rollout files
    def swe_fire_rate(rollout_glob: str) -> float:
        import glob
        n_tot = n_fire = 0
        for fp in glob.glob(rollout_glob):
            for line in open(fp):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("arm") != "tracegraph":
                    continue
                n_tot += 1
                if int(r.get("n_triggers", 0) or 0) > 0:
                    n_fire += 1
        return n_fire / max(n_tot, 1)

    swe_q_rate = swe_fire_rate("results/cxcmu/intervention/_archive_20260520_loose_jsonl/pilot_swe_qwen36_full/pilot_swe_qwen36_full_s*.jsonl")
    fig_rows.append({"experiment": "SWE-Qwen", "fire_rate": swe_q_rate, "spec_delta": pooled["SWE-Qwen36B"]["delta"], "spec_lo": pooled["SWE-Qwen36B"]["boot_lo"], "spec_hi": pooled["SWE-Qwen36B"]["boot_hi"]})

    swe_ds_rate = swe_fire_rate("results/cxcmu/intervention/_archive_20260520_loose_jsonl/pilot_swe_postfix/pilot_swe_postfix_s*.jsonl") or swe_fire_rate("results/cxcmu/intervention/_archive_20260520_loose_jsonl/pilot_swe_ext/pilot_swe_ext_s*.jsonl")
    fig_rows.append({"experiment": "SWE-DeepSeek", "fire_rate": swe_ds_rate, "spec_delta": pooled["SWE-DeepSeekV4"]["delta"], "spec_lo": pooled["SWE-DeepSeekV4"]["boot_lo"], "spec_hi": pooled["SWE-DeepSeekV4"]["boot_hi"]})

    with open(PAIRED_DIR / "figure_fire_vs_spec.csv", "w") as fh:
        fh.write("experiment\tfire_rate\tspec_delta\tspec_lo\tspec_hi\n")
        for r in fig_rows:
            fh.write(f"{r['experiment']}\t{r['fire_rate']:.4f}\t{r['spec_delta']:+.4f}\t{r['spec_lo']:+.4f}\t{r['spec_hi']:+.4f}\n")
            print(f"  {r['experiment']:<20s}  fire={r['fire_rate']:.2f}  spec={r['spec_delta']:+.3f}  CI=[{r['spec_lo']:+.3f}, {r['spec_hi']:+.3f}]")

    # PDF figure
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5.5, 3.5))
        xs = [r["fire_rate"] for r in fig_rows]
        ys = [r["spec_delta"] for r in fig_rows]
        lo = [r["spec_lo"] for r in fig_rows]
        hi = [r["spec_hi"] for r in fig_rows]
        labels = [r["experiment"] for r in fig_rows]
        err_lo = [y - l for y, l in zip(ys, lo)]
        err_hi = [h - y for y, h in zip(ys, hi)]
        plt.errorbar(xs, ys, yerr=[err_lo, err_hi], fmt="o", color="C0",
                     ecolor="gray", capsize=3)
        for x, y, lbl in zip(xs, ys, labels):
            plt.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
        plt.axhline(0, color="k", lw=0.5, ls="--")
        plt.xlabel("Trigger fire rate (fraction of rollouts where intervention activated)")
        plt.ylabel(r"Paired $\Delta$ resolved (tracegraph $-$ shuffled)")
        plt.title("Specificity scales with trigger coverage (95% bootstrap CI)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "intervention_fire_vs_spec.pdf")
        plt.savefig(FIG_DIR / "intervention_fire_vs_spec.png", dpi=150)
        print(f"\nwrote figs/intervention_fire_vs_spec.{{pdf,png}}")
    except ImportError:
        print("matplotlib not available; skipping figure")


if __name__ == "__main__":
    main()
