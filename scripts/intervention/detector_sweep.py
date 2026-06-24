#!/usr/bin/env python3
"""SWE intervention trigger — detector-only sweep.

Replays parameterized fire predicates against the *already-saved* per_step
logs of existing SWE intervention rollouts (no GPU). For each cell in the
predicate grid, reports per-arm:

  * fire_rate_rollout   — fraction of rollouts with >=1 fire (target 0.20-0.50)
  * fire_rate_step      — per-step fire rate
  * hard_at_step_prec   — P(obs_has_hard_error | fire) (target >= 0.50)
  * lookback_prec       — P(any hard_error in last k | fire)
  * is_trap_prec        — P(is_trap | fire)
  * intent_dist         — intent breakdown at fire steps
  * mean_step           — mean fire-step index
  * n_rollouts, n_fires

And per cell, the parity between tracegraph-source and shuffled-source arms:

  * parity_ratio = fire_rate_rollout(tracegraph) / fire_rate_rollout(shuffled)
                   (target near 1.0 so the matched-control arm isn't artificially harder)

Inputs (auto-detected; no CLI args needed for the default run):
  results/cxcmu/intervention/swe_eval_expand60_swe_qwen36_factorial60/pilot_swe_with_resolved.jsonl
  results/cxcmu/intervention/_archive_20260520_loose_jsonl/pilot_swe_qwen36_full/pilot_swe_qwen36_full_s*.jsonl

Outputs:
  results/cxcmu/intervention/swe_trigger_detector_sweep.json
  results/cxcmu/intervention/swe_trigger_detector_sweep.tsv

Usage:
  python scripts/129_swe_trigger_detector_sweep.py
"""
from __future__ import annotations

import glob
import itertools
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results/cxcmu/intervention"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Rollout loading


@dataclass
class Rollout:
    instance_id: str
    arm: str                      # arm as recorded: placebo/tracegraph/shuffled/tg_*/shuf_*
    arm_source: str               # normalized "trigger source" axis: tracegraph|shuffled|placebo
    per_step: List[Dict]
    source_file: str
    n_obs_steps: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_obs_steps = sum(
            1 for s in self.per_step if s.get("action") != "submit"
        )


def _arm_source(arm: str) -> str:
    a = (arm or "").lower()
    if a.startswith("tg") or a == "tracegraph" or a == "gate_hint":
        return "tracegraph"
    if a.startswith("shuf") or a == "shuffled":
        return "shuffled"
    if a in ("placebo", "generic_note"):
        return "placebo"
    return a


def load_rollouts() -> List[Rollout]:
    files: List[Path] = []
    files.append(
        ROOT
        / "results/cxcmu/intervention/swe_eval_expand60_swe_qwen36_factorial60"
        / "pilot_swe_with_resolved.jsonl"
    )
    files.extend(
        sorted(
            Path(p) for p in glob.glob(str(
                ROOT / "results/cxcmu/intervention/_archive_20260520_loose_jsonl"
                / "pilot_swe_qwen36_full" / "pilot_swe_qwen36_full_s*.jsonl"
            ))
        )
    )
    rollouts: List[Rollout] = []
    for fp in files:
        if not fp.exists():
            continue
        with open(fp) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                per_step = r.get("per_step") or []
                if not per_step:
                    continue
                rollouts.append(Rollout(
                    instance_id=str(r.get("instance_id")),
                    arm=str(r.get("arm") or ""),
                    arm_source=_arm_source(str(r.get("arm") or "")),
                    per_step=per_step,
                    source_file=fp.name,
                ))
    return rollouts


# ----------------------------------------------------------------------
# Predicate replay


@dataclass(frozen=True)
class PredicateConfig:
    sim_thr: float
    lookback_k: int                # 0 = no lookback requirement
    require_hard_at_step: bool
    allowed_intents: Optional[Tuple[str, ...]]
    warmup: int = 2
    cooldown: int = 4
    max_triggers: int = 1

    def as_dict(self) -> Dict:
        return {
            "sim_thr": self.sim_thr,
            "lookback_k": self.lookback_k,
            "require_hard_at_step": self.require_hard_at_step,
            "allowed_intents": list(self.allowed_intents) if self.allowed_intents else None,
            "warmup": self.warmup,
            "cooldown": self.cooldown,
            "max_triggers": self.max_triggers,
        }


def _fire_at(per_step: Sequence[Dict], idx: int, cfg: PredicateConfig) -> bool:
    s = per_step[idx]
    if float(s.get("sim_detect") or 0.0) < cfg.sim_thr:
        return False
    if cfg.require_hard_at_step and not s.get("obs_has_hard_error"):
        return False
    if cfg.lookback_k > 0:
        lo = max(0, idx - cfg.lookback_k + 1)
        recent = per_step[lo: idx + 1]
        if not any(t.get("obs_has_hard_error") for t in recent):
            return False
    if cfg.allowed_intents:
        if s.get("intent") not in cfg.allowed_intents:
            return False
    return True


def replay(per_step: Sequence[Dict], cfg: PredicateConfig) -> List[int]:
    """Return list of step indices where the predicate would have fired."""
    fired: List[int] = []
    cd_left = 0
    for idx, step in enumerate(per_step):
        if step.get("action") == "submit":
            break
        if idx < cfg.warmup:
            cd_left = max(0, cd_left - 1)
            continue
        if cd_left > 0 or len(fired) >= cfg.max_triggers:
            cd_left = max(0, cd_left - 1)
            continue
        if _fire_at(per_step, idx, cfg):
            fired.append(idx)
            cd_left = cfg.cooldown
        else:
            cd_left = max(0, cd_left - 1)
    return fired


# ----------------------------------------------------------------------
# Aggregation


def _kpi_for_arm(rollouts: Sequence[Rollout], cfg: PredicateConfig) -> Dict:
    n_rollouts = len(rollouts)
    if n_rollouts == 0:
        return {"n_rollouts": 0}
    n_with_fire = 0
    n_fires_total = 0
    n_obs_steps = 0
    hard_at = 0
    lookback_hit = 0
    is_trap_hit = 0
    intent_counter: Counter = Counter()
    cmd_counter: Counter = Counter()
    sum_step = 0
    for r in rollouts:
        n_obs_steps += r.n_obs_steps
        fired = replay(r.per_step, cfg)
        if fired:
            n_with_fire += 1
        n_fires_total += len(fired)
        for idx in fired:
            s = r.per_step[idx]
            hard_at += int(bool(s.get("obs_has_hard_error")))
            lo = max(0, idx - max(cfg.lookback_k, 1) + 1)
            recent = r.per_step[lo: idx + 1]
            lookback_hit += int(any(t.get("obs_has_hard_error") for t in recent))
            is_trap_hit += int(bool(s.get("is_trap")))
            intent_counter[str(s.get("intent") or "other")] += 1
            cmd_counter[str(s.get("cmd_class") or "other")] += 1
            sum_step += int(s.get("step") or idx)

    def _frac(num: int, den: int) -> Optional[float]:
        return None if den <= 0 else round(num / den, 4)

    return {
        "n_rollouts": n_rollouts,
        "n_with_fire": n_with_fire,
        "n_fires_total": n_fires_total,
        "n_obs_steps": n_obs_steps,
        "fire_rate_rollout": _frac(n_with_fire, n_rollouts),
        "fire_rate_step": _frac(n_fires_total, n_obs_steps),
        "hard_at_step_prec": _frac(hard_at, n_fires_total),
        "lookback_prec": _frac(lookback_hit, n_fires_total),
        "is_trap_prec": _frac(is_trap_hit, n_fires_total),
        "mean_step": (round(sum_step / n_fires_total, 2) if n_fires_total else None),
        "intent_dist": dict(intent_counter),
        "cmd_dist": dict(cmd_counter),
    }


def evaluate_cell(rollouts_by_source: Dict[str, List[Rollout]],
                  cfg: PredicateConfig) -> Dict:
    out: Dict = {"config": cfg.as_dict(), "per_source": {}}
    for source, rs in rollouts_by_source.items():
        out["per_source"][source] = _kpi_for_arm(rs, cfg)
    # Parity: tracegraph vs shuffled
    tg = out["per_source"].get("tracegraph", {})
    sh = out["per_source"].get("shuffled", {})
    if tg.get("fire_rate_rollout") is not None and sh.get("fire_rate_rollout"):
        out["parity_ratio_rollout"] = round(
            tg["fire_rate_rollout"] / sh["fire_rate_rollout"], 3
        )
    else:
        out["parity_ratio_rollout"] = None
    return out


# ----------------------------------------------------------------------
# Selection rule


def _passes(cell: Dict, *,
            fire_lo: float = 0.20, fire_hi: float = 0.50,
            hard_prec_min: float = 0.50,
            parity_lo: float = 0.5, parity_hi: float = 2.0) -> bool:
    tg = cell["per_source"].get("tracegraph") or {}
    fr = tg.get("fire_rate_rollout")
    hp = tg.get("hard_at_step_prec")
    if fr is None or hp is None:
        return False
    if not (fire_lo <= fr <= fire_hi):
        return False
    if hp < hard_prec_min:
        return False
    par = cell.get("parity_ratio_rollout")
    if par is None or not (parity_lo <= par <= parity_hi):
        return False
    return True


# ----------------------------------------------------------------------
# Main


def main() -> None:
    rollouts = load_rollouts()
    print(f"loaded {len(rollouts)} rollouts with per_step logs")
    by_source: Dict[str, List[Rollout]] = {}
    for r in rollouts:
        by_source.setdefault(r.arm_source, []).append(r)
    for k, v in sorted(by_source.items()):
        n_steps = sum(x.n_obs_steps for x in v)
        print(f"  source={k:<12s}  n_rollouts={len(v):4d}  n_obs_steps={n_steps}")

    sim_thrs = [0.25, 0.30, 0.35, 0.40, 0.45]
    lookbacks = [0, 1, 2, 3]
    req_hard = [False, True]
    intent_sets: List[Optional[Tuple[str, ...]]] = [
        None,
        ("edit", "submit"),
        ("edit", "submit", "test"),
    ]

    cells: List[Dict] = []
    for sim, k, rh, ais in itertools.product(sim_thrs, lookbacks, req_hard, intent_sets):
        cfg = PredicateConfig(
            sim_thr=sim,
            lookback_k=k,
            require_hard_at_step=rh,
            allowed_intents=ais,
        )
        cells.append(evaluate_cell(by_source, cfg))
    print(f"evaluated {len(cells)} predicate cells\n")

    # rank + print top candidates by the selection rule
    passing = [c for c in cells if _passes(c)]
    passing.sort(
        key=lambda c: (
            -(c["per_source"]["tracegraph"]["hard_at_step_prec"] or 0),
            -(c["per_source"]["tracegraph"]["fire_rate_rollout"] or 0),
        )
    )

    print("=" * 78)
    print("Top predicates passing selection rule (fire 0.20-0.50, hard_prec>=0.5,")
    print("                                       parity in [0.5, 2.0]):")
    print("=" * 78)
    if not passing:
        print("  NONE — no predicate satisfies all three KPIs simultaneously.")
        print("  Falling back to top 5 by hard_at_step_prec at any fire rate.")
        passing = sorted(
            cells,
            key=lambda c: (
                -((c["per_source"].get("tracegraph") or {}).get("hard_at_step_prec") or 0),
                -((c["per_source"].get("tracegraph") or {}).get("fire_rate_rollout") or 0),
            ),
        )[:5]

    hdr = (
        f"  {'sim':>5}  {'k':>2}  {'hard?':>5}  {'intents':<22s}  "
        f"{'fire_tg':>8}  {'hard_p':>7}  {'lkbck_p':>8}  {'trap_p':>7}  "
        f"{'parity':>7}  {'mean_step':>10}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c in passing[:20]:
        cf = c["config"]
        tg = c["per_source"]["tracegraph"]
        ai = ",".join(cf["allowed_intents"]) if cf["allowed_intents"] else "*"
        print(
            f"  {cf['sim_thr']:>5.2f}  {cf['lookback_k']:>2d}  "
            f"{str(cf['require_hard_at_step'])[:5]:>5s}  {ai:<22s}  "
            f"{tg['fire_rate_rollout'] or 0:>8.3f}  "
            f"{tg['hard_at_step_prec'] or 0:>7.3f}  "
            f"{tg['lookback_prec'] or 0:>8.3f}  "
            f"{tg['is_trap_prec'] or 0:>7.3f}  "
            f"{c.get('parity_ratio_rollout') or 0:>7.3f}  "
            f"{str(tg['mean_step']):>10s}"
        )

    # Print baseline (current expand60 config) for reference
    baseline = PredicateConfig(
        sim_thr=0.30,
        lookback_k=3,
        require_hard_at_step=False,
        allowed_intents=None,
    )
    print("\nBaseline (current expand60 config):")
    base_cell = evaluate_cell(by_source, baseline)
    tg = base_cell["per_source"].get("tracegraph", {})
    sh = base_cell["per_source"].get("shuffled", {})
    print(f"  TRACEGRAPH: fire={tg.get('fire_rate_rollout')}  "
          f"hard_p={tg.get('hard_at_step_prec')}  trap_p={tg.get('is_trap_prec')}  "
          f"mean_step={tg.get('mean_step')}  intents={tg.get('intent_dist')}")
    print(f"  SHUFFLED  : fire={sh.get('fire_rate_rollout')}  "
          f"hard_p={sh.get('hard_at_step_prec')}  trap_p={sh.get('is_trap_prec')}  "
          f"mean_step={sh.get('mean_step')}  intents={sh.get('intent_dist')}")

    # Write outputs
    out_json = OUT_DIR / "swe_trigger_detector_sweep.json"
    with open(out_json, "w") as fh:
        json.dump({
            "n_rollouts_total": len(rollouts),
            "n_rollouts_by_source": {k: len(v) for k, v in by_source.items()},
            "baseline_cell": base_cell,
            "all_cells": cells,
            "passing_cells": passing,
        }, fh, indent=2)
    print(f"\nwrote {out_json}")

    out_tsv = OUT_DIR / "swe_trigger_detector_sweep.tsv"
    with open(out_tsv, "w") as fh:
        fh.write(
            "sim_thr\tlookback_k\trequire_hard_at_step\tallowed_intents\t"
            "fire_tg\tfire_sh\thard_prec_tg\ttrap_prec_tg\tparity\tmean_step_tg\n"
        )
        for c in cells:
            cf = c["config"]
            tg = c["per_source"].get("tracegraph", {})
            sh = c["per_source"].get("shuffled", {})
            ai = ",".join(cf["allowed_intents"]) if cf["allowed_intents"] else "*"
            fh.write(
                f"{cf['sim_thr']:.2f}\t{cf['lookback_k']}\t"
                f"{cf['require_hard_at_step']}\t{ai}\t"
                f"{tg.get('fire_rate_rollout')}\t{sh.get('fire_rate_rollout')}\t"
                f"{tg.get('hard_at_step_prec')}\t{tg.get('is_trap_prec')}\t"
                f"{c.get('parity_ratio_rollout')}\t{tg.get('mean_step')}\n"
            )
    print(f"wrote {out_tsv}")


if __name__ == "__main__":
    main()
