"""Run the pre-registered v2 protocol (docs/V2_PROTOCOL.md, with Amendment 1).

Resumable: a step whose output file exists is skipped. All selection uses
validation and training rows; the second fresh holdout is evaluated once.
v1 is only replaced (after being kept as ``mindcheck_v1``) if gates A, B, C, E
and F pass.

Prerequisites: ``data/processed/allpos`` (train and val splits) and
``experiments/v2_B0_bias.json`` (v1 biases refitted on unfiltered validation).

Usage::

    .venv\\Scripts\\python.exe scripts/run_v2.py
"""

from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

CLASS_NAMES = ["K_STORM", "Q_EXP", "OUTPOST", "ROOK_FILE", "CHAIN_UNDERMINE", "NO_PLAN"]
CANDIDATES = ["F_allpos", "G_allpos_uniform", "H_allpos_strongreg"]
SEEDS = [42, 123, 999]
EPOCHS, PATIENCE = 16, 3
TIE_MARGIN = 0.005
MIN_GAIN = 0.010  # gates A and F
MAX_F1_GAP, MAX_ACC_GAP = 0.03, 0.02  # gate B
MAX_SEED_STD = 0.005  # gate C
MAX_CHAIN_GAP = 0.10  # gate D (target)
MAX_HOLDOUT_DIFF = 0.020  # gate E
PER_GAME = 25
V1 = {"variant": "D_random_balance", "seed": 999, "tag": "_loopr2"}

DATA = _REPO_ROOT / "data" / "processed" / "allpos"
OUT = _REPO_ROOT / "experiments" / "v2"
LOGS = _REPO_ROOT / "experiments" / "logs"
CKPT = _REPO_ROOT / "models" / "checkpoints"
B0_BIAS = _REPO_ROOT / "experiments" / "v2_B0_bias.json"
HOLDOUT2 = {
    "filtered": _REPO_ROOT / "data" / "processed" / "holdout2_filtered",
    "full": _REPO_ROOT / "data" / "processed" / "holdout2_full",
}
USED_HOLDOUT = _REPO_ROOT / "data" / "processed" / "holdout_full"


# ------------------------------ pure rules ------------------------------- #
def fold_mean(bias_result: Dict[str, Any]) -> float:
    folds = bias_result["cross_fold_check"]
    return float(np.mean([f["heldout_f1_after"] for f in folds.values()]))


def choose_winner(scores: Dict[str, float], chain_gaps: Dict[str, float]) -> str:
    """Highest score; candidates within ``TIE_MARGIN`` go to the smaller CHAIN gap."""
    best = max(scores.values())
    tied = [v for v, s in scores.items() if best - s <= TIE_MARGIN]
    return min(tied, key=lambda v: (chain_gaps[v], -scores[v]))


def validation_gates(seeds: Dict[int, Dict[str, float]], b0_score: float) -> Dict[str, Any]:
    """Gates A-D. ``seeds[s]`` needs score, raw_f1, f1_gap, acc_gap, chain_gap."""
    mean_score = float(np.mean([r["score"] for r in seeds.values()]))
    std = float(np.std([r["raw_f1"] for r in seeds.values()], ddof=1))
    bad_gap = [
        s for s, r in seeds.items() if r["f1_gap"] > MAX_F1_GAP or r["acc_gap"] > MAX_ACC_GAP
    ]
    bad_chain = [s for s, r in seeds.items() if r["chain_gap"] > MAX_CHAIN_GAP]
    return {
        "A": {"passed": mean_score >= b0_score + MIN_GAIN, "mean": mean_score, "b0": b0_score},
        "B": {"passed": not bad_gap, "failed_seeds": bad_gap},
        "C": {"passed": std <= MAX_SEED_STD, "std": std},
        "D": {"passed": not bad_chain, "failed_seeds": bad_chain, "blocks_promotion": False},
    }


def holdout_gates(
    seeds: Dict[int, Dict[str, float]], b0_biased: float, chosen: int
) -> Dict[str, Any]:
    """Gates E and F. ``seeds[s]`` needs val_raw_f1, holdout_raw_f1, holdout_biased_f1."""
    diffs = {s: abs(r["holdout_raw_f1"] - r["val_raw_f1"]) for s, r in seeds.items()}
    mean = float(np.mean([r["holdout_biased_f1"] for r in seeds.values()]))
    return {
        "E": {"passed": max(diffs.values()) <= MAX_HOLDOUT_DIFF, "abs_diff": diffs},
        "F": {
            "passed": mean >= b0_biased + MIN_GAIN
            and seeds[chosen]["holdout_biased_f1"] > b0_biased,
            "mean_biased": mean,
            "b0_biased": b0_biased,
        },
    }


def promotable(gates: Dict[str, Any]) -> bool:
    return all(gates[name]["passed"] for name in ("A", "B", "C", "E", "F"))


def per_game_sample(game_ids: Sequence[str], k: int, seed: int = 42) -> np.ndarray:
    """Row indices of a seeded uniform sample of up to ``k`` rows per game."""
    rows: Dict[str, List[int]] = defaultdict(list)
    for i, gid in enumerate(game_ids):
        rows[gid].append(i)
    rng = np.random.RandomState(seed)
    picked: List[int] = []
    for gid in sorted(rows):
        idx = rows[gid]
        picked.extend(rng.choice(idx, min(k, len(idx)), replace=False).tolist())
    return np.array(sorted(picked), dtype=np.int64)


def scores(y: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    labels = list(range(len(CLASS_NAMES)))
    per = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    return {
        "n": int(len(y)),
        "macro_f1": float(np.mean(per)),
        "accuracy": float(accuracy_score(y, pred)),
        "per_class_f1": {n: float(v) for n, v in zip(CLASS_NAMES, per)},
    }


# ------------------------------- steps ----------------------------------- #
def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _run(name: str, script: str, args: List[str], done: Path) -> None:
    if done.exists():
        print(f"skip {name} ({done.name} exists)", flush=True)
        return
    print(f"running {name}", flush=True)
    start = time.time()
    with (LOGS / f"v2_{name}.log").open("w", encoding="utf-8") as log:
        code = subprocess.call(
            [sys.executable, "-u", str(_REPO_ROOT / "scripts" / script), *args],
            cwd=_REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if code != 0 or not done.exists():
        raise SystemExit(f"FAILED {name} (exit {code}); see experiments/logs/v2_{name}.log")
    print(f"finished {name} in {(time.time() - start) / 60:.1f} min", flush=True)


def run_seed(variant: str, seed: int) -> Dict[str, Any]:
    """Train, fit biases and measure the gap for one variant/seed."""
    stem = f"{variant}_s{seed}"
    common = ["--variant", variant, "--tag", "_v2", "--data-dir", str(DATA)]
    train_out, bias_out, gap_out = (OUT / f"{k}_{stem}.json" for k in ("train", "bias", "gap"))
    _run(
        f"train_{stem}",
        "final_training.py",
        ["--phase", "train", *common, "--seeds", str(seed), "--epochs", str(EPOCHS)]
        + ["--patience", str(PATIENCE), "--threads", "8", "--out", str(train_out)],
        train_out,
    )
    _run(
        f"bias_{stem}",
        "tune_logit_bias.py",
        [*common, "--seed", str(seed), "--out", str(bias_out)],
        bias_out,
    )
    _run(
        f"gap_{stem}",
        "measure_gap.py",
        [*common, "--seed", str(seed), "--threads", "8", "--out", str(gap_out)],
        gap_out,
    )
    bias, gap, train = _read(bias_out), _read(gap_out), _read(train_out)[-1]
    return {
        "score": fold_mean(bias),
        "raw_f1": bias["val_macro_f1_before"],
        "biased_f1_full_fit": bias["val_macro_f1_after_full_fit"],
        "f1_gap": gap["macro_f1_gap_natural"],
        "acc_gap": gap["accuracy_gap_natural"],
        "chain_gap": gap["recall_gap_trained_on_minus_val"]["CHAIN_UNDERMINE"],
        "recall_gaps": gap["recall_gap_trained_on_minus_val"],
        "per_class_val_f1_biased": bias["per_class_f1_after"],
        "bias": bias["bias"],
        "best_epoch": train["best_epoch"],
        "epochs_run": train["epochs_run"],
    }


def _holdout_logits(variant: str, seed: int, tag: str) -> np.ndarray:
    """Raw logits of one checkpoint on every window of the second holdout."""
    import final_training as ft  # noqa: PLC0415 - needs torch only here
    import torch  # noqa: PLC0415

    from src.models.logger import load_checkpoint  # noqa: PLC0415

    cfg = ft.VARIANTS[variant]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ft._build_model(cfg)
    load_checkpoint(ft._ckpt_path(variant, seed, tag), model)
    model.to(device).eval()
    x = np.load(HOLDOUT2["full"] / "test_tensors.npy", mmap_mode="r")
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 2048):
            batch = np.array(x[i : i + 2048, -cfg["keep"] :], np.float32)
            out.append(model(torch.from_numpy(batch).to(device)).cpu().numpy())
    return np.concatenate(out)


def evaluate_holdout(winner: str, seeds: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Score every v2 seed and B0 once on the second holdout (guarded by a marker)."""
    result_path, marker = OUT / "holdout2_results.json", OUT / "HOLDOUT2_USED.marker"
    if result_path.exists():
        return _read(result_path)
    if marker.exists():
        raise SystemExit(f"{marker} exists but no result: inspect before touching the holdout.")
    marker.write_text("second holdout evaluation started\n", encoding="utf-8")
    import build_scaled_dataset as bsd  # noqa: PLC0415

    y = np.load(HOLDOUT2["full"] / "test_labels.npy")
    records = bsd._read_csv(HOLDOUT2["full"] / "test_records.csv")
    sample = per_game_sample([r["game_id"] for r in records], PER_GAME)
    # Both holdout directories use the same window ids for the same position.
    kept = {r["window_id"] for r in bsd._read_csv(HOLDOUT2["filtered"] / "test_records.csv")}
    filtered = np.array([i for i, r in enumerate(records) if r["window_id"] in kept])
    if len(filtered) != len(kept):
        raise SystemExit("filtered holdout windows do not all appear in the full holdout.")
    views = {"per_game_25": sample, "all_positions": np.arange(len(y)), "filtered": filtered}
    models = {"B0": (V1["variant"], V1["seed"], V1["tag"], _read(B0_BIAS)["bias"])}
    models.update({str(s): (winner, s, "_v2", r["bias"]) for s, r in seeds.items()})
    table: Dict[str, Any] = {}
    for name, (variant, seed, tag, bias) in models.items():
        logits = _holdout_logits(variant, seed, tag)
        vec = np.array([bias[c] for c in CLASS_NAMES], dtype=np.float32)
        table[name] = {
            view: {
                "raw": scores(y[idx], logits[idx].argmax(1)),
                "biased": scores(y[idx], (logits[idx] + vec).argmax(1)),
            }
            for view, idx in views.items()
        }
    # v1 exactly as shipped (its original filtered-validation biases), for reference.
    shipped = _read(CKPT / "mindcheck_final.json")["logit_bias"]
    logits = _holdout_logits(V1["variant"], V1["seed"], V1["tag"])
    vec = np.array([shipped[c] for c in CLASS_NAMES], dtype=np.float32)
    table["v1_as_shipped"] = {
        view: {"biased": scores(y[idx], (logits[idx] + vec).argmax(1))}
        for view, idx in views.items()
    }
    _write(result_path, table)
    return table


def export(winner: str, seed: int, run: Dict[str, Any], sidecar: Dict[str, Any]) -> None:
    """Keep v1 as ``mindcheck_v1``, then write v2 as ``mindcheck_final``."""
    import final_training as ft  # noqa: PLC0415

    from src.models.export import export_to_onnx, export_to_torchscript, verify_onnx_matches
    from src.models.inference import DeployableModel, load_deployable, save_deployable
    from src.models.logger import load_checkpoint

    for suffix in (".pt", ".onnx", ".ts.pt", ".json"):
        old, keep = CKPT / f"mindcheck_final{suffix}", CKPT / f"mindcheck_v1{suffix}"
        if old.exists() and not keep.exists():
            shutil.copy2(old, keep)
    if not all((CKPT / f"mindcheck_v1{s}").exists() for s in (".pt", ".onnx", ".ts.pt", ".json")):
        raise SystemExit("v1 copies are incomplete; refusing to overwrite mindcheck_final.")
    cfg = ft.VARIANTS[winner]
    bias = [run["bias"][c] for c in CLASS_NAMES]
    model = DeployableModel(keep=cfg["keep"], bias=bias, dropout=cfg["dropout"])
    load_checkpoint(ft._ckpt_path(winner, seed, "_v2"), model.backbone)
    model.eval()
    # Write and verify everything under staging names first, so a failure can never
    # leave a mixed v1/v2 set behind under the final names.
    stage = {s: CKPT / f"mindcheck_v2_staging{s}" for s in (".pt", ".onnx", ".ts.pt", ".json")}
    pt = save_deployable(model, stage[".pt"], {"variant": winner, "seed": seed})
    export_to_onnx(model, stage[".onnx"])
    if not verify_onnx_matches(load_deployable(pt), stage[".onnx"]):
        raise SystemExit("ONNX export does not match PyTorch output; v1 left in place.")
    export_to_torchscript(model, stage[".ts.pt"])
    _write(stage[".json"], sidecar)
    for suffix, path in stage.items():
        path.replace(CKPT / f"mindcheck_final{suffix}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run the pre-registered v2 protocol.")
    ap.parse_args(argv)
    for needed in (DATA / "meta_train.json", DATA / "meta_val.json", B0_BIAS):
        if not needed.exists():
            raise SystemExit(f"missing prerequisite: {needed}")
    for split in ("train", "val"):
        checks = _read(DATA / f"meta_{split}.json")["checks"]
        if not all(checks.values()):
            raise SystemExit(f"{split} integrity checks failed: {checks}")
    LOGS.mkdir(parents=True, exist_ok=True)
    if hasattr(ctypes, "windll"):  # keep Windows awake while running only
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)

    b0_fit = _read(B0_BIAS)
    if (b0_fit.get("variant"), b0_fit.get("seed"), b0_fit.get("tag")) != (
        V1["variant"],
        V1["seed"],
        V1["tag"],
    ):
        raise SystemExit(f"{B0_BIAS} is not for the v1 checkpoint {V1}.")
    b0 = fold_mean(b0_fit)
    round1 = {v: run_seed(v, 42) for v in CANDIDATES}
    winner = choose_winner(
        {v: r["score"] for v, r in round1.items()},
        {v: r["chain_gap"] for v, r in round1.items()},
    )
    _write(OUT / "round1.json", {"b0_score": b0, "candidates": round1, "winner": winner})
    print(f"decision round1: winner {winner}", flush=True)

    seeds = {s: run_seed(winner, s) for s in SEEDS}  # seed 42 reuses the round-1 run
    gates = validation_gates(seeds, b0)
    chosen = max(seeds, key=lambda s: seeds[s]["score"])
    _write(
        OUT / "round2.json",
        {"winner": winner, "chosen_seed": chosen, "seeds": seeds, "gates": gates},
    )
    print(f"decision round2: gates { {k: g['passed'] for k, g in gates.items()} }", flush=True)

    _run(
        "holdout2_build",
        "build_fresh_holdout.py",
        ["--out-filtered", str(HOLDOUT2["filtered"]), "--out-full", str(HOLDOUT2["full"])]
        + ["--after-holdout", str(USED_HOLDOUT), "--max-games", "6000"],
        HOLDOUT2["full"] / "meta.json",
    )
    table = evaluate_holdout(winner, seeds)
    gates.update(
        holdout_gates(
            {
                s: {
                    "val_raw_f1": r["raw_f1"],
                    "holdout_raw_f1": table[str(s)]["per_game_25"]["raw"]["macro_f1"],
                    "holdout_biased_f1": table[str(s)]["per_game_25"]["biased"]["macro_f1"],
                }
                for s, r in seeds.items()
            },
            table["B0"]["per_game_25"]["biased"]["macro_f1"],
            chosen,
        )
    )
    promoted = promotable(gates)
    _write(
        OUT / "final.json",
        {"winner": winner, "chosen_seed": chosen, "gates": gates, "promoted": promoted},
    )
    if promoted:
        export(
            winner,
            chosen,
            seeds[chosen],
            {
                "version": "v2",
                "variant": winner,
                "seed": chosen,
                "input": "(B,112,8,8) float32; the newest 56 channels are sliced internally",
                "keep_channels": 56,
                "class_names": CLASS_NAMES,
                "logit_bias": seeds[chosen]["bias"],
                "labels": "rule-engine weak labels; not validated against human annotation",
                "validation_unfiltered": seeds[chosen],
                "fresh_holdout2": table[str(chosen)],
                "gates": gates,
                "protocol": "docs/V2_PROTOCOL.md",
            },
        )
    print(
        f"COMPLETE. gates { {k: g['passed'] for k, g in gates.items()} } promoted={promoted}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
