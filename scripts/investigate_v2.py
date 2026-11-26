"""Diagnostic evaluation of the v1 and v2 models. Trains and tunes nothing for deployment.

Steps (each cached, so the script is resumable):

1. ``rule``: re-run the symbolic rule engine on validation and second-holdout games.
   It gives a past-only baseline prediction and, for diagnosis, the look-ahead
   aggregate scores behind each weak label (and checks the labels reproduce).
2. ``logits``: raw logits of every model on every population.
3. ``metrics``: all tables, written to ``experiments/v2/accuracy_investigation.json``.

The second holdout was already used for the v2 decision. Re-reading it here is
diagnostic only: nothing is selected or fitted on it. Biases fitted in this
script use validation data only.

Usage::

    .venv\\Scripts\\python.exe scripts/investigate_v2.py --work-dir <cache dir>
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_scaled_dataset as bsd  # noqa: E402
import numpy as np  # noqa: E402

NAMES = ["K_STORM", "Q_EXP", "OUTPOST", "ROOK_FILE", "CHAIN_UNDERMINE", "NO_PLAN"]
K, CHAIN, NO_PLAN = 6, 4, 5
PROCESSED = _REPO_ROOT / "data" / "processed"
EXP = _REPO_ROOT / "experiments"
OUT = EXP / "v2" / "accuracy_investigation.json"
V1_SEEDS = V2_SEEDS = (42, 123, 999)
V1_SAVED, V2_SAVED = 999, 42
MIN_CONFIDENCE = 0.55  # the existing dataset filter threshold


# ------------------------------- metrics --------------------------------- #
def confusion(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    cells = np.asarray(y, np.int64) * K + np.asarray(p, np.int64)
    return np.bincount(cells, minlength=K * K).reshape(K, K)


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros(len(a)), where=b > 0)


def from_confusion(cm: np.ndarray, classes: Sequence[int] = range(K)) -> Dict[str, Any]:
    """All metrics from a confusion matrix; averages run over ``classes`` present."""
    cm = np.asarray(cm, float)
    tp, support, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    precision, recall = _safe_div(tp, predicted), _safe_div(tp, support)
    f1 = _safe_div(2 * tp, support + predicted)
    used = [c for c in classes if support[c] > 0]
    n = cm.sum()
    return {
        "n": int(n),
        "accuracy": float(tp.sum() / n) if n else 0.0,
        "balanced_accuracy": float(np.mean(recall[used])),
        "macro_precision": float(np.mean(precision[used])),
        "macro_recall": float(np.mean(recall[used])),
        "macro_f1": float(np.mean(f1[used])),
        "weighted_f1": float((f1 * support).sum() / n) if n else 0.0,
        "per_class": {
            NAMES[c]: {
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
                "predicted": int(predicted[c]),
            }
            for c in range(K)
        },
    }


def evaluate(y: np.ndarray, p: np.ndarray, classes: Sequence[int] = range(K)) -> Dict[str, Any]:
    cm = confusion(y, p)
    return {**from_confusion(cm, classes), "confusion_matrix": cm.tolist()}


def stratified_random(train_share: np.ndarray, eval_share: np.ndarray) -> Dict[str, float]:
    """Expected scores of guessing with the training class frequencies (exact, no sampling)."""
    q, r = np.asarray(train_share, float), np.asarray(eval_share, float)
    f1 = _safe_div(2 * q * r, q + r)
    return {
        "accuracy": float((q * r).sum()),
        "balanced_accuracy": float(np.mean(q[r > 0])),
        "macro_f1": float(np.mean(f1[r > 0])),
    }


def share(y: np.ndarray) -> np.ndarray:
    return np.bincount(y, minlength=K) / max(len(y), 1)


def phase_of(ply: int) -> str:
    """The project's annotation phase bands (plies 12-14 count as opening)."""
    return "opening" if ply <= 25 else "middlegame" if ply <= 55 else "late"


def game_bootstrap(
    games: np.ndarray, y: np.ndarray, preds: Dict[str, np.ndarray], n_boot: int = 2000
) -> Dict[str, Any]:
    """Game-clustered 95% intervals for each model and for differences from the first."""
    ids, index = np.unique(games, return_inverse=True)
    per_game = {}
    for name, p in preds.items():
        cells = index * K * K + y * K + p
        per_game[name] = np.bincount(cells, minlength=len(ids) * K * K).reshape(len(ids), K, K)
    rng = np.random.RandomState(0)
    draws: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for _ in range(n_boot):
        weights = np.bincount(rng.randint(0, len(ids), len(ids)), minlength=len(ids))
        for name, mats in per_game.items():
            m = from_confusion(np.tensordot(weights, mats, axes=1))
            draws[name].append((m["accuracy"], m["macro_f1"], m["balanced_accuracy"]))
    base = next(iter(preds))
    out: Dict[str, Any] = {"games": int(len(ids)), "n_boot": n_boot, "reference": base}
    for name, values in draws.items():
        arr = np.array(values)
        for j, metric in enumerate(("accuracy", "macro_f1", "balanced_accuracy")):
            row = out.setdefault(name, {})
            row[metric] = np.percentile(arr[:, j], [2.5, 97.5]).tolist()
            if name != base:
                diff = arr[:, j] - np.array(draws[base])[:, j]
                row[f"{metric}_minus_reference"] = np.percentile(diff, [2.5, 97.5]).tolist()
    return out


# ------------------------------ rule engine ------------------------------ #
_W: Dict[str, Any] = {}


def _init_rule(wanted: Dict[str, str]) -> None:
    from src.data.pgn_parser import StreamingPGNParser
    from src.data.weak_labeler import RetrospectiveWeakLabeler

    _W.update(wanted=wanted, parser=StreamingPGNParser(), labeler=RetrospectiveWeakLabeler())


def _rule_game(text: str) -> Optional[Tuple[str, List[Tuple]]]:
    game = next(_W["parser"].iter_games(io.StringIO(text)), None)
    if game is None:
        return None
    game_id = _W["wanted"].get(bsd._move_hash(game))
    if game_id is None:
        return None
    lab = _W["labeler"]
    board = game.board()
    boards = [board.copy(stack=False)]
    for move in game.mainline_moves():
        board.push(move)
        boards.append(board.copy(stack=False))
    plies = range(lab.start_ply, len(boards))
    results = {
        t: lab.engine.evaluate_window(boards[max(0, t - lab.history_length + 1) : t + 1])
        for t in plies
    }
    plans = list(results[plies[0]].all_scores) if len(plies) else []
    rows = []
    for t in plies:
        span = [results[u].all_scores for u in range(t, t + lab.lookahead + 1) if u in results]
        agg = {p: sum(s[p] for s in span) / len(span) for p in plans}
        _name, label_id, _conf = lab._decide(agg)
        ordered = sorted(agg, key=lambda p: -agg[p])
        runner = next(p for p in ordered if p != "CHAIN_UNDERMINE")
        rows.append(
            (
                f"{game_id}#{t}",
                results[t].class_id,
                label_id,
                results[t].all_scores["CHAIN_UNDERMINE"],
                agg["CHAIN_UNDERMINE"],
                agg[runner],
                NAMES.index(runner),
            )
        )
    return game_id, rows


def rule_pass(wanted: Dict[str, str], skip: int, limit: int, out: Path, workers: int) -> None:
    if out.exists():
        return
    rows: List[Tuple] = []
    seen: set = set()
    with bsd.open_pgn_text(bsd.DEFAULT_ARCHIVE) as handle:
        texts = islice(bsd.iter_raw_games(handle), skip, skip + limit)
        with mp.get_context("spawn").Pool(workers, _init_rule, (wanted,)) as pool:
            for result in pool.imap(_rule_game, texts, chunksize=20):
                if result is not None and result[0] not in seen:
                    seen.add(result[0])
                    rows.extend(result[1])
    cols = list(zip(*rows))
    np.savez_compressed(
        out,
        window_id=np.array(cols[0]),
        rule_pred=np.array(cols[1], np.int8),
        label_again=np.array(cols[2], np.int8),
        own_chain=np.array(cols[3], np.float32),
        agg_chain=np.array(cols[4], np.float32),
        agg_runner=np.array(cols[5], np.float32),
        runner=np.array(cols[6], np.int8),
    )
    print(f"rule pass {out.name}: {len(seen)}/{len(wanted)} games, {len(rows)} windows")


def run_rule(work: Path, workers: int) -> None:
    ref = bsd._read_csv(PROCESSED / "scaled_80k" / "games.csv")
    scanned = json.loads((PROCESSED / "scaled_80k" / "meta.json").read_text())["games_scanned"]
    val = {g["move_hash"]: g["game_id"] for g in ref if g["split"] == "val"}
    rule_pass(val, 0, scanned, work / "rule_val.npz", workers)
    meta = json.loads((PROCESSED / "holdout2_full" / "meta.json").read_text())
    held = bsd._read_csv(PROCESSED / "holdout2_full" / "games.csv")
    wanted = {g["move_hash"]: g["game_id"] for g in held}
    rule_pass(
        wanted,
        meta["raw_games_skipped"],
        meta["raw_games_scanned_after_skip"],
        work / "rule_test.npz",
        workers,
    )


# -------------------------------- logits --------------------------------- #
def populations() -> Dict[str, Tuple[Path, str, Optional[np.ndarray]]]:
    """name -> (directory, split, row indices or None for all rows)."""
    trained = bsd.load_scaled_dataset(
        PROCESSED / "allpos",
        balance_train=True,
        seed=42,
        splits=("train",),
        balance_select="random",
        train_sample_kinds=("uniform",),
    )["train"].tensors.indices
    kinds = [r["sample_kind"] for r in bsd._read_csv(PROCESSED / "allpos" / "train_records.csv")]
    natural = np.array([i for i, kind in enumerate(kinds) if kind == "uniform"])
    return {
        "train_trained_rows": (PROCESSED / "allpos", "train", trained),
        "train_natural": (PROCESSED / "allpos", "train", natural),
        "val": (PROCESSED / "allpos", "val", None),
        "val_filtered": (PROCESSED / "scaled_80k", "val", None),
        "test_all": (PROCESSED / "holdout2_full", "test", None),
    }


def model_list() -> Dict[str, Tuple[str, int, str]]:
    models = {f"v1_s{s}": ("D_random_balance", s, "_loopr2") for s in V1_SEEDS}
    models.update({f"v2_s{s}": ("G_allpos_uniform", s, "_v2") for s in V2_SEEDS})
    return models


def run_logits(work: Path) -> None:
    import final_training as ft
    import torch

    from src.models.logger import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    todo = [
        (m, p)
        for m in model_list()
        for p in ("train_trained_rows", "train_natural", "val", "val_filtered", "test_all")
        if (p in ("val", "val_filtered", "test_all") or m == f"v2_s{V2_SAVED}")
        and not (work / f"logits_{m}_{p}.npy").exists()
    ]
    if not todo:
        return
    pops = populations()
    for pop, (directory, split, rows) in pops.items():
        names = [m for m, p in todo if p == pop]
        if not names:
            continue
        x = np.load(directory / f"{split}_tensors.npy", mmap_mode="r")
        index = np.arange(len(x)) if rows is None else rows
        np.save(work / f"rows_{pop}.npy", index)
        nets = {}
        for name in names:
            variant, seed, tag = model_list()[name]
            net = ft._build_model(ft.VARIANTS[variant])
            load_checkpoint(ft._ckpt_path(variant, seed, tag), net)
            nets[name] = net.to(device).eval()
        outs: Dict[str, List[np.ndarray]] = {n: [] for n in names}
        with torch.no_grad():
            for i in range(0, len(index), 4096):
                block = np.array(x[index[i : i + 4096], -56:], np.float32)
                batch = torch.from_numpy(block).to(device)
                for name, net in nets.items():
                    outs[name].append(net(batch).cpu().numpy())
        for name in names:
            np.save(work / f"logits_{name}_{pop}.npy", np.concatenate(outs[name]))
        print(f"logits {pop}: {len(index)} rows x {len(names)} model(s)", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Investigate v2 accuracy (no training).")
    ap.add_argument("--work-dir", required=True, help="Cache for logits and rule outputs.")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--steps", nargs="+", default=["rule", "logits", "metrics"])
    args = ap.parse_args(argv)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if "rule" in args.steps:
        run_rule(work, args.workers)
    if "logits" in args.steps:
        run_logits(work)
    if "metrics" in args.steps:
        from investigate_v2_report import build_report  # noqa: PLC0415

        OUT.write_text(json.dumps(build_report(work), indent=1), encoding="utf-8")
        print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
