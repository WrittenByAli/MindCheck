"""Does CHAIN_UNDERMINE's weak label depend on plies the model cannot see?

The weak label for window ``t`` is the mean of the rule engine's per-plan scores
over plies ``t..t+8`` (threshold 0.5, margin 0.15). The model only sees the
8-board history ending at ``t``. If the lever pawn push that triggers the rule
happens AFTER ``t``, the label is unobservable from the input.

This script re-derives the labels for sampled VALIDATION games (never test),
checks the re-derivation against the cached labels, and measures how often the
CHAIN_UNDERMINE evidence is visible in the input window versus only in the
future, then how well the seed-42 model recalls each group. Nothing is tuned and
no existing label or file is changed.

Usage::

    python scripts/analyze_chain_labels.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import multiprocessing as mp
import random
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import chess  # noqa: E402
import chess.pgn  # noqa: E402
import numpy as np  # noqa: E402
from build_scaled_dataset import _move_hash, iter_raw_games, open_pgn_text  # noqa: E402

from src.data.tensor_encoder import ChessTensorEncoder  # noqa: E402
from src.data.weak_labeler import PLAN_ORDER, RetrospectiveWeakLabeler  # noqa: E402
from src.data.window_extractor import PositionWindow  # noqa: E402
from src.rules.rule_engine import SymbolicRuleEngine  # noqa: E402

CHAIN = "CHAIN_UNDERMINE"
START_PLY = 12
HISTORY = 8
LOOKAHEAD = 8
THRESHOLD = 0.5
MARGIN = 0.15
PUSH_LEVEL = 0.8  # a lever push scores 1.0 (base) or 0.8 (apex)
DEFAULT_DATA = _REPO_ROOT / "data" / "processed" / "scaled_80k"
DEFAULT_ARCHIVE = _REPO_ROOT / "data" / "raw" / "lichess_elite_2025-11.zip"


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)                                                    #
# --------------------------------------------------------------------------- #
def first_push_offset(
    chain_scores: Sequence[Optional[float]], level: float = PUSH_LEVEL
) -> Optional[int]:
    """Smallest ``k`` with ``chain_scores[k] >= level`` (k=0 is the window itself)."""
    for k, score in enumerate(chain_scores):
        if score is not None and score >= level:
            return k
    return None


def assign_group(own_score: float, offset: Optional[int], threshold: float = THRESHOLD) -> str:
    """``visible``: input window already shows the lever; ``future_only``: only later plies do."""
    if own_score >= threshold:
        return "visible"
    if offset is not None:
        return "future_only"
    return "no_push_evidence"


def margin_stats(agg: Dict[str, float], label: str) -> Dict[str, Any]:
    """Aggregate score of ``label`` versus its strongest competitor."""
    others = {k: v for k, v in agg.items() if k != label}
    runner = max(others, key=lambda k: (others[k], k)) if others else None
    runner_score = others[runner] if runner else 0.0
    return {
        "top": agg[label],
        "runner_up": runner,
        "runner_up_score": runner_score,
        "margin": agg[label] - runner_score,
        "near_threshold": abs(agg[label] - THRESHOLD) <= 0.05,
    }


def fraction(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


# --------------------------------------------------------------------------- #
# Worker side: analyse the games we care about                                  #
# --------------------------------------------------------------------------- #
_W: Dict[str, Any] = {}


def _init(wanted: Dict[str, Dict[str, Any]]) -> None:
    _W["wanted"] = wanted
    _W["engine"] = SymbolicRuleEngine()
    _W["labeler"] = RetrospectiveWeakLabeler(start_ply=START_PLY, history_length=HISTORY)
    _W["encoder"] = ChessTensorEncoder(history_length=HISTORY)


def analyze_game(game: chess.pgn.Game, info: Dict[str, Any]) -> Dict[str, Any]:
    engine, labeler, encoder = _W["engine"], _W["labeler"], _W["encoder"]
    board = game.board()
    boards = [board.copy(stack=False)]
    sans: List[str] = []
    for move in game.mainline_moves():
        sans.append(board.san(move))
        board.push(move)
        boards.append(board.copy(stack=False))
    total = len(boards) - 1

    per_ply: Dict[int, Dict[str, float]] = {}
    for t in range(START_PLY, total + 1):
        lo = max(0, t - (HISTORY - 1))
        per_ply[t] = engine.evaluate_window(boards[lo : t + 1]).all_scores

    decided: Dict[int, Tuple[str, float, Dict[str, float]]] = {}
    for t in range(START_PLY, total + 1):
        wins = [per_ply[t]] + [per_ply[t + k] for k in range(1, LOOKAHEAD + 1) if t + k in per_ply]
        agg = {p: sum(w[p] for w in wins) / len(wins) for p in PLAN_ORDER}
        label, _lid, conf = labeler._decide(agg)
        decided[t] = (label, conf, agg)

    mismatches = None
    if info.get("verify"):
        records = labeler.label_game(game, game_id=info["game_id"])
        mismatches = sum(1 for r in records if decided[r.ply_index][0] != r.label_class)
        mismatches = {"windows": len(records), "mismatches": mismatches}

    rows = []
    for ply, (cached_label, cached_conf) in info["cached"].items():
        if ply not in decided:
            continue
        label, conf, agg = decided[ply]
        chain_future = [
            per_ply[ply + k][CHAIN] if (ply + k) in per_ply else None for k in range(LOOKAHEAD + 1)
        ]
        row: Dict[str, Any] = {
            "ply": ply,
            "cached_label": cached_label,
            "cached_conf": cached_conf,
            "new_label": label,
            "new_conf": conf,
            "agg": agg,
            "own": per_ply[ply],
            "chain_future": chain_future,
            "fen": boards[ply].fen(),
            "recent_sans": sans[max(0, ply - 8) : ply],
        }
        if cached_label == CHAIN:
            lo = max(0, ply - (HISTORY - 1))
            window = PositionWindow(
                game_id=info["game_id"],
                ply_index=ply,
                turn=boards[ply].turn,
                board_history=boards[lo : ply + 1],
                last_move=None,
            )
            tensor, _mask = encoder.encode_position_window(window)
            row["tensor56"] = tensor[-56:].astype(np.uint8)
        rows.append(row)
    return {"game_id": info["game_id"], "role": info["role"], "rows": rows, "verify": mismatches}


def _process_chunk(texts: List[str]) -> Tuple[int, List[Tuple[str, Dict[str, Any]]]]:
    found = []
    for text in texts:
        game = chess.pgn.read_game(io.StringIO(text))
        if game is None:
            continue
        digest = _move_hash(game)
        info = _W["wanted"].get(digest)
        if info is not None:
            found.append((digest, analyze_game(game, info)))
    return len(texts), found


# --------------------------------------------------------------------------- #
# Main-process pipeline                                                         #
# --------------------------------------------------------------------------- #
def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_wanted(data_dir: Path, n_chain: int, n_control: int, seed: int, n_verify: int):
    games = [g for g in _read_csv(data_dir / "games.csv") if g["split"] == "val"]
    hash_of = {g["game_id"]: g["move_hash"] for g in games}
    cached: Dict[str, Dict[int, Tuple[str, float]]] = defaultdict(dict)
    for r in _read_csv(data_dir / "val_records.csv"):
        cached[r["game_id"]][int(r["ply_index"])] = (r["label_class"], float(r["weak_confidence"]))
    chain_games = sorted(
        g for g, rows in cached.items() if any(v[0] == CHAIN for v in rows.values())
    )
    rng = random.Random(seed)
    chain_sample = rng.sample(chain_games, min(n_chain, len(chain_games)))
    rest = sorted(set(hash_of) - set(chain_sample))
    control_sample = rng.sample(rest, min(n_control, len(rest)))

    wanted: Dict[str, Dict[str, Any]] = {}
    for role, ids in (("chain", chain_sample), ("control", control_sample)):
        for i, gid in enumerate(ids):
            wanted[hash_of[gid]] = {
                "game_id": gid,
                "role": role,
                "cached": cached[gid],
                "verify": role == "chain" and i < n_verify,
            }
    return wanted, {
        "val_games_total": len(games),
        "val_games_with_chain_windows": len(chain_games),
        "chain_games_sampled": len(chain_sample),
        "control_games_sampled": len(control_sample),
    }


def scan_archive(archive: Path, wanted: Dict[str, Dict[str, Any]], workers: int, max_scan: int):
    results: Dict[str, Dict[str, Any]] = {}
    scanned = 0
    chunk_size = 150
    pending: deque = deque()
    with open_pgn_text(archive) as handle:
        pool = mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(wanted,))
        try:
            buf: List[str] = []

            def drain(block: bool) -> None:
                nonlocal scanned
                while pending and (block or pending[0].ready() or len(pending) >= workers * 3):
                    n, found = pending.popleft().get()
                    scanned += n
                    for digest, res in found:
                        results.setdefault(digest, res)

            for text in iter_raw_games(handle):
                buf.append(text)
                if len(buf) >= chunk_size:
                    pending.append(pool.apply_async(_process_chunk, (buf,)))
                    buf = []
                    drain(False)
                    if len(results) >= len(wanted) or scanned >= max_scan:
                        break
            if buf and len(results) < len(wanted):
                pending.append(pool.apply_async(_process_chunk, (buf,)))
            drain(True)
        finally:
            pool.terminate()
            pool.join()
    return results, scanned


def summarize(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rows = [(res["role"], r) for res in results.values() for r in res["rows"]]
    match = sum(1 for _, r in rows if r["new_label"] == r["cached_label"])
    conf_diff = max((abs(r["new_conf"] - r["cached_conf"]) for _, r in rows), default=0.0)
    verify = [res["verify"] for res in results.values() if res["verify"]]
    repro = {
        "windows_compared": len(rows),
        "label_match": match,
        "label_match_rate": fraction(match, len(rows)),
        "max_abs_confidence_diff": round(conf_diff, 6),
        "label_game_check": {
            "games": len(verify),
            "windows": sum(v["windows"] for v in verify),
            "mismatches": sum(v["mismatches"] for v in verify),
        },
    }

    chain_rows = [r for _, r in rows if r["cached_label"] == CHAIN]
    own_vals = [r["own"][CHAIN] for r in chain_rows]
    own_hist = {
        "zero": sum(1 for v in own_vals if v == 0),
        "between_0_and_0.5": sum(1 for v in own_vals if 0 < v < THRESHOLD),
        "at_least_0.5": sum(1 for v in own_vals if v >= THRESHOLD),
    }
    groups: Counter = Counter()
    offsets: Counter = Counter()
    runner_up: Counter = Counter()
    margins: List[float] = []
    near = 0
    for r in chain_rows:
        offset = first_push_offset(r["chain_future"])
        r["offset"] = offset
        r["group"] = assign_group(r["own"][CHAIN], offset)
        groups[r["group"]] += 1
        offsets[offset if offset is not None else "none"] += 1
        stats = margin_stats(r["agg"], CHAIN)
        r["margin_stats"] = stats
        runner_up[stats["runner_up"]] += 1
        margins.append(stats["margin"])
        near += int(stats["near_threshold"])
    n = len(chain_rows)

    per_class: Dict[str, Dict[str, Any]] = {}
    for scope, scoped in (
        ("control_sample", [r for role, r in rows if role == "control"]),
        ("all_sampled", [r for _, r in rows]),
    ):
        table = {}
        for plan in PLAN_ORDER:
            sel = [r for r in scoped if r["cached_label"] == plan]
            own = [r["own"][plan] for r in sel]
            table[plan] = {
                "n": len(sel),
                "own_score_zero": fraction(sum(1 for v in own if v == 0), len(sel)),
                "own_score_below_threshold": fraction(
                    sum(1 for v in own if v < THRESHOLD), len(sel)
                ),
            }
        per_class[scope] = table

    return {
        "reproduction": repro,
        "chain_windows": n,
        "a_own_window_chain_score": {
            **own_hist,
            "fractions": {k: fraction(v, n) for k, v in own_hist.items()},
        },
        "b_evidence_timing": {
            "groups": dict(groups),
            "group_fractions": {k: fraction(v, n) for k, v in groups.items()},
            "first_push_offset_plies_ahead": {
                str(k): v for k, v in sorted(offsets.items(), key=str)
            },
            "predicted_future_only_fraction": round(4 / 7, 4),
        },
        "c_margin_and_overlap": {
            "mean_margin_to_runner_up": round(float(np.mean(margins)), 4) if margins else None,
            "median_margin_to_runner_up": round(float(np.median(margins)), 4) if margins else None,
            "runner_up_class_counts": dict(runner_up.most_common()),
            "runner_up_class_fractions": {k: fraction(v, n) for k, v in runner_up.most_common()},
            "within_0.05_of_threshold": near,
            "within_0.05_of_threshold_fraction": fraction(near, n),
        },
        "d_own_window_score_by_label_class": per_class,
    }, chain_rows


def model_recall(chain_rows: List[Dict[str, Any]], seed: int, tag: str, bias_file: Path):
    import torch
    from final_training import CLASS_NAMES, VARIANTS, _build_model, _ckpt_path

    from src.models.logger import load_checkpoint

    variant = "C_4ply_mirror_reg"
    model = _build_model(VARIANTS[variant])
    load_checkpoint(_ckpt_path(variant, seed, tag), model)
    model.eval()
    torch.set_num_threads(8)
    bias = json.loads(bias_file.read_text(encoding="utf-8"))["bias"]
    vec = np.array([bias[name] for name in CLASS_NAMES], dtype=np.float32)
    x = np.stack([r["tensor56"] for r in chain_rows]).astype(np.float32)
    with torch.no_grad():
        logits = np.concatenate(
            [model(torch.from_numpy(x[i : i + 512])).numpy() for i in range(0, len(x), 512)]
        )
    chain_id = CLASS_NAMES.index(CHAIN)
    raw = logits.argmax(1)
    adj = (logits + vec).argmax(1)
    out: Dict[str, Any] = {"seed": seed, "bias_file": str(bias_file.name)}
    for group in ("all", "visible", "future_only", "no_push_evidence"):
        idx = [i for i, r in enumerate(chain_rows) if group == "all" or r["group"] == group]
        out[group] = {
            "n": len(idx),
            "recall_raw": fraction(int(sum(raw[i] == chain_id for i in idx)), len(idx)),
            "recall_with_bias": fraction(int(sum(adj[i] == chain_id for i in idx)), len(idx)),
        }
    out["raw_predicted_class_counts"] = {
        CLASS_NAMES[k]: int(v) for k, v in Counter(raw.tolist()).most_common()
    }
    for r, p_raw, p_adj in zip(chain_rows, raw, adj):
        r["pred_raw"] = CLASS_NAMES[int(p_raw)]
        r["pred_adj"] = CLASS_NAMES[int(p_adj)]
    return out


def pick_examples(chain_rows: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in chain_rows:
        by_group[r["group"]].append(r)
    picked: List[Dict[str, Any]] = []
    for group in ("visible", "future_only"):
        pool = by_group.get(group, [])
        picked += rng.sample(pool, min(5, len(pool)))
    rest = [r for r in chain_rows if r not in picked]
    while len(picked) < 10 and rest:
        picked.append(rest.pop(rng.randrange(len(rest))))
    return [
        {
            "fen": r["fen"],
            "ply": r["ply"],
            "recent_san_moves": r["recent_sans"],
            "group": r["group"],
            "own_window_chain_score": r["own"][CHAIN],
            "first_push_offset": r["offset"],
            "aggregate_scores": {k: round(v, 3) for k, v in r["agg"].items()},
            "runner_up": r["margin_stats"]["runner_up"],
            "model_pred_raw": r.get("pred_raw"),
            "model_pred_with_bias": r.get("pred_adj"),
        }
        for r in picked
    ]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CHAIN_UNDERMINE label-visibility analysis.")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    ap.add_argument("--n-chain-games", type=int, default=600)
    ap.add_argument("--n-control-games", type=int, default=600)
    ap.add_argument("--n-verify-games", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-scan", type=int, default=90000)
    ap.add_argument("--model-seed", type=int, default=42)
    ap.add_argument("--tag", default="_80k")
    ap.add_argument("--out", default=str(_REPO_ROOT / "experiments" / "analysis_chain_labels.json"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 1
    workers = max(1, min(args.workers, 8))
    data_dir = Path(args.data_dir)

    t0 = time.time()
    wanted, sample_info = build_wanted(
        data_dir, args.n_chain_games, args.n_control_games, args.seed, args.n_verify_games
    )
    print(sample_info, flush=True)
    results, scanned = scan_archive(Path(args.archive), wanted, workers, args.max_scan)
    print(f"found {len(results)}/{len(wanted)} games after scanning {scanned}", flush=True)

    summary, chain_rows = summarize(results)
    bias_file = (
        _REPO_ROOT
        / "experiments"
        / (f"logit_bias{args.tag}_C_4ply_mirror_reg_s{args.model_seed}.json")
    )
    recall = model_recall(chain_rows, args.model_seed, args.tag, bias_file)
    report = {
        "split_used": "validation only (test never loaded)",
        "seed": args.seed,
        "sample": {
            **sample_info,
            "games_found_in_archive": len(results),
            "games_requested": len(wanted),
            "archive_games_scanned": scanned,
        },
        **summary,
        "model_recall_on_chain_windows": recall,
        "examples": pick_examples(chain_rows, args.seed),
        "seconds": round(time.time() - t0, 1),
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
