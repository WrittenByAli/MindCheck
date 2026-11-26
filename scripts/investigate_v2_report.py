"""Build the tables of the v2 accuracy investigation from cached logits.

Used by ``investigate_v2.py --steps metrics``. Every bias fitted here uses
validation data (or training labels, for the prior correction) only.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import build_scaled_dataset as bsd
import numpy as np
from investigate_v2 import (
    CHAIN,
    EXP,
    MIN_CONFIDENCE,
    NAMES,
    NO_PLAN,
    PROCESSED,
    V1_SAVED,
    V2_SAVED,
    V2_SEEDS,
    K,
    confusion,
    evaluate,
    from_confusion,
    game_bootstrap,
    phase_of,
    share,
    stratified_random,
)
from tune_logit_bias import fit_bias

STRATEGY = list(range(NO_PLAN))
_ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _vector(bias: Dict[str, float]) -> np.ndarray:
    return np.array([bias[n] for n in NAMES], dtype=np.float32)


class Data:
    """Labels, record metadata and logits for every population."""

    def __init__(self, work: Path) -> None:
        self.work = work
        self.records: Dict[str, List[Dict[str, str]]] = {}
        self.y: Dict[str, np.ndarray] = {}
        self.games: Dict[str, np.ndarray] = {}
        sources = {
            "val": (PROCESSED / "allpos", "val"),
            "val_filtered": (PROCESSED / "scaled_80k", "val"),
            "test_all": (PROCESSED / "holdout2_full", "test"),
        }
        for name, (directory, split) in sources.items():
            self.records[name] = bsd._read_csv(directory / f"{split}_records.csv")
            self.y[name] = np.load(directory / f"{split}_labels.npy")
            self.games[name] = np.array([r["game_id"] for r in self.records[name]])
        allpos_train = np.load(PROCESSED / "allpos" / "train_labels.npy")
        for name in ("train_trained_rows", "train_natural"):
            self.y[name] = allpos_train[np.load(work / f"rows_{name}.npy")]
        kept = {
            r["window_id"]
            for r in bsd._read_csv(PROCESSED / "holdout2_filtered" / "test_records.csv")
        }
        self.filtered = np.array(
            [i for i, r in enumerate(self.records["test_all"]) if r["window_id"] in kept]
        )
        assert len(self.filtered) == len(kept)
        self.y["test_filtered"] = self.y["test_all"][self.filtered]
        self.filtered_train_y = np.load(PROCESSED / "scaled_80k" / "train_labels.npy")

    def logits(self, model: str, population: str) -> np.ndarray:
        if population == "test_filtered":
            return self.logits(model, "test_all")[self.filtered]
        return np.load(self.work / f"logits_{model}_{population}.npy")


def biases() -> Dict[str, np.ndarray]:
    out = {
        "v1_shipped_filtered_val": _vector(
            _json(_ROOT / "models" / "checkpoints" / "mindcheck_final.json")["logit_bias"]
        ),
        "v1_B0_unfiltered_val": _vector(_json(EXP / "v2_B0_bias.json")["bias"]),
    }
    for seed in V2_SEEDS:
        path = EXP / "v2" / f"bias_G_allpos_uniform_s{seed}.json"
        out[f"v2_s{seed}_unfiltered_val"] = _vector(_json(path)["bias"])
    return out


def prior_correction(natural_train_y: np.ndarray) -> np.ndarray:
    """Training-labels-only logit adjustment back to the natural class prior.

    Balanced sampling plus inverse-frequency loss weights make the effective
    training prior uniform, so adding log(natural train share) restores it.
    """
    bias = np.log(share(natural_train_y))
    return (bias - bias.mean()).astype(np.float32)


def pred(logits: np.ndarray, bias: Any = 0) -> np.ndarray:
    return (logits + bias).argmax(1)


# ------------------------------- sections -------------------------------- #
def distribution(d: Data) -> Dict[str, Any]:
    out = {}
    for name, y in {**d.y, "train_filtered_natural": d.filtered_train_y}.items():
        counts = np.bincount(y, minlength=K)
        out[name] = {
            "n": int(len(y)),
            "counts": dict(zip(NAMES, counts.tolist())),
            "share": dict(zip(NAMES, (counts / len(y)).round(4).tolist())),
        }
    return out


def baselines(d: Data, rule: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Every baseline is built from TRAINING labels (or past-only rules), never test."""
    out: Dict[str, Any] = {}
    for pop in ("train_trained_rows", "train_natural", "val", "test_all", "test_filtered"):
        filtered = pop == "test_filtered"
        train_y = d.filtered_train_y if filtered else d.y["train_natural"]
        y = d.y[pop]
        majority = int(share(train_y).argmax())
        trained_counts = np.bincount(d.y["train_trained_rows"], minlength=K)
        top = np.flatnonzero(trained_counts == trained_counts.max())
        row = {
            "majority_class_from_natural_train": {
                "class": NAMES[majority],
                **_brief(evaluate(y, np.full_like(y, majority))),
            },
            "always_no_plan": _brief(evaluate(y, np.full_like(y, NO_PLAN))),
            "stratified_random_expected": stratified_random(share(train_y), share(y)),
            "most_frequent_in_balanced_training_rows": [
                {"class": NAMES[int(c)], **_brief(evaluate(y, np.full_like(y, c)))} for c in top
            ],
        }
        if pop in rule:
            row["rule_engine_past_only"] = evaluate(y, rule[pop])
        out[pop] = row
    return out


def _brief(m: Dict[str, Any]) -> Dict[str, float]:
    return {k: m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}


def v2_metrics(d: Data, bias: Dict[str, np.ndarray]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for seed in V2_SEEDS:
        name = f"v2_s{seed}"
        pops = ["val", "test_all", "test_filtered"]
        if seed == V2_SAVED:
            pops = ["train_trained_rows", "train_natural"] + pops
        vec = bias[f"{name}_unfiltered_val"]
        out[name] = {
            pop: {
                "raw": evaluate(d.y[pop], pred(d.logits(name, pop))),
                "biased": evaluate(d.y[pop], pred(d.logits(name, pop), vec)),
            }
            for pop in pops
        }
    return out


def _no_plan_row(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    m = from_confusion(confusion(y, p))["per_class"]["NO_PLAN"]
    strategy, none = y != NO_PLAN, y == NO_PLAN
    return {
        "true_share": float(none.mean()),
        "predicted_share": float((p == NO_PLAN).mean()),
        "precision": m["precision"],
        "recall": m["recall"],
        "strategy_positions_called_no_plan": float((p[strategy] == NO_PLAN).mean()),
        "no_plan_positions_given_a_strategy": float((p[none] != NO_PLAN).mean()),
    }


def no_plan(d: Data, bias: Dict[str, np.ndarray]) -> Dict[str, Any]:
    name = f"v2_s{V2_SAVED}"
    out = {}
    for pop in ("train_trained_rows", "val", "test_all", "test_filtered"):
        logits = d.logits(name, pop)
        out[pop] = {
            "raw": _no_plan_row(d.y[pop], pred(logits)),
            "biased": _no_plan_row(d.y[pop], pred(logits, bias[f"{name}_unfiltered_val"])),
        }
    return out


def _gain_by_class(y: np.ndarray, raw: np.ndarray, adj: np.ndarray) -> Dict[str, float]:
    """Accuracy points gained or lost per TRUE class when the bias is applied."""
    delta = (adj == y).astype(int) - (raw == y).astype(int)
    return {NAMES[c]: float(delta[y == c].sum() / len(y)) for c in range(K)}


def bias_audit(d: Data, bias: Dict[str, np.ndarray]) -> Dict[str, Any]:
    name = f"v2_s{V2_SAVED}"
    stored = bias[f"{name}_unfiltered_val"]
    refit = fit_bias(d.logits(name, "val"), d.y["val"]).astype(np.float32)
    train_only = prior_correction(d.y["train_natural"])
    filtered_fit = fit_bias(d.logits(name, "val_filtered"), d.y["val_filtered"])
    options = {
        "raw": np.zeros(K, np.float32),
        "fit_on_unfiltered_val_macro_f1 (shipped)": stored,
        "train_labels_only_prior_correction": train_only,
        "fit_on_filtered_val_macro_f1": filtered_fit.astype(np.float32),
    }
    table = {}
    for pop in ("val", "test_all", "test_filtered"):
        logits = d.logits(name, pop)
        raw = pred(logits)
        table[pop] = {}
        for label, vec in options.items():
            m = evaluate(d.y[pop], pred(logits, vec))
            table[pop][label] = {
                **_brief(m),
                "accuracy_change_by_true_class": _gain_by_class(d.y[pop], raw, pred(logits, vec)),
                "recall": {n: m["per_class"][n]["recall"] for n in NAMES},
            }
    val_games = {
        g["move_hash"]
        for g in bsd._read_csv(PROCESSED / "scaled_80k" / "games.csv")
        if g["split"] == "val"
    }
    held = bsd._read_csv(PROCESSED / "holdout2_full" / "games.csv")
    bias_file = EXP / "v2" / f"bias_G_allpos_uniform_s{V2_SAVED}.json"
    return {
        "values": {k: dict(zip(NAMES, np.round(v, 4).tolist())) for k, v in options.items()},
        "stored_bias_reproduced_from_validation": {
            "max_abs_diff": float(np.abs(refit - stored).max())
        },
        "val_games_in_holdout2": len(val_games & {g["move_hash"] for g in held}),
        "bias_file_written_before_holdout2_built": bool(
            os.path.getmtime(bias_file)
            < os.path.getmtime(PROCESSED / "holdout2_full" / "meta.json")
        ),
        "comparisons": table,
    }


def subsets(d: Data, rule: Dict[str, np.ndarray], bias: Dict[str, np.ndarray]) -> Dict[str, Any]:
    recs = d.records["test_all"]
    y = d.y["test_all"]
    conf = np.array([float(r["weak_confidence"]) for r in recs])
    in_filtered = np.zeros(len(y), bool)
    in_filtered[d.filtered] = True
    masks = {
        "A_all_positions": np.ones(len(y), bool),
        "B_strategy_labelled": y != NO_PLAN,
        "C_clear_strategy (existing filter)": in_filtered & (y != NO_PLAN),
        "D1_no_plan_labelled": y == NO_PLAN,
        "D2_low_confidence (<0.55)": conf < MIN_CONFIDENCE,
    }
    train_y = d.y["train_natural"]
    majority_strategy = int(share(train_y)[:NO_PLAN].argmax())
    out: Dict[str, Any] = {}
    for label, mask in masks.items():
        classes = STRATEGY if label.startswith(("B_", "C_")) else range(K)
        row: Dict[str, Any] = {"n": int(mask.sum()), "models": {}}
        for key, model, vec_name in (
            (f"v2_s{V2_SAVED}", f"v2_s{V2_SAVED}", f"v2_s{V2_SAVED}_unfiltered_val"),
            (f"v1_s{V1_SAVED}", f"v1_s{V1_SAVED}", "v1_shipped_filtered_val"),
            ("v1_unfiltered_val_bias", f"v1_s{V1_SAVED}", "v1_B0_unfiltered_val"),
        ):
            logits = d.logits(model, "test_all")[mask]
            variants = {"raw": pred(logits), "biased": pred(logits, bias[vec_name])}
            if label.startswith(("B_", "C_")):
                variants["forced_choice_among_5_plans_raw"] = logits[:, :NO_PLAN].argmax(1)
            row["models"][key] = {k: evaluate(y[mask], p, classes) for k, p in variants.items()}
        row["rule_engine_past_only"] = _brief(evaluate(y[mask], rule["test_all"][mask], classes))
        if label.startswith(("B_", "C_")):
            row["majority_strategy_baseline"] = {
                "class": NAMES[majority_strategy],
                **_brief(evaluate(y[mask], np.full(mask.sum(), majority_strategy), classes)),
            }
        out[label] = row
    return out


def compare_models(d: Data, bias: Dict[str, np.ndarray]) -> Dict[str, Any]:
    v1, v2 = f"v1_s{V1_SAVED}", f"v2_s{V2_SAVED}"
    out: Dict[str, Any] = {}
    for pop in ("test_all", "test_filtered"):
        filtered_bias = fit_bias(d.logits(v2, "val_filtered"), d.y["val_filtered"])
        runs = {
            "v1_raw": pred(d.logits(v1, pop)),
            "v2_raw": pred(d.logits(v2, pop)),
            "v1_unfiltered_val_bias": pred(d.logits(v1, pop), bias["v1_B0_unfiltered_val"]),
            "v2_unfiltered_val_bias": pred(d.logits(v2, pop), bias[f"{v2}_unfiltered_val"]),
            "v1_filtered_val_bias": pred(d.logits(v1, pop), bias["v1_shipped_filtered_val"]),
            "v2_filtered_val_bias": pred(d.logits(v2, pop), filtered_bias),
        }
        games = d.games["test_all"] if pop == "test_all" else d.games["test_all"][d.filtered]
        boot = {}
        for a, b in (
            ("v1_raw", "v2_raw"),
            ("v1_unfiltered_val_bias", "v2_unfiltered_val_bias"),
            ("v1_filtered_val_bias", "v2_filtered_val_bias"),
        ):
            boot[f"{b} vs {a}"] = game_bootstrap(games, d.y[pop], {a: runs[a], b: runs[b]})
        out[pop] = {
            "metrics": {k: evaluate(d.y[pop], p) for k, p in runs.items()},
            "game_bootstrap_95ci": boot,
        }
    return out


def versus_rule(d: Data, rule: Dict[str, np.ndarray], bias: Dict[str, np.ndarray]) -> Dict:
    """Paired game-bootstrap of each v2 seed against the past-only rule engine."""
    out: Dict[str, Any] = {}
    for pop in ("test_all", "test_filtered"):
        games = d.games["test_all"] if pop == "test_all" else d.games["test_all"][d.filtered]
        for seed in V2_SEEDS:
            name = f"v2_s{seed}"
            logits = d.logits(name, pop)
            vec = bias[f"{name}_unfiltered_val"]
            for kind, p in (("raw", pred(logits)), ("biased", pred(logits, vec))):
                runs = {"rule_engine_past_only": rule[pop], f"{name}_{kind}": p}
                out[f"{pop}:{name}_{kind}"] = game_bootstrap(games, d.y[pop], runs)
    return out


def _chain_pr(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    tp = int(((y == CHAIN) & (p == CHAIN)).sum())
    pos, guessed = int((y == CHAIN).sum()), int((p == CHAIN).sum())
    precision, recall = tp / guessed if guessed else 0.0, tp / pos if pos else 0.0
    f1 = 2 * tp / (pos + guessed) if pos + guessed else 0.0
    return {
        "support": pos,
        "predicted": guessed,
        "true_positive": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _concentration(y: np.ndarray, p: np.ndarray, games: np.ndarray) -> Dict[str, Any]:
    positives = Counter(games[y == CHAIN].tolist())
    ranked = [g for g, _ in positives.most_common()]
    total = sum(positives.values())
    out: Dict[str, Any] = {
        "games_with_chain_positions": len(positives),
        "games_total": int(len(set(games.tolist()))),
        "max_positions_in_one_game": max(positives.values()),
        "share_in_top_10_games": sum(positives[g] for g in ranked[:10]) / total,
    }
    for k in (0, 10, 50):
        keep = ~np.isin(games, ranked[:k])
        out[f"without_top_{k}_games"] = _chain_pr(y[keep], p[keep])
    return out


def _visibility(y: np.ndarray, preds: Dict[str, np.ndarray], z: Any) -> Dict[str, Any]:
    chain = y == CHAIN
    own = z["own_chain"][chain]
    margin = z["agg_chain"][chain] - z["agg_runner"][chain]
    out: Dict[str, Any] = {
        "chain_labelled": int(chain.sum()),
        "past_only_chain_score_zero": float((own == 0).mean()),
        "past_only_engine_also_says_chain": float((z["rule_pred"][chain] == CHAIN).mean()),
        "runner_up_class_share": {
            NAMES[int(c)]: float(v / chain.sum())
            for c, v in Counter(z["runner"][chain].tolist()).items()
        },
        "label_margin_percentiles_10_50_90": np.percentile(margin, [10, 50, 90]).tolist(),
    }
    for name, p in preds.items():
        hit = p[chain] == CHAIN
        out[f"{name}_recall_evidence_visible"] = float(hit[own > 0].mean())
        out[f"{name}_recall_evidence_future_only"] = float(hit[own == 0].mean())
    return out


def _examples(d: Data, logits: np.ndarray, vec: np.ndarray, z: Any) -> Dict[str, Any]:
    y, recs = d.y["val"], d.records["val"]
    p = pred(logits, vec)
    rng = np.random.RandomState(0)
    out = {}
    for kind, mask in (
        ("false_negative", (y == CHAIN) & (p != CHAIN)),
        ("false_positive", (y != CHAIN) & (p == CHAIN)),
    ):
        idx = np.flatnonzero(mask)
        picks = rng.choice(idx, min(8, len(idx)), replace=False)
        out[kind] = [
            {
                "window_id": recs[i]["window_id"],
                "ply": int(recs[i]["ply_index"]),
                "fen": recs[i]["fen"],
                "label": NAMES[y[i]],
                "predicted": NAMES[p[i]],
                "weak_confidence": float(recs[i]["weak_confidence"]),
                "past_only_chain_score": float(z["own_chain"][i]),
                "lookahead_chain_score": float(z["agg_chain"][i]),
            }
            for i in picks
        ]
    return out


def chain_study(d: Data, raw_rule: Dict[str, Any], bias: Dict[str, np.ndarray]) -> Dict[str, Any]:
    name = f"v2_s{V2_SAVED}"
    vec = bias[f"{name}_unfiltered_val"]
    out: Dict[str, Any] = {
        "windows": {
            pop: int((y == CHAIN).sum())
            for pop, y in {**d.y, "train_filtered_natural": d.filtered_train_y}.items()
        }
    }
    for pop in ("val", "test_all"):
        y, logits = d.y[pop], d.logits(name, pop)
        preds = {"raw": pred(logits), "biased": pred(logits, vec)}
        plies = np.array([int(r["ply_index"]) for r in d.records[pop]])
        phases = np.array([phase_of(p) for p in plies])
        block: Dict[str, Any] = {}
        for kind, p in preds.items():
            cm = confusion(y, p)
            block[kind] = {
                **_chain_pr(y, p),
                "missed_chain_predicted_as": {
                    NAMES[c]: int(cm[CHAIN, c]) for c in range(K) if c != CHAIN
                },
                "false_alarms_true_class": {
                    NAMES[c]: int(cm[c, CHAIN]) for c in range(K) if c != CHAIN
                },
                "by_phase": {
                    ph: _chain_pr(y[phases == ph], p[phases == ph])
                    for ph in ("opening", "middlegame", "late")
                },
            }
        block["game_concentration_biased"] = _concentration(y, preds["biased"], d.games[pop])
        block["label_evidence"] = _visibility(y, preds, raw_rule[pop])
        out[pop] = block
    out["val_examples"] = _examples(d, d.logits(name, "val"), vec, raw_rule["val"])
    return out


def label_predictability(d: Data, rule: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """How often the past-only rule engine matches the look-ahead weak label."""
    out = {}
    for pop in ("val", "test_all"):
        y, p = d.y[pop], rule[pop]
        out[pop] = {
            "overall_agreement": float((y == p).mean()),
            "by_true_class": {
                NAMES[c]: float((p[y == c] == c).mean()) for c in range(K) if (y == c).any()
            },
        }
    return out


def _align_rule(d: Data, work: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Re-order rule-engine output to record order and check labels reproduce."""
    out = {}
    for pop, filename in (("val", "rule_val.npz"), ("test_all", "rule_test.npz")):
        z = np.load(work / filename)
        where = {w: i for i, w in enumerate(z["window_id"].tolist())}
        order = np.array([where[r["window_id"]] for r in d.records[pop]])
        out[pop] = {k: z[k][order] for k in z.files}
    return out


def build_report(work: Path) -> Dict[str, Any]:
    d = Data(work)
    bias = biases()
    raw_rule = _align_rule(d, work)
    rule = {pop: z["rule_pred"].astype(np.int64) for pop, z in raw_rule.items()}
    rule["test_filtered"] = rule["test_all"][d.filtered]
    return {
        "note": (
            "Diagnostic only; weak labels. Holdout2 was already used for the v2 decision; "
            "nothing here is fitted or selected on it."
        ),
        "label_reproduction": {
            pop: float((z["label_again"] == d.y[pop]).mean()) for pop, z in raw_rule.items()
        },
        "distribution": distribution(d),
        "baselines": baselines(d, rule),
        "v2_metrics": v2_metrics(d, bias),
        "no_plan": no_plan(d, bias),
        "bias_audit": bias_audit(d, bias),
        "subsets": subsets(d, rule, bias),
        "v2_vs_rule_engine": versus_rule(d, rule, bias),
        "v1_vs_v2": compare_models(d, bias),
        "chain_undermine": chain_study(d, raw_rule, bias),
        "label_predictability": label_predictability(d, rule),
    }
