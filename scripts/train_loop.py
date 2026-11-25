"""Resumable train -> evaluate -> gate loop that produces the final model.

Pre-registered design (gates are the constants below; every decision uses TRAIN
and VALIDATION only):

* Round 1 screens C, D and E on seed 42.
* Round 2 retrains the winner on 3 seeds with a longer epoch cap.
* Round 3 runs only if gate G1 fails and a stronger-regularization variant exists.
* The final test uses a FRESH holdout from archive games the 80k build never
  scanned (filtered and full-position versions). Each seed is evaluated once on
  each, after the final seed has been chosen from validation, guarded by marker
  files. The old 80k test split is never loaded.

Finished steps are recorded in ``experiments/loop/state.json`` and skipped on
restart. Launch with::

    .venv/Scripts/python.exe -u scripts/train_loop.py --wait-for-data
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CLASS_NAMES = ["K_STORM", "Q_EXP", "OUTPOST", "ROOK_FILE", "CHAIN_UNDERMINE", "NO_PLAN"]

# ---------------------------- pre-registered ------------------------------ #
ROUND1_VARIANTS = ["C_4ply_mirror_reg", "D_random_balance", "E_random_balance_sqrtw"]
ROUND1_SEED = 42
ROUND1_EPOCHS, ROUND1_PATIENCE = 8, 3
ROUND2_SEEDS = [42, 123, 999]
ROUND2_EPOCHS, ROUND2_PATIENCE = 16, 3
MAX_ROUNDS = 3
MAX_MACRO_F1_GAP = 0.03  # fair natural train-vs-val macro-F1 gap
MAX_ACCURACY_GAP = 0.02  # fair natural train-vs-val accuracy gap
TIE_MARGIN = 0.005
G2_MAX_SEED_STD = 0.005  # sample std of raw val macro-F1 across seeds
G3_MAX_TEST_VAL_DIFF = 0.02  # |test - val| raw macro-F1, per seed
CLASS_LIMITATION_F1 = 0.15  # reported as a known limitation, not a gate
# Round-3 variant per winner. Used only if final_training.py actually offers it.
ROUND3_VARIANTS = {name: f"{name}_strongreg" for name in ROUND1_VARIANTS}

GATE_TEXT = [
    f"Round-1 eligibility: fair macro-F1 gap <= {MAX_MACRO_F1_GAP} and accuracy gap "
    f"<= {MAX_ACCURACY_GAP}.",
    "Winner: highest bias-adjusted held-out-fold mean val macro-F1 among eligible "
    f"variants (all variants if none is eligible); ties within {TIE_MARGIN} go to the "
    "smaller fair macro-F1 gap.",
    "G1: every confirmation seed passes both fair-gap thresholds.",
    f"G2: sample std of raw val macro-F1 across seeds <= {G2_MAX_SEED_STD}.",
    f"G3 (once): |filtered fresh-holdout - val| raw macro-F1 <= {G3_MAX_TEST_VAL_DIFF} for "
    "every seed. Full-position holdout metrics are reported without a gate; nothing is tuned "
    "on either holdout.",
    f"Any class with val F1 < {CLASS_LIMITATION_F1} is a known limitation, not a gate.",
    f"At most {MAX_ROUNDS} rounds.",
]
TEST_CAVEAT = (
    "The final test is a fresh holdout from archive games the 80k build never scanned. The old "
    "80k test split is not evaluated (it was evaluated once for variant C in an earlier "
    "session). Validation and the filtered holdout keep only confident, balanced windows with a "
    "top-25-by-confidence cap, and that confidence uses look-ahead, so they are not natural "
    "positions and are likely optimistic; the full-position holdout is closer to live use. "
    "All labels are rule-engine weak labels."
)
HOLDOUT_KINDS = ("filtered", "full")


class LoopError(RuntimeError):
    """A step failed; the loop stops."""


class GateStop(RuntimeError):
    """A pre-registered gate stopped the loop (reported, not hidden)."""


# --------------------------------- gates ---------------------------------- #
def passes_gap(gap: Dict[str, Any]) -> bool:
    return (
        gap["macro_f1_gap_natural"] <= MAX_MACRO_F1_GAP
        and gap["accuracy_gap_natural"] <= MAX_ACCURACY_GAP
    )


def heldout_score(bias: Dict[str, Any]) -> float:
    """Mean held-out-fold val macro-F1 after the bias (fit on the other half)."""
    folds = bias["cross_fold_check"].values()
    return float(statistics.mean(f["heldout_f1_after"] for f in folds))


def select_winner(candidates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """``candidates[name] = {"score": float, "gap": gap_json}`` -> decision."""
    if not candidates:
        raise LoopError("no candidates to select from.")
    eligible = [n for n, c in candidates.items() if passes_gap(c["gap"])]
    pool = eligible or list(candidates)
    best = max(candidates[n]["score"] for n in pool)
    tied = [n for n in pool if candidates[n]["score"] >= best - TIE_MARGIN]
    winner = min(
        tied,
        key=lambda n: (candidates[n]["gap"]["macro_f1_gap_natural"], -candidates[n]["score"]),
    )
    return {
        "winner": winner,
        "eligible": eligible,
        "eligibility_gate_failed": not eligible,
        "tied": tied,
    }


def gate_g1(gaps: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    failed = [seed for seed, gap in gaps.items() if not passes_gap(gap)]
    return {"passed": not failed and bool(gaps), "failed_seeds": failed}


def gate_g2(val_scores: Sequence[float]) -> Dict[str, Any]:
    std = float(statistics.stdev(val_scores)) if len(val_scores) > 1 else float("inf")
    return {"passed": std <= G2_MAX_SEED_STD, "std": std, "mean": statistics.mean(val_scores)}


def gate_g3(val: Dict[int, float], test: Dict[int, float]) -> Dict[str, Any]:
    diffs = {seed: abs(test[seed] - val[seed]) for seed in val}
    return {"passed": all(d <= G3_MAX_TEST_VAL_DIFF for d in diffs.values()), "abs_diff": diffs}


def class_limitations(per_class_f1: Dict[str, float]) -> List[str]:
    return [name for name, f1 in per_class_f1.items() if f1 < CLASS_LIMITATION_F1]


def holdout_summary(test: Optional[Dict[str, Any]]) -> List[str]:
    """Status lines, full-position numbers first."""
    if not test:
        return []
    lines: List[str] = []
    for kind, title in (
        ("full", "FULL-POSITION fresh holdout (closest to live use; no gate)"),
        ("filtered", "Filtered fresh holdout (comparable to validation; gate G3)"),
    ):
        lines += ["", title + ":"]
        for seed, row in test["seeds"].items():
            m = row[f"holdout_{kind}"]
            lines.append(
                f"  seed {seed}: raw macro-F1 {m['raw_macro_f1']:.4f}, acc "
                f"{m['raw_accuracy']:.4f} | with bias macro-F1 {m['bias_macro_f1']:.4f}, "
                f"acc {m['bias_accuracy']:.4f}"
            )
    return lines


# --------------------------------- loop ----------------------------------- #
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def subprocess_runner(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        return subprocess.call(cmd, cwd=_REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)


class Loop:
    """The state machine. ``runner`` and ``exporter`` are injectable for tests."""

    def __init__(
        self,
        root: Path,
        data_dir: Path,
        python: str,
        variants: Sequence[str],
        device: Optional[str] = None,
        threads: int = 8,
        batch_size: Optional[int] = None,
        accept_gate_failure: bool = False,
        holdout_games: int = 6000,
        holdout_workers: int = 14,
        runner: Callable[[List[str], Path], int] = subprocess_runner,
        exporter: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.root = Path(root)
        self.loop_dir = self.root / "experiments" / "loop"
        self.results = self.loop_dir / "results"
        self.markers = self.loop_dir / "test_markers"
        self.logs = self.root / "experiments" / "logs"
        self.ckpt_dir = self.root / "models" / "checkpoints"
        self.data_dir = Path(data_dir)
        self.holdout_dirs = {
            kind: self.root / "data" / "processed" / f"holdout_{kind}" for kind in HOLDOUT_KINDS
        }
        self.holdout_games = holdout_games
        self.holdout_workers = holdout_workers
        self.python = python
        self.variants = list(variants)
        self.device = device
        self.threads = threads
        self.batch_size = batch_size
        self.accept_gate_failure = accept_gate_failure
        self.runner = runner
        self.exporter = exporter or export_final
        state_path = self.loop_dir / "state.json"
        self.state: Dict[str, Any] = (
            _read_json(state_path) if state_path.exists() else {"steps": {}, "decisions": {}}
        )

    # -- bookkeeping -------------------------------------------------------- #
    def _save(self, message: str) -> None:
        _write_atomic(self.loop_dir / "state.json", json.dumps(self.state, indent=2))
        lines = ["MindCheck training loop", f"updated: {time.strftime('%Y-%m-%d %H:%M:%S')}"]
        lines += [f"now: {message}", "", "Pre-registered gates:"]
        lines += [f"  - {g}" for g in GATE_TEXT]
        lines += ["", f"Caveat: {TEST_CAVEAT}"]
        lines += holdout_summary(self.state["decisions"].get("test"))
        lines += ["", "Finished steps:"]
        lines += [f"  - {name}" for name in self.state["steps"]]
        lines += ["", "Decisions:", json.dumps(self.state["decisions"], indent=2)]
        _write_atomic(self.loop_dir / "status.txt", "\n".join(lines) + "\n")
        print(message, flush=True)

    def _decide(self, key: str, value: Any) -> None:
        # JSON round-trip so in-memory state matches what a resumed run reads.
        self.state["decisions"][key] = json.loads(json.dumps(value))
        self._save(f"decision {key}")

    def step(self, name: str, cmd: List[str], output: Path, requires: Sequence[Path] = ()) -> Any:
        """Run ``cmd`` unless its finished ``output`` already exists; return the JSON."""
        complete = output.exists() and all(p.exists() for p in requires)
        if not complete:
            self._save(f"running {name} (log: experiments/logs/{name}.log)")
            started = time.time()
            code = self.runner(cmd, self.logs / f"{name}.log")
            if code != 0:
                self._save(f"FAILED {name} with exit code {code}")
                raise LoopError(f"step {name} failed (exit {code}); see {self.logs}/{name}.log")
            if not output.exists() or not all(p.exists() for p in requires):
                self._save(f"FAILED {name}: expected output missing")
                raise LoopError(f"step {name} produced no output at {output}")
            seconds = round(time.time() - started, 1)
        else:
            seconds = self.state["steps"].get(name, {}).get("seconds")
        data = _read_json(output)
        if name not in self.state["steps"]:
            self.state["steps"][name] = {"output": str(output), "seconds": seconds}
            self._save(f"{'adopted' if complete else 'finished'} {name}")
        return data

    # -- commands ----------------------------------------------------------- #
    def _script(self, name: str) -> List[str]:
        return [self.python, "-u", str(_REPO_ROOT / "scripts" / name)]

    def ckpt(self, variant: str, seed: int, tag: str) -> Path:
        return self.ckpt_dir / f"scaled{tag}_{variant}_s{seed}.pt"

    def train_cmd(
        self,
        phase: str,
        variant: str,
        seed: int,
        tag: str,
        epochs: int,
        patience: int,
        out: Path,
        data_dir: Optional[Path] = None,
    ) -> List[str]:
        cmd = self._script("final_training.py") + [
            "--phase", phase, "--variant", variant, "--seeds", str(seed),
            "--epochs", str(epochs), "--patience", str(patience),
            "--threads", str(self.threads), "--tag", tag,
            "--data-dir", str(data_dir or self.data_dir), "--out", str(out),
        ]  # fmt: skip
        if self.batch_size:
            cmd += ["--batch-size", str(self.batch_size)]
        if self.device:
            cmd += ["--device", self.device]
        return cmd

    def _eval_cmd(self, script: str, variant: str, tag: str, out: Path) -> List[str]:
        return self._script(script) + [
            "--variant", variant, "--tag", tag,
            "--data-dir", str(self.data_dir), "--out", str(out),
        ]  # fmt: skip

    def candidate(
        self, variant: str, seed: int, tag: str, epochs: int, patience: int
    ) -> Dict[str, Any]:
        """Train one (variant, seed), tune its bias and measure its fair gap."""
        key = f"{tag.strip('_')}_{variant}_s{seed}"
        train_out = self.results / f"train_{key}.json"
        bias_out = self.results / f"bias_{key}.json"
        gap_out = self.results / f"gap_{key}.json"
        train = self.step(
            f"train_{key}",
            self.train_cmd("train", variant, seed, tag, epochs, patience, train_out),
            train_out,
            requires=[self.ckpt(variant, seed, tag)],
        )[-1]
        bias = self.step(
            f"bias_{key}",
            self._eval_cmd("tune_logit_bias.py", variant, tag, bias_out) + ["--seed", str(seed)],
            bias_out,
        )
        gap = self.step(
            f"gap_{key}",
            self._eval_cmd("measure_gap.py", variant, tag, gap_out)
            + ["--seed", str(seed), "--threads", str(self.threads)],
            gap_out,
        )
        return {
            "variant": variant,
            "seed": seed,
            "tag": tag,
            "score": heldout_score(bias),
            "val_raw_macro_f1": train["val"]["macro_f1"],
            "per_class_val_f1": {n: c["f1"] for n, c in train["val"]["per_class"].items()},
            "best_epoch": train["best_epoch"],
            "gap": gap,
            "bias": bias,
            "bias_path": str(bias_out),
        }

    # -- rounds ------------------------------------------------------------- #
    def round1(self) -> str:
        found = {
            v: self.candidate(v, ROUND1_SEED, "_loopr1", ROUND1_EPOCHS, ROUND1_PATIENCE)
            for v in ROUND1_VARIANTS
        }
        decision = select_winner(found)
        decision["table"] = {
            v: {
                "score": c["score"],
                "val_raw_macro_f1": c["val_raw_macro_f1"],
                "macro_f1_gap": c["gap"]["macro_f1_gap_natural"],
                "accuracy_gap": c["gap"]["accuracy_gap_natural"],
            }
            for v, c in found.items()
        }
        self._decide("round1", decision)
        return decision["winner"]

    def confirm(self, variant: str, tag: str, label: str) -> Dict[str, Any]:
        runs = {
            s: self.candidate(variant, s, tag, ROUND2_EPOCHS, ROUND2_PATIENCE) for s in ROUND2_SEEDS
        }
        ens_out = self.results / f"ensemble_{tag.strip('_')}_{variant}.json"
        ensemble = self.step(
            f"ensemble_{tag.strip('_')}_{variant}",
            self._eval_cmd("ensemble_eval.py", variant, tag, ens_out)
            + ["--seeds", *map(str, ROUND2_SEEDS)],
            ens_out,
        )
        # Chosen from validation, and recorded before any test evaluation happens.
        chosen = max(runs, key=lambda s: runs[s]["score"])
        summary = {
            "variant": variant,
            "tag": tag,
            "G1": gate_g1({s: r["gap"] for s, r in runs.items()}),
            "G2": gate_g2([r["val_raw_macro_f1"] for r in runs.values()]),
            "chosen_seed": chosen,
            "seeds": {
                s: {
                    "val_raw_macro_f1": r["val_raw_macro_f1"],
                    "bias_heldout_macro_f1": r["score"],
                    "macro_f1_gap": r["gap"]["macro_f1_gap_natural"],
                    "accuracy_gap": r["gap"]["accuracy_gap_natural"],
                    "best_epoch": r["best_epoch"],
                    "hit_epoch_cap": r["best_epoch"] >= ROUND2_EPOCHS,
                    "per_class_val_f1": r["per_class_val_f1"],
                    "known_limitations": class_limitations(r["per_class_val_f1"]),
                }
                for s, r in runs.items()
            },
            "ensemble_val_raw_macro_f1": ensemble["ensemble_val_macro_f1_raw"],
            "ensemble_val_bias_heldout": ensemble["ensemble_val_macro_f1_bias_heldout_folds"],
        }
        self._decide(label, summary)
        return {"summary": summary, "runs": runs}

    # -- final -------------------------------------------------------------- #
    def holdout_cmd(self) -> List[str]:
        return self._script("build_fresh_holdout.py") + [
            "--exclude-dir", str(self.data_dir),
            "--out-filtered", str(self.holdout_dirs["filtered"]),
            "--out-full", str(self.holdout_dirs["full"]),
            "--max-games", str(self.holdout_games), "--workers", str(self.holdout_workers),
        ]  # fmt: skip

    def build_holdout(self) -> Dict[str, Any]:
        full = self.holdout_dirs["full"] / "meta.json"
        filtered = self.holdout_dirs["filtered"] / "meta.json"
        self.step("holdout_build", self.holdout_cmd(), full, requires=[filtered])
        for directory in self.holdout_dirs.values():
            verify_data(directory)
        return {k: _read_json(d / "meta.json") for k, d in self.holdout_dirs.items()}

    def _claim_test(self, variant: str, tag: str, seed: int, kind: str, output: Path) -> None:
        """Allow each holdout evaluation to start at most once, ever."""
        identity = {"variant": variant, "tag": tag}
        for marker in sorted(self.markers.glob("*.json")):
            other = _read_json(marker)
            if {k: other[k] for k in identity} != identity:
                raise LoopError(
                    f"test guard: the holdout was already used for {other}; refusing "
                    f"to evaluate {identity}."
                )
        marker = self.markers / f"s{seed}_{kind}.json"
        if marker.exists() and not output.exists():
            raise LoopError(
                f"test guard: a {kind}-holdout run for seed {seed} was started but left no "
                f"result. Refusing to run it again. Inspect the log, then decide about {marker}."
            )
        if output.exists() and not marker.exists():
            raise LoopError(f"test guard: {output} exists without its marker.")
        if not marker.exists():
            record = {**identity, "seed": seed, "kind": kind, "started": time.time()}
            _write_atomic(marker, json.dumps(record))

    def _evaluate_holdout(self, summary: Dict[str, Any], run: Dict[str, Any], kind: str) -> Dict:
        variant, tag, seed = summary["variant"], summary["tag"], run["seed"]
        key = f"test_{tag.strip('_')}_{variant}_s{seed}_{kind}"
        out = self.results / f"{key}.json"
        self._claim_test(variant, tag, seed, kind, out)
        cmd = self.train_cmd(
            "test", variant, seed, tag, ROUND2_EPOCHS, ROUND2_PATIENCE, out,
            data_dir=self.holdout_dirs[kind],
        )  # fmt: skip
        row = self.step(key, cmd + ["--bias-file", run["bias_path"]], out)[-1]
        raw, biased = row["test"], row["test_bias_adjusted"]
        return {
            "raw_macro_f1": raw["macro_f1"],
            "raw_accuracy": raw["accuracy"],
            "raw_per_class_f1": {n: c["f1"] for n, c in raw["per_class"].items()},
            "bias_macro_f1": biased["macro_f1"],
            "bias_accuracy": biased["accuracy"],
            "bias_per_class_f1": {n: c["f1"] for n, c in biased["per_class"].items()},
        }

    def final_test(self, confirmed: Dict[str, Any]) -> Dict[str, Any]:
        """Build the fresh holdout, then evaluate every confirmation seed once on each."""
        summary, runs = confirmed["summary"], confirmed["runs"]
        holdout = self.build_holdout()
        seeds = {
            seed: {
                f"holdout_{kind}": self._evaluate_holdout(summary, run, kind)
                for kind in HOLDOUT_KINDS
            }
            for seed, run in runs.items()
        }
        g3 = gate_g3(
            {s: r["val_raw_macro_f1"] for s, r in runs.items()},
            {s: seeds[s]["holdout_filtered"]["raw_macro_f1"] for s in runs},
        )
        result = {
            "G3": g3,
            "G3_applies_to": "filtered fresh holdout raw macro-F1 vs validation raw macro-F1",
            "caveat": TEST_CAVEAT,
            "holdout": {
                kind: {
                    "games": meta.get("games_valid"),
                    "windows": meta.get("windows"),
                    "class_counts": meta.get("integrity", {}).get("class_counts"),
                }
                for kind, meta in holdout.items()
            },
            "seeds": seeds,
        }
        self._decide("test", result)
        return result

    def run(self) -> None:
        winner = self.round1()
        confirmed = self.confirm(winner, "_loopr2", "round2")
        if not confirmed["summary"]["G1"]["passed"]:
            stronger = ROUND3_VARIANTS.get(winner)
            if stronger in self.variants:
                confirmed = self.confirm(stronger, "_loopr3", "round3")
            if not confirmed["summary"]["G1"]["passed"]:
                reason = (
                    "G1 FAILED: the fair train/val gap exceeds the pre-registered threshold. "
                    + (
                        "Round 3 also failed."
                        if stronger in self.variants
                        else "No stronger-regularization variant is available for round 3."
                    )
                )
                self._decide("g1_failure", reason)
                if not self.accept_gate_failure:
                    raise GateStop(
                        reason + " Stopping before test/export. Re-run with "
                        "--accept-gate-failure to evaluate and export with the failure recorded."
                    )
        test = self.final_test(confirmed)
        exported = self.exporter(self, confirmed, test)
        self._decide("export", exported)
        gates = {
            "G1": confirmed["summary"]["G1"]["passed"],
            "G2": confirmed["summary"]["G2"]["passed"],
            "G3": test["G3"]["passed"],
        }
        self._decide("gates", gates)
        self._save(f"COMPLETE. Gates passed: {gates}")


# -------------------------------- export ---------------------------------- #
def export_final(loop: Loop, confirmed: Dict[str, Any], test: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap the validation-chosen seed to the spec input and export it."""
    import final_training  # noqa: PLC0415 - only needed at export time

    from src.models.export import export_to_onnx, export_to_torchscript, verify_onnx_matches
    from src.models.inference import DeployableModel, load_deployable, save_deployable
    from src.models.logger import load_checkpoint

    summary = confirmed["summary"]
    seed, variant, tag = summary["chosen_seed"], summary["variant"], summary["tag"]
    run = confirmed["runs"][seed]
    cfg = final_training.VARIANTS[variant]
    bias = [run["bias"]["bias"][name] for name in CLASS_NAMES]
    model = DeployableModel(keep=cfg["keep"], bias=bias, dropout=cfg["dropout"])
    load_checkpoint(loop.ckpt(variant, seed, tag), model.backbone)
    model.eval()

    base = loop.ckpt_dir / "mindcheck_final"
    pt = save_deployable(
        model, base.with_suffix(".pt"), {"variant": variant, "seed": seed, "tag": tag}
    )
    onnx_path = export_to_onnx(model, base.with_suffix(".onnx"))
    if not verify_onnx_matches(load_deployable(pt), onnx_path):
        raise LoopError("ONNX export does not match PyTorch output.")
    ts_path = export_to_torchscript(model, loop.ckpt_dir / "mindcheck_final.ts.pt")

    meta_path = loop.data_dir / "meta.json"
    meta = _read_json(meta_path) if meta_path.exists() else {}
    sidecar = {
        "variant": variant,
        "seed": seed,
        "selected_by": "validation bias-adjusted held-out-fold macro-F1",
        "input": "(B,112,8,8) float32; the newest channels are sliced internally",
        "keep_channels": cfg["keep"],
        "config": cfg,
        "class_names": CLASS_NAMES,
        "logit_bias": dict(zip(CLASS_NAMES, bias)),
        "labels": "rule-engine weak labels; not validated against human annotation",
        "validation": summary["seeds"][seed],
        "holdout_full_closest_to_live_use": test["seeds"][seed]["holdout_full"],
        "holdout_filtered": test["seeds"][seed]["holdout_filtered"],
        "holdout_sizes": test.get("holdout"),
        "gates": {"G1": summary["G1"], "G2": summary["G2"], "G3": test["G3"]},
        "test_caveat": TEST_CAVEAT,
        "dataset": {
            k: meta.get(k) for k in ("games_valid", "windows_per_split", "games_per_split")
        },
    }
    _write_atomic(base.with_suffix(".json"), json.dumps(sidecar, indent=2))
    return {
        "checkpoint": str(pt),
        "onnx": str(onnx_path),
        "torchscript": str(ts_path),
        "sidecar": str(base.with_suffix(".json")),
        "onnx_matches_pytorch": True,
    }


# ------------------------------ environment ------------------------------- #
def has_cuda(python: str) -> bool:
    code = "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
    try:
        return subprocess.call([python, "-c", code], stderr=subprocess.DEVNULL) == 0
    except OSError:
        return False


def choose_python(override: Optional[str] = None) -> str:
    if override:
        return override
    venv = _REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    return str(venv) if venv.exists() and has_cuda(str(venv)) else sys.executable


def probe_training_cli(python: str) -> Dict[str, Any]:
    """Ask final_training.py which variants and flags it currently supports."""
    text = subprocess.check_output(
        [python, str(_REPO_ROOT / "scripts" / "final_training.py"), "--help"],
        cwd=_REPO_ROOT,
        text=True,
        env=dict(os.environ, COLUMNS="2000"),
    )
    block = text.split("--variant {", 1)[1].split("}", 1)[0]
    variants = [v.strip() for v in block.split(",")]
    return {"variants": variants, "has_device": "--device" in text}


def wait_for_data(data_dir: Path, timeout_hours: float = 6.0, poll_seconds: float = 60.0) -> None:
    meta = data_dir / "meta.json"
    deadline = time.time() + timeout_hours * 3600
    while not meta.exists():
        if time.time() > deadline:
            raise LoopError(f"timed out waiting for {meta}")
        print(f"waiting for {meta} ...", flush=True)
        time.sleep(poll_seconds)


def verify_data(data_dir: Path) -> None:
    meta = data_dir / "meta.json"
    if not meta.exists():
        raise LoopError(f"{meta} not found; build the dataset or pass --wait-for-data.")
    checks = _read_json(meta).get("integrity", {}).get("checks", {})
    failed = [name for name, ok in checks.items() if ok is not True]
    if not checks or failed:
        raise LoopError(f"dataset integrity checks not all true: {failed or 'none recorded'}")


@contextlib.contextmanager
def keep_awake() -> Iterator[None]:
    """Stop Windows from sleeping while the loop runs; no persistent setting changes."""
    if sys.platform != "win32":
        yield
        return
    import ctypes

    continuous, system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(continuous | system_required)
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(continuous)


def command_plan(loop: Loop) -> List[str]:
    lines = []
    for variant in ROUND1_VARIANTS:
        out = loop.results / f"train_loopr1_{variant}_s{ROUND1_SEED}.json"
        cmd = loop.train_cmd(
            "train", variant, ROUND1_SEED, "_loopr1", ROUND1_EPOCHS, ROUND1_PATIENCE, out
        )
        lines += [" ".join(cmd), f"  -> tune_logit_bias.py, measure_gap.py ({variant})"]
    for seed in ROUND2_SEEDS:
        out = loop.results / f"train_loopr2_<winner>_s{seed}.json"
        cmd = loop.train_cmd(
            "train", "<winner>", seed, "_loopr2", ROUND2_EPOCHS, ROUND2_PATIENCE, out
        )
        lines += [" ".join(cmd), "  -> tune_logit_bias.py, measure_gap.py"]
    lines += [
        "ensemble_eval.py on the 3 seeds; gates G1, G2",
        "round 3 only if G1 fails and a *_strongreg variant exists",
        " ".join(loop.holdout_cmd()),
        "once per seed and holdout (guarded): final_training --phase test on holdout_filtered "
        "and holdout_full with that seed's validation bias; gate G3 on filtered only",
        "export models/checkpoints/mindcheck_final.{pt,json,onnx,ts.pt}",
    ]
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run the pre-registered training loop.")
    ap.add_argument("--data-dir", default=str(_REPO_ROOT / "data" / "processed" / "scaled_80k"))
    ap.add_argument("--python", default=None, help="Interpreter for subprocesses.")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--holdout-games", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--wait-for-data", action="store_true")
    ap.add_argument("--wait-timeout-hours", type=float, default=6.0)
    ap.add_argument("--accept-gate-failure", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if str(_REPO_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    python = choose_python(args.python)
    cli = probe_training_cli(python)
    missing = [v for v in ROUND1_VARIANTS if v not in cli["variants"]]
    if missing:
        print(f"final_training.py lacks variants: {missing}", file=sys.stderr)
        return 1
    device = ("cuda" if has_cuda(python) else "cpu") if cli["has_device"] else None
    loop = Loop(
        _REPO_ROOT,
        Path(args.data_dir),
        python,
        cli["variants"],
        device=device,
        threads=args.threads,
        batch_size=args.batch_size,
        accept_gate_failure=args.accept_gate_failure,
        holdout_games=args.holdout_games,
    )
    print(f"interpreter: {python}\ndevice flag: {device or 'not supported by CLI'}", flush=True)
    if args.dry_run:
        print("\n".join(GATE_TEXT + [""] + command_plan(loop)))
        return 0
    try:
        with keep_awake():
            if args.wait_for_data:
                loop._save("waiting for dataset")
                wait_for_data(loop.data_dir, args.wait_timeout_hours)
            verify_data(loop.data_dir)
            loop.run()
    except GateStop as exc:
        loop._save(f"STOPPED BY GATE: {exc}")
        return 2
    except LoopError as exc:
        loop._save(f"STOPPED: {exc}")
        return 1
    except Exception as exc:  # report unexpected errors in status.txt, then re-raise
        loop._save(f"CRASHED: {exc!r}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
