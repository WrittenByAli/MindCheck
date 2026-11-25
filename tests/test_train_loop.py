"""Unit tests for the loop's gates, resume logic and test-once guard.

Every number here is a synthetic input to the decision logic, not a model metric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from scripts import train_loop as tl


def _gap(f1: float, acc: float = 0.01) -> dict:
    return {"macro_f1_gap_natural": f1, "accuracy_gap_natural": acc}


def test_gap_thresholds_are_inclusive():
    assert tl.passes_gap(_gap(0.03, 0.02))
    assert not tl.passes_gap(_gap(0.031, 0.01))
    assert not tl.passes_gap(_gap(0.01, 0.021))


def test_winner_prefers_eligible_over_higher_scoring_ineligible():
    got = tl.select_winner(
        {"a": {"score": 0.90, "gap": _gap(0.10)}, "b": {"score": 0.70, "gap": _gap(0.01)}}
    )
    assert got["winner"] == "b" and not got["eligibility_gate_failed"]


def test_winner_records_gate_failure_when_none_eligible():
    got = tl.select_winner(
        {"a": {"score": 0.70, "gap": _gap(0.10)}, "b": {"score": 0.60, "gap": _gap(0.05)}}
    )
    assert got["winner"] == "a" and got["eligibility_gate_failed"]


def test_tie_goes_to_smaller_gap_but_only_within_margin():
    tie = {"a": {"score": 0.704, "gap": _gap(0.025)}, "b": {"score": 0.700, "gap": _gap(0.005)}}
    assert tl.select_winner(tie)["winner"] == "b"
    tie["a"]["score"] = 0.710
    assert tl.select_winner(tie)["winner"] == "a"


def test_gates_g1_g2_g3_and_limitations():
    assert tl.gate_g1({1: _gap(0.01), 2: _gap(0.02)})["passed"]
    assert tl.gate_g1({1: _gap(0.01), 2: _gap(0.09)})["failed_seeds"] == [2]
    assert tl.gate_g2([0.650, 0.652, 0.651])["passed"]
    assert not tl.gate_g2([0.60, 0.65, 0.70])["passed"]
    assert tl.gate_g3({1: 0.65}, {1: 0.64})["passed"]
    assert not tl.gate_g3({1: 0.65}, {1: 0.60})["passed"]
    assert tl.class_limitations({"x": 0.149, "y": 0.15}) == ["x"]


class FakeRunner:
    """Writes the JSON each real script would write, without training anything."""

    def __init__(self, gap_f1: float = 0.01, fail_on: str = "") -> None:
        self.calls: List[str] = []
        self.commands: List[List[str]] = []
        self.gap_f1 = gap_f1
        self.fail_on = fail_on

    def __call__(self, cmd: List[str], log: Path) -> int:
        name = log.stem
        self.calls.append(name)
        self.commands.append(cmd)
        if self.fail_on and name.startswith(self.fail_on):
            return 1
        opt = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}
        if name == "holdout_build":
            meta = {"integrity": {"checks": {"ok": True}}, "games_valid": 1, "windows": 1}
            for key in ("--out-filtered", "--out-full"):
                directory = Path(opt[key])
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "meta.json").write_text(json.dumps(meta))
            return 0
        out = Path(opt["--out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        classes = {n: {"f1": 0.5} for n in tl.CLASS_NAMES}
        rep = {"macro_f1": 0.5, "accuracy": 0.5, "per_class": classes}
        if name.startswith("train_"):
            ckpt = out.parents[3] / "models" / "checkpoints"
            ckpt.mkdir(parents=True, exist_ok=True)
            tag, variant, seed = opt["--tag"], opt["--variant"], opt["--seeds"]
            (ckpt / f"scaled{tag}_{variant}_s{seed}.pt").write_text("x")
            payload: object = [{"val": rep, "best_epoch": 3}]
        elif name.startswith("bias_"):
            payload = {"cross_fold_check": {"a": {"heldout_f1_after": 0.6}}, "bias": {}}
        elif name.startswith("gap_"):
            payload = _gap(self.gap_f1)
        elif name.startswith("ensemble_"):
            payload = {
                "ensemble_val_macro_f1_raw": 0.5,
                "ensemble_val_macro_f1_bias_heldout_folds": {},
            }
        else:
            payload = [{"test": rep, "test_bias_adjusted": rep}]
        out.write_text(json.dumps(payload))
        return 0


def _loop(tmp_path: Path, runner: FakeRunner, **kwargs) -> tl.Loop:
    return tl.Loop(
        tmp_path,
        tmp_path / "data",
        "python",
        tl.ROUND1_VARIANTS + kwargs.pop("extra_variants", []),
        runner=runner,
        exporter=lambda *_: {"exported": True},
        **kwargs,
    )


def _tests(calls: List[str]) -> List[str]:
    return [c for c in calls if c.startswith("test_")]


def test_full_run_then_resume_skips_everything(tmp_path):
    first = FakeRunner()
    _loop(tmp_path, first).run()
    assert len(_tests(first.calls)) == 6  # 3 seeds x (filtered, full)
    build = first.calls.index("holdout_build")
    assert build > max(i for i, c in enumerate(first.calls) if c.startswith("ensemble_"))
    assert build < min(i for i, c in enumerate(first.calls) if c.startswith("test_"))
    for name, cmd in zip(first.calls, first.commands):
        if name.startswith("test_"):  # only the fresh holdout, never the old 80k test split
            data_dir = Path(cmd[cmd.index("--data-dir") + 1])
            assert data_dir.name == f"holdout_{name.rsplit('_', 1)[1]}"
            assert "--bias-file" in cmd
    second = FakeRunner()
    _loop(tmp_path, second).run()
    assert second.calls == []
    status = (tmp_path / "experiments" / "loop" / "status.txt").read_text()
    assert "COMPLETE" in status and tl.TEST_CAVEAT in status
    assert status.index("FULL-POSITION fresh holdout") < status.index("Filtered fresh holdout")


def test_failure_stops_loop_and_resume_continues_from_there(tmp_path):
    broken = FakeRunner(fail_on="gap_loopr1_D")
    with pytest.raises(tl.LoopError):
        _loop(tmp_path, broken).run()
    assert not any(c.startswith("train_loopr2") for c in broken.calls)
    fixed = FakeRunner()
    _loop(tmp_path, fixed).run()
    assert fixed.calls[0].startswith("gap_loopr1_D")
    assert not any(c.startswith("train_loopr1_C") for c in fixed.calls)


def test_g1_failure_without_round3_variant_stops_before_test(tmp_path):
    runner = FakeRunner(gap_f1=0.2)
    with pytest.raises(tl.GateStop):
        _loop(tmp_path, runner).run()
    assert _tests(runner.calls) == []


def test_round3_runs_once_when_variant_exists(tmp_path):
    runner = FakeRunner(gap_f1=0.2)
    extra = list(tl.ROUND3_VARIANTS.values())
    with pytest.raises(tl.GateStop):
        _loop(tmp_path, runner, extra_variants=extra).run()
    assert sum(c.startswith("train_loopr3") for c in runner.calls) == 3
    assert _tests(runner.calls) == []


def test_accepting_gate_failure_records_it(tmp_path):
    loop = _loop(tmp_path, FakeRunner(gap_f1=0.2), accept_gate_failure=True)
    loop.run()
    assert loop.state["decisions"]["gates"]["G1"] is False
    assert "G1 FAILED" in loop.state["decisions"]["g1_failure"]


def test_interrupted_test_run_is_never_repeated(tmp_path):
    with pytest.raises(tl.LoopError):
        _loop(tmp_path, FakeRunner(fail_on="test_")).run()
    retry = FakeRunner()
    with pytest.raises(tl.LoopError, match="test guard"):
        _loop(tmp_path, retry).run()
    assert retry.calls == []


def test_test_split_cannot_be_reused_for_another_model(tmp_path):
    loop = _loop(tmp_path, FakeRunner())
    loop._claim_test("a", "_loopr2", 42, "full", tmp_path / "none.json")
    with pytest.raises(tl.LoopError, match="already used"):
        loop._claim_test("b", "_loopr2", 123, "full", tmp_path / "none2.json")


def test_g3_uses_the_filtered_holdout_not_the_full_one(tmp_path):
    class Diverging(FakeRunner):
        def __call__(self, cmd, log):
            code = super().__call__(cmd, log)
            if log.stem.startswith("test_") and log.stem.endswith("_full"):
                out = Path(cmd[cmd.index("--out") + 1])
                rows = json.loads(out.read_text())
                rows[0]["test"] = {**rows[0]["test"], "macro_f1": 0.1}
                out.write_text(json.dumps(rows))
            return code

    loop = _loop(tmp_path, Diverging())
    loop.run()
    assert loop.state["decisions"]["gates"]["G3"] is True


def test_verify_data_requires_all_checks_true(tmp_path):
    with pytest.raises(tl.LoopError):
        tl.verify_data(tmp_path)
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"integrity": {"checks": {"a": True, "b": False}}}))
    with pytest.raises(tl.LoopError):
        tl.verify_data(tmp_path)
    meta.write_text(json.dumps({"integrity": {"checks": {"a": True}}}))
    tl.verify_data(tmp_path)


def test_round2_and_round1_tags_never_collide(tmp_path):
    loop = _loop(tmp_path, FakeRunner())
    assert loop.ckpt("v", 42, "_loopr1") != loop.ckpt("v", 42, "_loopr2")
