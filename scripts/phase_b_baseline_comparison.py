"""Phase B: compare the trained neural model against the rule-only baseline
on the exact same (deterministic, seed=42) test split.

Not a permanent pipeline module -- a one-off comparison script for the
locked-in training plan's Phase B.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from src.data.weak_labeler import PLAN_IDS  # noqa: E402
from src.models.logger import load_checkpoint  # noqa: E402
from src.models.residual_cnn import HistoryChessResNet  # noqa: E402
from src.models.train import prepare_datasets  # noqa: E402
from src.rules.rule_engine import SymbolicRuleEngine  # noqa: E402

NUM_CLASSES = 6
ID_TO_NAME = {v: k for k, v in PLAN_IDS.items()}


def main() -> int:
    t0 = time.time()
    print("Re-running prepare_datasets (seed=42) to reproduce the exact test split ...")
    ds = prepare_datasets("data/raw/lichess_elite_subset.pgn", seed=42)
    t1 = time.time()
    print(f"  train: {len(ds['train'])}  val: {len(ds['val'])}  test: {len(ds['test'])}")
    print(f"  (took {t1-t0:.1f}s)")

    test_ds = ds["test"]
    y_true = list(test_ds.labels)

    # --- Neural model on the test split ---
    print("\nEvaluating trained checkpoint on test split ...")
    model = HistoryChessResNet()
    load_checkpoint("models/checkpoints/best_resnet.pt", model)
    model.eval()

    y_pred_nn = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            tensor, _mask, _label = test_ds[i]
            logits = model(tensor.unsqueeze(0))
            y_pred_nn.append(int(logits.argmax(-1).item()))
    t2 = time.time()
    print(f"  (took {t2-t1:.1f}s)")

    # --- Rule-only baseline on the exact same windows ---
    print("\nEvaluating rule-only baseline on the same test windows ...")
    engine = SymbolicRuleEngine()
    y_pred_rule = []
    for window in test_ds.windows:
        result = engine.evaluate_window(window.board_history)
        y_pred_rule.append(result.class_id)
    t3 = time.time()
    print(f"  (took {t3-t2:.1f}s)")

    labels_range = list(range(NUM_CLASSES))

    def report(name: str, y_pred: list[int]) -> None:
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, labels=labels_range, average="macro", zero_division=0)
        per_class = f1_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
        print(f"\n=== {name} ===")
        print(f"  accuracy:   {acc:.4f}")
        print(f"  macro-F1:   {macro_f1:.4f}")
        print("  per-class F1:")
        for cid, f1 in enumerate(per_class):
            print(f"    {ID_TO_NAME[cid]:16s} {f1:.4f}")
        return macro_f1

    nn_f1 = report("Neural model (HistoryChessResNet, epoch 5 checkpoint)", y_pred_nn)
    rule_f1 = report("Rule-only baseline (SymbolicRuleEngine)", y_pred_rule)

    print("\n=== Verdict ===")
    print(f"  neural macro-F1 - rule-only macro-F1 = {nn_f1 - rule_f1:+.4f}")
    if nn_f1 > rule_f1:
        print("  Neural model beats the rule-only baseline on held-out test data.")
    else:
        print("  WARNING: neural model does NOT beat the rule-only baseline.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
