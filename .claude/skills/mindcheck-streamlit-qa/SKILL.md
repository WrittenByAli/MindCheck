---
name: mindcheck-streamlit-qa
description: Streamlit application engineering, model-integration and quality assurance for MindCheck's interactive UI. Load when working on src/ui, wiring checkpoints into the app, or testing the end-to-end user flow.
---

# MindCheck Streamlit QA

## Application scope
- PGN upload + sample-game selection, ply-by-ply navigation, SVG board rendering, plan timeline, evidence panel, prediction scores.
- Load the **actual** trained checkpoint (`.pt`/`.onnx`) from disk; run **real** forward-pass inference through the model code in `src/models` / `src/tracking`.

## Error handling
- Missing-history (early plies < 8): show the padded/masked state honestly, don't fabricate history.
- Malformed PGN: surface a clear parse error, don't crash or silently drop the game.
- Missing checkpoint: show an explicit "model not available" state — do not fall back to rule-engine output pretending to be the neural prediction.
- Test on a clean install (fresh venv) before considering a feature done.

## Testing requirements
- Test the complete flow: upload → parse → window extraction → inference → tracking → explanation → render. Not just isolated widgets.
- Cover: malformed PGN, missing checkpoint, zero-history game start, and a normal full game.

## Hard restrictions
- Never simulate, mock, or hardcode a neural prediction in the UI — it must come from a real model forward pass.
- Never silently substitute rule-based inference for the neural model when the model/checkpoint is unavailable; report the failure instead.
- Report runtime failures and missing artifacts honestly in the UI and in status output — no optimistic masking of errors.
