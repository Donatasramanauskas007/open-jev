# Chess calibrated-decision experiment

This folder extends the existing chess LoRA experiment into a reproducible calibration benchmark. It compares the base model and saved chess adapter, each before and after validation-fitted temperature scaling. It reuses the existing data preparation and training commands without changing them.

The predicted event is **matching the reference puzzle solution among the listed legal moves**. Probabilities are not estimates of winning the game or proof that alternative moves are bad. This is the supervised/calibration baseline for future outcome-based RL experiments, not an implementation of TypeSafe's undisclosed RLCD algorithm.

## Run

From the repository root, with the existing environment, model, adapter, and chess splits:

```sh
# Audit all splits without loading a model or requiring MLX.
python3 -m chess_finetune.run --audit-only --out runs/chess-audit-001

# Small integration check. Results are explicitly marked as a smoke test.
.venv/bin/python -m chess_finetune.run --limit 8 --out runs/chess-smoke-001

# Full comparison: fit on valid.jsonl, evaluate on test.jsonl.
.venv/bin/python -m chess_finetune.run --out runs/chess-calibration-001

# Base model only, if no adapter is available.
.venv/bin/python -m chess_finetune.run --base-only --out runs/chess-base-001
```

Every output directory must be new. `--data`, `--model`, `--adapter`, and `--batch-size` customize inputs. The current saved adapter uses MLX and requires Apple silicon with Metal access. No training, download, or model mutation happens during evaluation.

If starting from scratch, the existing `make chess-data` and `make chess-train` prepare the legacy puzzle data and train the adapter. See [the original experiment](../finetune/README.md) for requirements. Do not rerun data preparation over a locked benchmark without assigning it a new dataset version.

## Method

1. Read and validate train/valid/test records. Audit cross-split positions, puzzle IDs, and optional source-game identifiers. Abort on overlap; do not silently remove or move examples.
2. Rank all listed moves with `OptionScorer`, using summed token log probabilities and a leading space before every UCI move, matching the existing training/evaluation convention.
3. Fit one positive temperature separately for each model on validation NLL. Search T in [0.05, 20], include T=1 as a candidate, and retain the temperature with lowest validation loss.
4. Evaluate untouched test labels using raw and temperature-scaled distributions. Positive temperature preserves the move ranking; any accuracy improvement comes from the adapter, not this calibrator.
5. Save raw scores, probabilities, timings, calibration artifacts, data/model hashes, and metrics.

No guarantee is made that validation-fitted scaling improves test calibration. A fitted boundary temperature suggests checking score scale and model quality. A tiny smoke test cannot establish a useful calibrator.

## Outputs

- `report.md`: comparison table and reliability-bin tables.
- `metrics.json`: accuracy/top-3, categorical NLL, multiclass Brier, top-label ECE, reliability bins, and selective error at several coverage levels.
- `{base,lora}-{valid,test}-predictions.jsonl`: per-position FEN, puzzle ID, candidate moves, gold index, raw scores, raw/calibrated probabilities, and scoring duration.
- `{base,lora}-calibrator.json`: fitted temperature, validation identity, objective, and link to the run manifest. Do not apply this temperature to another model, prompt, or normalization mode.
- `manifest.json`: configuration, model/adapter content hashes, dataset hashes, code revision/dirty status, platform, and installed library versions.
- `split-audit.json`: overlap checks and source-game metadata coverage.

Brier is summed across classes, then averaged across positions (range 0–2). ECE uses 10 equal-width confidence bins; empty bins have null statistics. Coverage calculations retain all predictions tied at the cutoff and report actual coverage. Forced single-move positions receive probability 1; aggregate and nonforced-only results are separated in the JSON report. Per-position timings are warmed scoring observations, not an end-to-end serving benchmark.

## Split limitations and experimental discipline

The legacy files split by puzzle. The new audit additionally checks the first four FEN fields (ignoring move clocks). It cannot prove source-game independence if `game_id`/`game_url` is absent, detect near-duplicates, or establish that a pretrained model never saw these puzzles. Preserve source-game IDs when producing a stronger dataset version and assign entire games to one split. Canonical FEN comparison is conservative, not a chess transposition detector.

The existing test set has already been inspected in earlier experiments. Use it for development comparisons; create a new locked test set before claiming a final research result. Keep a separate calibration partition if validation is also heavily used for model/prompt selection. The current pipeline intentionally does not change splits automatically.

## Next research step

Use this benchmark to establish whether the adapter improves decision quality and whether its uncertainty is useful. Then add an explicitly defined outcome-feedback experiment, with supervised and post-hoc calibration baselines retained. Game reward and win probability require a defined opponent, continuation policy, and outcome horizon; they must not silently replace the puzzle-label event.

## Offline checks

```sh
python3 -m unittest discover -s tests -p test_chess_calibration.py -v
```
