# Accuracy-efficiency evaluation

Compares fixed K, random allocation, repeated sampling, and adaptive inference at matched costs. This folder is separate from GPM and changes none of its code. See `results/REPORT.md` for current findings.

## Run the pilot

No GPU is needed. From this folder, with numpy, pandas, scikit-learn, and matplotlib installed:

```sh
python -m unittest -v test_evaluate.py
python evaluate.py --predictions ../sampling_benefit_diagnostic/results/predictions.npz --output results
python plot_results.py --results results
```

## Follow-up analyses

`extra_analyses.py` compares uncertainty scores, incremental batches of K = 16, and a two-stage K = 16 to K = 128 rule. It also needs no GPU.

```sh
python extra_analyses.py --predictions ../sampling_benefit_diagnostic/results/predictions.npz --output results
```

## Add real timing on Colab

Follow the steps at the top of `colab_timing.ipynb`. It installs GPM's packages with `colab_setup.py`, checks that the test nodes and accuracy match the saved predictions, runs `measure_timing.py`, and saves everything to Google Drive.

The resulting adaptive time is an estimate: measured time per K combined with replayed stopping decisions. Arvind's live run supplies the true adaptive time.

## Combine final runs

Run `evaluate.py` once for each dataset and checkpoint, then combine the output folders. Each run is written as `dataset:checkpoint=folder`.

```sh
python aggregate_runs.py --output combined computers:1=results_c1 computers:2=results_c2 computers:3=results_c3 wikics:1=results_w1 wikics:2=results_w2 wikics:3=results_w3
```

This writes `combined/combined_curve.csv` and `combined/combined_success.csv`. Results are matched by method and fixed setting or cost target, because adaptive thresholds differ between checkpoints.

## Files the adaptive script must produce

**Predictions, for final evaluation:** an `.npz` file with these fields.

| Field | Shape | Meaning |
|---|---|---|
| `probabilities` | [K, repetition, node, class] | Softmax outputs, for at least K = 16, 32, 64, 128 |
| `labels` | [node] | True classes |
| `node_ids` | [node] | Graph node IDs |
| `k_values` | [K] | Budgets, in the same order as the first axis |
| `node_split` | [node] | `val` or `test` |
| `nested` | scalar | True when each K uses the first K patterns of the same pool |

With `node_split` present, `evaluate.py` switches to final mode. It tunes on `val` nodes and reports on `test` nodes.

**Live adaptive decisions:** pass this file with `--adaptive-decisions`.

| Field | Shape | Meaning |
|---|---|---|
| `node_ids` | [node] | Graph node IDs |
| `final_k` | [repetition, node] | Budget where each node stopped: 16, 32, 64, or 128 |
| `predicted` | [repetition, node] | Final predicted class |
| `seconds_total` | scalar, optional | Wall-clock time for the whole run |

## Cost

The default unit is pattern-passes: patterns sent through GPM, counted every time. A node reaching K = 128 therefore costs 16 + 32 + 64 + 128 = 240 if every step reruns from scratch. `reuse_cost` in `curve.csv` is the optimistic figure if sampling and encoding are reused, and is not valid until the multiscale issue in the report is handled.
