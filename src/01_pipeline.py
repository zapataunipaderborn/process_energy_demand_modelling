"""
01_pipeline.py — Orchestrate multiple 02_modelling.py runs with different configurations.

Each entry in EXPERIMENTS defines one run. Fields:
  data_experiment       str        which gold data folder to load  (e.g. '1' → data/gold/experiment_1)
  run_name              str        label stored in info.json and used as the results sub-folder suffix
  processes_to_run      list       process folders to include
  temporal_resolution   str        'original' | '1min' | '5min' | '15min'
  run_process_modelling bool       whether to run the process modelling loop (default: False)
  mining_algorithms     list|None  which pm4py process-discovery algorithms to test/compare,
                                   e.g. ['heuristic', 'inductive']. Subset of
                                   'alpha' | 'heuristic' | 'inductive' | 'ilp'.
                                   None (default) → use 02_modelling.py's own default list.
  run_energy_modelling  bool       whether to run energy profile/curve modelling
                                   (sensor curve fitting + curve-quality benchmark).
                                   Set False to run process modelling only. Default: True.
  run_joint_duration_eval bool     whether to run the (slow) per-instance-matched
                                   "Joint Duration + Profile Evaluation" heatmaps.
                                   Superseded by the energy-distribution metrics
                                   (energy_distribution_results/), which don't rely
                                   on instance matching. Default: False.
  run_schedule_profile_eval bool   whether to run the "Schedule Profile Evaluation"
                                   (Best/mine vs. Schedule-direct vs. Stochastic
                                   generator, per process) — writes to
                                   schedule_profile_eval_results/. Default: False.
  run_autoregressive_eval bool     whether to add the autoregressive test-time
                                   rollout rows ("… (autoreg)") to the Curve-Only
                                   Evaluation. Only adds those extra comparison
                                   rows; affects nothing else. Default: False.
                                   Inert unless the lagged-energy prev-activity
                                   features are switched on explicitly
                                   (split_curves_with_prev_activity
                                   include_prev_energy=True), which reported runs
                                   do not do.
  save_predicted_curves bool       whether to also persist the actual real/predicted
                                   curve arrays (not just the aggregated Wasserstein
                                   distances) behind complete-curve eval and schedule
                                   profile eval, for recomputing other metrics or
                                   plotting later. Default: False.
  complete_curve_approaches list|None restrict the complete-curve eval to these
                                   approaches only. None (default) → every
                                   trained approach is assembled into complete
                                   case profiles and scored, per simulation
                                   mode. e.g. ['ml_step_dtw'] evaluates just
                                   that one — the complete-curve stage is
                                   (process × mode × approach) inference over
                                   every simulated case, so each dropped
                                   approach cuts that stage proportionally.
                                   The schedule-profile "Best, mine" comparator
                                   follows: it uses 'ml_external' when listed,
                                   otherwise the first listed approach.
  train_ratio           float      fraction of cases used for training in the
                                   train/test split (the rest go to test).
                                   e.g. 0.8 for an 80/20 split. Default: 0.70
                                   (02_modelling.py's own default, used when omitted).
  random_seed           int        master seed for the whole run — python, numpy,
                                   torch, the train/test split and every
                                   per-combo training seed derive from it, and
                                   PYTHONHASHSEED is pinned to it in the child
                                   process. Default: 42.
  split_type            str        how cases are assigned to train vs. test:
                                   'temporal' (default) — earliest train_ratio
                                   fraction of cases by start time → train, the
                                   rest (the "future") → test, no leakage.
                                   'random' — fixed-seed shuffle of case IDs at
                                   the same ratio, no time ordering. Same
                                   case-id partition is applied to every
                                   evaluation (process, energy/profile,
                                   schedule-profile) either way.
  curve_approaches      list|None  which curve-fitting models to train, e.g.
                                   ['ml_external']. Subset of the
                                   APPROACHES list in 02_modelling.py ('baseline'
                                   — "Baseline", ONE median curve per sensor
                                   (pooled over all activities, the naive
                                   floor); 'median_activity_sensor' — "Median
                                   per Activity & Sensor", median curve per
                                   sensor+activity+object, no model;
                                   'ml_dtw' — DBA + DTW +
                                   regression; 'ml_external',
                                   'ml_only', 'seq2seq', 'seq2seq_only',
                                   'seq2seq_external', 'seq2seq_iom' (the
                                   competing paper's own selection rule:
                                   generate + score against a softDTW reference
                                   every 10 epochs); plus the train/eval-gap
                                   variants of ml_external:
                                   'ml_external_wcounts',
                                   'ml_external_wmetric',
                                   'ml_external_calib', 'ml_rawspace' —
                                   see the EXPERIMENTS block for what each
                                   changes). None
                                   (default) → use 02_modelling.py's own list.
                                   Fewer models = much faster runs. The
                                   schedule-profile / complete-curve eval needs
                                   'ml_external' present.
  curve_models          list|None  which regressors compete for every curve
                                   approach, per (sensor, activity, object) —
                                   the best is kept by validation MAE. Subset of
                                   'Linear Regression' | 'Ridge' |
                                   'Random Forest' | 'XGBoost' |
                                   'Hist Gradient Boosting' | 'MLP' (short
                                   aliases also accepted: linear, ridge, rf,
                                   xgb, hgb, mlp). None (default) → all six
                                   compete. Fewer = faster: the curve stage
                                   trains one model per name per combo, times
                                   curve_n_optuna_trials.
  seq2seq_cells         list|None  which cells compete inside every seq2seq
                                   approach: 'lstm' and/or 'transformer'. Both
                                   are trained per combo and the lower
                                   validation loss wins. None (default) → both.
                                   Set ['lstm'] to reproduce a pre-transformer
                                   run or to halve the seq2seq cost.
  curve_optimize_hyperparams bool  whether curve models run the (slow) Optuna
                                   hyperparameter search. None → 02_modelling.py
                                   default (True). Set False for fast test runs.
  curve_n_optuna_trials int|None   Optuna trials per (sensor, activity, object)
                                   when the search is on. None → default (50).
  curve_loss str|None              Training loss for the sklearn curve models:
                                   'absolute_error' (default) fits the conditional
                                   MEDIAN, matching the L1 metrics (sMAE, WAPE)
                                   and the val-MAE model selection.
                                   'squared_error' restores the old conditional-
                                   MEAN behaviour for ablation.
  curve_select_by_realism bool|None which candidate regressor wins each
                                   (sensor, activity, object) leaf. None/False =
                                   validation MAE on the CANONICAL rows (the
                                   historical behaviour). True = the realism
                                   'Overall' the results notebooks report, scored
                                   by running each candidate through its REAL
                                   predictor (decode included) on the held-out
                                   curves and comparing per-curve features
                                   against the REAL curve values. Fixes the
                                   space/unit/metric mismatch between what is
                                   selected on and what is reported.
  curve_selection_metrics list|None which per-curve features are averaged into
                                   that 'Overall'. Subset of 'sum' | 'max' |
                                   'mean' | 'std' | 'roughness' | 'acf1'.
                                   None -> ['sum','max','mean','std']. Changing
                                   this changes the selection objective.
                                   Only used when curve_select_by_realism.
  dtw_shape_blind_decode bool|None decode canonical predictions back to the test
                                   timeline using ONLY the curve's length
                                   (linear resample), instead of DTW-warping
                                   against the curve's own values. Puts the DTW
                                   and non-DTW approaches on the same test-time
                                   information, so DTW is measured purely as a
                                   training-time representation. None → False
                                   (original behaviour).
  curve_median_floor    bool|None  fall back to the "Median per Activity &
                                   Sensor" curve whenever a trained approach
                                   fails to beat that leaf's own median on
                                   held-out curves. None -> False (OFF).
                                   Acceptance is on POINTWISE error, so turning
                                   it on deletes any method that trades
                                   pointwise accuracy for curve realism: it fired
                                   on 33/40 leaves for 'exemplar', replacing its
                                   real measured curve with the smooth median.
                                   Leave it off to see what each approach really
                                   predicts; turn it on only for a
                                   "never worse than the naive floor" run, and
                                   not alongside exemplar*/realism reporting.
  curve_median_floor_ratio float|None
                                   margin the model must win by: kept only when
                                   its held-out MAE < ratio x the floor's.
                                   None -> 1.0 (strictly better). 0.95 would
                                   demand a 5% improvement.
  step_dtw_fallback_ratio  float|str|None
                                   the ml_step_dtw* do-no-harm gate: a leaf is
                                   routed to its ml_external pipeline iff the
                                   step model's held-out MAE > ratio x
                                   ml_external's. None -> 1.25 (sim_extractor
                                   default). 'inf' DISABLES the gate — the
                                   step/smooth model is kept on every leaf,
                                   even where it clearly loses. Note the gate
                                   is silently inert anyway when ml_external
                                   is not in curve_approaches (nothing to
                                   route to) — that, not retraining noise, is
                                   why the smooth rows differed between
                                   experiment_999 and experiment_1000.
  save_curve_values     bool|None  persist the full predicted / real curve on
                                   every scored curve in curve_eval_results.parquet,
                                   so any metric can be computed offline later
                                   without re-running. None → True (ON). The per-curve
                                   SHAPE STATISTICS (std / lag-1 acf / zero
                                   fraction / peak, for both the real and the
                                   predicted curve) are written either way — those
                                   are what the realism measures need, and they
                                   cost ~10 floats a row. Turn this on only when
                                   you want arbitrary future metrics: it inflates
                                   the file by roughly the curve length per row.
"""

import os
import subprocess
import sys
from pathlib import Path

setting = True

# ── Experiment definitions ────────────────────────────────────────────────────
EXPERIMENTS = [

    # ── experiment_1001: 1000's full comparison + seq2seq_only ────────────────
    # FIRST of two queued runs. Identical to experiment_1000 (same data, split,
    # seed, tuning: six model families x 10 Optuna trials, gate OFF) with
    # 'seq2seq_only' added, so the neural baseline sits in the same tables as
    # everything else. seq2seq trains on CPU here (CUDA_VISIBLE_DEVICES='' in
    # the child env — the fork-deadlock fix), so expect this run to be MUCH
    # longer than 1000; it is queued first so its table is complete before the
    # iom run starts.
    {
        'data_experiment':       '1',
        'run_name':              'experiment_1001',
        'processes_to_run':      ['process_1', 'process_2', 'process_3',
                                  'process_4_1', 'process_4_2', 'process_5'],
        'temporal_resolution':   '1min',
        'run_process_modelling': True,
        'mining_algorithms':     ['heuristic', 'alpha'],
        'run_energy_modelling':  True,
        'run_joint_duration_eval':   False,
        'run_schedule_profile_eval': True,
        'run_autoregressive_eval':   False,
        'save_predicted_curves': True,
        'train_ratio':           0.70,
        'split_type':            'temporal',
        'random_seed':           42,     # identical split to 999/1000, so all
                                         # tables are comparable row for row
        'curve_approaches': [
            'baseline',
            'median_activity_sensor',
            'ml_external',
            'ml_only',
            'ml_step_dtw_smooth',
            'exemplar',
            'seq2seq_only',
        ],
        'curve_optimize_hyperparams': True,
        'curve_n_optuna_trials': 10,
        # Complete-profile stage: smooth (~40 s per process_1 combo, per-instance
        # model calls), baseline (~1 s, median resample), seq2seq_only (network
        # forward passes per instance).
        'complete_curve_approaches': ['ml_step_dtw_smooth', 'baseline',
                                      'seq2seq_only'],
        'curve_median_floor':    False,
        'save_curve_values':     True,
        'step_dtw_fallback_ratio': 'inf',  # gate OFF: the smooth rows show the
                                           # step model everywhere, no ml_external
                                           # rescue on its weak leaves
    },

    # ── experiment_1002: seq2seq_iom alone ────────────────────────────────────
    # SECOND of the two: the competing paper's approach with its own selection
    # rule (generate + score against a softDTW reference every 10 epochs), and
    # nothing else. Same data/split/seed as 1001, so its rows drop straight into
    # the same comparison.
    {
        'data_experiment':       '1',
        'run_name':              'experiment_1002',
        'processes_to_run':      ['process_1', 'process_2', 'process_3',
                                  'process_4_1', 'process_4_2', 'process_5'],
        'temporal_resolution':   '1min',
        'run_process_modelling': True,
        'mining_algorithms':     ['heuristic', 'alpha'],
        'run_energy_modelling':  True,
        'run_joint_duration_eval':   False,
        'run_schedule_profile_eval': True,
        'run_autoregressive_eval':   False,
        'save_predicted_curves': True,
        'train_ratio':           0.70,
        'split_type':            'temporal',
        'random_seed':           42,
        'curve_approaches': [
            'seq2seq_iom',
        ],
        'curve_optimize_hyperparams': True,
        'curve_n_optuna_trials': 10,
        'complete_curve_approaches': ['seq2seq_iom'],
        'curve_median_floor':    False,
        'save_curve_values':     True,
    },

    # ── experiment_999 + experiment_1000 (both ran 2026-07-29/30) ─────────────
    # # 999: ml_step_dtw_smooth alone (tuned: six families x 10 trials) on all
    # #      six processes; no ml_external, so the do-no-harm gate was inert.
    # # 1000: the full comparison (baseline, median_activity_sensor, ml_external,
    # #      ml_only, ml_step_dtw_smooth, exemplar), same split/seed as 999.
    # #      NOTE the smooth rows differ between the two BECAUSE of the gate:
    # #      with ml_external present it rescued ~39 weak leaves (sMAE mean
    # #      4.47 vs 5.43, medians ~equal). Superseded by 1001's config, which
    # #      pins the gate OFF ('step_dtw_fallback_ratio': 'inf').

    # ── experiment_997 (superseded by 998) ────────────────────────────────────
    # # ── experiment_996: canonical smooth run (global-warp reconstruction) ─────
    # # Clean rerun after the 995 detour. 995 (DEAD END — do not cite) swapped the
    # # smooth variant's global medoid warp for per-segment resampling +
    # # interpolated gains; it RE-TERRACED the simulated heat declines (tread
    # # pairs + ~1050 kW cliffs vs the warp's steady ~500 kW/min; near-flat-frac
    # # 0.50 vs 0.35) because the leaf medoid is too coarse for any per-segment
    # # replay once instances run longer than it. The warp also costs nothing on
    # # levels (steam Wasserstein value median 73.3 warp vs 74.6 per-segment).
    # # sim_extractor was reverted to the warp, so this run should reproduce
    # # 994's outputs (same data/split/seed/code path).
    # #
    # # (994 context:) first run on the process_1 data regenerated
    # # 2026-07-29 with the holding duty tied to the schedule. Two generator
    # # changes measured there, both in simulation_process/:
    # #   * hold level: sterilization_profile now takes a `hold_duty_factor`, and
    # #     generate_process_1 derives it from how long that autoclave sat idle since
    # #     its own previous cycle (1 - 0.35*exp(-idle_min/480)). A vessel that starts
    # #     straight after the last batch is still hot and loses less. The gap is
    # #     written to every event as object_attributes['idle_min'], so it is IN the
    # #     feature frame — the holding level went from 0% explainable (pure A_U
    # #     nuisance) to R^2 0.74-0.81 against that driver.
    # #   * duration: AUTOCLAVE_CYCLE_VARIABILITY 0.15 -> 0.07 and LOAD_FIXED_FRAC
    # #     0.30 -> 0.20, lifting R^2(volume -> cycle duration) from 0.60 to 0.93.
    # # NOT weather: WEATHER_ENERGY_COUPLING stays False, so the ef_* columns still
    # # carry no signal for this process.
    # # Hold energy is ~18 kW here against ~22 kW in 993 — nominal A_U is now the
    # # COLD-vessel case — so the hold rows are not level-comparable with 993/991.
    # # Same scope, split and seed as 993 so everything else is.
    # {
    #     'data_experiment':       '1',
    #     'run_name':              'experiment_997',
    #     'processes_to_run':      ['process_1'],
    #     'temporal_resolution':   '1min',
    #     'run_process_modelling': True,
    #     'mining_algorithms':     ['heuristic'],
    #     'run_energy_modelling':  True,
    #     'run_joint_duration_eval':   False,
    #     'run_schedule_profile_eval': True,   # notebooks 05 and 07 read
    #                                          # schedule_profile_eval_results/<proc>/predicted_curves.parquet
    #     'run_autoregressive_eval':   False,
    #     'save_predicted_curves': True,
    #     'train_ratio':           0.70,
    #     'split_type':            'temporal',
    #     'random_seed':           42,
    #     'curve_approaches': [
    #         #'baseline',
    #         #'ml_step_dtw',
    #         'ml_step_dtw_smooth',
    #     ],
    #     'curve_optimize_hyperparams': False,
    #     'complete_curve_approaches': ['ml_step_dtw'],#, 'ml_step_dtw_smooth', 'baseline'],
    #     'curve_median_floor':    False,
    #     'save_curve_values':     True,
    # },

    # ── experiment_993 (superseded by 994 — same config, pre-schedule-coupling data)
    # # ── experiment_991: process_1-only check of the step-DTW fixes ────────────
    # # Re-runs ONLY the synthetic process after (a) the segment-fill
    # # interpolation fix in sim_extractor.predict_raw_curve_step_dtw (stretched
    # # segments no longer staircase when upsampled past the reference's
    # # resolution) and (b) — IF the generator task in
    # # simulation_process/opus_task_predictable_levels.md has been executed and
    # # data/gold/experiment_1/process_1 regenerated — volume-driven autoclave
    # # levels. Without (b) this run measures the smoothing fix alone.
    # # Lean scope: exactly what notebook 06's combined figure and the
    # # complete-profile metrics need — mining + simulation + curve eval +
    # # predicted curves. No joint-duration heatmaps, no schedule-profile stage,
    # # no seq2seq. Same data/split/seed as 988/989, so the process_1 rows drop
    # # straight into their comparison tables.
    # {
    #     'data_experiment':       '1',
    #     'run_name':              'experiment_993',   # 992 (regenerated process_1 data of
    #                                                  # 2026-07-29) + ml_step_dtw_smooth
    #                                                  # side-by-side comparison
    #     'processes_to_run':      ['process_1'],
    #     'temporal_resolution':   '1min',
    #     'run_process_modelling': True,   # the (b) simulated panels need the nets + logs
    #     'mining_algorithms':     ['heuristic'],   # single miner — fast check run
    #     'run_energy_modelling':  True,
    #     'run_joint_duration_eval':   False,  # slow per-instance heatmaps — not this run's job
    #     'run_schedule_profile_eval': True,   # ON: notebooks 05 and 07 read
    #                                          # schedule_profile_eval_results/<proc>/predicted_curves.parquet
    #                                          # (05 takes its 'real' reference series from there too)
    #     'run_autoregressive_eval':   False,
    #     'save_predicted_curves': True,   # gates the complete-curve eval notebook 06 reads
    #                                      # (predicted_curves_ml_step_dtw.parquet +
    #                                      # predicted_logs + complete-curve metrics)
    #     'train_ratio':           0.70,
    #     'split_type':            'temporal',
    #     'random_seed':           42,     # bit-identical split to 988/989
    #     'curve_approaches': [
    #         'baseline',                 # kept: essentially free (one median curve per
    #                                     # sensor, no ML) and required so the complete-curve
    #                                     # 'Baseline' row exists (988 listed ml_step_dtw
    #                                     # alone and that row came out '--')
    #         'ml_step_dtw',              # the approach under test. NOTE: with ml_external
    #                                     # absent, the do-no-harm fallback gate has nothing
    #                                     # to route to and is inactive — every leaf keeps
    #                                     # the pure step model, which is the point here
    #         'ml_step_dtw_smooth',       # same training, gains interpolated at
    #                                     # reconstruction (no boundary jumps) — the
    #                                     # staircase fix, run side by side so the smoothing
    #                                     # can be judged (and dropped) in isolation
    #     ],
    #     'curve_optimize_hyperparams': False,  # OFF for speed — no-op for both approaches
    #                                           # anyway (fixed models)
    #     'complete_curve_approaches': ['ml_step_dtw', 'ml_step_dtw_smooth', 'baseline'],
    #     'curve_median_floor':    False,  # must stay OFF (see 989's note)
    #     'save_curve_values':     True,   # keep y_true/y_pred for offline realism metrics
    # },

    # ── Reportable comparison: Step DTW vs incumbents vs literature ─────────
    # experiment_984 established ml_step_dtw (std/rough ratios 0.74/0.67 vs
    # exemplar's 0.59/0.50, sMAE mean 4.05 beating exemplar 4.12 and 982's
    # ml_external 4.43, best tail q99 43.5). This run assembles the reportable
    # table in ONE eval pass:
    #   the two naive floors, the pointwise incumbent (ml_external — also the
    #   target ablation for ml_step_dtw: same features, per-position target),
    #   the literature baseline 'seq2seq' (DTW + encoder-decoder, the
    #   competing-paper approach ml_step_dtw has to beat), the realism
    #   reference 'exemplar', and ml_step_dtw itself.
    # Also turns the (slow) per-instance-matched Joint Duration + Profile
    # Evaluation ON so the complete-profile heatmaps come out of this run too.
    # {
    # 'data_experiment':       '1',
    # 'run_name':              'experiment_989',
    # 'processes_to_run':      ['process_1', 'process_2', 'process_3',
    # 'process_4_1', 'process_4_2', 'process_5'],
    # 'temporal_resolution':   '1min',
    # 'run_process_modelling': True,
    # 'mining_algorithms':     ['heuristic', 'alpha', 'inductive'],
    # 'run_energy_modelling':  True,
    # 'run_joint_duration_eval':  True,   # ON: full per-instance-matched
                                            # complete-profile evaluation (slow)
    # 'run_schedule_profile_eval': True,  # keys off 'ml_external', which is trained
    # 'run_autoregressive_eval':   False,
    # 'save_predicted_curves': True,
    # 'train_ratio':           0.70,
    # 'split_type':            'temporal',
    # 'random_seed':           42,     # same seed as the full runs, so the
                                         # train/test split is bit-identical
    # 'curve_approaches': [
    # 'baseline',                 # coarsest floor: ONE median per sensor
    # 'median_activity_sensor',   # the per-leaf naive floor
    # 'ml_external',              # the incumbent per-position predictor — the
                                        # direct parent; same curve set, same features,
                                        # so the gap is attributable to the segment
                                        # reparameterisation alone
    # 'ml_only',                  # plain ML on the curve: linear resample encode +
                                        # decode, no DBA and no DTW anywhere. The no-DTW
                                        # ablation — pairs with ml_dtw the way
                                        # seq2seq_only pairs with seq2seq, so the two
                                        # together price what DTW is worth on its own
            #'seq2seq',                  # DTW + encoder-decoder — the competing-paper
                                        # baseline ml_step_dtw has to beat; LSTM and
                                        # transformer cells compete per leaf (see
                                        # 'seq2seq_cells' to pin one)
            #'seq2seq_iom',              # OFF after experiment_986: the seq2seq worker
                                        # pool deadlocked on fork (7h, all workers in
                                        # futex_wait). The CUDA-hiding env fix below is
                                        # in place but unvalidated — run the seq2seq
                                        # family in a separate dedicated run first, so
                                        # a repeat cannot take the main table with it.
                                        # (Was: the competing paper run THEIR way —
                                        # vanilla targets + IOM selection against a
                                        # softDTW reference, full epoch budget.)
    # 'ml_step_dtw',              # segment durations+levels via DTW
                                        # correspondence, reconstruction instead of decode
    # 'exemplar',                 # the realism reference (real replayed curve)
    # ],
    # 'curve_optimize_hyperparams': True,   # Optuna search per (sensor, activity,
                                              # object) — publication-grade fits
    # 'curve_n_optuna_trials': 10,          # 10 trials like the previous full
                                              # reportable runs; the default 50 is a
                                              # ~5x multiplier on the whole ml_external
                                              # training stage for marginal gains.
                                              # Note: ml_step_dtw/exemplar have fixed
                                              # models and ignore the search entirely.
    # 'complete_curve_approaches': ['ml_step_dtw', 'baseline'],
                                         # complete-profile coupling for the new method
                                         # PLUS the naive floor — every approach still gets
                                         # the Curve-Only Evaluation, but only these two are
                                         # assembled into complete case profiles per mode,
                                         # cutting the simulation-eval stage to 2/N.
                                         # Comment this line out to evaluate all of them.
                                         # 'baseline' is what the complete-profile notebook
                                         # (05) reports as its Baseline row: the same
                                         # simulated process with ONE median curve per
                                         # sensor, so the Baseline-vs-ml_step_dtw gap is
                                         # attributable to the curve generator alone. The
                                         # 988 run of 2026-07-28 listed ml_step_dtw only,
                                         # which is why that row came out empty ('--').
                                         # It costs one more pass of
                                         # (process × mode) inference over every simulated
                                         # case (~1.5-2h at this run's size).
                                         # 'Best, mine' in the schedule-profile eval then
                                         # sources ml_step_dtw instead of ml_external.
    # 'curve_median_floor':    False,  # must stay OFF: the floor accepts on
                                         # POINTWISE error, so it would delete the
                                         # realism gain this approach exists for
    # 'save_curve_values':     True,   # keep y_true/y_pred so the realism
                                         # metrics can be recomputed offline
    # },

    # ── Dedicated seq2seq_iom run — the isolation the 988 comment asks for ────
    # experiment_986 lost 7 hours to the fork deadlock (all 16 seq2seq workers in
    # futex_wait) and the CUDA-hiding env fix in the runner below has never been
    # exercised. This run exists to exercise it on the cheapest configuration that
    # still produces the literature baseline's numbers, and it is deliberately
    # SEPARATE so a repeat of that hang cannot take the reportable table with it.
    #
    # Curve-Only Evaluation exclusively: no mining, no simulation, no
    # complete-profile or schedule-profile stage — those need neither seq2seq_iom
    # nor each other, and every one of them is a multiple of the curve stage.
    # Same data_experiment / processes / train_ratio / split_type / random_seed as
    # 988, so the train/test partition is bit-identical and the resulting
    # curve-only rows drop straight into 988's comparison table.
    #
    # NOTE on 'curve_optimize_hyperparams': Optuna only ever touches the sklearn
    # curve workers — seq2seq trains from the SEQ2SEQ_* module constants in
    # 02_modelling.py (hidden 128, 2 layers, 80 epochs, patience 10), which no
    # pipeline key overrides. The flag is set False anyway so the run stays
    # tuning-free if an sklearn approach is ever added to the list.
    #
    # To run ONLY this one, comment out the experiment_988 block above —
    # EXPERIMENTS runs top to bottom and 988 is a ~13h job.
    # {
    # 'data_experiment':       '1',
    # 'run_name':              'experiment_989_seq2seq_iom',
    # 'processes_to_run':      ['process_1', 'process_2', 'process_3',
    # 'process_4_1', 'process_4_2', 'process_5'],
    # 'temporal_resolution':   '1min',
    # 'run_process_modelling': False,  # curve comparison only — no mining, no simulation
    # 'run_energy_modelling':  True,
    # 'run_joint_duration_eval':    False,
    # 'run_schedule_profile_eval':  False,  # needs a complete-curve approach; not this run's job
    # 'run_autoregressive_eval':    False,
    # 'save_predicted_curves': False,  # nothing downstream of the curve eval to persist for
    # 'train_ratio':           0.70,
    # 'split_type':            'temporal',
    # 'random_seed':           42,     # identical split to 988 — the rows are comparable
    # 'curve_approaches':      ['seq2seq_iom'],   # the competing paper run THEIR way:
                                         # vanilla targets, model SELECTED every 10 epochs by
                                         # generating whole curves and scoring them against a
                                         # softDTW barycenter (MSE + sigma length)
    # 'seq2seq_cells':         ['lstm'],   # Woerrlein & Strassburger's own cell, and it
                                             # halves the cost. Drop this line to let the
                                             # transformer compete per leaf as in 988.
    # 'curve_optimize_hyperparams': False,  # see the NOTE above — no-op for seq2seq today
    # 'save_curve_values':     True,   # keep y_true/y_pred so the realism metrics can be
                                         # recomputed offline without re-running
    # 'curve_median_floor':    False,  # same reason as 988: the floor accepts on POINTWISE
                                         # error and would overwrite the texture being measured
    # },

    # ── Previous full run (kept for reference; uncomment to re-run) ───────────
    # {
    #     'data_experiment':       '1',
    #     'run_name':              'experiment_981',
        # FULL REPORTABLE RUN: all 6 processes, heuristic + alpha miners, Optuna
        # hyperparameter search ON. Tests the budget over-generation fix
        # (utils/simulation.py BUDGET_EXIT_DISCOUNT=0.02 while under budget,
        # BUDGET_EXIT_BOOST=25.0 once spent, TIME backstop BUDGET_MAX_TIME_RATIO=1.5;
        # the activity-count cap BUDGET_MAX_LENGTH_RATIO is OFF, see
        # BUDGET_USE_LENGTH_CAP) and the DBA zero-calibration
        # (sim_extractor._zero_calibrate_barycenter) end-to-end, and is directly
        # comparable to experiment_944 (same split, same miners) for the
        # before/after leakage check.
    #     'processes_to_run':      ['process_1', 'process_2', 'process_3', 'process_4_1', 'process_4_2', 'process_5'],
    #     'temporal_resolution':   '1min',
    #     'run_process_modelling': True,
    #     'mining_algorithms':     ['heuristic', 'alpha', 'inductive'],#, 'alpha'],#, 'alpha'],   # matches experiment_944 for apples-to-apples comparison
    #     'run_energy_modelling':  setting,    # ON: needed so the curve pipelines exist for the schedule-profile "Best, mine" comparison
    #     'run_joint_duration_eval': False, # slow, per-instance-matched heatmaps; superseded by energy_distribution_results
    #     'run_schedule_profile_eval':setting, # ON: Best/mine vs. Schedule-direct vs. Stochastic generator, per process
    #     'run_autoregressive_eval': False, # OFF: skip the "… (autoreg)" curve-eval rows
    #     'save_predicted_curves': setting, # ON: persists predicted_curves.parquet the schedule-profile comparison reads
    #     'train_ratio':           0.70, # fraction of cases used for training (e.g. 0.8 for 80/20)
    #     'split_type':            'temporal',#'temporal', # 'temporal' (default, no leakage) or 'random' (fixed-seed shuffle)
        # All standard curve approaches (the uncommented defaults in
        # 02_modelling.py's APPROACHES list — same set experiment_944 produced),
        # renamed to a consistent 'ml_*' prefix for every sklearn/DTW-regression
        # approach, with each paired against its seq2seq counterpart (see the
        # full reference table below). NOTE: this trains ~10 curve families per
        # (sensor, activity, object) instead of one, so with Optuna ON this is a
        # big runtime multiplier — the seq2seq_* families especially are slow.
        # The schedule-profile / complete-curve eval still keys off
        # 'ml_external'.
    #     'curve_approaches':      [
    #         'baseline',              # "Baseline": ONE median curve per SENSOR, pooled over all activities/objects (naive floor)
    #         'median_activity_sensor', # "Median per Activity & Sensor": median curve per (sensor, activity, object), no model
    #         'ml_dtw',                # DBA barycenter + DTW alignment + regression (formerly just 'baseline')
    #         'ml_external',  # DBA + DTW + regression + external factors + prev-activity NAME (no lagged energy)
    #         'ml_only',                # no DTW at all (formerly 'ml_linear')
    #         'exemplar',               # real training curve (DTW medoid of a shape cluster),
                                      # picked by a classifier over attributes + ef_*, scaled
                                      # by an L1 level model — the only approach that never
                                      # averages, so the only one whose output keeps realistic
                                      # texture. Expect WORSE sMAE and a far better std ratio.
    #         'ml_cluster_dtw',         # 'ml_external' fitted PER DTW shape cluster and decoded
                                      # through a warp PREDICTED from the attributes instead of
                                      # a uniform resample. Keeps exemplar_dtw's two structural
                                      # DTW uses (elastic clustering, learned warp) but every
                                      # output value comes from a regression, so the curve is
                                      # novel instead of a replayed one. Expect sMAE at or a
                                      # little better than 'ml_external', and a std ratio
                                      # between it and 'exemplar' — a conditional mean stays
                                      # smoother than any single real curve.

            # ── Train/eval-gap variants of 'ml_external' ──────────────────────
            # Every canonical approach FITS in barycenter space (DTW-warped,
            # position-averaged) but is SCORED pointwise on the real curve after
            # a decode the model never saw. Each of these changes exactly one
            # thing about that gap, so each is attributable against 'ml_external'
            # — keep 'ml_external' enabled or there is nothing to compare to.
            #'ml_external_wcounts',    # row weight = how many raw samples the DTW path folded
                                      # into that canonical position (they are not equally
                                      # informative, but the fit treats them as if they were)
    #         'ml_external_wmetric',    # ... additionally divided by the curve's own sigma,
                                      #which is exactly what sMAE divides residuals by
    #         'ml_external_calib',      # (gain, offset) fitted in RAW space after the decode,
                                      # correcting the encode's flattening of peaks — the
                                      # canonical loss cannot see that bias at all
    #         'ml_rawspace',            # no encode/decode: one row per raw sample, DTW/DBA
                                      # demoted from target transform to feature. Fitted
                                      # objective == evaluated objective. Also immune to
                                      # dtw_shape_blind_decode, since it has no decode.

    #         'seq2seq',
    #         'seq2seq_only',
    #         'seq2seq_external',
    #     ],
        # Full reference — every valid approach name (uncomment to enable the
        # experimental ones, which are commented out in 02_modelling.py by
        # default). Kept here so the whole universe is togglable in future.
        # Naming: every sklearn/DTW-regression approach now starts with 'ml_'.
        # One-to-one pairing with the seq2seq family (same conditioning, same
        # DTW/no-DTW alignment choice, different regressor):
        #   ml_dtw                 <-> seq2seq
        #   ml_only                <-> seq2seq_only
        #   ml_external  <-> seq2seq_external
        # The two median baselines sit outside that chain as the naive floors
        # everything else should beat: 'baseline' = "Baseline" (ONE median per
        # sensor, pooled over all activities/objects — the coarsest floor),
        # 'median_activity_sensor' = "Median per Activity & Sensor" (median per
        # sensor+activity+object):
        #   'baseline', 'median_activity_sensor', 'ml_dtw', 'ml_external',
        #   'ml_only', 'exemplar', 'exemplar_only', 'exemplar_dtw', 'ml_cluster_dtw',
        #   'seq2seq', 'seq2seq_only', 'seq2seq_external'
        # 'exemplar' sits outside the ml_*/seq2seq pairing: it is the only
        # approach that predicts a REAL measured curve rather than an average of
        # them, so it is the one to look at when the question is whether the
        # output resembles a real load profile rather than whether it minimises
        # pointwise error.
        # Regressors competed for every curve approach, per (sensor, activity,
        # object); best kept by validation MAE. Comment a line out to turn that
        # model off. None / omitted = all six.
    #     'curve_models': [
    #         'Linear Regression',   # unpenalised OLS reference (nothing to tune —
                                   # its Optuna trials are all identical)
    #         'Ridge',               # penalised counterpart; the one-hot design
                                   # matrix is high-dimensional and collinear
            #'Random Forest',
    #         'XGBoost',             # the gradient-boosting member
    #         'Hist Gradient Boosting',  # native-categorical, histogram-binned GB
    #         'MLP',                 # feed-forward net (sklearn MLPRegressor)
    #     ],
        # Cells competed inside every seq2seq approach; lower validation loss
        # wins. ['lstm'] reproduces the pre-transformer behaviour.
    #     'seq2seq_cells': ['lstm', 'transformer'],
    #     'curve_optimize_hyperparams': True,  # ON: proper tuned run (slow, publication-grade)
    #     'curve_n_optuna_trials': 10,         # trials per (sensor, activity, object)

        # ── Which candidate regressor wins each leaf ──────────────────────────
        # False (historical): validation MAE on the CANONICAL rows — a warped,
        #   position-averaged surrogate, scored per ROW, in raw units.
        # True: the realism 'Overall' the results notebooks report — each
        #   candidate is run through its REAL predictor (decode included) on the
        #   held-out curves and its per-curve features are compared against the
        #   REAL curve values. Selection then measures what the paper measures.
        # Cheap while dtw_shape_blind_decode is True (the decode is a linear
        # resample); turning that off makes this a DTW per candidate per curve.
    #     'curve_select_by_realism': True,
        # Which per-curve features are averaged into that 'Overall'. Subset of
        # 'sum' | 'max' | 'mean' | 'std' | 'roughness' | 'acf1'. Changing this
        # changes the selection objective -- e.g. ['std'] alone selects purely
        # for reproducing the amount of variation, ['std', 'roughness'] adds
        # "and by real dynamics rather than added noise".
    #     'curve_selection_metrics': ['sum', 'max', 'mean', 'std', 'roughness'],
        # Fall back to the per-(sensor, activity, object) median curve whenever a
        # trained approach loses to it on held-out curves. OFF: acceptance is on
        # pointwise error, so it deletes the realism-oriented approaches (it fired
        # on 33/40 leaves for 'exemplar'). Flip to True for a "never worse than the
        # naive floor" run — but then do not read the realism tables from it.
    #     'curve_median_floor': False,
        # Persist the predicted curve for EVERY scored test curve, so any metric
        # can be computed from the saved results instead of by re-running.
        # y_pred goes to curve_eval_results.parquet; the REAL curves are written
        # once to real_test_curves.parquet (they are shared by every approach) and
        # the results notebooks re-join them. Set False if a run needs a small file.
    #     'save_curve_values': True,
        # 'curve_median_floor_ratio': 1.0,   # margin required; 0.95 = beat it by 5%
    # },

    # ── Exemplar comparison ──────────────────────────────────────────────────
    # Uncomment this dict to run it. Trains ONLY the cheap approaches plus the
    # new 'exemplar' one, so it finishes in a fraction of the full run and is a
    # direct answer to "does the exemplar method produce a curve that looks
    # real, and what does that cost in pointwise error?".
    #
    # What to expect, measured over 90 (sensor, activity, object) leaves and
    # 4,187 test curves on data_experiment 1 with this same 70/30 temporal split:
    #
    #     approach                 sMAE    WAPE %   std ratio (want 1.0)
    #     median_activity_sensor   0.845   7.70     0.110
    #     ml_dtw                   0.848   7.70     0.133
    #     ml_external              0.852   7.71     0.135
    #     ml_only                  0.844   7.62     0.086
    #     exemplar                 1.018   8.92     0.738
    #
    # i.e. the exemplar curve carries ~74% of the real curves' standard
    # deviation against ~9-14% for every averaging approach, at a cost of ~20%
    # on sMAE. That trade is the whole point: sMAE rewards a smooth line through
    # the middle of a spiky signal, so it cannot see the difference between a
    # usable load profile and a flat one. Report the two together.
    #
    # dtw_shape_blind_decode is left at its default (True) — no approach may use
    # the test curve's own values to decode, which is the only setting under
    # which these columns are comparable to each other or reachable in simulation.
    #
    # {
    #     'data_experiment':       '1',
    #     'run_name':              'experiment_970_exemplar',
    #     'processes_to_run':      ['process_1', 'process_2', 'process_3',
    #                               'process_4_1', 'process_4_2', 'process_5'],
    #     'temporal_resolution':   '1min',
    #     'run_process_modelling': False,   # curve comparison only — no mining needed
    #     'run_energy_modelling':  True,
    #     'run_schedule_profile_eval': True,
    #     'save_predicted_curves': True,    # needed to plot the curves and see the texture
    #     'train_ratio':           0.70,
    #     'split_type':            'temporal',
    #     'curve_approaches':      [
    #         'median_activity_sensor',   # the floor everything is measured against
    #         'ml_dtw',                   # the incumbent DTW + regression approach
    #         'ml_external',              # + external factors, the strongest incumbent
    #         'exemplar',                 # the new one
    #     ],
    #     'curve_models': ['Ridge', 'Hist Gradient Boosting'],
    #     'curve_optimize_hyperparams': False,
    #     'save_curve_values': True,   # keep every y_true/y_pred so new metrics
    #                                  # can be computed from the parquet later
    # },

]

# ── Runner ────────────────────────────────────────────────────────────────────
modelling_script = Path(__file__).parent / '02_modelling.py'

for i, exp in enumerate(EXPERIMENTS, start=1):
    mining_algorithms = exp.get('mining_algorithms')
    run_energy_modelling = exp.get('run_energy_modelling', True)
    run_joint_duration_eval = exp.get('run_joint_duration_eval', False)
    run_schedule_profile_eval = exp.get('run_schedule_profile_eval', False)
    run_autoregressive_eval = exp.get('run_autoregressive_eval', False)
    save_predicted_curves = exp.get('save_predicted_curves', False)
    curve_approaches = exp.get('curve_approaches')
    complete_curve_approaches = exp.get('complete_curve_approaches')
    curve_models = exp.get('curve_models')
    seq2seq_cells = exp.get('seq2seq_cells')
    curve_optimize_hyperparams = exp.get('curve_optimize_hyperparams')
    curve_n_optuna_trials = exp.get('curve_n_optuna_trials')
    curve_loss = exp.get('curve_loss')
    curve_select_by_realism = exp.get('curve_select_by_realism')
    curve_selection_metrics = exp.get('curve_selection_metrics')
    dtw_shape_blind_decode = exp.get('dtw_shape_blind_decode')
    save_curve_values = exp.get('save_curve_values')
    curve_median_floor = exp.get('curve_median_floor')
    curve_median_floor_ratio = exp.get('curve_median_floor_ratio')
    step_dtw_fallback_ratio = exp.get('step_dtw_fallback_ratio')
    train_ratio = exp.get('train_ratio')
    split_type = exp.get('split_type', 'temporal')
    random_seed = int(exp.get('random_seed', 42))
    if split_type not in ('temporal', 'random'):
        print(f"[pipeline] ERROR: {exp['run_name']} has split_type={split_type!r}, "
              f"must be 'temporal' or 'random'. Stopping.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Running experiment {i}/{len(EXPERIMENTS)}: {exp['run_name']}")
    print(f"  data_experiment={exp['data_experiment']}  "
          f"temporal_resolution={exp['temporal_resolution']}  "
          f"processes={exp['processes_to_run']}  "
          f"run_process_modelling={exp.get('run_process_modelling', False)}  "
          f"mining_algorithms={mining_algorithms or '(default)'}  "
          f"run_energy_modelling={run_energy_modelling}  "
          f"run_joint_duration_eval={run_joint_duration_eval}  "
          f"run_schedule_profile_eval={run_schedule_profile_eval}  "
          f"run_autoregressive_eval={run_autoregressive_eval}  "
          f"save_predicted_curves={save_predicted_curves}  "
          f"curve_approaches={curve_approaches or '(default)'}  "
          f"curve_models={curve_models or '(all)'}  "
          f"seq2seq_cells={seq2seq_cells or '(all)'}  "
          f"curve_optimize_hyperparams={curve_optimize_hyperparams if curve_optimize_hyperparams is not None else '(default)'}  "
          f"curve_loss={curve_loss or '(default absolute_error)'}  "
          f"curve_select_by_realism={curve_select_by_realism if curve_select_by_realism is not None else '(default False)'}  "
          f"curve_selection_metrics={curve_selection_metrics or '(default sum,max,mean,std)'}  "
          f"dtw_shape_blind_decode={dtw_shape_blind_decode if dtw_shape_blind_decode is not None else '(default True)'}  "
          f"save_curve_values={save_curve_values if save_curve_values is not None else '(default True)'}  "
          f"curve_median_floor={curve_median_floor if curve_median_floor is not None else '(default False = OFF)'}  "
          f"curve_median_floor_ratio={curve_median_floor_ratio if curve_median_floor_ratio is not None else '(default 1.0)'}  "
          f"train_ratio={train_ratio if train_ratio is not None else '(default 0.70)'}  "
          f"split_type={split_type}  "
          f"random_seed={random_seed}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    # ── CPU-only, fork-safe child ────────────────────────────────────────────
    # Hide the GPUs from the modelling child BEFORE torch is ever imported.
    # Everything in the run is CPU-only by design, but a parent that has merely
    # probed CUDA (set_global_seeds' torch.cuda.is_available at startup) forks
    # poisoned children: experiment_986 lost 7 hours with all 16 seq2seq
    # workers and the parent stuck in futex_wait at 4s CPU each — the
    # in-worker guards (_train_seq2seq_worker) run only AFTER the fork, which
    # is too late for locks inherited FROM the fork. With no visible GPU the
    # parent never initialises CUDA and there is nothing to inherit.
    env['CUDA_VISIBLE_DEVICES']           = ''
    env['PYTORCH_NVML_BASED_CUDA_CHECK']  = '1'
    # ── Reproducibility ──────────────────────────────────────────────────────
    # One master seed for the child run: 02_modelling.py seeds python/numpy/torch
    # from it and derives every per-combo training seed from it.
    env['PIPELINE_RANDOM_SEED']       = str(random_seed)
    env['PIPELINE_RANDOM_SPLIT_SEED'] = str(random_seed)
    # PYTHONHASHSEED only takes effect at interpreter start, so it has to be set
    # HERE, on the child's environment — str hashing is randomised per process
    # otherwise, and any set-of-strings iterated without sorting would order
    # differently between two runs that are identical in every other respect.
    env['PYTHONHASHSEED']             = str(random_seed)
    env['PIPELINE_DATA_EXPERIMENT']       = exp['data_experiment']
    env['PIPELINE_RUN_NAME']              = exp['run_name']
    env['PIPELINE_PROCESSES_TO_RUN']      = ','.join(exp['processes_to_run'])
    env['PIPELINE_TEMPORAL_RESOLUTION']   = exp['temporal_resolution']
    env['PIPELINE_RUN_PROCESS_MODELLING'] = 'true' if exp.get('run_process_modelling', False) else 'false'
    env['PIPELINE_RUN_ENERGY_MODELLING']  = 'true' if run_energy_modelling else 'false'
    env['PIPELINE_RUN_JOINT_DURATION_EVAL'] = 'true' if run_joint_duration_eval else 'false'
    env['PIPELINE_RUN_SCHEDULE_PROFILE_EVAL'] = 'true' if run_schedule_profile_eval else 'false'
    env['PIPELINE_RUN_AUTOREGRESSIVE_EVAL'] = 'true' if run_autoregressive_eval else 'false'
    env['PIPELINE_SAVE_PREDICTED_CURVES'] = 'true' if save_predicted_curves else 'false'
    if mining_algorithms:
        env['PIPELINE_MINING_ALGORITHMS'] = ','.join(mining_algorithms)
    if curve_approaches:
        env['PIPELINE_CURVE_APPROACHES'] = ','.join(curve_approaches)
    if complete_curve_approaches:
        env['PIPELINE_COMPLETE_CURVE_APPROACHES'] = ','.join(complete_curve_approaches)
    if curve_models:
        env['PIPELINE_CURVE_MODELS'] = ','.join(curve_models)
    if seq2seq_cells:
        env['PIPELINE_SEQ2SEQ_CELLS'] = ','.join(seq2seq_cells)
    if curve_optimize_hyperparams is not None:
        env['PIPELINE_CURVE_OPTIMIZE_HYPERPARAMS'] = 'true' if curve_optimize_hyperparams else 'false'
    if curve_n_optuna_trials is not None:
        env['PIPELINE_CURVE_N_OPTUNA_TRIALS'] = str(curve_n_optuna_trials)
    if curve_loss is not None:
        env['PIPELINE_CURVE_LOSS'] = str(curve_loss)
    if curve_select_by_realism is not None:
        env['PIPELINE_CURVE_SELECT_BY_REALISM'] = 'true' if curve_select_by_realism else 'false'
    if curve_selection_metrics:
        env['PIPELINE_CURVE_SELECTION_METRICS'] = ','.join(curve_selection_metrics)
    if dtw_shape_blind_decode is not None:
        env['PIPELINE_DTW_SHAPE_BLIND_DECODE'] = 'true' if dtw_shape_blind_decode else 'false'
    if save_curve_values is not None:
        env['PIPELINE_SAVE_CURVE_VALUES'] = 'true' if save_curve_values else 'false'
    if curve_median_floor is not None:
        env['PIPELINE_CURVE_MEDIAN_FLOOR'] = 'true' if curve_median_floor else 'false'
    if curve_median_floor_ratio is not None:
        env['PIPELINE_CURVE_MEDIAN_FLOOR_RATIO'] = str(curve_median_floor_ratio)
    if step_dtw_fallback_ratio is not None:
        # 'inf' disables the step-DTW do-no-harm gate: no leaf is ever routed
        # to ml_external, the step/smooth model is kept everywhere.
        env['PIPELINE_STEP_DTW_FALLBACK_RATIO'] = str(step_dtw_fallback_ratio)
    if train_ratio is not None:
        env['PIPELINE_TRAIN_RATIO'] = str(train_ratio)
    env['PIPELINE_SPLIT_TYPE'] = split_type

    result = subprocess.run(
        [sys.executable, str(modelling_script)],
        env=env,
    )

    if result.returncode != 0:
        print(f"\n[pipeline] ERROR: {exp['run_name']} exited with code {result.returncode}. Stopping.")
        sys.exit(result.returncode)

    print(f"\n[pipeline] Finished: {exp['run_name']}")

print(f"\n{'='*60}")
print(f"  All {len(EXPERIMENTS)} experiment(s) completed.")
print(f"{'='*60}\n")
