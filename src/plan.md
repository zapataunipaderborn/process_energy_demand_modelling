Petri-Constrained Warped-Curve ML Plan

Objective
- Use warping only (fixed-size curve features), no encoder.
- Use Petri net to constrain valid next activities.
- Use ML classifier probabilities to select among Petri-enabled options.
- Use ML regressor to predict next duration with stochastic variability.
- Minimize reliance on old statistical transition logic.

Core Principle
- Petri net is the hard constraint.
- ML is the decision policy inside that constraint.

At each simulation step
1. Get enabled labeled transitions from current Petri marking.
2. Build features from the current simulated activity and its warped curve.
3. Classifier predicts next-activity probabilities for all labels plus END.
4. Mask probabilities to enabled labels only, renormalize, sample next label.
5. Fire a Petri transition with that label.
6. Regressor predicts next duration and sample with uncertainty.

Data Construction for Training
1. Build supervised samples as event pairs k to k+1 per case.
2. Features from current event k:
- warped_curve_0 to warped_curve_(L-1)
- activity, object, object_type, higher_level_activity
- object attributes
- optional short history: last labels, last durations
- optional index within case
3. Targets from next event k+1:
- y_transition: next label or END
- y_duration: next duration in minutes

Models
1. Transition model
- Probabilistic classifier with predict_proba.
- Recommended start: XGBoost classifier.

2. Duration model
- Regressor for next duration.
- Recommended start: XGBoost regressor.

3. Uncertainty model for stochastic duration
- Option A: residual sigma model.
- Option B: quantile models q10, q50, q90.
- Option C: per-activity calibrated residual standard deviation.

Petri + ML Fusion (No Statistical Transition Dependence)
1. Let E_t be enabled labels at time t from Petri net.
2. Let p_cls be classifier probabilities over all labels.
3. Mask and renormalize over E_t only.
4. Optional blend with Petri replay prior p_petri:
- p_final proportional to p_cls^alpha times p_petri^(1-alpha)
5. Sample from p_final (not argmax).

Fallback policy (minimal statistical usage)
1. If classifier available and masked mass > 0:
- use masked ML probabilities.
2. Else if Petri replay prior available:
- sample from Petri prior over enabled labels.
3. Else:
- uniform over enabled labels.
4. Only final emergency fallback:
- old statistical transition table.

Non-Deterministic Simulation
1. Transition variability
- Sample from probabilities with temperature.
- temperature < 1 sharper, > 1 flatter.

2. Duration variability
- Base mean prediction mu from regressor.
- Sample duration from uncertainty-aware distribution.
- Clip duration to train percentile bounds per activity.

Pseudo-Algorithm (Per Event Step)
1. Compute enabled labels from Petri marking.
2. Build feature vector from warped current curve + context.
3. Get classifier probabilities.
4. Mask to enabled labels.
5. If empty after mask, apply fallback chain.
6. Sample next label.
7. Fire matching Petri transition.
8. Predict next duration.
9. Sample duration with uncertainty.
10. Log event, continue until END or final marking.

Implementation Mapping
1. sim_modeller.py
- Add train_transition_with_warped_curve.
- Add train_duration_with_warped_curve.
- Add predict_transition_proba_with_curve.
- Add predict_duration_with_curve.
- Add sample_duration_with_curve.

2. simulation.py
- Add mode ml_warped_curve_petri.
- In Petri simulation loop, replace transition choice with masked classifier sampling.
- Keep Petri marking semantics unchanged.
- Use duration sampler from ML model.

3. pipeline.py
- Enable transition training for this mode.
- Keep temporal case split.
- Simulate test production plan and compare to real test log.

Evaluation
1. Transition metrics
- top-1 accuracy, macro F1, log-loss, calibration.

2. Duration metrics
- MAE, RMSE, WAPE.
- interval coverage if quantiles are used.

3. Process metrics
- fitness, precision, generalization, simplicity.
- event ratio and case-length distribution similarity.

4. Ablation study
- context only (no curve)
- warped curve only
- warped curve + context
- warped curve + context + Petri prior blend

Leakage Controls
- Use only information available up to event k when predicting k+1.
- No future curve values in features.
- Fit any scaling on train split only.

Recommended First Build
1. Use fixed-length warped vectors as-is.
2. Train XGBoost classifier + XGBoost regressor.
3. Add temperature sampling for transitions.
4. Add residual-based duration sampling.
5. Integrate Petri masking first, then optional Petri prior blending.
