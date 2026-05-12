import copy
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from datetime import datetime, timedelta

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net import semantics as pn_semantics

from sim_extractor import sample_from_dist


class ProcessSimulation:
    """
    Simulate a process using either statistical or ML-based models.

    Parameters
    ----------
    activity_stats_df : pd.DataFrame
        Output of ``sim_extractor.extract_process`` (the stats DataFrame).
    production_plan : pd.DataFrame
        One row per case to simulate.
    mode : {'statistical', 'ml', 'ml_duration_only', 'petri_net'}
        'statistical'      – sample durations/transitions from fitted
        distributions (or normal as fallback).
        'ml'               – use XGBoost models from *ml_models* when
        available; fall back to statistical automatically.
        'ml_duration_only' – use XGBoost only for the duration *median*;
        the standard deviation and transition probabilities still come
        from the statistical extraction (``activity_stats_df``).
        'petri_net'        – simulate using the Petri net token game;
        transitions are chosen based on stochastic weights from log
        replay; durations from fitted distributions.
    ml_models : SimModeller | None
        A trained ``SimModeller`` instance (required when mode='ml').
    process_models : dict | None
        Mined Petri nets from ``sim_extractor.extract_process()``.
        Required when mode='petri_net'.
    random_seed : int
        NumPy / random seed for reproducibility.
    """

    def __init__(self, activity_stats_df, production_plan,
                 mode='statistical', base_simulation_mode='statistical', ml_models=None,
                 process_models=None, random_seed=42,
                 energy_duration_modifiers=None,
                 energy_transition_modifiers=None,
                 energy_state_columns=None,
                 energy_pipelines=None,
                 activity_exog_means=None,
                 duration_scale_clip=None,
                 logit_bias_clip=None,
                 temporal_resolution_minutes=15.0,
                 verbose=True):
        # Backward-compatible input handling: extract_process now returns
        # (stats_df, raw_df, process_models), while older callers pass stats_df only.
        if isinstance(activity_stats_df, (tuple, list)):
            if len(activity_stats_df) == 0:
                raise ValueError("activity_stats_df tuple/list is empty")
            self.activity_stats = activity_stats_df[0]
            if process_models is None and len(activity_stats_df) >= 3:
                process_models = activity_stats_df[2]
        else:
            self.activity_stats = activity_stats_df
            
        self.verbose = verbose

        if not isinstance(self.activity_stats, pd.DataFrame):
            raise TypeError(
                "activity_stats_df must be a pandas DataFrame or the tuple returned by extract_process"
            )

        self.production_plan = production_plan
        self.mode = mode
        self.base_simulation_mode = base_simulation_mode
        self.ml_models = ml_models
        self.process_models = process_models  # Petri nets from sim_extractor
        print(f"[DEBUG __init__] mode={self.mode}, "
              f"process_models is None: {process_models is None}, "
              f"process_models len: {len(process_models) if process_models else 'N/A'}")
        random.seed(random_seed)
        np.random.seed(random_seed)

        if self.mode in ('ml', 'ml_duration_only',
                        'ml_duration_only_with_activity_past',
                        'ml_duration_only_with_activity_past_point_estimate',
                        'ml_global_model') \
                and self.ml_models is None:
            print(f"[ProcessSimulation] WARNING: mode='{self.mode}' but no ml_models "
                  "provided – falling back to statistical mode.")
            self.mode = 'statistical'

        # Petri net mode validation
        if self.mode in ('petri_net', 'petri_net_statistical',
                         'petri_net_statistical_memory') \
                and (self.process_models is None or len(self.process_models) == 0):
            raise ValueError(
                f"mode='{self.mode}' requires process_models but none were provided "
                f"or all failed to mine. Check sim_extractor output.")

        # Energy-aware mode validation
        _ENERGY_MODES = (
            'petri_net_energy_aware',
            'petri_net_energy_duration_aware',
            'petri_net_energy_transition_aware',
            'petri_net_energy_direct',
            'petri_net_energy_direct_duration_only',
            'petri_net_energy_direct_transition_only',
            'petri_net_energy_direct_global',
            'petri_net_quantile_blend',
        )
        if self.mode in _ENERGY_MODES:
            if self.process_models is None or len(self.process_models) == 0:
                raise ValueError(
                    f"mode='{self.mode}' requires process_models (a mined Petri net). "
                    "Pass the best base Petri net from extract_process()."
                )
            if energy_duration_modifiers is None and energy_transition_modifiers is None:
                raise ValueError(
                    f"mode='{self.mode}' requires at least one of "
                    "energy_duration_modifiers or energy_transition_modifiers."
                )

        # Store energy-aware attributes
        self.energy_duration_modifiers    = energy_duration_modifiers or {}
        self.energy_transition_modifiers  = energy_transition_modifiers or {}
        self.energy_state_columns         = energy_state_columns or []
        self.energy_pipelines             = energy_pipelines or {}
        self.activity_exog_means          = activity_exog_means or {}
        self.duration_scale_clip          = duration_scale_clip
        self.logit_bias_clip              = logit_bias_clip
        self.temporal_resolution_minutes  = float(temporal_resolution_minutes) if temporal_resolution_minutes else 15.0

        # Verification: confirm global model is loaded for ml_global_model
        if self.mode == 'ml_global_model' and self.ml_models is not None:
            has_global = self.ml_models.global_duration_model is not None
            print(f"[ProcessSimulation] ml_global_model: global_duration_model loaded = {has_global}")
            if not has_global:
                print("[ProcessSimulation] WARNING: global model is None! "
                      "Will fall back to statistical for every activity.")

        self.env = None
        self.events = []

        # Process configuration - build from activity_stats only
        self._build_activity_config()
        
    def _build_activity_config(self):
        """Build activity configuration from stats DataFrame"""
        self.activity_config = {}
        
        print(f"Building config from {len(self.activity_stats)} activities:")
        for _, row in self.activity_stats.iterrows():
            activity_name = str(row['activity']).strip()
            object_name = str(row['object']).strip()
            object_type = str(row['object_type']).strip()
            higher_level = str(row['higher_level_activity']).strip() if pd.notna(row['higher_level_activity']) else None
            
            key = (activity_name, object_name, object_type, higher_level)
            self.activity_config[key] = {
                'duration':     row['duration'],
                'duration_std': row['duration_std'],
                # Best-fit distribution (added by enhanced sim_extractor)
                'dist_name':    row.get('dist_name', 'norm'),
                'dist_params':  row.get('dist_params', (row['duration'], row['duration_std'])),
                'transitions':  row['transition'],
                'is_start':     row['is_start'],
                'is_end':       row['is_end'],
                'n_events':     row['n_events'],
            }
            
            print(f"  Config: {key}")
            print(f"    Transitions: {row['transition']}")
            print(f"    Start: {row['is_start']}")
    
    def _get_activity_duration(self, activity, object_name, object_type,
                               higher_level_activity, object_attributes=None,
                               activity_history=None, activity_index=0):
        """
        Generate a sampled duration (minutes) for one activity instance.

        In 'ml' mode the XGBoost duration model is tried first; it falls back
        to the statistical path when no model exists for this key.
        In 'ml_duration_only' mode, the ML model provides only the median
        prediction while the noise (std) comes from the statistical config.
        In 'ml_duration_only_with_activity_past' mode, same as ml_duration_only
        but the last 2 activities & durations are passed as extra features.
        In 'statistical' mode the best-fit distribution stored in
        """
    def _get_activity_duration(self, activity, object_name, object_type,
                               higher_level_activity=None,
                               object_attributes=None,
                               activity_history=None,
                               activity_index=0,
                               override_mode=None):
        """
        Sample the duration.

        If `override_mode` is provided, it uses that instead of `self.mode`.
        """
        eval_mode = override_mode if override_mode else self.mode
        activity              = str(activity).strip()
        object_name           = str(object_name).strip()
        object_type           = str(object_type).strip()
        higher_level_activity = (str(higher_level_activity).strip()
                                 if pd.notna(higher_level_activity) else None)
        object_attributes     = object_attributes or {}

        key = (activity, object_name, object_type, higher_level_activity)

        # ── Full ML path (duration + ML std) ──────────────────────────────
        if eval_mode == 'ml' and self.ml_models is not None:
            ml_dur = self.ml_models.predict_duration(
                activity, object_name, object_type,
                higher_level_activity, object_attributes
            )
            if ml_dur is not None:
                return ml_dur
            # else: fall through to statistical

        # ── ML duration-only paths (with or without activity history) ─────
        _ML_DUR_MODES = ('ml_duration_only',
                         'ml_duration_only_with_activity_past',
                         'ml_duration_only_with_activity_past_point_estimate')
        if eval_mode in _ML_DUR_MODES and self.ml_models is not None:
            use_hist = eval_mode in ('ml_duration_only_with_activity_past',
                                     'ml_duration_only_with_activity_past_point_estimate')
            hist = activity_history if use_hist else None
            ml_median = self.ml_models.predict_duration_median(
                activity, object_name, object_type,
                higher_level_activity, object_attributes,
                activity_history=hist,
            )
            if ml_median is not None:
                # Point estimate mode → return raw prediction, no noise
                if eval_mode == 'ml_duration_only_with_activity_past_point_estimate':
                    return max(0.1, float(ml_median))
                # Otherwise add statistical std as noise
                stat_std = 0.0
                if key in self.activity_config:
                    stat_std = self.activity_config[key].get('duration_std', 0.0)
                if stat_std > 0:
                    sampled = np.random.normal(ml_median, stat_std)
                else:
                    sampled = ml_median
                return max(0.1, float(sampled))
            # else: fall through to statistical

        # ── Global model path (single model across all activities) ───────
        if eval_mode == 'ml_global_model' and self.ml_models is not None:
            global_pred = self.ml_models.predict_duration_global(
                activity, object_name, object_type,
                higher_level_activity, object_attributes or {},
                activity_history=activity_history,
                activity_index=activity_index,
            )
            if global_pred is not None:
                return global_pred
            # else: fall through to statistical

        # ── Statistical path ──────────────────────────────────────────────
        if key in self.activity_config:
            config      = self.activity_config[key]
            dist_name   = config.get('dist_name', 'norm')
            dist_params = config.get('dist_params')

            if dist_params and any(p != 0 for p in dist_params[1:]):
                return sample_from_dist(dist_name, dist_params)
            else:
                # Degenerate case: zero variance – return the median
                return max(0.1, config['duration'])
        else:
            print(f"WARNING: No config found for {key}")
            return 10.0
    
    def _get_next_activity(self, current_activity, object_name, object_type,
                           higher_level_activity, object_attributes=None):
        """
        Determine next activity based on transition probabilities.
        Handles probabilistic END transitions.

        In 'ml' mode the XGBoost transition classifier is tried first; it
        falls back to the statistical (frequency-based) path automatically.
        In 'ml_duration_only' mode, transitions always come from the
        statistical (frequency-based) path.
        """
        current_activity      = str(current_activity).strip()
        object_name           = str(object_name).strip()
        object_type           = str(object_type).strip()
        higher_level_activity = (str(higher_level_activity).strip()
                                 if pd.notna(higher_level_activity) else None)
        object_attributes     = object_attributes or {}

        key = (current_activity, object_name, object_type, higher_level_activity)

        print(f"    Looking for transitions from: {current_activity}")

        # ── resolve transition probability dict ───────────────────────────
        transitions = None

        # ML path (full ml mode only – ml_duration_only skips this)
        if self.mode == 'ml' and self.ml_models is not None:
            ml_tr = self.ml_models.predict_transitions(
                current_activity, object_name, object_type,
                higher_level_activity, object_attributes
            )
            if ml_tr is not None:
                transitions = ml_tr
                print(f"    ML transitions: {transitions}")

        # Statistical path (used by 'statistical', 'ml_duration_only',
        # and as fallback for 'ml' mode)
        if transitions is None:
            if key in self.activity_config:
                transitions = self.activity_config[key]['transitions']
                print(f"    Statistical transitions: {transitions}")
            else:
                print(f"    Key not found in config: {key}")
                return None

        # ── sample next activity ──────────────────────────────────────────
        if not transitions:
            print(f"    No transitions available - ending process")
            return None

        # Sort to make sampling independent from dict insertion order.
        transition_items = sorted(transitions.items(), key=lambda kv: str(kv[0]))
        activities    = [a for a, _ in transition_items]
        probabilities = [p for _, p in transition_items]
        prob_sum      = sum(probabilities)

        if prob_sum <= 0:
            return None

        probabilities   = [p / prob_sum for p in probabilities]
        next_transition = np.random.choice(activities, p=probabilities)

        if next_transition == '__END__':
            print(f"    Selected END transition - stopping process")
            return None

        print(f"    Selected next activity: {next_transition}")
        return str(next_transition).strip()
    
    def _get_start_activities(self, object_name, object_type, higher_level_activity):
        """Get all possible start activities for a given object configuration"""
        object_name = str(object_name).strip()
        object_type = str(object_type).strip()
        higher_level_activity = str(higher_level_activity).strip() if pd.notna(higher_level_activity) else None
        
        start_activities = []
        
        print(f"  Looking for start activities for: {object_name} ({object_type}) with higher_level: {higher_level_activity}")
        
        for key, config in self.activity_config.items():
            activity, obj, obj_type, higher_level = key
            if (obj == object_name and obj_type == object_type and 
                higher_level == higher_level_activity and config['is_start']):
                start_activities.append(str(activity).strip())
                print(f"    Found start activity: {activity}")
        
        if not start_activities:
            print(f"    NO START ACTIVITIES FOUND! Available keys:")
            for key in self.activity_config.keys():
                activity, obj, obj_type, higher_level = key
                print(f"      {key} - is_start: {self.activity_config[key]['is_start']}")
        
        return start_activities
    
    def _log_event(self, case_id, activity, timestamp_start, timestamp_end, 
                   object_name, object_type, higher_level_activity, object_attributes):
        """Log a simulation event"""
        self.events.append({
            'case_id': str(case_id).strip(),
            'activity': str(activity).strip(),
            'timestamp_start': timestamp_start,
            'timestamp_end': timestamp_end,
            'object': str(object_name).strip(),
            'object_type': str(object_type).strip(),
            'higher_level_activity': str(higher_level_activity).strip() if pd.notna(higher_level_activity) else None,
            'object_attributes': object_attributes
        })
    
    # ------------------------------------------------------------------
    # Petri net token-game simulation
    # ------------------------------------------------------------------

    def _get_enabled_transitions(self, net, marking):
        """
        Return the set of transitions enabled in the current marking.
        A transition is enabled when every input place has at least
        one token.
        """
        enabled = set()
        for t in net.transitions:
            if all(marking.get(arc.source, 0) >= 1 for arc in t.in_arcs):
                enabled.add(t)
        return enabled

    def _fire_transition(self, marking, transition):
        """
        Fire *transition* in *marking*: consume tokens from input places,
        produce tokens in output places.  Returns new marking.
        """
        new_marking = copy.copy(marking)
        for arc in transition.in_arcs:
            new_marking[arc.source] -= 1
            if new_marking[arc.source] == 0:
                del new_marking[arc.source]
        for arc in transition.out_arcs:
            if arc.target not in new_marking:
                new_marking[arc.target] = 0
            new_marking[arc.target] += 1
        return new_marking

    def _choose_transition(self, enabled, stochastic_map):
        """
        Given a set of enabled transitions, pick one using stochastic
        weights.  Falls back to uniform random if no weights available.
        """
        # Sort to make sampling independent from set iteration order.
        enabled_list = sorted(
            list(enabled),
            key=lambda t: (str(t.label) if t.label is not None else '', str(t.name)),
        )
        weights = [stochastic_map.get(t, 1.0) for t in enabled_list]
        total = sum(weights)
        if total <= 0:
            return random.choice(enabled_list)
        probs = [w / total for w in weights]
        return np.random.choice(enabled_list, p=probs)

    def _simulate_petri_net_for_case(self, case_id, object_attributes,
                                     start_time):
        """
        Simulate one case using the Petri net token game.

        For each (object, object_type, higher_level_activity) group that
        has a mined Petri net, we:
          1. Place a token in the initial marking.
          2. Find enabled transitions.
          3. Choose one (stochastic weights).
          4. Fire it → update marking.
          5. If the transition has a label (= activity), sample a duration
             and log the event.
          6. Repeat until the final marking is reached or no transitions
             are enabled.
        """
        current_sim_time = start_time.timestamp()
        unique_objects = (
            self.activity_stats[['object', 'object_type',
                                 'higher_level_activity']]
            .drop_duplicates(subset=['higher_level_activity'])
        )

        for _, obj_config in unique_objects.iterrows():
            object_name = str(obj_config['object']).strip()
            object_type = str(obj_config['object_type']).strip()
            higher_level_activity = (
                str(obj_config['higher_level_activity']).strip()
                if pd.notna(obj_config['higher_level_activity']) else None
            )

            key = (object_name, object_type, higher_level_activity)
            model = self.process_models.get(key) if self.process_models else None

            if model is None:
                available = list(self.process_models.keys()) if self.process_models else []
                raise RuntimeError(
                    f"petri_net mode: no Petri net found for key {key}. "
                    f"Available keys: {available}")

            net = model['net']
            im  = model['im']
            fm  = model['fm']
            stochastic_map = model.get('stochastic_map', {})
            duration_map   = model.get('duration_map', {})
            max_case_length = model.get('max_case_length', 200)

            # Start the token game
            marking = copy.copy(im)
            activity_count = 0
            max_steps = max(max_case_length * 2, 50)  # guard from training data
            step = 0

            print(f"\nCase {case_id}: Petri net simulation for {object_name} "
                  f"({object_type})")

            while step < max_steps:
                step += 1

                # Check if we reached the final marking
                if marking == fm:
                    print(f"    Final marking reached after {activity_count} "
                          f"activities.")
                    break

                # Check also if fm is a subset of marking (common pattern)
                fm_reached = all(
                    marking.get(p, 0) >= fm[p] for p in fm
                )
                if fm_reached and activity_count > 0:
                    print(f"    Final marking subset reached after "
                          f"{activity_count} activities.")
                    break

                # Find enabled transitions
                enabled = self._get_enabled_transitions(net, marking)
                if not enabled:
                    print(f"    No enabled transitions — deadlock after "
                          f"{activity_count} activities.")
                    break

                # Choose which transition to fire
                chosen = self._choose_transition(enabled, stochastic_map)

                # Fire the transition (update marking)
                marking = self._fire_transition(marking, chosen)

                # If this is a visible transition (has a label), log event
                if chosen.label is not None:
                    activity_label = str(chosen.label).strip()

                    # Get duration from the duration map or activity config
                    act_key = (activity_label, object_name, object_type,
                               higher_level_activity)

                    if act_key in self.activity_config:
                        config = self.activity_config[act_key]
                        dist_name = config.get('dist_name', 'norm')
                        dist_params = config.get('dist_params')
                        if dist_params and any(p != 0 for p in dist_params[1:]):
                            activity_duration = sample_from_dist(
                                dist_name, dist_params
                            )
                        else:
                            activity_duration = max(0.1, config['duration'])
                    elif activity_label in duration_map:
                        dn, dp = duration_map[activity_label]
                        activity_duration = sample_from_dist(dn, dp)
                    else:
                        print(f"    WARNING: No duration info for "
                              f"'{activity_label}' — using 10 min default.")
                        activity_duration = 10.0

                    # Calculate timestamps
                    start_time_obj = datetime.fromtimestamp(current_sim_time)
                    current_sim_time += activity_duration * 60
                    end_time_obj = datetime.fromtimestamp(current_sim_time)

                    # Log the event
                    self._log_event(
                        case_id=case_id,
                        activity=activity_label,
                        timestamp_start=start_time_obj,
                        timestamp_end=end_time_obj,
                        object_name=object_name,
                        object_type=object_type,
                        higher_level_activity=higher_level_activity,
                        object_attributes=object_attributes,
                    )

                    activity_count += 1
                    current_sim_time += 1  # 1 second gap

                    print(f"    [{activity_count}] Fired '{activity_label}' "
                          f"(dur={activity_duration:.1f} min)")
                else:
                    # Silent (tau) transition — no event logged
                    pass

            if step >= max_steps:
                print(f"    WARNING: Max steps ({max_steps}) reached for "
                      f"{object_name} — stopping.")

            print(f"  Completed {object_name} Petri net simulation with "
                  f"{activity_count} activities")

    # ------------------------------------------------------------------
    # Combined: Petri net structure + statistical/ML probabilities
    # ------------------------------------------------------------------

    def _simulate_petri_net_statistical_for_case(self, case_id,
                                                  object_attributes,
                                                  start_time):
        """
        Simulate one case using the Petri net structure as a constraint
        on which activities can follow, while **blending** the Petri
        net's stochastic weights (from token replay) with the
        statistical transition probabilities from the log.

        At each step:
        1.  Find enabled Petri net transitions → extract their labels
            as the set of *structurally valid* next activities.
        2.  Look up the statistical transition probabilities AND the
            Petri net stochastic weights for the current decision.
        3.  Blend them using  alpha * stat_prob + (1-alpha) * pn_weight
            (both independently normalised first).
        4.  Fire the corresponding Petri net transition to advance the
            marking.
        5.  Use the full `_get_activity_duration` pipeline for the
            sampled duration.

        alpha = 1.0  →  pure statistical (Petri net only constrains structure)
        alpha = 0.0  →  pure Petri net stochastic weights
        alpha = 0.5  →  equal blend (default)
        """
        # ── blending weight: tune between 0.0 and 1.0 ────────────────
        alpha = 0

        current_sim_time = start_time.timestamp()
        unique_objects = (
            self.activity_stats[['object', 'object_type',
                                 'higher_level_activity']]
            .drop_duplicates(subset=['higher_level_activity'])
        )

        for _, obj_config in unique_objects.iterrows():
            object_name = str(obj_config['object']).strip()
            object_type = str(obj_config['object_type']).strip()
            higher_level_activity = (
                str(obj_config['higher_level_activity']).strip()
                if pd.notna(obj_config['higher_level_activity']) else None
            )

            key = (object_name, object_type, higher_level_activity)
            pm_keys = list(self.process_models.keys()) if self.process_models else []
            print(f"  [DEBUG] Looking up key: {key}")
            print(f"  [DEBUG] Available process_models keys: {pm_keys}")
            model = self.process_models.get(key) if self.process_models else None

            if model is None:
                available = list(self.process_models.keys()) if self.process_models else []
                raise RuntimeError(
                    f"petri_net_statistical mode: no Petri net found for key {key}. "
                    f"Available keys: {available}")
            
            print(f"  ✅ Petri net FOUND for {key}")
            print(f"  [DEBUG] label_stochastic: {model.get('label_stochastic', {})}")

            net = model['net']
            im  = model['im']
            fm  = model['fm']
            label_stochastic = model.get('label_stochastic', {})
            decision_weights = model.get('decision_weights', {})
            max_case_length = model.get('max_case_length', 200)

            # Start the token game
            marking = copy.copy(im)
            activity_count = 0
            activity_history = []
            max_steps = max(max_case_length * 2, 50)  # Change 4: guard from data
            step = 0

            print(f"\nCase {case_id}: Petri-net-statistical simulation "
                  f"for {object_name} ({object_type})  [alpha={alpha}]")

            while step < max_steps:
                step += 1

                # Check final marking
                if marking == fm:
                    print(f"    Final marking reached after "
                          f"{activity_count} activities.")
                    break
                fm_reached = all(
                    marking.get(p, 0) >= fm[p] for p in fm
                )
                if fm_reached and activity_count > 0:
                    print(f"    Final marking subset reached after "
                          f"{activity_count} activities.")
                    break

                # Find enabled transitions
                enabled = self._get_enabled_transitions(net, marking)
                if not enabled:
                    print(f"    No enabled transitions — deadlock after "
                          f"{activity_count} activities.")
                    break

                # ── Build label → [transitions] map for enabled set ───
                label_to_transitions = defaultdict(list)
                silent_transitions = []
                for t in enabled:
                    if t.label is not None:
                        label_to_transitions[str(t.label).strip()].append(t)
                    else:
                        silent_transitions.append(t)

                # If only silent transitions are enabled, fire one and loop
                if not label_to_transitions:
                    chosen = random.choice(silent_transitions)
                    marking = self._fire_transition(marking, chosen)
                    continue

                valid_labels = set(label_to_transitions.keys())

                # ── First step: pick a start activity ─────────────────
                if activity_count == 0:
                    start_acts = self._get_start_activities(
                        object_name, object_type, higher_level_activity
                    )
                    constrained_starts = [
                        a for a in start_acts if a in valid_labels
                    ]
                    if not constrained_starts:
                        constrained_starts = list(valid_labels)
                    if not constrained_starts:
                        print(f"    No valid start activities — ending.")
                        break

                    chosen_label = random.choice(constrained_starts)
                    print(f"  Starting with constrained activity: "
                          f"{chosen_label}")
                else:
                    # ── Blend statistical + Petri net weights ──────────
                    prev_activity = self.events[-1]['activity']
                    prev_key = (prev_activity, object_name, object_type,
                                higher_level_activity)

                    # Source 1: statistical transition probabilities
                    stat_probs = {}
                    if prev_key in self.activity_config:
                        raw_tr = self.activity_config[prev_key].get(
                            'transitions', {}
                        )
                        stat_probs = {
                            k: v for k, v in raw_tr.items()
                            if k in valid_labels or k == '__END__'
                        }

                    # Source 2: Decision-point weights (context-aware)
                    dp_key = frozenset(valid_labels)
                    dp_weights = decision_weights.get(dp_key, {})

                    # Fallback to label_stochastic if no decision-point data
                    if not dp_weights:
                        dp_weights = {
                            lbl: label_stochastic.get(lbl, 0.0)
                            for lbl in valid_labels
                        }

                    # Include __END__ from decision-point weights
                    pn_weights = {
                        lbl: dp_weights.get(lbl, 0.0)
                        for lbl in valid_labels
                    }
                    # Add __END__ if the decision-point data has it
                    if '__END__' in dp_weights:
                        pn_weights['__END__'] = dp_weights['__END__']

                    # Normalise each source independently
                    all_labels = list(valid_labels)
                    has_end = '__END__' in stat_probs or '__END__' in pn_weights
                    if has_end and '__END__' not in all_labels:
                        all_labels.append('__END__')

                    stat_total = sum(stat_probs.get(l, 0.0)
                                     for l in all_labels) or 1.0
                    pn_total = sum(pn_weights.get(l, 0.0)
                                   for l in all_labels) or 1.0

                    # Blend into a single probability per label (+ __END__)
                    blended = {}
                    for lbl in all_labels:
                        s = stat_probs.get(lbl, 0.0) / stat_total
                        p = pn_weights.get(lbl, 0.0) / pn_total
                        blended[lbl] = alpha * s + (1.0 - alpha) * p

                    # Sample from blended probabilities (including __END__)
                    acts = list(blended.keys())
                    probs = [blended[a] for a in acts]
                    psum = sum(probs)
                    if psum <= 0:
                        # Fallback: uniform over valid labels
                        acts = list(valid_labels)
                        probs = [1.0 / len(acts)] * len(acts)
                    else:
                        probs = [p / psum for p in probs]

                    chosen_label = str(
                        np.random.choice(acts, p=probs)
                    ).strip()

                    # If __END__ was chosen, stop the process
                    if chosen_label == '__END__':
                        print(f"    Selected END transition — stopping.")
                        break

                    print(f"    Constrained next: {chosen_label}")

                # ── Fire the Petri net transition for chosen_label ─────
                candidates = label_to_transitions.get(chosen_label, [])
                if not candidates:
                    print(f"    WARNING: label '{chosen_label}' not in "
                          f"enabled transitions — ending.")
                    break
                chosen_transition = random.choice(candidates)
                marking = self._fire_transition(marking, chosen_transition)

                # ── Sample duration using full pipeline ────────────────
                _HIST_MODES = (
                    'ml_duration_only_with_activity_past',
                    'ml_duration_only_with_activity_past_point_estimate',
                    'ml_global_model',
                )
                activity_duration = self._get_activity_duration(
                    chosen_label, object_name, object_type,
                    higher_level_activity, object_attributes,
                    activity_history=(
                        activity_history if self.mode in _HIST_MODES
                        else None
                    ),
                    activity_index=activity_count,
                )

                # Timestamps
                start_time_obj = datetime.fromtimestamp(current_sim_time)
                current_sim_time += activity_duration * 60
                end_time_obj = datetime.fromtimestamp(current_sim_time)

                self._log_event(
                    case_id=case_id, activity=chosen_label,
                    timestamp_start=start_time_obj,
                    timestamp_end=end_time_obj,
                    object_name=object_name,
                    object_type=object_type,
                    higher_level_activity=higher_level_activity,
                    object_attributes=object_attributes,
                )

                activity_history.insert(
                    0, (chosen_label, activity_duration)
                )
                activity_history = activity_history[:2]
                activity_count += 1
                current_sim_time += 1  # 1 second gap

                print(f"    [{activity_count}] '{chosen_label}' "
                      f"(dur={activity_duration:.1f} min)")

            if step >= max_steps:
                print(f"    WARNING: Max steps ({max_steps}) reached.")

            print(f"  Completed {object_name} petri_net_statistical "
                  f"with {activity_count} activities")

    # ------------------------------------------------------------------
    # Memory-augmented: GSPN with history-dependent stochastic weights
    # ------------------------------------------------------------------

    def _simulate_petri_net_statistical_memory_for_case(self, case_id,
                                                        object_attributes,
                                                        start_time):
        """
        Simulate one case using the Petri net structure with
        **history-dependent** transition selection (GSPN-M).

        At each decision point the transition weights are resolved using
        a priority cascade:
          1. **Bigram** — P(next | prev_activity, current_activity)
             Uses 2nd-order Markov statistics mined from the event log.
          2. **Activity-count** — P(next | current_activity, times_seen)
             Uses repetition-aware statistics (how many times the
             current activity has already been executed in this case).
          3. **Label stochastic (1st-order)** — standard Petri net
             stochastic weights from token replay.

        The highest-priority source with sufficient data is **blended**
        with the 1st-order label stochastic weights (MEMORY_BLEND_ALPHA
        controls the mix) to avoid overfitting to rare contexts.

        Durations are sampled identically to the non-memory mode.
        """
        MIN_CONTEXT_OBSERVATIONS = 5   # need ≥5 total obs to trust a context
        MEMORY_BLEND_ALPHA = 0.5     # 70% memory, 30% label_stochastic

        current_sim_time = start_time.timestamp()
        unique_objects = (
            self.activity_stats[['object', 'object_type',
                                 'higher_level_activity']]
            .drop_duplicates(subset=['higher_level_activity'])
        )

        for _, obj_config in unique_objects.iterrows():
            object_name = str(obj_config['object']).strip()
            object_type = str(obj_config['object_type']).strip()
            higher_level_activity = (
                str(obj_config['higher_level_activity']).strip()
                if pd.notna(obj_config['higher_level_activity']) else None
            )

            key = (object_name, object_type, higher_level_activity)
            model = self.process_models.get(key) if self.process_models else None

            if model is None:
                available = list(self.process_models.keys()) if self.process_models else []
                raise RuntimeError(
                    f"petri_net_statistical_memory mode: no Petri net found for key {key}. "
                    f"Available keys: {available}")

            net = model['net']
            im  = model['im']
            fm  = model['fm']
            label_stochastic       = model.get('label_stochastic', {})
            bigram_transitions     = model.get('bigram_transitions', {})
            activity_count_trans   = model.get('activity_count_transitions', {})
            decision_weights       = model.get('decision_weights', {})
            max_case_length        = model.get('max_case_length', 200)

            # ── Per-case memory state ─────────────────────────────────
            marking = copy.copy(im)
            activity_count = 0
            activity_history = []        # ordered list of fired labels
            activity_counts_map = defaultdict(int)  # {label: times_fired}
            prev_activity = '__START__'  # sentinel for first step
            max_steps = max(max_case_length * 2, 50)  # Change 4
            step = 0

            print(f"\nCase {case_id}: Petri-net-memory simulation "
                  f"for {object_name} ({object_type})")

            while step < max_steps:
                step += 1

                # ── Check final marking ───────────────────────────────
                if marking == fm:
                    print(f"    Final marking reached after "
                          f"{activity_count} activities.")
                    break
                fm_reached = all(
                    marking.get(p, 0) >= fm[p] for p in fm
                )
                if fm_reached and activity_count > 0:
                    print(f"    Final marking subset reached after "
                          f"{activity_count} activities.")
                    break

                # ── Enabled transitions ───────────────────────────────
                enabled = self._get_enabled_transitions(net, marking)
                if not enabled:
                    print(f"    No enabled transitions — deadlock after "
                          f"{activity_count} activities.")
                    break

                # ── Separate labelled vs silent ───────────────────────
                label_to_transitions = defaultdict(list)
                silent_transitions = []
                for t in enabled:
                    if t.label is not None:
                        label_to_transitions[str(t.label).strip()].append(t)
                    else:
                        silent_transitions.append(t)

                # Only silent transitions enabled → fire one and loop
                if not label_to_transitions:
                    chosen = random.choice(silent_transitions)
                    marking = self._fire_transition(marking, chosen)
                    continue

                valid_labels = set(label_to_transitions.keys())

                # ── First step: pick a start activity ─────────────────
                if activity_count == 0:
                    start_acts = self._get_start_activities(
                        object_name, object_type, higher_level_activity
                    )
                    constrained_starts = [
                        a for a in start_acts if a in valid_labels
                    ]
                    if not constrained_starts:
                        constrained_starts = list(valid_labels)
                    if not constrained_starts:
                        print(f"    No valid start activities — ending.")
                        break
                    chosen_label = random.choice(constrained_starts)
                    weight_source = 'start'

                else:
                    # ── Memory-based weight resolution with blending ──
                    memory_weights = None
                    weight_source = None

                    # --- Try bigram (2nd-order Markov) -----------------
                    bigram_prev = (activity_history[-2]
                                   if len(activity_history) >= 2
                                   else '__START__')

                    bigram_context = bigram_transitions.get(
                        (bigram_prev, prev_activity), {}
                    )
                    if bigram_context:
                        filtered = {
                            k: v for k, v in bigram_context.items()
                            if k in valid_labels or k == '__END__'
                        }
                        total = sum(filtered.values())
                        if total >= MIN_CONTEXT_OBSERVATIONS and filtered:
                            memory_weights = filtered
                            weight_source = 'bigram'

                    # --- Try activity-count if bigram insufficient -----
                    if memory_weights is None:
                        times_seen = max(
                            activity_counts_map.get(prev_activity, 1) - 1, 0
                        )
                        count_context = activity_count_trans.get(
                            (prev_activity, times_seen), {}
                        )
                        if count_context:
                            filtered = {
                                k: v for k, v in count_context.items()
                                if k in valid_labels or k == '__END__'
                            }
                            total = sum(filtered.values())
                            if total >= MIN_CONTEXT_OBSERVATIONS and filtered:
                                memory_weights = filtered
                                weight_source = 'activity_count'

                    # --- Build base weights (decision-point-aware) -----
                    dp_key = frozenset(valid_labels)
                    dp_weights = decision_weights.get(dp_key, {})
                    if dp_weights:
                        base_weights = {
                            lbl: dp_weights.get(lbl, 0.0)
                            for lbl in valid_labels
                        }
                        # Include __END__ from decision-point weights
                        if '__END__' in dp_weights:
                            base_weights['__END__'] = dp_weights['__END__']
                    else:
                        # Fallback to global label_stochastic
                        base_weights = {
                            lbl: label_stochastic.get(lbl, 1.0)
                            for lbl in valid_labels
                        }

                    # --- Blend memory with base weights ───────────────
                    if memory_weights is not None:
                        all_keys = set(base_weights.keys())
                        all_keys.update(memory_weights.keys())

                        mem_total = sum(memory_weights.values())
                        base_total = sum(base_weights.values())

                        weights = {}
                        for k in all_keys:
                            mem_p = (memory_weights.get(k, 0) / mem_total
                                     if mem_total > 0 else 0)
                            base_p = (base_weights.get(k, 0) / base_total
                                      if base_total > 0 else 0)
                            weights[k] = (MEMORY_BLEND_ALPHA * mem_p
                                          + (1 - MEMORY_BLEND_ALPHA) * base_p)
                        weight_source = f'{weight_source}+blend'
                    else:
                        weights = base_weights
                        weight_source = 'decision_point' if dp_weights else 'label_stochastic'

                    # ── Normalise and sample ──────────────────────────
                    all_labels = list(weights.keys())
                    vals = [weights[l] for l in all_labels]
                    psum = sum(vals)
                    if psum <= 0:
                        all_labels = list(valid_labels)
                        vals = [1.0] * len(all_labels)
                        psum = sum(vals)
                    probs = [v / psum for v in vals]

                    chosen_label = str(
                        np.random.choice(all_labels, p=probs)
                    ).strip()

                    if chosen_label == '__END__':
                        print(f"    Selected END transition "
                              f"(source={weight_source}) — stopping.")
                        break

                    print(f"    Constrained next: {chosen_label} "
                          f"(source={weight_source})")

                # ── Fire the Petri net transition ─────────────────────
                candidates = label_to_transitions.get(chosen_label, [])
                if not candidates:
                    print(f"    WARNING: label '{chosen_label}' not in "
                          f"enabled transitions — ending.")
                    break
                chosen_transition = random.choice(candidates)
                marking = self._fire_transition(marking, chosen_transition)

                # ── Update memory state ───────────────────────────────
                activity_history.append(chosen_label)
                activity_counts_map[chosen_label] += 1
                prev_activity = chosen_label

                # ── Sample duration ───────────────────────────────────
                _HIST_MODES = (
                    'ml_duration_only_with_activity_past',
                    'ml_duration_only_with_activity_past_point_estimate',
                    'ml_global_model',
                )
                hist_for_dur = (
                    [(a, 0.0) for a in activity_history[-2:]]
                    if self.mode in _HIST_MODES else None
                )
                activity_duration = self._get_activity_duration(
                    chosen_label, object_name, object_type,
                    higher_level_activity, object_attributes,
                    activity_history=hist_for_dur,
                    activity_index=activity_count,
                )

                # Timestamps
                start_time_obj = datetime.fromtimestamp(current_sim_time)
                current_sim_time += activity_duration * 60
                end_time_obj = datetime.fromtimestamp(current_sim_time)

                self._log_event(
                    case_id=case_id, activity=chosen_label,
                    timestamp_start=start_time_obj,
                    timestamp_end=end_time_obj,
                    object_name=object_name,
                    object_type=object_type,
                    higher_level_activity=higher_level_activity,
                    object_attributes=object_attributes,
                )

                activity_count += 1
                current_sim_time += 1  # 1 second gap

                print(f"    [{activity_count}] '{chosen_label}' "
                      f"(dur={activity_duration:.1f} min, "
                      f"src={weight_source})")

            if step >= max_steps:
                print(f"    WARNING: Max steps ({max_steps}) reached.")

            print(f"  Completed {object_name} petri_net_statistical_memory "
                  f"with {activity_count} activities")

    def _simulate_statistical_for_object(self, case_id, object_attributes,
                                          current_sim_time, object_name,
                                          object_type,
                                          higher_level_activity):
        """
        Statistical simulation for a single object group.
        Extracted so petri_net mode can fall back to it per-group.
        """
        start_activities = self._get_start_activities(
            object_name, object_type, higher_level_activity
        )
        if not start_activities:
            print(f"  WARNING: No start activities for {object_name} — SKIP")
            return

        current_activity = random.choice(start_activities)
        activity_count = 0
        activity_history = []

        while current_activity:
            _HIST_MODES = ('ml_duration_only_with_activity_past',
                           'ml_duration_only_with_activity_past_point_estimate',
                           'ml_global_model')
            activity_duration = self._get_activity_duration(
                current_activity, object_name, object_type,
                higher_level_activity, object_attributes,
                activity_history=activity_history if self.mode in _HIST_MODES else None,
                activity_index=activity_count,
            )

            start_time_obj = datetime.fromtimestamp(current_sim_time)
            current_sim_time += activity_duration * 60
            end_time_obj = datetime.fromtimestamp(current_sim_time)

            self._log_event(
                case_id=case_id, activity=current_activity,
                timestamp_start=start_time_obj, timestamp_end=end_time_obj,
                object_name=object_name, object_type=object_type,
                higher_level_activity=higher_level_activity,
                object_attributes=object_attributes,
            )

            activity_history.insert(0, (current_activity, activity_duration))
            activity_history = activity_history[:2]
            activity_count += 1

            next_activity = self._get_next_activity(
                current_activity, object_name, object_type,
                higher_level_activity, object_attributes,
            )
            if not next_activity:
                break
            current_activity = next_activity
            current_sim_time += 1

    def _simulate_petri_net_energy_aware_for_case(
        self, case_id, object_attributes, start_time,
        enable_duration=True, enable_transitions=True,
    ):
        """
        Simulate one case using the Petri net token game where the energy
        state (predicted sensor curves from the previous activity) feeds
        back into:
          - transition weight blending  (if enable_transitions=True)
          - duration distribution mean shift  (if enable_duration=True)

        The spread (σ) of the duration distribution is NEVER modified so
        variability is preserved.

        After every visible transition fires, the energy pipeline predicts
        a sensor curve for the chosen activity.  That curve is summarised to
        (mean, end, std) per sensor and stored as `current_energy_state`.
        A ValueError is raised if any expected feature column is missing.
        """
        from sim_extractor import _energy_summary

        current_sim_time = start_time.timestamp()
        unique_objects = (
            self.activity_stats[['object', 'object_type', 'higher_level_activity']]
            .drop_duplicates(subset=['higher_level_activity'])
        )

        for _, obj_config in unique_objects.iterrows():
            object_name = str(obj_config['object']).strip()
            object_type = str(obj_config['object_type']).strip()
            higher_level_activity = (
                str(obj_config['higher_level_activity']).strip()
                if pd.notna(obj_config['higher_level_activity']) else None
            )

            key = (object_name, object_type, higher_level_activity)
            model = self.process_models.get(key) if self.process_models else None

            if model is None:
                available = list(self.process_models.keys()) if self.process_models else []
                raise RuntimeError(
                    f"petri_net_energy mode: no Petri net found for key {key}. "
                    f"Available keys: {available}")

            net             = model['net']
            im              = model['im']
            fm              = model['fm']
            stochastic_map    = model.get('stochastic_map', {})
            label_stochastic  = model.get('label_stochastic', {})
            decision_weights  = model.get('decision_weights', {})
            max_case_length   = model.get('max_case_length', 200)

            marking          = copy.copy(im)
            activity_count   = 0
            activity_history = []
            prev_activity    = None
            max_steps        = max(max_case_length * 2, 50)
            step             = 0

            # Seed energy state from the training-time means of start activities
            # for this object group.  This is more realistic than averaging all
            # activities: the first fired transition will immediately update the
            # state, so we want a seed that reflects what the process looks like
            # at the beginning.  Falls back to the global mean if no start-activity
            # modifier has a stored mean.
            if self.energy_state_columns:
                _start_acts = set(self._get_start_activities(
                    object_name, object_type, higher_level_activity
                ))
                all_mods = {**self.energy_duration_modifiers, **self.energy_transition_modifiers}
                # Prefer means from start-activity modifiers; fall back to all modifiers
                _candidates = {k: m for k, m in all_mods.items() if k in _start_acts and hasattr(m, '_train_feature_mean')}
                if not _candidates:
                    _candidates = {k: m for k, m in all_mods.items() if hasattr(m, '_train_feature_mean')}
                all_means: dict[str, list] = {c: [] for c in self.energy_state_columns}
                for _mod in _candidates.values():
                    for c, v in _mod._train_feature_mean.items():
                        if c in all_means:
                            all_means[c].append(v)
                seed_state = {
                    c: float(np.mean(vals)) if vals else 0.0
                    for c, vals in all_means.items()
                }
                # Seed ef_* using the global mean across all start-activity exog means
                if self.activity_exog_means:
                    _ef_seed_vals: dict[str, list] = {}
                    for _sa in (_start_acts or self.activity_exog_means.keys()):
                        for _col, _v in self.activity_exog_means.get(_sa, {}).items():
                            _ef_seed_vals.setdefault(_col, []).append(_v)
                    if not _ef_seed_vals:  # fallback: all activities
                        for _act_means in self.activity_exog_means.values():
                            for _col, _v in _act_means.items():
                                _ef_seed_vals.setdefault(_col, []).append(_v)
                    for _col, _vs in _ef_seed_vals.items():
                        seed_state[_col] = float(np.mean(_vs))
                current_energy_state = seed_state if seed_state else None
            else:
                current_energy_state = None

            if self.verbose:
                mode_tag = ('dur+tr' if enable_duration and enable_transitions
                            else 'dur' if enable_duration else 'tr')
                print(f"\nCase {case_id}: energy-aware PN simulation "
                      f"[{mode_tag}] for {object_name} ({object_type})")

            while step < max_steps:
                step += 1

                # ── Check final marking ───────────────────────────────
                if marking == fm:
                    print(f"    Final marking reached after {activity_count} activities.")
                    break
                if all(marking.get(p, 0) >= fm[p] for p in fm) and activity_count > 0:
                    print(f"    Final marking (subset) reached after {activity_count} activities.")
                    break

                # ── Enabled transitions ───────────────────────────────
                enabled = self._get_enabled_transitions(net, marking)
                if not enabled:
                    print(f"    Deadlock after {activity_count} activities.")
                    break

                # Sort to make sampling deterministic relative to seed
                enabled_list = sorted(
                    list(enabled),
                    key=lambda t: (str(t.label) if t.label is not None else '', str(t.name)),
                )

                # ── Fetch base PN weights ─────────────────────────────
                weights = [stochastic_map.get(t, 1.0) for t in enabled_list]
                weight_source = 'stochastic_map'

                # ── Energy bias on transition weights ─────────────────
                if (enable_transitions
                        and current_energy_state is not None
                        and prev_activity in self.energy_transition_modifiers):
                    clf = self.energy_transition_modifiers[prev_activity]
                    energy_vec = [current_energy_state[c] for c in self.energy_state_columns]
                    try:
                        log_proba = clf.predict_log_proba([energy_vec])[0]
                        logit_map = dict(zip(clf.classes_, log_proba))
                        lo, hi    = self.logit_bias_clip

                        # Bias ONLY visible transitions (silent transitions 'label is None' use base PN weights)
                        for i, t in enumerate(enabled_list):
                            if t.label is not None:
                                lbl = str(t.label).strip()
                                bias = logit_map.get(lbl, 0.0)
                                bias = max(lo, min(hi, bias))
                                weights[i] = weights[i] * np.exp(bias)

                        weight_source += '+energy'
                    except Exception as exc:
                        print(f"    WARNING: transition energy bias failed: {exc}")

                # ── Sample the NEXT transition ────────────────────────
                total = sum(weights)
                if total <= 0:
                    chosen_transition = random.choice(enabled_list)
                else:
                    probs = [w / total for w in weights]
                    chosen_transition = np.random.choice(enabled_list, p=probs)

                # ── Fire the transition ───────────────────────────────
                marking = self._fire_transition(marking, chosen_transition)

                if chosen_transition.label is not None:
                    chosen_label = str(chosen_transition.label).strip()
                    if self.verbose:
                        print(f"    Next: {chosen_label} (source={weight_source})")

                    # ── Evaluate duration using the BASE ML or STAT config
                    base_dur = self._get_activity_duration(
                        chosen_label, object_name, object_type,
                        higher_level_activity, object_attributes,
                        activity_history=activity_history,
                        activity_index=activity_count,
                        override_mode=self.base_simulation_mode
                    )

                    # ── Apply Multiplicative Energy Scale ─────────────
                    if (enable_duration
                            and current_energy_state is not None
                            and chosen_label in self.energy_duration_modifiers):
                        mdl = self.energy_duration_modifiers[chosen_label]
                        energy_vec = [current_energy_state[c] for c in self.energy_state_columns]
                        try:
                            log_scale = float(mdl.predict([energy_vec])[0])
                            lo, hi    = self.duration_scale_clip
                            scale     = max(lo, min(hi, np.exp(log_scale)))

                            # Mathematically correct duration scaling:
                            activity_duration = max(0.1, base_dur * scale)
                            if self.verbose:
                                print(f"    Duration: {activity_duration:.1f} min "
                                      f"(base={base_dur:.1f}, scale={scale:.3f})")
                        except Exception as exc:
                            if self.verbose:
                                print(f"    WARNING: duration energy scale failed: {exc}")
                            activity_duration = base_dur
                    else:
                        activity_duration = base_dur

                    # ── Timestamps + log ───────────────────────────────
                    start_time_obj   = datetime.fromtimestamp(current_sim_time)
                    current_sim_time += activity_duration * 60
                    end_time_obj     = datetime.fromtimestamp(current_sim_time)

                    self._log_event(
                        case_id=case_id, activity=chosen_label,
                        timestamp_start=start_time_obj, timestamp_end=end_time_obj,
                        object_name=object_name, object_type=object_type,
                        higher_level_activity=higher_level_activity,
                        object_attributes=object_attributes,
                    )
                    
                    # Update history to support ML Base models
                    activity_history.insert(0, (chosen_label, activity_duration))
                    activity_history = activity_history[:2]
                
                else:
                    # Silent transition — no log, 0 duration jump
                    chosen_label = None


                # ── Update energy state ────────────────────────────────
                # energy_pipelines is keyed [sensor][activity][object].
                # Fall back to the first available object pipeline when the
                # exact object is not found (handles unseen objects at test time).
                if self.energy_pipelines and chosen_label is not None:
                    new_energy_state = {}
                    for sensor, act_map in self.energy_pipelines.items():
                        try:
                            obj_map = act_map.get(chosen_label, {})
                            ep = obj_map.get(object_name) or (
                                next(iter(obj_map.values())) if obj_map else None
                            )
                            if ep is None:
                                # No pipeline for this sensor/activity — carry forward previous state
                                for sfx in ('_mean', '_end', '_std'):
                                    key = f'{sensor}{sfx}'
                                    if key in current_energy_state:
                                        new_energy_state[key] = current_energy_state[key]
                                continue
                            ref_curve  = ep.get('reference_curve')
                            if ref_curve is None:
                                raise ValueError(
                                    f"energy_pipelines['{sensor}']['{chosen_label}']['{object_name}'] "
                                    f"has no 'reference_curve'."
                                )
                            predict_fn = ep.get('predict_fn')
                            # Build exog_values dict for pipelines that use external factors
                            _exog_vals = {}
                            if ep.get('exog_cols') and self.activity_exog_means:
                                _act_means = self.activity_exog_means.get(chosen_label, {})
                                _exog_vals = {
                                    col: np.array([v])
                                    for col, v in _act_means.items()
                                    if col in ep['exog_cols']
                                }
                            # Use predicted duration to set curve_length so the model
                            # receives the same feature value it saw during training.
                            # Resample ref_curve to the expected number of timesteps.
                            _n_ts = max(2, round(activity_duration / self.temporal_resolution_minutes))
                            _input_curve = np.interp(
                                np.linspace(0, 1, _n_ts),
                                np.linspace(0, 1, len(ref_curve)),
                                ref_curve,
                            )
                            curve = (predict_fn(
                                         raw_values=_input_curve,
                                         activity=chosen_label,
                                         object_attributes=object_attributes,
                                         exog=_exog_vals if _exog_vals else None,
                                     ) if predict_fn is not None
                                     else _input_curve)

                            # Log the full curve for evaluation later
                            if self.events and self.events[-1]['activity'] == chosen_label:
                                if 'simulated_energy_curves' not in self.events[-1]:
                                    self.events[-1]['simulated_energy_curves'] = {}
                                self.events[-1]['simulated_energy_curves'][sensor] = curve

                            from sim_extractor import _energy_summary
                            new_energy_state.update(_energy_summary(curve, sensor))
                        except Exception as exc:
                            raise ValueError(
                                f"Energy pipeline prediction failed for sensor '{sensor}', "
                                f"activity '{chosen_label}', object '{object_name}': {exc}"
                            ) from exc

                    # Populate ef_* columns from precomputed activity means
                    if self.activity_exog_means and chosen_label in self.activity_exog_means:
                        new_energy_state.update(self.activity_exog_means[chosen_label])
                    # Always carry forward any state columns not yet populated (e.g. sensors
                    # with no trained pipeline, or weather sensors absent from energy_pipelines)
                    if current_energy_state:
                        for _k, _v in current_energy_state.items():
                            if _k not in new_energy_state:
                                new_energy_state[_k] = _v

                    # Validate — raise if any expected column is missing
                    if self.energy_state_columns:
                        missing = [c for c in self.energy_state_columns
                                   if c not in new_energy_state]
                        if missing:
                            raise ValueError(
                                f"Energy state missing columns: {missing}. "
                                f"Check sensor config for activity '{chosen_label}'."
                            )

                    current_energy_state = new_energy_state

                # Always track prev_activity (needed by transition modifier on next step)
                if chosen_label is not None:
                    prev_activity = chosen_label

                activity_count   += 1
                current_sim_time += 1  # 1-second gap between activities

                if chosen_label is not None:
                    if self.verbose:
                        print(f"    [{activity_count}] '{chosen_label}' "
                              f"dur={activity_duration:.1f} min")

            if step >= max_steps:
                print(f"    WARNING: max steps ({max_steps}) reached.")

            print(f"  Completed {object_name} energy-aware PN with "
                  f"{activity_count} activities")

    def _simulate_petri_net_energy_direct_for_case(
        self, case_id, object_attributes, start_time,
        enable_duration=True, enable_transitions=True,
    ):
        """
        Simulate one case using the Petri net for structure, but with ML models
        that *directly* predict duration and next-activity from the energy state
        of the previous activity — no multiplicative correction, full prediction.

        Duration:   energy_state → regressor → duration_minutes
                    Falls back to statistical per-activity mean when no ML model.
        Transition: energy_state → classifier → sample from predict_proba
                    Falls back to base PN stochastic weights when no ML model.

        Models come from ``self.energy_duration_modifiers`` /
        ``self.energy_transition_modifiers`` which must have been populated by
        ``extract_energy_direct_models`` (not the modifier variant).
        """
        current_sim_time = start_time.timestamp()
        unique_objects = (
            self.activity_stats[['object', 'object_type', 'higher_level_activity']]
            .drop_duplicates(subset=['higher_level_activity'])
        )

        for _, obj_config in unique_objects.iterrows():
            object_name             = str(obj_config['object']).strip()
            object_type             = str(obj_config['object_type']).strip()
            higher_level_activity   = (
                str(obj_config['higher_level_activity']).strip()
                if pd.notna(obj_config['higher_level_activity']) else None
            )

            key   = (object_name, object_type, higher_level_activity)
            model = self.process_models.get(key) if self.process_models else None

            if model is None:
                available = list(self.process_models.keys()) if self.process_models else []
                raise RuntimeError(
                    f"petri_net_energy_direct mode: no Petri net found for key {key}. "
                    f"Available keys: {available}")

            net             = model['net']
            im              = model['im']
            fm              = model['fm']
            stochastic_map  = model.get('stochastic_map', {})
            max_case_length = model.get('max_case_length', 200)

            marking          = copy.copy(im)
            activity_count   = 0
            activity_history = []
            prev_activity    = None
            max_steps        = max(max_case_length * 2, 50)
            step             = 0

            # Seed energy state from start-activity modifier means (same logic as modifier method)
            if self.energy_state_columns:
                _start_acts = set(self._get_start_activities(
                    object_name, object_type, higher_level_activity
                ))
                all_mods = {**self.energy_duration_modifiers, **self.energy_transition_modifiers}
                _candidates = {k: m for k, m in all_mods.items() if k in _start_acts and hasattr(m, '_train_feature_mean')}
                if not _candidates:
                    _candidates = {k: m for k, m in all_mods.items() if hasattr(m, '_train_feature_mean')}
                all_means = {c: [] for c in self.energy_state_columns}
                for _mod in _candidates.values():
                    for c, v in _mod._train_feature_mean.items():
                        if c in all_means:
                            all_means[c].append(v)
                seed_state = {
                    c: float(np.mean(vals)) if vals else 0.0
                    for c, vals in all_means.items()
                }
                # Seed ef_* columns using start-activity exog means
                if self.activity_exog_means:
                    _ef_seed_vals_d: dict[str, list] = {}
                    for _sa in (_start_acts or self.activity_exog_means.keys()):
                        for _col, _v in self.activity_exog_means.get(_sa, {}).items():
                            _ef_seed_vals_d.setdefault(_col, []).append(_v)
                    if not _ef_seed_vals_d:
                        for _act_means in self.activity_exog_means.values():
                            for _col, _v in _act_means.items():
                                _ef_seed_vals_d.setdefault(_col, []).append(_v)
                    for _col, _vs in _ef_seed_vals_d.items():
                        seed_state[_col] = float(np.mean(_vs))
                # Zero-init prev_activity and curr_activity one-hots at case start
                for _col in self.energy_state_columns:
                    if _col.startswith('prev_act_') or _col.startswith('curr_act_'):
                        seed_state[_col] = 0.0
                current_energy_state = seed_state if seed_state else None
            else:
                current_energy_state = None

            while marking != fm and step < max_steps:
                step += 1
                enabled = self._get_enabled_transitions(net, marking)
                if not enabled:
                    print(f"    Deadlock after {activity_count} activities.")
                    break

                enabled_list = sorted(
                    enabled,
                    key=lambda t: (str(t.label) if t.label is not None else '', str(t.name)),
                )

                # ── Transition sampling ───────────────────────────────
                # If ML classifier exists for prev_activity AND enable_transitions:
                #   sample next activity directly from predict_proba, then pick
                #   the enabled transition whose label matches best.
                # Else: use base PN stochastic weights.
                chosen_transition = None
                visible_enabled = [t for t in enabled_list if t.label is not None]
                _has_global_tr = '__global__' in self.energy_transition_modifiers
                _tr_lookup_key = '__global__' if _has_global_tr else (prev_activity if prev_activity else None)
                if len(visible_enabled) <= 1:
                    pass  # deterministic or single-option step — PN fallback below is correct
                elif (enable_transitions
                        and current_energy_state is not None
                        and prev_activity is not None
                        and _tr_lookup_key is not None
                        and _tr_lookup_key in self.energy_transition_modifiers):
                    clf = self.energy_transition_modifiers[_tr_lookup_key]
                    if _has_global_tr:
                        # Global model: set curr_act_* to the query activity (prev_activity here)
                        _ev = dict(current_energy_state)
                        for _c in self.energy_state_columns:
                            if _c.startswith('curr_act_'):
                                _ev[_c] = 1.0 if _c == f'curr_act_{prev_activity}' else 0.0
                        energy_vec = [_ev.get(c, 0.0) for c in self.energy_state_columns]
                    else:
                        energy_vec = [current_energy_state[c] for c in self.energy_state_columns]
                    try:
                        proba_vec  = clf.predict_proba([energy_vec])[0]
                        label_prob = dict(zip(clf.classes_, proba_vec))

                        # Score each visible enabled transition by ML probability
                        weights = []
                        for t in enabled_list:
                            if t.label is not None:
                                weights.append(label_prob.get(str(t.label).strip(), 1e-6))
                            else:
                                # Silent transitions: use base PN weight
                                weights.append(stochastic_map.get(t, 1e-6))

                        total = sum(weights)
                        if total > 0:
                            probs = [w / total for w in weights]
                            chosen_transition = np.random.choice(enabled_list, p=probs)
                    except Exception as exc:
                        if self.verbose:
                            print(f"    WARNING: direct transition sampling failed: {exc}")

                if chosen_transition is None:
                    # Fallback: base PN stochastic weights
                    weights = [stochastic_map.get(t, 1.0) for t in enabled_list]
                    total   = sum(weights)
                    if total <= 0:
                        chosen_transition = random.choice(enabled_list)
                    else:
                        probs = [w / total for w in weights]
                        chosen_transition = np.random.choice(enabled_list, p=probs)

                marking = self._fire_transition(marking, chosen_transition)

                if chosen_transition.label is not None:
                    chosen_label = str(chosen_transition.label).strip()
                    activity_count += 1

                    # ── Duration: direct ML prediction or statistical ──
                    _has_global_dur = '__global__' in self.energy_duration_modifiers
                    _dur_lookup_key = '__global__' if _has_global_dur else chosen_label
                    if (enable_duration
                            and current_energy_state is not None
                            and _dur_lookup_key in self.energy_duration_modifiers):
                        mdl = self.energy_duration_modifiers[_dur_lookup_key]
                        if _has_global_dur:
                            # Global model: set curr_act_* to the activity whose duration we want
                            _ev = dict(current_energy_state)
                            for _c in self.energy_state_columns:
                                if _c.startswith('curr_act_'):
                                    _ev[_c] = 1.0 if _c == f'curr_act_{chosen_label}' else 0.0
                            energy_vec = [_ev.get(c, 0.0) for c in self.energy_state_columns]
                        else:
                            energy_vec = [current_energy_state[c] for c in self.energy_state_columns]
                        try:
                            _raw = float(mdl.predict([energy_vec])[0])
                            if getattr(mdl, '_log_duration', False):
                                _raw = np.exp(_raw)
                            activity_duration = max(0.1, _raw)
                            if self.verbose:
                                print(f"    [{chosen_label}] direct ML duration: "
                                      f"{activity_duration:.1f} min")
                        except Exception as exc:
                            if self.verbose:
                                print(f"    WARNING: direct duration prediction failed: {exc}")
                            activity_duration = self._get_activity_duration(
                                chosen_label, object_name, object_type,
                                higher_level_activity, object_attributes,
                                activity_history=activity_history,
                                activity_index=activity_count,
                                override_mode=self.base_simulation_mode,
                            )
                    else:
                        activity_duration = self._get_activity_duration(
                            chosen_label, object_name, object_type,
                            higher_level_activity, object_attributes,
                            activity_history=activity_history,
                            activity_index=activity_count,
                            override_mode=self.base_simulation_mode,
                        )

                    start_time_obj   = datetime.fromtimestamp(current_sim_time)
                    current_sim_time += activity_duration * 60
                    end_time_obj     = datetime.fromtimestamp(current_sim_time)

                    self._log_event(
                        case_id=case_id, activity=chosen_label,
                        timestamp_start=start_time_obj, timestamp_end=end_time_obj,
                        object_name=object_name, object_type=object_type,
                        higher_level_activity=higher_level_activity,
                        object_attributes=object_attributes,
                    )

                    activity_history.insert(0, (chosen_label, activity_duration))
                    activity_history = activity_history[:2]
                    prev_activity    = chosen_label

                    # ── Update energy state from simulated curve ──────────────
                    # Predict the sensor curve for the just-fired activity and
                    # compress it to (mean, end, std) so the next iteration has
                    # a realistic energy state to drive transition and duration
                    # predictions — this is the closed feedback loop.
                    if self.energy_pipelines and chosen_label is not None:
                        new_energy_state = {}
                        for sensor, act_map in self.energy_pipelines.items():
                            try:
                                obj_map = act_map.get(chosen_label, {})
                                ep = obj_map.get(object_name) or (
                                    next(iter(obj_map.values())) if obj_map else None
                                )
                                if ep is None:
                                    for sfx in ('_mean', '_end', '_std'):
                                        key = f'{sensor}{sfx}'
                                        if current_energy_state and key in current_energy_state:
                                            new_energy_state[key] = current_energy_state[key]
                                    continue
                                ref_curve = ep.get('reference_curve')
                                if ref_curve is None:
                                    raise ValueError(
                                        f"energy_pipelines['{sensor}']['{chosen_label}']['{object_name}'] "
                                        f"has no 'reference_curve'."
                                    )
                                predict_fn = ep.get('predict_fn')
                                _exog_vals = {}
                                if ep.get('exog_cols') and self.activity_exog_means:
                                    _act_means = self.activity_exog_means.get(chosen_label, {})
                                    _exog_vals = {
                                        col: np.array([v])
                                        for col, v in _act_means.items()
                                        if col in ep['exog_cols']
                                    }
                                _n_ts = max(2, round(activity_duration / self.temporal_resolution_minutes))
                                _input_curve = np.interp(
                                    np.linspace(0, 1, _n_ts),
                                    np.linspace(0, 1, len(ref_curve)),
                                    ref_curve,
                                )
                                curve = (predict_fn(
                                             raw_values=_input_curve,
                                             activity=chosen_label,
                                             object_attributes=object_attributes,
                                             exog=_exog_vals if _exog_vals else None,
                                         ) if predict_fn is not None
                                         else _input_curve)

                                if self.events and self.events[-1]['activity'] == chosen_label:
                                    if 'simulated_energy_curves' not in self.events[-1]:
                                        self.events[-1]['simulated_energy_curves'] = {}
                                    self.events[-1]['simulated_energy_curves'][sensor] = curve

                                from sim_extractor import _energy_summary
                                new_energy_state.update(_energy_summary(curve, sensor))
                            except Exception as exc:
                                raise ValueError(
                                    f"Energy pipeline prediction failed for sensor '{sensor}', "
                                    f"activity '{chosen_label}', object '{object_name}': {exc}"
                                ) from exc

                        if self.activity_exog_means and chosen_label in self.activity_exog_means:
                            new_energy_state.update(self.activity_exog_means[chosen_label])
                        if current_energy_state:
                            for _k, _v in current_energy_state.items():
                                if _k not in new_energy_state:
                                    new_energy_state[_k] = _v

                        if self.energy_state_columns:
                            missing = [c for c in self.energy_state_columns
                                       if c not in new_energy_state]
                            if missing:
                                raise ValueError(
                                    f"Energy state missing columns: {missing}. "
                                    f"Check sensor config for activity '{chosen_label}'."
                                )

                        current_energy_state = new_energy_state
                    elif current_energy_state is not None and self.activity_exog_means:
                        _act_ef = self.activity_exog_means.get(chosen_label, {})
                        if _act_ef:
                            current_energy_state = {**current_energy_state, **_act_ef}

                    # Update prev_activity one-hot; keep curr_act_* zeroed (set on-the-fly per call)
                    if current_energy_state is not None:
                        _prev_act_cols = [c for c in self.energy_state_columns
                                          if c.startswith('prev_act_')]
                        if _prev_act_cols:
                            for _pc in _prev_act_cols:
                                current_energy_state[_pc] = 0.0
                            _pa_key = f'prev_act_{chosen_label}'
                            if _pa_key in self.energy_state_columns:
                                current_energy_state[_pa_key] = 1.0
                        for _cc in self.energy_state_columns:
                            if _cc.startswith('curr_act_'):
                                current_energy_state[_cc] = 0.0

            if step >= max_steps:
                print(f"    WARNING: max steps ({max_steps}) reached.")

            print(f"  Completed {object_name} energy-direct PN with "
                  f"{activity_count} activities")

    # ------------------------------------------------------------------
    # Quantile-blend: quantile-conditioned duration + entropy-blended transitions
    # ------------------------------------------------------------------

    def _simulate_petri_net_quantile_blend_for_case(
        self, case_id, object_attributes, start_time,
    ):
        """
        petri_net_quantile_blend:

        Duration  — ML predicts a quantile q in [0,1] for the fitted distribution;
                    small Gaussian jitter is added to preserve variance; then
                    duration = dist.ppf(q).  Falls back to dist.sample() when no
                    model is stored for the activity.

        Transition — ML predict_proba is entropy-weighted blended with PN stochastic
                    weights.  High-entropy (uncertain) predictions yield weight ≈ 0
                    and the simulation behaves like petri_net_inductive.
        """
        from sim_extractor import _energy_summary, _DIST_MAP
        import scipy.stats as _scipy_stats

        current_sim_time = start_time.timestamp()
        unique_objects = (
            self.activity_stats[['object', 'object_type', 'higher_level_activity']]
            .drop_duplicates(subset=['higher_level_activity'])
        )

        for _, obj_config in unique_objects.iterrows():
            object_name           = str(obj_config['object']).strip()
            object_type           = str(obj_config['object_type']).strip()
            higher_level_activity = (
                str(obj_config['higher_level_activity']).strip()
                if pd.notna(obj_config['higher_level_activity']) else None
            )

            key   = (object_name, object_type, higher_level_activity)
            model = self.process_models.get(key) if self.process_models else None
            if model is None:
                raise RuntimeError(
                    f"petri_net_quantile_blend: no Petri net for key {key}. "
                    f"Available: {list(self.process_models.keys()) if self.process_models else []}"
                )

            net              = model['net']
            im               = model['im']
            fm               = model['fm']
            stochastic_map   = model.get('stochastic_map', {})
            max_case_length  = model.get('max_case_length', 200)

            marking          = copy.copy(im)
            activity_count   = 0
            activity_history = []
            prev_activity    = None
            max_steps        = max(max_case_length * 2, 50)
            step             = 0

            # Seed energy state
            if self.energy_state_columns:
                _start_acts = set(self._get_start_activities(
                    object_name, object_type, higher_level_activity
                ))
                all_mods = {**self.energy_duration_modifiers, **self.energy_transition_modifiers}
                _cands = {k: m for k, m in all_mods.items()
                          if k in _start_acts and hasattr(m, '_train_feature_mean')}
                if not _cands:
                    _cands = {k: m for k, m in all_mods.items()
                              if hasattr(m, '_train_feature_mean')}
                _all_means: dict[str, list] = {c: [] for c in self.energy_state_columns}
                for _mod in _cands.values():
                    for c, v in _mod._train_feature_mean.items():
                        if c in _all_means:
                            _all_means[c].append(v)
                seed_state = {c: float(np.mean(vs)) if vs else 0.0
                              for c, vs in _all_means.items()}
                for _col in self.energy_state_columns:
                    if _col.startswith('prev_act_'):
                        seed_state[_col] = 0.0
                current_energy_state = seed_state if seed_state else None
            else:
                current_energy_state = None

            if self.verbose:
                print(f"\nCase {case_id}: quantile-blend PN simulation "
                      f"for {object_name} ({object_type})")

            while step < max_steps:
                step += 1

                if marking == fm:
                    if self.verbose:
                        print(f"    Final marking after {activity_count} activities.")
                    break
                if all(marking.get(p, 0) >= fm[p] for p in fm) and activity_count > 0:
                    break

                enabled = self._get_enabled_transitions(net, marking)
                if not enabled:
                    if self.verbose:
                        print(f"    Deadlock after {activity_count} activities.")
                    break

                enabled_list = sorted(
                    enabled,
                    key=lambda t: (str(t.label) if t.label is not None else '', str(t.name)),
                )

                # ── Transition sampling: entropy-weighted blend ───────
                chosen_transition = None
                visible_enabled   = [t for t in enabled_list if t.label is not None]

                if (len(visible_enabled) > 1
                        and current_energy_state is not None
                        and prev_activity is not None
                        and prev_activity in self.energy_transition_modifiers):
                    clf = self.energy_transition_modifiers[prev_activity]
                    energy_vec = [current_energy_state.get(c, 0.0)
                                  for c in self.energy_state_columns]
                    try:
                        proba_vec  = clf.predict_proba([energy_vec])[0]
                        label_prob = dict(zip(clf.classes_, proba_vec))

                        # Entropy-based confidence weight
                        H       = -float(np.sum(proba_vec * np.log(proba_vec + 1e-10)))
                        H_max   = np.log(max(len(proba_vec), 2))
                        conf    = float(np.clip(1.0 - H / H_max, 0.0, 1.0))

                        # Blend: (1-conf)*PN + conf*ML
                        pn_total = sum(stochastic_map.get(t, 1.0) for t in enabled_list) or 1.0
                        weights  = []
                        for t in enabled_list:
                            pn_w = stochastic_map.get(t, 1.0) / pn_total
                            if t.label is not None:
                                ml_p = label_prob.get(str(t.label).strip(), 1e-6)
                            else:
                                ml_p = pn_w
                            weights.append((1.0 - conf) * pn_w + conf * ml_p)

                        total = sum(weights)
                        if total > 0:
                            probs = [w / total for w in weights]
                            chosen_transition = np.random.choice(enabled_list, p=probs)
                    except Exception as exc:
                        if self.verbose:
                            print(f"    WARNING: quantile-blend transition failed: {exc}")

                if chosen_transition is None:
                    # PN fallback
                    weights = [stochastic_map.get(t, 1.0) for t in enabled_list]
                    total   = sum(weights)
                    if total <= 0:
                        chosen_transition = random.choice(enabled_list)
                    else:
                        probs = [w / total for w in weights]
                        chosen_transition = np.random.choice(enabled_list, p=probs)

                marking = self._fire_transition(marking, chosen_transition)

                if chosen_transition.label is not None:
                    chosen_label  = str(chosen_transition.label).strip()
                    activity_count += 1

                    # ── Duration: quantile-conditioned ───────────────────
                    act_key = (chosen_label, object_name, object_type, higher_level_activity)
                    dur_mdl = self.energy_duration_modifiers.get(chosen_label)

                    if (dur_mdl is not None
                            and getattr(dur_mdl, '_is_quantile', False)
                            and current_energy_state is not None):
                        try:
                            energy_vec = [current_energy_state.get(c, 0.0)
                                          for c in self.energy_state_columns]
                            q_pred   = float(np.clip(dur_mdl.predict([energy_vec])[0], 0.01, 0.99))
                            jitter   = getattr(dur_mdl, '_jitter_std', 0.15)
                            q_final  = float(np.clip(q_pred + np.random.normal(0, jitter), 0.01, 0.99))
                            dn       = dur_mdl._dist_name
                            dp       = dur_mdl._dist_params
                            dist_obj = _DIST_MAP.get(dn, _scipy_stats.norm)
                            activity_duration = max(0.1, float(dist_obj.ppf(q_final, *dp)))
                            if self.verbose:
                                print(f"    [{chosen_label}] quantile q={q_final:.3f} "
                                      f"→ {activity_duration:.1f} min")
                        except Exception as exc:
                            if self.verbose:
                                print(f"    WARNING: quantile duration failed: {exc}")
                            activity_duration = self._get_activity_duration(
                                chosen_label, object_name, object_type,
                                higher_level_activity, object_attributes,
                            )
                    else:
                        activity_duration = self._get_activity_duration(
                            chosen_label, object_name, object_type,
                            higher_level_activity, object_attributes,
                        )

                    start_time_obj   = datetime.fromtimestamp(current_sim_time)
                    current_sim_time += activity_duration * 60
                    end_time_obj     = datetime.fromtimestamp(current_sim_time)

                    self._log_event(
                        case_id=case_id, activity=chosen_label,
                        timestamp_start=start_time_obj, timestamp_end=end_time_obj,
                        object_name=object_name, object_type=object_type,
                        higher_level_activity=higher_level_activity,
                        object_attributes=object_attributes,
                    )

                    activity_history.insert(0, (chosen_label, activity_duration))
                    activity_history = activity_history[:2]

                    # Update energy state from pipeline
                    if self.energy_pipelines and current_energy_state is not None:
                        new_energy_state = {}
                        for sensor, act_map in self.energy_pipelines.items():
                            try:
                                obj_map   = act_map.get(chosen_label, {})
                                ep        = obj_map.get(object_name) or (
                                    next(iter(obj_map.values())) if obj_map else None
                                )
                                if ep is None:
                                    for sfx in ('_mean', '_end', '_std', '_start', '_delta', '_integral'):
                                        k = f'{sensor}{sfx}'
                                        if k in current_energy_state:
                                            new_energy_state[k] = current_energy_state[k]
                                    continue
                                ref_curve = ep.get('reference_curve')
                                if ref_curve is None:
                                    raise ValueError(f"No reference_curve for {sensor}/{chosen_label}")
                                new_energy_state.update(_energy_summary(np.array(ref_curve), sensor))
                            except Exception as exc:
                                if self.verbose:
                                    print(f"    WARNING: energy pipeline update failed: {exc}")
                        if new_energy_state:
                            current_energy_state = {**current_energy_state, **new_energy_state}

                    # Update prev_act one-hots
                    for _col in self.energy_state_columns:
                        if _col.startswith('prev_act_'):
                            current_energy_state[_col] = (
                                1.0 if _col == f'prev_act_{chosen_label}' else 0.0
                            )
                    prev_activity = chosen_label

            if step >= max_steps and self.verbose:
                print(f"    WARNING: max steps ({max_steps}) reached.")
            if self.verbose:
                print(f"  Completed {object_name} quantile-blend PN with "
                      f"{activity_count} activities")

    def _simulate_process_for_case(self, case_id, object_attributes, start_time):
        """Simulate process for one case using probabilistic end transitions"""
        print(f"[DEBUG _simulate_process_for_case] self.mode = '{self.mode}'")
        # ── Petri net mode delegates to its own method ────────────────
        if self.mode == 'petri_net':
            self._simulate_petri_net_for_case(
                case_id, object_attributes, start_time
            )
            return

        # ── Petri net + statistical combined mode ─────────────────────
        if self.mode == 'petri_net_statistical':
            self._simulate_petri_net_statistical_for_case(
                case_id, object_attributes, start_time
            )
            return

        # ── Petri net + memory (history-dependent) mode ───────────────
        if self.mode == 'petri_net_statistical_memory':
            self._simulate_petri_net_statistical_memory_for_case(
                case_id, object_attributes, start_time
            )
            return

        # ── Energy-aware Petri net modes (modifier approach) ─────────
        if self.mode in ('petri_net_energy_aware',
                         'petri_net_energy_duration_aware',
                         'petri_net_energy_transition_aware'):
            enable_dur = self.mode != 'petri_net_energy_transition_aware'
            enable_tr  = self.mode != 'petri_net_energy_duration_aware'
            self._simulate_petri_net_energy_aware_for_case(
                case_id, object_attributes, start_time,
                enable_duration=enable_dur,
                enable_transitions=enable_tr,
            )
            return

        # ── Energy-direct Petri net modes (full ML prediction) ───────
        if self.mode in ('petri_net_energy_direct',
                         'petri_net_energy_direct_duration_only',
                         'petri_net_energy_direct_transition_only',
                         'petri_net_energy_direct_global'):
            enable_dur = self.mode != 'petri_net_energy_direct_transition_only'
            enable_tr  = self.mode != 'petri_net_energy_direct_duration_only'
            self._simulate_petri_net_energy_direct_for_case(
                case_id, object_attributes, start_time,
                enable_duration=enable_dur,
                enable_transitions=enable_tr,
            )
            return

        # ── Quantile-blend: quantile-conditioned duration + entropy-blended transitions
        if self.mode == 'petri_net_quantile_blend':
            self._simulate_petri_net_quantile_blend_for_case(
                case_id, object_attributes, start_time,
            )
            return

        # ── All other modes: statistical / ML ────────────────────────
        unique_objects = self.activity_stats[['object', 'object_type', 'higher_level_activity']].drop_duplicates(subset=['higher_level_activity'])
        
        current_sim_time = start_time.timestamp()
        
        print(f"Case {case_id}: Found {len(unique_objects)} unique object configurations:")
        for _, obj_config in unique_objects.iterrows():
            print(f"  - {obj_config['object']} ({obj_config['object_type']}) with higher_level: {obj_config['higher_level_activity']}")
        
        for _, obj_config in unique_objects.iterrows():
            object_name = str(obj_config['object']).strip()
            object_type = str(obj_config['object_type']).strip()
            higher_level_activity = str(obj_config['higher_level_activity']).strip() if pd.notna(obj_config['higher_level_activity']) else None
            
            print(f"\nCase {case_id}: Simulating {object_name} ({object_type}) process")
            
            # Get start activities
            start_activities = self._get_start_activities(object_name, object_type, higher_level_activity)
            
            if not start_activities:
                print(f"  WARNING: No start activities found for {object_name} ({object_type}) - SKIPPING!")
                continue
                
            # Choose a random start activity
            current_activity = random.choice(start_activities)
            print(f"  Starting with activity: {current_activity}")
            
            # Follow the process flow with probabilistic ending
            activity_count = 0
            # Track last 2 activities/durations for the _with_activity_past mode
            activity_history = []   # [(activity_name, duration), ...] most recent first

            while current_activity:
                print(f"    Processing activity {activity_count + 1}: {current_activity}")
                
                # Get duration (pass history for modes that use it)
                _HIST_MODES = ('ml_duration_only_with_activity_past',
                               'ml_duration_only_with_activity_past_point_estimate',
                               'ml_global_model')
                activity_duration = self._get_activity_duration(
                    current_activity, object_name, object_type,
                    higher_level_activity, object_attributes,
                    activity_history=activity_history if self.mode in _HIST_MODES else None,
                    activity_index=activity_count,
                )
                
                # Calculate timestamps
                start_time_obj = datetime.fromtimestamp(current_sim_time)
                current_sim_time += activity_duration * 60
                end_time_obj = datetime.fromtimestamp(current_sim_time)
                
                # Log the event
                self._log_event(
                    case_id=case_id,
                    activity=current_activity,
                    timestamp_start=start_time_obj,
                    timestamp_end=end_time_obj,
                    object_name=object_name,
                    object_type=object_type,
                    higher_level_activity=higher_level_activity,
                    object_attributes=object_attributes
                )
                
                # Update activity history (most-recent-first, keep last 2)
                activity_history.insert(0, (current_activity, activity_duration))
                activity_history = activity_history[:2]

                activity_count += 1
                
                # Get next activity (which may return None if __END__ is chosen)
                next_activity = self._get_next_activity(
                    current_activity, object_name, object_type,
                    higher_level_activity, object_attributes
                )
                
                if not next_activity:
                    print(f"    Process ended after {current_activity}")
                    break
                
                current_activity = next_activity
                current_sim_time += 1  # 1 second gap
            
            print(f"  Completed {object_name} process with {activity_count} activities")
    
    def run(self):
        """Run the simulation with probabilistic end transitions"""
        self.events = []
        
        print(f"Starting simulation with {len(self.production_plan)} cases...")
        print(f"Activity config has {len(self.activity_config)} activity configurations")
        print(f"Simulation mode: {self.mode}")
        print("Using probabilistic end transitions!")
        
        # Simulate each case
        for i, (_, order) in enumerate(self.production_plan.iterrows()):
            case_id = str(order['case_id']).strip()
            object_attributes = order.get('object_attributes', {})
            start_time = order['timestamp_start']
            
            print(f"\n{'='*50}")
            print(f"CASE {i+1}/{len(self.production_plan)}: {case_id}")
            print(f"{'='*50}")
            
            self._simulate_process_for_case(case_id, object_attributes, start_time)
            
            print(f"Events generated so far: {len(self.events)}")
        
        print(f"\n{'='*50}")
        print(f"SIMULATION COMPLETED")
        print(f"Total events generated: {len(self.events)}")
        print(f"{'='*50}")
        
        # Convert to DataFrame
        simulated_df = pd.DataFrame(self.events)
        
        if not simulated_df.empty:
            simulated_df['case_id'] = simulated_df['case_id'].astype(str)
            simulated_df['activity'] = simulated_df['activity'].astype(str)
            simulated_df = simulated_df.sort_values('timestamp_start').reset_index(drop=True)
        
        return simulated_df

def compare_simulation_with_real(simulated_df, real_df):
    """Compare simulation results with real data"""
    comparison = {}
    
    print("=== SIMULATION vs REAL DATA COMPARISON ===\n")
    
    # Basic counts
    comparison['total_events'] = {
        'simulated': len(simulated_df),
        'real': len(real_df)
    }
    
    print(f"Total Events:")
    print(f"  Simulated: {comparison['total_events']['simulated']}")
    print(f"  Real: {comparison['total_events']['real']}")
    print(f"  Difference: {comparison['total_events']['simulated'] - comparison['total_events']['real']}")
    
    # Cases
    sim_cases = simulated_df['case_id'].nunique() if 'case_id' in simulated_df.columns and len(simulated_df) > 0 else 0
    real_cases = real_df['case_id'].nunique()
    
    comparison['total_cases'] = {
        'simulated': sim_cases,
        'real': real_cases
    }
    
    print(f"\nTotal Cases:")
    print(f"  Simulated: {sim_cases}")
    print(f"  Real: {real_cases}")
    
    # Events per case
    if sim_cases > 0:
        avg_events_per_case_sim = len(simulated_df) / sim_cases
        avg_events_per_case_real = len(real_df) / real_cases
        
        print(f"\nEvents per Case:")
        print(f"  Simulated: {avg_events_per_case_sim:.1f} avg")
        print(f"  Real: {avg_events_per_case_real:.1f} avg")
    
    # Activities comparison
    if not simulated_df.empty:
        sim_activities = set(simulated_df['activity'].unique())
        real_activities = set(real_df['activity'].unique())
        
        comparison['activities'] = {
            'simulated': sim_activities,
            'real': real_activities,
            'common': sim_activities.intersection(real_activities),
            'sim_only': sim_activities - real_activities,
            'real_only': real_activities - sim_activities
        }
        
        print(f"\nActivities:")
        print(f"  Simulated: {len(sim_activities)} activities")
        print(f"  Real: {len(real_activities)} activities")
        print(f"  Common: {len(comparison['activities']['common'])} activities")
        if comparison['activities']['sim_only']:
            print(f"  Only in simulation: {comparison['activities']['sim_only']}")
        if comparison['activities']['real_only']:
            print(f"  Only in real: {comparison['activities']['real_only']}")
    
    return comparison