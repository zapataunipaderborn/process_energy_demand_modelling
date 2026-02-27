from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import warnings

# ---------------------------------------------------------------------------
# pm4py imports (used when mining_algorithm != 'manual')
# ---------------------------------------------------------------------------
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

_DIST_MAP = {
    'norm':        scipy_stats.norm,
    'lognorm':     scipy_stats.lognorm,
    'expon':       scipy_stats.expon,
    'gamma':       scipy_stats.gamma,
    'weibull_min': scipy_stats.weibull_min,
}


def fit_best_distribution(data):
    """
    Fit multiple distributions to *data* (array of positive durations in minutes)
    and return (dist_name, dist_params) for the best fit by KS p-value.

    Falls back to ('norm', (mean, std)) when there isn't enough data or all
    fits fail.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]

    if len(data) < 3:
        loc = float(np.mean(data)) if len(data) > 0 else 0.0
        scale = float(np.std(data)) if len(data) > 1 else 0.0
        return 'norm', (loc, scale)

    best_name = 'norm'
    best_params = None
    best_pvalue = -1.0

    for name, dist in _DIST_MAP.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Fix location at 0 for non-negative distributions
                if name in ('lognorm', 'expon', 'gamma', 'weibull_min'):
                    params = dist.fit(data, floc=0)
                else:
                    params = dist.fit(data)
                _, pvalue = scipy_stats.kstest(data, name, args=params)
        except Exception:
            continue

        if pvalue > best_pvalue:
            best_pvalue = pvalue
            best_name = name
            best_params = params

    if best_params is None:
        loc = float(np.mean(data))
        scale = float(np.std(data))
        best_params = (loc, scale)

    return best_name, tuple(float(p) for p in best_params)


def sample_from_dist(dist_name, dist_params):
    """Draw one positive sample from a previously fitted distribution."""
    dist = _DIST_MAP.get(dist_name, scipy_stats.norm)
    try:
        return max(0.1, float(dist.rvs(*dist_params)))
    except Exception:
        # Last-resort: treat params as (mean, std)
        mean = dist_params[0] if dist_params else 1.0
        std  = dist_params[1] if len(dist_params) > 1 else 0.0
        return max(0.1, float(np.random.normal(mean, std)))


# ---------------------------------------------------------------------------
# pm4py mining helpers
# ---------------------------------------------------------------------------

def _mine_petri_net(sub_log, algorithm='inductive', noise_threshold=0.2):
    """
    Mine a Petri net from a pm4py-formatted event log sub-group.

    Parameters
    ----------
    sub_log : pd.DataFrame
        pm4py-formatted event log (case:concept:name, concept:name, time:timestamp).
    algorithm : str
        'inductive', 'heuristic', or 'alpha'.
    noise_threshold : float
        Noise filtering for the Inductive Miner (0.0 = keep all, 1.0 = max filtering).
        Higher values produce stricter models that filter out infrequent paths.

    Returns
    -------
    net, im, fm : PetriNet, Marking, Marking
    """
    if algorithm == 'inductive':
        if noise_threshold == 'tune':
            print(f"    Tuning noise_threshold for Inductive Miner...")
            best_net, best_im, best_fm = None, None, None
            results = []
            
            # Test a grid of thresholds
            for nt in [0.0, 0.2, 0.4, 0.6, 0.8]:
                try:
                    net_cand, im_cand, fm_cand = pm4py.discover_petri_net_inductive(
                        sub_log, noise_threshold=nt
                    )
                    # Compute fitness using token replay
                    replay_res = token_replay.apply(
                        sub_log, net_cand, im_cand, fm_cand,
                        parameters={'consider_remaining_in_fitness': True}
                    )
                    fitness = sum(r.get('trace_fitness', 0.0) for r in replay_res) / len(replay_res) if replay_res else 0.0
                    results.append((nt, fitness, net_cand, im_cand, fm_cand))
                    print(f"      nt={nt:.1f} -> fitness={fitness:.4f}")
                except Exception as e:
                    print(f"      nt={nt:.1f} -> Failed: {e}")
                    
            if not results:
                # Fallback if tuning fails
                print(f"      Tuning failed, falling back to nt=0.2")
                net, im, fm = pm4py.discover_petri_net_inductive(sub_log, noise_threshold=0.2)
            else:
                # We want the highest noise threshold (simplest model) that maintains good fitness (> 0.8)
                acceptable = [r for r in results if r[1] >= 0.8]
                if acceptable:
                    best_r = max(acceptable, key=lambda x: x[0])
                else:
                    best_r = max(results, key=lambda x: x[1])
                
                print(f"    Selected best noise_threshold: {best_r[0]:.1f} with fitness {best_r[1]:.4f}")
                net, im, fm = best_r[2], best_r[3], best_r[4]
        else:
            net, im, fm = pm4py.discover_petri_net_inductive(
                sub_log, noise_threshold=noise_threshold
            )
    elif algorithm == 'heuristic':
        net, im, fm = pm4py.discover_petri_net_heuristics(sub_log)
    elif algorithm == 'alpha':
        net, im, fm = pm4py.discover_petri_net_alpha(sub_log)
    else:
        raise ValueError(f"Unknown mining algorithm: {algorithm}")

    return net, im, fm


def _get_stochastic_map(net, im, fm, log):
    """
    Replay the log on the Petri net via token replay and compute
    transition firing frequencies → stochastic weights.

    Returns
    -------
    stochastic_map : dict
        {Transition: weight} where weight is the count of times that
        transition fired during replay, normalised per decision point.
    replay_results : list
        Raw token-replay results for further analysis.
    """
    # Run token-based replay
    replay_results = token_replay.apply(
        log, net, im, fm,
        parameters={
            'consider_remaining_in_fitness': True
        }
    )

    # Count how many times each transition fired across all traces
    firing_counts = defaultdict(int)
    for result in replay_results:
        for transition in result.get('activated_transitions', []):
            firing_counts[transition] += 1

    # Build stochastic map (raw counts — simulation will normalise per choice)
    stochastic_map = dict(firing_counts)

    return stochastic_map, replay_results


def _compute_decision_point_weights(net, im, fm, case_sorted):
    """
    Replay each case on the Petri net step-by-step and record,
    for every *decision point* (defined by the frozenset of enabled
    labelled transitions), which activity was actually chosen.

    When a case ends (no more events), record an ``__END__`` choice
    at the current decision point.

    Returns
    -------
    decision_weights : dict
        {frozenset(enabled_labels): {chosen_label_or___END__: count, ...}}
    max_case_length : int
        Longest case (number of labelled activities) seen in training.
    """
    import copy

    def _enabled(net_, marking_):
        enabled_ = set()
        for t in net_.transitions:
            if all(marking_.get(arc.source, 0) >= 1 for arc in t.in_arcs):
                enabled_.add(t)
        return enabled_

    def _fire(marking_, transition_):
        m = copy.copy(marking_)
        for arc in transition_.in_arcs:
            m[arc.source] -= 1
            if m[arc.source] == 0:
                del m[arc.source]
        for arc in transition_.out_arcs:
            if arc.target not in m:
                m[arc.target] = 0
            m[arc.target] += 1
        return m

    decision_weights = defaultdict(lambda: defaultdict(int))
    max_case_length = 0

    for case_id, case_df in case_sorted.items():
        activities_in_case = case_df['activity'].tolist()
        if not activities_in_case:
            continue
        max_case_length = max(max_case_length, len(activities_in_case))

        marking = copy.copy(im)
        act_idx = 0
        max_replay_steps = 500

        for _ in range(max_replay_steps):
            enabled = _enabled(net, marking)
            if not enabled:
                break  # deadlock

            # Separate labelled and silent transitions
            label_map = defaultdict(list)   # label -> [transition, ...]
            silent = []
            for t in enabled:
                if t.label is not None:
                    label_map[str(t.label).strip()].append(t)
                else:
                    silent.append(t)

            # If only silent transitions enabled, fire one and continue
            if not label_map:
                if silent:
                    marking = _fire(marking, silent[0])
                    continue
                else:
                    break

            enabled_labels = frozenset(label_map.keys())

            # Case has ended — record __END__
            if act_idx >= len(activities_in_case):
                decision_weights[enabled_labels]['__END__'] += 1
                break

            # Match the next activity to an enabled transition
            next_act = str(activities_in_case[act_idx]).strip()

            if next_act in label_map:
                decision_weights[enabled_labels][next_act] += 1
                # Fire the corresponding transition
                chosen_t = label_map[next_act][0]
                marking = _fire(marking, chosen_t)
                act_idx += 1
            else:
                # Activity not enabled — skip silent transitions first
                if silent:
                    marking = _fire(marking, silent[0])
                    continue
                else:
                    # Can't match — skip this activity (misalignment)
                    act_idx += 1
                    continue

        # If we consumed all activities but didn't record __END__ yet
        if act_idx >= len(activities_in_case):
            enabled = _enabled(net, marking)
            label_map = defaultdict(list)
            for t in enabled:
                if t.label is not None:
                    label_map[str(t.label).strip()].append(t)
            if label_map:
                enabled_labels = frozenset(label_map.keys())
                decision_weights[enabled_labels]['__END__'] += 1

    # Convert to plain dicts
    decision_weights = {k: dict(v) for k, v in decision_weights.items()}

    return decision_weights, max_case_length


def _derive_transitions_from_net(net, im, fm, sub_log, sub_df):
    """
    Derive transition probabilities from the Petri net by analysing
    the directly-follows relationships in the log filtered through the model.

    Uses pm4py's DFG + the net structure to produce per-activity transition
    probabilities that respect the mined model.

    Returns
    -------
    transitions_dict : dict
        {activity_label: {next_activity: probability}}
    start_activities : set
        Activities that appear as start activities.
    end_activities : set
        Activities that appear as end activities.
    """
    # Get DFG from the log
    dfg, start_acts, end_acts = pm4py.discover_dfg(sub_log)

    # Convert DFG to transition probabilities
    # dfg is {(act_a, act_b): count, ...}
    outgoing_counts = defaultdict(lambda: defaultdict(int))
    outgoing_total = defaultdict(int)

    for (src, tgt), count in dfg.items():
        outgoing_counts[src][tgt] += count
        outgoing_total[src] += count

    # Add __END__ transitions from end_acts
    # end_acts is {activity: count}
    for act, count in end_acts.items():
        outgoing_counts[act]['__END__'] += count
        outgoing_total[act] += count

    # Normalise to probabilities
    transitions_dict = {}
    for act in outgoing_counts:
        transitions_dict[act] = {}
        total = outgoing_total[act]
        if total > 0:
            for next_act, count in outgoing_counts[act].items():
                transitions_dict[act][next_act] = count / total

    return transitions_dict, set(start_acts.keys()), set(end_acts.keys())


# ---------------------------------------------------------------------------
# History-dependent transition extraction (for memory-augmented simulation)
# ---------------------------------------------------------------------------

def _extract_history_weights(case_sorted):
    """
    Walk per-case activity sequences and build history-dependent
    transition statistics.

    Returns
    -------
    bigram_transitions : dict
        {(prev_activity, current_activity): {next_activity: count}}
        Second-order Markov: P(next | current, previous).
    activity_count_transitions : dict
        {(current_activity, times_current_seen_so_far): {next_activity: count}}
        Repetition-aware: P(next | current, how many times current has
        already appeared in this case).
    """
    bigram_transitions = defaultdict(lambda: defaultdict(int))
    activity_count_transitions = defaultdict(lambda: defaultdict(int))

    for case_id, case_acts in case_sorted.items():
        if len(case_acts) == 0:
            continue

        activity_counter = defaultdict(int)  # how many times each activity seen so far

        for idx in range(len(case_acts)):
            current_act = case_acts.iloc[idx]['activity']

            # Previous activity (for bigram)
            prev_act = case_acts.iloc[idx - 1]['activity'] if idx > 0 else '__START__'

            # How many times this activity has been seen before in this case
            times_seen = activity_counter[current_act]
            activity_counter[current_act] += 1

            # Next activity
            if idx + 1 < len(case_acts):
                next_act = case_acts.iloc[idx + 1]['activity']
            else:
                next_act = '__END__'

            # Record bigram: (prev, current) -> next
            bigram_transitions[(prev_act, current_act)][next_act] += 1

            # Record activity-count: (current, times_seen) -> next
            activity_count_transitions[(current_act, times_seen)][next_act] += 1

    # Convert defaultdicts to plain dicts for serialisation safety
    bigram_transitions = {k: dict(v) for k, v in bigram_transitions.items()}
    activity_count_transitions = {k: dict(v) for k, v in activity_count_transitions.items()}

    print(f"    History weights: {len(bigram_transitions)} bigram keys, "
          f"{len(activity_count_transitions)} activity-count keys")

    return bigram_transitions, activity_count_transitions


# ---------------------------------------------------------------------------
# Shared: duration + raw-row extraction per activity (used by ALL modes)
# ---------------------------------------------------------------------------

def _extract_duration_and_raw(group, object_name, object_type,
                              higher_level_activity, case_sorted):
    """
    For every activity in *group*, compute duration stats and collect
    per-instance raw rows for ML training.

    This is shared between 'manual' and pm4py modes — the duration
    fitting and raw-row collection are independent of the process model.

    Returns
    -------
    duration_info : dict  {activity: {duration, duration_std, dist_name, dist_params, n_events}}
    raw_rows : list[dict]
    """
    activities = group['activity'].unique()
    duration_info = {}
    raw_rows = []

    for activity in activities:
        activity_data = group[group['activity'] == activity]

        # ── Basic statistics ──────────────────────────────────────────
        durations = (
            activity_data['timestamp_end'] - activity_data['timestamp_start']
        ).dt.total_seconds() / 60
        duration_median = float(durations.median())
        duration_std    = float(durations.std()) if len(durations) > 1 else 0.0
        n_events = len(activity_data)

        # ── Best-fit distribution ─────────────────────────────────────
        dist_name, dist_params = fit_best_distribution(durations.values)
        print(f"    {activity}: best fit = {dist_name} {dist_params}")

        duration_info[activity] = {
            'duration':     duration_median,
            'duration_std': duration_std,
            'dist_name':    dist_name,
            'dist_params':  dist_params,
            'n_events':     n_events,
        }

        # ── Raw rows for ML training ──────────────────────────────────
        for case_id, case_acts in case_sorted.items():
            current_indices = case_acts[case_acts['activity'] == activity].index

            for idx in current_indices:
                row_here = case_acts.iloc[idx]

                # resolved next activity for this instance
                if idx + 1 < len(case_acts):
                    next_act = case_acts.iloc[idx + 1]['activity']
                else:
                    next_act = '__END__'

                inst_duration = (
                    (row_here['timestamp_end'] - row_here['timestamp_start'])
                    .total_seconds() / 60
                )
                attr_raw = row_here.get('object_attributes', {}) or {}
                attr_flat = {f'attr_{k}': v for k, v in attr_raw.items()}

                # ── lag features: last 2 activities & durations ────────
                prev_act_1, prev_dur_1 = '__NONE__', 0.0
                prev_act_2, prev_dur_2 = '__NONE__', 0.0
                if idx >= 1:
                    prev_row = case_acts.iloc[idx - 1]
                    prev_act_1 = prev_row['activity']
                    prev_dur_1 = (
                        (prev_row['timestamp_end'] - prev_row['timestamp_start'])
                        .total_seconds() / 60
                    )
                if idx >= 2:
                    prev_row2 = case_acts.iloc[idx - 2]
                    prev_act_2 = prev_row2['activity']
                    prev_dur_2 = (
                        (prev_row2['timestamp_end'] - prev_row2['timestamp_start'])
                        .total_seconds() / 60
                    )

                raw_rows.append({
                    'case_id':               case_id,
                    'activity':              activity,
                    'object':                object_name,
                    'object_type':           object_type,
                    'higher_level_activity': higher_level_activity,
                    'duration':              inst_duration,
                    'next_activity':         next_act,
                    'timestamp_start':       row_here['timestamp_start'],
                    'activity_index':        idx,
                    'hour_of_day':           row_here['timestamp_start'].hour if hasattr(row_here['timestamp_start'], 'hour') else 0,
                    'day_of_week':           row_here['timestamp_start'].weekday() if hasattr(row_here['timestamp_start'], 'weekday') else 0,
                    'prev_activity_1':       prev_act_1,
                    'prev_duration_1':       prev_dur_1,
                    'prev_activity_2':       prev_act_2,
                    'prev_duration_2':       prev_dur_2,
                    **attr_flat,
                })

    return duration_info, raw_rows


# ---------------------------------------------------------------------------
# Manual process extraction (original approach)
# ---------------------------------------------------------------------------

def _extract_manual(group, object_name, object_type, higher_level_activity,
                    case_sorted, duration_info):
    """
    Original manual extraction: walk cases to determine start/end flags
    and transition probabilities by counting.
    """
    activities = group['activity'].unique()
    stats = []

    for activity in activities:
        # ── Start detection ───────────────────────────────────────────
        is_start = False
        for case_id, case_acts in case_sorted.items():
            if len(case_acts) > 0 and case_acts.iloc[0]['activity'] == activity:
                is_start = True
                break

        # ── Probabilistic transition detection ───────────────────────
        transition_counts = defaultdict(int)
        end_counts       = 0
        total_occurrences = 0

        for case_id, case_acts in case_sorted.items():
            current_indices = case_acts[case_acts['activity'] == activity].index

            for idx in current_indices:
                total_occurrences += 1
                if idx + 1 < len(case_acts):
                    next_act = case_acts.iloc[idx + 1]['activity']
                    transition_counts[next_act] += 1
                else:
                    end_counts += 1

        # ── Transition probabilities ──────────────────────────────────
        transitions = {}
        if total_occurrences > 0:
            for next_act, count in transition_counts.items():
                transitions[next_act] = count / total_occurrences
            if end_counts > 0:
                transitions['__END__'] = end_counts / total_occurrences

        is_end = end_counts > 0
        d = duration_info[activity]

        print(f"      Events: {d['n_events']}, Start: {is_start}, Can End: {is_end}")
        print(f"      Total occurrences: {total_occurrences}")
        print(f"      Transitions: {transitions}")

        stats.append({
            'activity':              activity,
            'object':                object_name,
            'object_type':           object_type,
            'higher_level_activity': higher_level_activity,
            'duration':              d['duration'],
            'duration_std':          d['duration_std'],
            'dist_name':             d['dist_name'],
            'dist_params':           d['dist_params'],
            'n_events':              d['n_events'],
            'transition':            transitions,
            'is_start':              is_start,
            'is_end':                is_end,
        })

    return stats, None  # No process_model for manual


# ---------------------------------------------------------------------------
# pm4py-based process extraction
# ---------------------------------------------------------------------------

def _extract_with_pm4py(group, object_name, object_type, higher_level_activity,
                        case_sorted, duration_info, algorithm='inductive',
                        noise_threshold=0.2):
    """
    Mine a Petri net from the sub-log and derive transitions, start/end
    from the mined model.

    Returns
    -------
    stats : list[dict]
    process_model : dict
        {'net': PetriNet, 'im': Marking, 'fm': Marking,
         'stochastic_map': dict, 'duration_map': dict}
    """
    # ── Format the sub-log for pm4py ──────────────────────────────────
    sub_df = group[['case_id', 'activity', 'timestamp_start', 'timestamp_end']].copy()
    sub_df = sub_df.dropna(subset=['case_id'])

    if len(sub_df) < 2:
        print(f"    WARNING: Too few events ({len(sub_df)}) for pm4py mining — "
              f"falling back to manual extraction.")
        return _extract_manual(group, object_name, object_type,
                               higher_level_activity, case_sorted, duration_info)

    sub_log = pm4py.format_dataframe(
        sub_df,
        case_id='case_id',
        activity_key='activity',
        timestamp_key='timestamp_start'
    )

    # ── Mine the Petri net ────────────────────────────────────────────
    print(f"    Mining Petri net with '{algorithm}' algorithm "
          f"(noise_threshold={noise_threshold})...")
    net, im, fm = _mine_petri_net(sub_log, algorithm,
                                  noise_threshold=noise_threshold)

    print(f"    Petri net: {len(net.places)} places, "
          f"{len(net.transitions)} transitions, "
          f"{len(net.arcs)} arcs")

    # ── Get stochastic map via token replay ───────────────────────────
    stochastic_map, _ = _get_stochastic_map(net, im, fm, sub_log)
    print(f"    Stochastic map: {len(stochastic_map)} transition weights")

    # ── Decision-point-aware weights (Changes 2+3) ────────────────────
    decision_weights, max_case_length = _compute_decision_point_weights(
        net, im, fm, case_sorted
    )
    n_dp = len(decision_weights)
    n_end = sum(1 for dp in decision_weights.values() if '__END__' in dp)
    print(f"    Decision-point weights: {n_dp} decision points, "
          f"{n_end} with __END__ probability")
    print(f"    Max case length in training: {max_case_length}")

    # ── Derive transitions from the mined model ──────────────────────
    transitions_dict, start_acts, end_acts = _derive_transitions_from_net(
        net, im, fm, sub_log, sub_df
    )

    # ── Build duration map for the Petri net simulation ───────────────
    # Maps transition labels → (dist_name, dist_params)
    duration_map = {}
    for act, d in duration_info.items():
        duration_map[act] = (d['dist_name'], d['dist_params'])

    # ── Build stats rows ──────────────────────────────────────────────
    activities = group['activity'].unique()
    stats = []

    for activity in activities:
        is_start = activity in start_acts
        is_end = activity in end_acts
        transitions = transitions_dict.get(activity, {})
        d = duration_info[activity]

        print(f"      {activity}: Start={is_start}, End={is_end}, "
              f"Transitions={transitions}")

        stats.append({
            'activity':              activity,
            'object':                object_name,
            'object_type':           object_type,
            'higher_level_activity': higher_level_activity,
            'duration':              d['duration'],
            'duration_std':          d['duration_std'],
            'dist_name':             d['dist_name'],
            'dist_params':           d['dist_params'],
            'n_events':              d['n_events'],
            'transition':            transitions,
            'is_start':              is_start,
            'is_end':                is_end,
        })

    # ── Build label-level stochastic weights for blending ───────────
    # stochastic_map keys are Transition objects; convert to labels
    label_stochastic = defaultdict(float)
    for t, weight in stochastic_map.items():
        if t.label is not None:
            label_stochastic[str(t.label).strip()] += weight

    # ── Extract history-dependent transition weights ───────────────
    bigram_transitions, activity_count_transitions = _extract_history_weights(
        case_sorted
    )

    process_model = {
        'net': net,
        'im': im,
        'fm': fm,
        'stochastic_map': stochastic_map,
        'label_stochastic': dict(label_stochastic),
        'duration_map': duration_map,
        'bigram_transitions': bigram_transitions,
        'activity_count_transitions': activity_count_transitions,
        'decision_weights': decision_weights,
        'max_case_length': max_case_length,
    }

    return stats, process_model


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_process(df, mining_algorithm='inductive', noise_threshold=0.2):
    """
    Extract process statistics from an event log.

    Parameters
    ----------
    df : pd.DataFrame
        Event log with columns: case_id, activity, timestamp_start,
        timestamp_end, object, object_type, higher_level_activity,
        object_attributes.
    mining_algorithm : str
        Process mining algorithm to use:
        - 'inductive' (default) — pm4py Inductive Miner → sound Petri net
        - 'heuristic' — pm4py Heuristics Miner → noise-tolerant
        - 'alpha' — pm4py Alpha Miner → classic algorithm
        - 'manual' — original manual extraction (no process mining)
    noise_threshold : float
        Noise filtering for the Inductive Miner (0.0 = keep all,
        1.0 = max filtering). Default 0.2.

    Returns
    -------
    stats_df : pd.DataFrame
        One row per (activity, object, object_type, higher_level_activity).
        Columns: activity, object, object_type, higher_level_activity,
        duration, duration_std, dist_name, dist_params, n_events,
        transition (dict), is_start, is_end.
    raw_df : pd.DataFrame
        One row per activity instance with raw duration, next_activity,
        lag features, and flattened object_attributes.
    process_models : dict or None
        When mining_algorithm != 'manual', contains the mined Petri nets:
        {(object, object_type, higher_level_activity): {
            'net': PetriNet, 'im': Marking, 'fm': Marking,
            'stochastic_map': dict, 'duration_map': dict
        }}
        None when mining_algorithm == 'manual'.
    """
    use_pm4py = mining_algorithm != 'manual'

    all_stats = []
    all_raw_rows = []
    process_models = {} if use_pm4py else None

    # Group by the combination that defines a unique process configuration
    grouped = df.groupby(['object', 'object_type', 'higher_level_activity'])

    for (object_name, object_type, higher_level_activity), group in grouped:
        print(f"\nProcessing: {object_name} ({object_type}) - {higher_level_activity}")
        print(f"  Mining algorithm: {mining_algorithm}")

        activities = group['activity'].unique()
        print(f"  Found {len(activities)} unique activities: {activities}")

        # Pre-sort cases once for the whole group
        case_sorted = {}
        for case_id in group['case_id'].dropna().unique():
            case_sorted[case_id] = (
                group[group['case_id'] == case_id]
                .sort_values('timestamp_start')
                .reset_index(drop=True)
            )

        # ── Duration + raw row extraction (shared by all modes) ───────
        duration_info, raw_rows = _extract_duration_and_raw(
            group, object_name, object_type, higher_level_activity, case_sorted
        )
        all_raw_rows.extend(raw_rows)

        # ── Process model extraction ──────────────────────────────────
        if use_pm4py:
            stats, process_model = _extract_with_pm4py(
                group, object_name, object_type, higher_level_activity,
                case_sorted, duration_info, algorithm=mining_algorithm,
                noise_threshold=noise_threshold
            )
            if process_model is not None:
                key = (object_name, object_type, higher_level_activity)
                process_models[key] = process_model
        else:
            stats, _ = _extract_manual(
                group, object_name, object_type, higher_level_activity,
                case_sorted, duration_info
            )

        all_stats.extend(stats)

    stats_df = pd.DataFrame(all_stats)
    raw_df   = pd.DataFrame(all_raw_rows)

    return stats_df, raw_df, process_models
