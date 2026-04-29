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
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

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

def _mine_petri_net(sub_log, algorithm='inductive', noise_threshold=0.2,
                    heuristic_params=None):
    """
    Mine a Petri net from a pm4py-formatted event log sub-group.

    Parameters
    ----------
    sub_log : pd.DataFrame
        pm4py-formatted event log (case:concept:name, concept:name, time:timestamp).
    algorithm : str
        'inductive', 'heuristic', 'alpha', or 'ilp'.
    noise_threshold : float
        Noise filtering for the Inductive Miner (0.0 = keep all, 1.0 = max filtering).
        Higher values produce stricter models that filter out infrequent paths.
    heuristic_params : dict or None
        Optional parameter dict for Heuristics Miner.

    Returns
    -------
    net, im, fm : PetriNet, Marking, Marking
    """
    if algorithm == 'inductive':
        net, im, fm = pm4py.discover_petri_net_inductive(
            sub_log, noise_threshold=noise_threshold
        )
    elif algorithm == 'heuristic':
        h_params = heuristic_params or {}
        if h_params:
            try:
                net, im, fm = pm4py.discover_petri_net_heuristics(
                    sub_log, **h_params
                )
            except TypeError:
                # Keep backward compatibility across pm4py versions.
                net, im, fm = pm4py.discover_petri_net_heuristics(sub_log)
        else:
            net, im, fm = pm4py.discover_petri_net_heuristics(sub_log)
    elif algorithm == 'alpha':
        net, im, fm = pm4py.discover_petri_net_alpha(sub_log)
    elif algorithm == 'ilp':
        net, im, fm = pm4py.discover_petri_net_ilp(sub_log)
    else:
        raise ValueError(f"Unknown mining algorithm: {algorithm}")

    return net, im, fm


def _evaluate_mined_model(sub_log, net, im, fm):
    """
    Score one mined Petri net using an overall quality metric:
    mean(fitness, precision, generalization, simplicity).
    """
    fitness = 0.0
    precision = None
    generalization = None
    simplicity = None

    try:
        replay = token_replay.apply(
            sub_log, net, im, fm,
            parameters={'consider_remaining_in_fitness': True}
        )
        trace_scores = []
        for item in replay:
            if 'trace_fitness' in item and item['trace_fitness'] is not None:
                trace_scores.append(float(item['trace_fitness']))
            elif item.get('trace_is_fit') is True:
                trace_scores.append(1.0)
            else:
                trace_scores.append(0.0)
        if trace_scores:
            fitness = float(np.mean(trace_scores))
    except Exception:
        fitness = 0.0

    try:
        precision = float(pm4py.precision_token_based_replay(sub_log, net, im, fm))
    except Exception:
        precision = None

    if precision is None:
        precision = fitness

    try:
        generalization = float(
            generalization_evaluator.apply(sub_log, net, im, fm)
        )
    except Exception:
        generalization = None

    try:
        simplicity = float(simplicity_evaluator.apply(net))
    except Exception:
        simplicity = None

    components = [fitness, precision, generalization, simplicity]
    valid_components = [
        max(0.0, min(1.0, float(v)))
        for v in components
        if v is not None and np.isfinite(v)
    ]

    score = float(np.mean(valid_components)) if valid_components else 0.0
    return {
        'score': score,
        'overall_metric': score,
        'fitness': fitness,
        'precision': precision,
        'generalization': generalization,
        'simplicity': simplicity,
    }


def _build_mining_candidates(algorithm, noise_threshold=0.2,
                             heuristic_params=None,
                             optimize_mining_hyperparams=False,
                             mining_search_space=None):
    """Build candidate miner parameter sets for local tuning."""
    base = {
        'noise_threshold': float(noise_threshold),
        'heuristic_params': dict(heuristic_params or {}),
    }

    if not optimize_mining_hyperparams or algorithm not in ('inductive', 'heuristic'):
        return [base]

    search_space = mining_search_space or {}
    candidates = []

    if algorithm == 'inductive':
        default_grid = [0.05, 0.10, 0.20, 0.30, 0.40]
        noise_grid = search_space.get('inductive_noise_thresholds', default_grid)
        for n in noise_grid:
            n = float(max(0.0, min(1.0, n)))
            candidates.append({'noise_threshold': n, 'heuristic_params': {}})

    elif algorithm == 'heuristic':
        default_grid = [
            {'dependency_threshold': 0.3, 'and_threshold': 0.65, 'loop_two_threshold': 0.5},
            {'dependency_threshold': 0.5, 'and_threshold': 0.65, 'loop_two_threshold': 0.5},
            {'dependency_threshold': 0.7, 'and_threshold': 0.65, 'loop_two_threshold': 0.5},
            {'dependency_threshold': 0.5, 'and_threshold': 0.50, 'loop_two_threshold': 0.5},
            {'dependency_threshold': 0.5, 'and_threshold': 0.80, 'loop_two_threshold': 0.5},
        ]
        h_grid = search_space.get('heuristic_params_grid', default_grid)
        for params in h_grid:
            candidates.append({'noise_threshold': float(noise_threshold),
                               'heuristic_params': dict(params or {})})

    if not candidates:
        candidates = [base]

    # Deduplicate candidates while preserving order.
    seen = set()
    unique = []
    for cand in candidates:
        key = (
            round(float(cand.get('noise_threshold', 0.0)), 6),
            tuple(sorted((cand.get('heuristic_params') or {}).items())),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)

    return unique


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
                        noise_threshold=0.2,
                        heuristic_params=None,
                        optimize_mining_hyperparams=False,
                        mining_search_space=None):
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

    # ── Mine the Petri net (with optional local hyperparameter search) ─────
    candidates = _build_mining_candidates(
        algorithm,
        noise_threshold=noise_threshold,
        heuristic_params=heuristic_params,
        optimize_mining_hyperparams=optimize_mining_hyperparams,
        mining_search_space=mining_search_space,
    )

    best = None
    best_score = float('-inf')
    for idx, cand in enumerate(candidates, start=1):
        cand_noise = cand.get('noise_threshold', noise_threshold)
        cand_heur = cand.get('heuristic_params', {})
        print(
            f"    Mining Petri net with '{algorithm}' "
            f"(candidate {idx}/{len(candidates)}): "
            f"noise={cand_noise}, heur={cand_heur}"
        )

        net_i, im_i, fm_i = _mine_petri_net(
            sub_log,
            algorithm,
            noise_threshold=cand_noise,
            heuristic_params=cand_heur,
        )
        eval_i = _evaluate_mined_model(sub_log, net_i, im_i, fm_i)
        print(
            f"      quality: score={eval_i['score']:.4f}, "
            f"fitness={eval_i['fitness']:.4f}, "
            f"precision={eval_i['precision']:.4f}, "
            f"generalization={eval_i['generalization'] if eval_i['generalization'] is not None else 'n/a'}, "
            f"simplicity={eval_i['simplicity'] if eval_i['simplicity'] is not None else 'n/a'}"
        )

        if eval_i['score'] > best_score:
            best_score = eval_i['score']
            best = {
                'net': net_i,
                'im': im_i,
                'fm': fm_i,
                'eval': eval_i,
                'cand': cand,
            }

    if best is None:
        raise RuntimeError("No candidate Petri net could be mined.")

    assert best is not None

    net, im, fm = best['net'], best['im'], best['fm']
    print(
        f"    Selected params: noise={best['cand'].get('noise_threshold')}, "
        f"heur={best['cand'].get('heuristic_params', {})} "
        f"(score={best['eval']['score']:.4f})"
    )

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
        'mining_hyperparams': {
            'algorithm': algorithm,
            'noise_threshold': best['cand'].get('noise_threshold'),
            'heuristic_params': best['cand'].get('heuristic_params', {}),
            'optimization_metrics': best['eval'],
        },
        'bigram_transitions': bigram_transitions,
        'activity_count_transitions': activity_count_transitions,
        'decision_weights': decision_weights,
        'max_case_length': max_case_length,
    }

    return stats, process_model


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_process(df, mining_algorithm='inductive', noise_threshold=0.2,
                    heuristic_params=None,
                    optimize_mining_hyperparams=False,
                    mining_search_space=None):
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
        - 'ilp' — pm4py ILP Miner → precise/sound, can be strict
        - 'manual' — original manual extraction (no process mining)
    noise_threshold : float
        Noise filtering for the Inductive Miner (0.0 = keep all,
        1.0 = max filtering). Default 0.2.
    heuristic_params : dict or None
        Optional fixed Heuristics Miner params when
        ``optimize_mining_hyperparams`` is False.
    optimize_mining_hyperparams : bool
        If True and ``mining_algorithm`` is in {'inductive', 'heuristic'},
        run a local search over candidate miner params and pick the best
        model by an overall quality metric (fitness, precision,
        generalization, simplicity).
    mining_search_space : dict or None
        Optional search-space override:
        - 'inductive_noise_thresholds': list[float]
        - 'heuristic_params_grid': list[dict]

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
                noise_threshold=noise_threshold,
                heuristic_params=heuristic_params,
                optimize_mining_hyperparams=optimize_mining_hyperparams,
                mining_search_space=mining_search_space,
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


# ---------------------------------------------------------------------------
# Energy modifier extraction
# ---------------------------------------------------------------------------

def _energy_summary(curve: np.ndarray, sensor_name: str) -> dict:
    """Reduce a 1-D sensor curve to 3 scalar features."""
    return {
        f'{sensor_name}_mean': float(np.mean(curve)),
        f'{sensor_name}_end':  float(curve[-1]),
        f'{sensor_name}_std':  float(np.std(curve)),
    }


def _build_energy_state_matrix(df_expanded, sensors, activity_col='activity_log',
                                timestamp_start_col='timestamp_start_log',
                                datetime_energy_col='datetime_energy'):
    """
    For every activity instance in df_expanded, compute the energy-state
    feature vector from the *actual* sensor measurements.

    Returns
    -------
    records : list[dict]
        Each dict has:
          - 'activity'  : str
          - 'duration'  : float  (minutes)
          - 'next_activity' : str  (or '__END__')
          - one key per sensor × 3 features  (mean, end, std)
    energy_state_columns : list[str]
        Ordered list of the 3×N feature names.
    """
    records = []

    # Determine feature column order once
    energy_state_columns = []
    for s in sensors:
        energy_state_columns += [f'{s}_mean', f'{s}_end', f'{s}_std']

    # Group by activity instance
    group_cols = ['case_id_log', 'object_log', activity_col, timestamp_start_col]
    available_group_cols = [c for c in group_cols if c in df_expanded.columns]

    df = df_expanded.dropna(subset=[activity_col]).copy()
    df[datetime_energy_col] = pd.to_datetime(df[datetime_energy_col])

    df['_instance_id'] = df.groupby(available_group_cols).ngroup()

    for instance_id, grp in df.groupby('_instance_id'):
        grp = grp.sort_values(datetime_energy_col)

        activity = str(grp[activity_col].iloc[0]).strip()

        # Duration in minutes
        ts_col = 'timestamp_start_log'
        te_col = 'timestamp_end_log'
        if ts_col in grp.columns and te_col in grp.columns:
            ts = pd.to_datetime(grp[ts_col].iloc[0])
            te = pd.to_datetime(grp[te_col].iloc[0])
            duration = max(0.1, (te - ts).total_seconds() / 60)
        else:
            duration = None

        if duration is None:
            continue

        # Build energy state from actual sensor curves
        row = {'activity': activity, 'duration': duration}
        ok = True
        for sensor in sensors:
            if sensor not in grp.columns:
                raise ValueError(
                    f"extract_energy_modifiers: sensor column '{sensor}' not found "
                    f"in df_expanded. Available columns: {list(grp.columns)}"
                )
            curve = grp[sensor].dropna().values
            if len(curve) < 2:
                ok = False
                break
            row.update(_energy_summary(curve, sensor))

        if not ok:
            continue

        records.append(row)

    # Derive next_activity per (case, object, activity sequence)
    # We do a simple lag on the sorted records per case
    df_recs = pd.DataFrame(records)
    return df_recs, energy_state_columns


class StatisticalDurationBaseline:
    """
    Sklearn-compatible regressor that ignores the energy-state features and
    instead samples from a fitted normal distribution (mean, std) of the
    training durations.  Used as a named competitor in the model-selection
    loop so the report can show its R² and it can win when ML adds no value.

    For the modifier path the target is log(duration/mean_duration), so the
    fitted distribution is on that log-ratio space.  For the direct path the
    target is raw minutes; set ``log_ratio=False`` in that case.
    """

    def __init__(self, log_ratio=True):
        self.log_ratio = log_ratio
        self._mean = 0.0
        self._std  = 1.0

    def fit(self, X, y):
        self._mean = float(np.mean(y))
        self._std  = max(float(np.std(y)), 1e-6)
        return self

    def predict(self, X):
        return np.random.normal(self._mean, self._std, size=len(X))

    # sklearn needs these for get_params / set_params (e.g. cross_val_score)
    def get_params(self, deep=True):
        return {'log_ratio': self.log_ratio}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def _extract_feature_importance(model, feature_names: list) -> dict:
    """Return a {feature: importance} dict for tree/linear models; empty dict otherwise."""
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_.tolist()))
    if hasattr(model, 'coef_'):
        coefs = model.coef_
        if coefs.ndim > 1:
            coefs = np.abs(coefs).mean(axis=0)
        return dict(zip(feature_names, np.abs(coefs).tolist()))
    return {}


def extract_energy_modifiers(
    df_expanded,
    sensors,
    activity_col='activity_log',
    duration_models=None,
    transition_models=None,
    min_samples=30,
    timestamp_start_col='timestamp_start_log',
    datetime_energy_col='datetime_energy',
):
    """
    Mine energy-modifier models from *training* df_expanded.

    For each activity label:
      - duration modifier  : regressor that predicts log(duration/mean_duration)
                             from the energy-state vector.
      - transition modifier: classifier that predicts P(next_activity | energy_state).

    Parameters
    ----------
    df_expanded : pd.DataFrame
        Training expanded df (must contain *_log columns and sensor *_energy columns).
    sensors : list[str]
        Sensor column names to use (must exist in df_expanded).
    activity_col : str
    duration_model_class : sklearn regressor class  (default: Lasso)
    duration_model_params : dict
    transition_model_class : sklearn classifier class  (default: LogisticRegression)
    transition_model_params : dict
    min_transitions : int
        Minimum number of observed transitions required to fit a transition modifier.

    Returns
    -------
    energy_duration_modifiers : dict[activity → fitted model]
    energy_transition_modifiers : dict[activity → fitted model]
    energy_state_columns : list[str]   (3 × N, ordered)
    """
    print("\n" + "=" * 70)
    print("ENERGY MODIFIER EXTRACTION")
    print("=" * 70)
    print(f"  Sensors          : {sensors}")
    print(f"  Duration models  : {duration_models}")
    print(f"  Transition models: {transition_models}")

    # ── Build energy-state matrix from actual sensor curves ───────────────
    df_recs, energy_state_columns = _build_energy_state_matrix(
        df_expanded, sensors,
        activity_col=activity_col,
        timestamp_start_col=timestamp_start_col,
        datetime_energy_col=datetime_energy_col,
    )

    if df_recs.empty:
        print("  WARNING: no valid activity instances found — returning empty modifiers.")
        return {}, {}, energy_state_columns

    if duration_models is None: duration_models = ['xgboost']
    if transition_models is None: transition_models = ['logistic']

    # Local registry of models
    from xgboost import XGBRegressor
    from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, accuracy_score
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    def get_regressor(name):
        if name == 'xgboost': return XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, verbosity=0, random_state=42)
        if name == 'linear': return LinearRegression()
        if name == 'lasso': return Lasso(alpha=0.1, random_state=42)
        if name == 'mlp': return MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
        if name == 'statistical': return StatisticalDurationBaseline(log_ratio=True)
        return XGBRegressor(n_estimators=100, random_state=42) # fallback

    def get_classifier(name):
        if name == 'logistic': return LogisticRegression(max_iter=1000, random_state=42)
        if name == 'random_forest': return RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        if name == 'gradient_boosting': return GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        return LogisticRegression(max_iter=1000, random_state=42) # fallback

    df_recs = _build_energy_state_matrix_with_next(
        df_expanded, sensors, activity_col, timestamp_start_col, datetime_energy_col
    )
    
    energy_state_columns = []
    for s in sensors:
        energy_state_columns += [f'{s}_mean', f'{s}_end', f'{s}_std']

    if df_recs.empty:
        print("  WARNING: no valid instances with next_activity — returning empty modifiers.")
        return {}, {}, energy_state_columns, {}

    print(f"  Total instances  : {len(df_recs)}")

    # ── Per-activity modifiers ────────────────────────────────────────────
    energy_duration_modifiers = {}
    energy_transition_modifiers = {}
    model_choices_report = {}

    for activity, grp_act in df_recs.groupby('activity'):
        X = grp_act[energy_state_columns].values
        n = len(grp_act)
        
        # Training-time column means — fallback energy state
        train_feature_mean = dict(zip(energy_state_columns, X.mean(axis=0)))
        
        act_report = {
            'Duration Approach': f'Statistical Base Method (n={n}<{min_samples})',
            'Transition Approach': f'Statistical Base Method (n={n}<{min_samples})'
        }

        # ── ML Eligibility Check ───────────────────────────────────────
        if n >= min_samples:
            # Prepare split for competition
            y_dur_raw = grp_act['duration'].values
            mean_dur = y_dur_raw.mean()
            if mean_dur > 0:
                y_dur = np.log(y_dur_raw / mean_dur)
            else:
                y_dur = y_dur_raw
                
            y_tr = grp_act['next_activity'].values
            n_classes = len(set(y_tr))

            # Train/Val Split
            X_tr, X_val, yd_tr, yd_val, yt_tr, yt_val = train_test_split(
                X, y_dur, y_tr, test_size=0.2, random_state=42
            )

            # --- Duration Model Selection ---
            # Statistical baseline: R²=0 (predicting the mean, i.e. no energy effect).
            # ML modifier is only kept if it beats R²>0 on the validation split.
            if mean_dur > 0:
                best_dur_score = -float('inf')
                best_dur_name = None  # None → statistical wins

                for model_name in duration_models:
                    try:
                        mdl = get_regressor(model_name)
                        mdl.fit(X_tr, yd_tr)
                        preds = mdl.predict(X_val)
                        score = r2_score(yd_val, preds)
                        if score > best_dur_score:
                            best_dur_score = score
                            best_dur_name = model_name
                    except Exception:
                        pass

                # Statistical baseline R² = 0 (always predicts the mean).
                # Keep ML only if it genuinely improves over the statistical baseline.
                if best_dur_name is not None and best_dur_score > 0.05:
                    try:
                        best_mdl = get_regressor(best_dur_name)
                        best_mdl.fit(X, y_dur)
                        best_mdl._mean_duration = mean_dur
                        best_mdl._train_feature_mean = train_feature_mean
                        best_mdl._feature_importance = _extract_feature_importance(best_mdl, energy_state_columns)
                        energy_duration_modifiers[str(activity)] = best_mdl
                        act_report['Duration Approach'] = f'{best_dur_name} (R²={best_dur_score:.3f})'
                        if best_mdl._feature_importance:
                            top = sorted(best_mdl._feature_importance.items(), key=lambda kv: -kv[1])[:3]
                            act_report['Duration Top Features'] = ', '.join(f'{k}:{v:.3f}' for k, v in top)
                        print(f"  [{activity}] Duration -> ML:{best_dur_name} "
                              f"(val R²={best_dur_score:.3f}) > statistical (R²=0) ✓")
                    except Exception as exc:
                        print(f"  [{activity}] Duration FAILED: {exc}")
                        act_report['Duration Approach'] = 'Statistical (ML fit failed)'
                else:
                    act_report['Duration Approach'] = (
                        f'Statistical (best ML R²={best_dur_score:.3f} ≤ 0.05)'
                    )
                    print(f"  [{activity}] Duration -> statistical "
                          f"(best ML val R²={best_dur_score:.3f} ≤ 0.05, no improvement)")

            # --- Transition Model Selection ---
            # Statistical baseline: majority-class accuracy (predict the most common next activity).
            # ML modifier is only kept if it beats that baseline.
            if n_classes >= 2:
                from collections import Counter
                majority_class_acc = Counter(yt_val).most_common(1)[0][1] / len(yt_val)

                best_tr_score = -float('inf')
                best_tr_name = None  # None → statistical wins

                for model_name in transition_models:
                    try:
                        clf = get_classifier(model_name)
                        clf.fit(X_tr, yt_tr)
                        preds = clf.predict(X_val)
                        score = accuracy_score(yt_val, preds)
                        if score > best_tr_score:
                            best_tr_score = score
                            best_tr_name = model_name
                    except Exception:
                        pass

                if best_tr_name is not None and best_tr_score > majority_class_acc:
                    try:
                        best_clf = get_classifier(best_tr_name)
                        best_clf.fit(X, y_tr)
                        best_clf._train_feature_mean = train_feature_mean
                        best_clf._feature_importance = _extract_feature_importance(best_clf, energy_state_columns)
                        energy_transition_modifiers[str(activity)] = best_clf
                        act_report['Transition Approach'] = (
                            f'{best_tr_name} (Acc={best_tr_score:.3f})'
                        )
                        if best_clf._feature_importance:
                            top = sorted(best_clf._feature_importance.items(), key=lambda kv: -kv[1])[:3]
                            act_report['Transition Top Features'] = ', '.join(f'{k}:{v:.3f}' for k, v in top)
                        print(f"  [{activity}] Transition -> ML:{best_tr_name} "
                              f"(val Acc={best_tr_score:.3f}) > majority baseline "
                              f"({majority_class_acc:.3f}) ✓")
                    except Exception as exc:
                        print(f"  [{activity}] Transition FAILED: {exc}")
                        act_report['Transition Approach'] = 'Statistical (ML fit failed)'
                else:
                    act_report['Transition Approach'] = (
                        f'Statistical (best ML Acc={best_tr_score:.3f} '
                        f'≤ majority baseline {majority_class_acc:.3f})'
                    )
                    print(f"  [{activity}] Transition -> statistical "
                          f"(best ML val Acc={best_tr_score:.3f} ≤ majority "
                          f"baseline {majority_class_acc:.3f})")
            else:
                act_report['Transition Approach'] = 'Statistical (1 class only)'

        model_choices_report[str(activity)] = act_report

    print(f"\n  Duration modifiers  : {len(energy_duration_modifiers)} activities use ML "
          f"(rest fall back to statistical — no modifier applied)")
    print(f"  Transition modifiers: {len(energy_transition_modifiers)} activities use ML "
          f"(rest fall back to statistical — base PN weights unchanged)")
    
    return energy_duration_modifiers, energy_transition_modifiers, energy_state_columns, model_choices_report


def extract_energy_direct_models(
    df_expanded,
    sensors,
    activity_col='activity_log',
    duration_models=None,
    transition_models=None,
    min_samples=30,
    timestamp_start_col='timestamp_start_log',
    datetime_energy_col='datetime_energy',
):
    """
    Train per-activity ML models that *directly* predict duration and next
    activity from the energy-state vector of the *previous* activity.

    Unlike ``extract_energy_modifiers`` (which learns a multiplicative
    correction on top of a statistical baseline), these models are the full
    prediction:

      energy_state_of_prev_activity → duration_minutes   (regressor)
      energy_state_of_prev_activity → next_activity      (classifier, used
                                                           for direct sampling)

    For each activity the best ML model is chosen by validation score and
    kept only if it beats the statistical baseline:
      - Duration  : kept when val R² > 0  (baseline = always predict mean)
      - Transition: kept when val Acc > majority-class accuracy

    Returns
    -------
    duration_models_direct   : dict[activity → fitted regressor]
        Model predicts duration_minutes directly.  Has attribute
        ``_mean_duration`` (fallback) and ``_train_feature_mean``.
    transition_models_direct : dict[activity → fitted classifier]
        Model predicts P(next_activity | energy_state) for direct sampling.
        Has attribute ``_train_feature_mean``.
    energy_state_columns     : list[str]   (3 × N features, ordered)
    model_choices_report     : dict[activity → {Duration, Transition approach}]
    """
    print("\n" + "=" * 70)
    print("ENERGY DIRECT MODEL EXTRACTION")
    print("=" * 70)
    print(f"  Sensors          : {sensors}")
    print(f"  Duration models  : {duration_models}")
    print(f"  Transition models: {transition_models}")

    from xgboost import XGBRegressor
    from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, accuracy_score
    from collections import Counter
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    if duration_models is None:   duration_models   = ['xgboost']
    if transition_models is None: transition_models = ['logistic']

    def get_regressor(name):
        if name == 'xgboost': return XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, verbosity=0, random_state=42)
        if name == 'linear':  return LinearRegression()
        if name == 'lasso':   return Lasso(alpha=0.1, random_state=42)
        if name == 'mlp':     return MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
        if name == 'statistical': return StatisticalDurationBaseline(log_ratio=False)
        return XGBRegressor(n_estimators=100, random_state=42)

    def get_classifier(name):
        if name == 'logistic':          return LogisticRegression(max_iter=1000, random_state=42)
        if name == 'random_forest':     return RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        if name == 'gradient_boosting': return GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        return LogisticRegression(max_iter=1000, random_state=42)

    # Build records: each row = one activity instance with its energy state
    # from the PREVIOUS activity (what's available before this activity fires)
    # and the duration + next_activity of THIS activity.
    df_recs = _build_energy_state_matrix_with_next(
        df_expanded, sensors, activity_col, timestamp_start_col, datetime_energy_col
    )

    energy_state_columns = []
    for s in sensors:
        energy_state_columns += [f'{s}_mean', f'{s}_end', f'{s}_std']

    if df_recs.empty:
        print("  WARNING: no valid instances — returning empty models.")
        return {}, {}, energy_state_columns, {}

    print(f"  Total instances  : {len(df_recs)}")

    duration_models_direct   = {}
    transition_models_direct = {}
    model_choices_report     = {}

    for activity, grp in df_recs.groupby('activity'):
        X       = grp[energy_state_columns].values
        y_dur   = grp['duration'].values          # raw minutes — direct target
        y_tr    = grp['next_activity'].values
        n       = len(grp)
        n_classes = len(set(y_tr))

        train_feature_mean = dict(zip(energy_state_columns, X.mean(axis=0)))
        mean_dur           = float(y_dur.mean())

        act_report = {
            'Duration Approach':   f'Statistical (n={n}<{min_samples})',
            'Transition Approach': f'Statistical (n={n}<{min_samples})',
        }

        if n < min_samples:
            model_choices_report[str(activity)] = act_report
            continue

        X_tr, X_val, yd_tr, yd_val, yt_tr, yt_val = train_test_split(
            X, y_dur, y_tr, test_size=0.2, random_state=42
        )

        # ── Duration: direct regression on minutes ────────────────────
        best_dur_score = -float('inf')
        best_dur_name  = None

        for model_name in duration_models:
            try:
                mdl = get_regressor(model_name)
                mdl.fit(X_tr, yd_tr)
                score = r2_score(yd_val, mdl.predict(X_val))
                if score > best_dur_score:
                    best_dur_score = score
                    best_dur_name  = model_name
            except Exception:
                pass

        if best_dur_name is not None and best_dur_score > 0.05:
            try:
                best_mdl = get_regressor(best_dur_name)
                best_mdl.fit(X, y_dur)
                best_mdl._mean_duration      = mean_dur
                best_mdl._train_feature_mean = train_feature_mean
                best_mdl._feature_importance = _extract_feature_importance(best_mdl, energy_state_columns)
                duration_models_direct[str(activity)] = best_mdl
                act_report['Duration Approach'] = f'{best_dur_name} (R²={best_dur_score:.3f})'
                if best_mdl._feature_importance:
                    top = sorted(best_mdl._feature_importance.items(), key=lambda kv: -kv[1])[:3]
                    act_report['Duration Top Features'] = ', '.join(f'{k}:{v:.3f}' for k, v in top)
                print(f"  [{activity}] Duration -> ML:{best_dur_name} "
                      f"(val R²={best_dur_score:.3f}) > statistical ✓")
            except Exception as exc:
                print(f"  [{activity}] Duration FAILED: {exc}")
                act_report['Duration Approach'] = 'Statistical (ML fit failed)'
        else:
            act_report['Duration Approach'] = (
                f'Statistical (best ML R²={best_dur_score:.3f} ≤ 0.05)'
            )
            print(f"  [{activity}] Duration -> statistical "
                  f"(best ML val R²={best_dur_score:.3f} ≤ 0.05)")

        # ── Transition: direct classifier for sampling ────────────────
        if n_classes >= 2:
            majority_acc   = Counter(yt_val).most_common(1)[0][1] / len(yt_val)
            best_tr_score  = -float('inf')
            best_tr_name   = None

            for model_name in transition_models:
                try:
                    clf   = get_classifier(model_name)
                    clf.fit(X_tr, yt_tr)
                    score = accuracy_score(yt_val, clf.predict(X_val))
                    if score > best_tr_score:
                        best_tr_score = score
                        best_tr_name  = model_name
                except Exception:
                    pass

            if best_tr_name is not None and best_tr_score > majority_acc:
                try:
                    best_clf = get_classifier(best_tr_name)
                    best_clf.fit(X, y_tr)
                    best_clf._train_feature_mean = train_feature_mean
                    best_clf._feature_importance = _extract_feature_importance(best_clf, energy_state_columns)
                    transition_models_direct[str(activity)] = best_clf
                    act_report['Transition Approach'] = (
                        f'{best_tr_name} (Acc={best_tr_score:.3f})'
                    )
                    if best_clf._feature_importance:
                        top = sorted(best_clf._feature_importance.items(), key=lambda kv: -kv[1])[:3]
                        act_report['Transition Top Features'] = ', '.join(f'{k}:{v:.3f}' for k, v in top)
                    print(f"  [{activity}] Transition -> ML:{best_tr_name} "
                          f"(val Acc={best_tr_score:.3f}) > majority ({majority_acc:.3f}) ✓")
                except Exception as exc:
                    print(f"  [{activity}] Transition FAILED: {exc}")
                    act_report['Transition Approach'] = 'Statistical (ML fit failed)'
            else:
                act_report['Transition Approach'] = (
                    f'Statistical (best ML Acc={best_tr_score:.3f} ≤ majority {majority_acc:.3f})'
                )
                print(f"  [{activity}] Transition -> statistical "
                      f"(best ML val Acc={best_tr_score:.3f} ≤ majority {majority_acc:.3f})")
        else:
            act_report['Transition Approach'] = 'Statistical (1 class only)'

        model_choices_report[str(activity)] = act_report

    print(f"\n  Duration direct models  : {len(duration_models_direct)} activities use ML")
    print(f"  Transition direct models: {len(transition_models_direct)} activities use ML")

    return duration_models_direct, transition_models_direct, energy_state_columns, model_choices_report


def _build_energy_state_matrix_with_next(
    df_expanded, sensors, activity_col, timestamp_start_col, datetime_energy_col
):
    """
    Like _build_energy_state_matrix but also resolves next_activity
    from the chronological order within each (case, object) group.
    """
    records = []

    energy_state_columns = []
    for s in sensors:
        energy_state_columns += [f'{s}_mean', f'{s}_end', f'{s}_std']

    group_cols = ['case_id_log', 'object_log', activity_col, timestamp_start_col]
    available_group_cols = [c for c in group_cols if c in df_expanded.columns]

    df = df_expanded.dropna(subset=[activity_col]).copy()
    df[datetime_energy_col] = pd.to_datetime(df[datetime_energy_col])
    df[timestamp_start_col] = pd.to_datetime(df[timestamp_start_col])
    df['_instance_id'] = df.groupby(available_group_cols).ngroup()

    # Build ordered list of instances per (case, object) for next_activity
    order_cols = [c for c in ['case_id_log', 'object_log'] if c in df.columns]
    instance_info = (
        df.groupby('_instance_id')
        .agg(
            activity=(activity_col, 'first'),
            ts=(timestamp_start_col, 'first'),
            **{c: (c, 'first') for c in order_cols}
        )
        .sort_values(order_cols + ['ts'])
        .reset_index()
    )
    # next_activity within each (case, object)
    if order_cols:
        instance_info['next_activity'] = (
            instance_info.groupby(order_cols)['activity'].shift(-1).fillna('__END__')
        )
    else:
        instance_info['next_activity'] = '__END__'

    next_map = dict(zip(instance_info['_instance_id'], instance_info['next_activity']))

    for instance_id, grp in df.groupby('_instance_id'):
        grp = grp.sort_values(datetime_energy_col)
        activity = str(grp[activity_col].iloc[0]).strip()

        ts_col_end = 'timestamp_end_log'
        if timestamp_start_col in grp.columns and ts_col_end in grp.columns:
            ts = pd.to_datetime(grp[timestamp_start_col].iloc[0])
            te = pd.to_datetime(grp[ts_col_end].iloc[0])
            duration = max(0.1, (te - ts).total_seconds() / 60)
        else:
            continue

        row = {'activity': activity, 'duration': duration,
               'next_activity': str(next_map.get(instance_id, '__END__'))}
        ok = True
        for sensor in sensors:
            if sensor not in grp.columns:
                raise ValueError(
                    f"Sensor column '{sensor}' not found in df_expanded. "
                    f"Available: {list(grp.columns)}"
                )
            curve = grp[sensor].dropna().values
            if len(curve) < 2:
                ok = False
                break
            row.update(_energy_summary(curve, sensor))

        if ok:
            records.append(row)

    return pd.DataFrame(records)


# %%
# Modelling the curves

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from dtw import dtw
from tslearn.barycenters import dtw_barycenter_averaging
import optuna
import matplotlib.pyplot as plt


# =============================================================================
# LEAKAGE AUDIT (confirmed clean)
# =============================================================================
# 1. split_curves      — split happens first; ngroup() is a pure index, not a
#                        learned statistic, so computing it pre-split is safe.
# 2. DBA barycenter    — computed from train_curves only.
# 3. DTW alignment     — applied to train_curves only.
# 4. _infer_key_types  — called on train_curves only; test curves use the
#                        resulting key_types dict at predict time.
# 5. get_dummies       — defined on training data; at predict time, test columns
#                        are aligned to feature_columns (missing → 0, extra → dropped).
#                        Val instances share training-level categories (correct).
# 6. scaler            — numeric columns (including curve_length) are scaled
#                        with statistics fitted on train only.
# 7. predict_raw_curve — only the pipeline (fitted on train) is used; no test
#                        value influences any fitted statistic.
# =============================================================================


# =============================================================================
# STEP 1 — RAW SPLIT (before ANY preprocessing touches the data)
# =============================================================================

# Canonical reference length used for DBA/training by default.
# Change this value once to use a different reference grid length.
REFERENCE_LENGTH = 100

def split_curves(df_expanded, variable, activities, objects,
                 test_size=0.15, random_state=42, verbose=1,
                 exog_columns=None):
    """
    Extract all activity curves and split into train/test sets.

    This is the very first operation. No DTW, no DBA, no normalisation —
    nothing touches the data before this split. Test curves are set aside as
    raw numpy arrays and never used again until final evaluation.

    Parameters
    ----------
    df_expanded  : pd.DataFrame
    variable     : str           — target energy column
    activities   : list[str]     — activity filter
    objects      : list[str]     — object_log filter
    test_size    : float         — fraction for test (default 0.15)
    random_state : int
    verbose      : int           — 0 = silent, 1 = full output
    exog_columns : list[str] | None
        Optional list of external-factor columns (e.g. 'ef_temp_energy') whose
        raw time series should be stored per curve for use by exog-aware pipelines.

    Returns
    -------
    train_curves : list[dict]
    test_curves  : list[dict]
        Each dict: 'instance_id', 'activity', 'original_values' (np.ndarray),
                   'original_length', 'attributes'
        When exog_columns is not None, each dict also contains:
            'exog_values': {col: np.ndarray}  — one array per exog column,
                           same length as original_values
    """
    if verbose:
        print("=" * 80)
        print("STEP 1 — RAW TRAIN/TEST SPLIT (before any preprocessing)")
        print("=" * 80)

    exog_cols_present = [c for c in (exog_columns or []) if c in df_expanded.columns]

    df = df_expanded.copy()
    keep_cols = ['case_id_log', 'activity_log', 'object_log', 'timestamp_start_log',
                 'datetime_energy', variable, 'object_attributes_log'] + exog_cols_present
    df = df[keep_cols]
    df = df[df['object_log'].isin(objects)]
    df = df[df['activity_log'].isin(activities)]
    df['timestamp_start'] = pd.to_datetime(df['timestamp_start_log'])
    df['datetime_energy'] = pd.to_datetime(df['datetime_energy'])
    df = df.sort_values(['case_id_log', 'timestamp_start_log', 'datetime_energy'])

    # ngroup() assigns a pure integer index — no statistic is learned here,
    # so computing it on the full (pre-split) df is safe.
    df['activity_instance_id'] = (
        df.groupby(['case_id_log', 'object_log', 'activity_log',
                    'timestamp_start_log']).ngroup()
    )

    curves_data = []
    for instance_id, group in df.groupby('activity_instance_id'):
        group  = group.sort_values('datetime_energy').reset_index(drop=True)
        values = np.asarray(group[variable].dropna().values).squeeze()
        if values.ndim != 1:
            continue
        if len(values) >= 5:
            attributes = (
                group['object_attributes_log'].iloc[0]
                if not group['object_attributes_log'].empty else {}
            )
            entry = {
                'instance_id':     instance_id,
                'activity':        group['activity_log'].iloc[0],
                'original_values': values,
                'original_length': len(values),
                'attributes':      attributes,
            }
            if exog_cols_present:
                entry['exog_values'] = {
                    col: group[col].values for col in exog_cols_present
                }
            curves_data.append(entry)

    if verbose:
        print(f"Total curves extracted : {len(curves_data)}")

    all_ids = list(range(len(curves_data)))
    if test_size <= 0:
        train_ids, test_ids = all_ids, []
    elif test_size >= 1.0:
        train_ids, test_ids = [], all_ids
    else:
        train_ids, test_ids = train_test_split(
            all_ids, test_size=test_size, random_state=random_state
        )
    train_curves = [curves_data[i] for i in train_ids]
    test_curves  = [curves_data[i] for i in test_ids]

    if verbose:
        print(f"Train curves : {len(train_curves)}")
        print(f"Test curves  : {len(test_curves)}  <- set aside, never touched until evaluation")

    return train_curves, test_curves


# =============================================================================
# STEP 1b — VISUALISE RAW TRAINING CURVES (verbose=1 only, before DTW/DBA)
# =============================================================================

def plot_training_curves(train_curves, n_plot=12, figsize_per_row=(14, 2.5), verbose=1):
    """
    Plot a sample of raw training curves BEFORE any DTW/DBA transformation.
    Call this after split_curves() and before build_and_train_pipeline().

    Parameters
    ----------
    train_curves    : list[dict]  — from split_curves()
    n_plot          : int         — how many curves to show (default 12)
    figsize_per_row : tuple       — (width, height) per subplot row
    verbose         : int         — if 0, does nothing
    """
    if not verbose:
        return

    n_plot = min(n_plot, len(train_curves))
    n_cols = 3
    n_rows = int(np.ceil(n_plot / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows)
    )
    axes = np.array(axes).flatten()

    # Sample evenly across the full training set for a representative overview
    indices = np.linspace(0, len(train_curves) - 1, n_plot, dtype=int)

    for ax, idx in zip(axes[:n_plot], indices):
        curve = train_curves[idx]
        ax.plot(curve['original_values'], color='steelblue', linewidth=1.5)
        ax.set_title(
            f"id={curve['instance_id']} | {curve['activity']}\n"
            f"len={curve['original_length']}",
            fontsize=8
        )
        ax.set_xlabel("Time step", fontsize=7)
        ax.set_ylabel("Energy",    fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes[n_plot:]:
        ax.set_visible(False)

    plt.suptitle(
        f"Raw Training Curves — BEFORE DTW/DBA  "
        f"(showing {n_plot} of {len(train_curves)})",
        fontsize=11, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig("tmp.png"); plt.close()


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _align_curve_with_dtw(query, reference):
    """DTW-align a single query curve onto the reference grid (train only)."""
    alignment = dtw(query, reference, keep_internals=True)
    aligned = np.zeros(len(reference))
    counts  = np.zeros(len(reference))
    for qi, ri in zip(alignment.index1, alignment.index2):
        aligned[ri] += query[qi]
        counts[ri]  += 1
    counts  = np.where(counts == 0, 1, counts)
    aligned /= counts
    return aligned


def _infer_key_types(curves):
    """
    Determine numeric vs categorical for each attribute key.
    Always called only on train_curves — key_types is stored in the pipeline
    and reused at predict time without touching test data.
    """
    all_keys = sorted({k for c in curves for k in c['attributes'].keys()})
    key_types = {}
    for key in all_keys:
        is_numeric = True
        for curve in curves:
            if key in curve['attributes']:
                try:
                    float(curve['attributes'][key])
                except (ValueError, TypeError):
                    is_numeric = False
                    break
        key_types[key] = 'numeric' if is_numeric else 'category'
    return all_keys, key_types


def _build_feature_matrix(curves, all_keys, key_types, fixed_length,
                           value_key='resampled_values', include_target=True):
    """
    Flatten DTW-aligned train curves into a regression dataset.
    Each curve contributes `fixed_length` rows (one per barycenter position).
    Only ever called on train_curves.
    """
    _rel_denom = max(fixed_length - 1, 1)
    rows = []
    for curve in curves:
        for position_idx in range(fixed_length):
            row = {
                'instance_id':       curve['instance_id'],
                'activity':          curve['activity'],
                'position_idx':      position_idx,
                'relative_pos':      position_idx / _rel_denom,
                'curve_length':      curve['original_length'],
            }
            if include_target:
                row['y'] = curve[value_key][position_idx]

            for key in all_keys:
                value = curve['attributes'].get(key, None)
                if key_types[key] == 'numeric':
                    try:
                        row[key] = float(value) if value is not None else np.nan
                    except (ValueError, TypeError):
                        row[key] = np.nan
                else:
                    row[key] = str(value) if value is not None else 'None'
            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# STEP 2 — BUILD + TRAIN PIPELINE  (train curves only)
# =============================================================================

def build_and_train_pipeline(
    train_curves,
    variable,
    fixed_length=100,
    val_size=0.2,
    random_state=42,
    models=None,
    optimize_hyperparams=False,
    n_trials=50,
    verbose=1,
    n_jobs=-1,
):
    """
    Build the full preprocessing + training pipeline using ONLY train curves.

    Stages
    ------
    2a. DBA barycenter  — computed from train curves only
    2b. DTW alignment   — aligns each train curve to the barycenter
    2c. Feature matrix  — position + attribute features (train only)
    2d. Model training  — val split is done by instance_id (no leakage)
    2e. Best model      — selected by validation R2

    Parameters
    ----------
    train_curves         : list[dict]  — from split_curves()
    variable             : str         — informational label only
    fixed_length         : int         — DBA barycenter length
    val_size             : float       — fraction of train instances for val
    random_state         : int
    models               : dict        — {name: ModelClass}
    optimize_hyperparams : bool
    n_trials             : int
    verbose              : int         — 0 = silent, 1 = full output

    Returns
    -------
    pipeline : dict
        'model'            — best trained sklearn model
        'model_name'       — str
        'reference_curve'  — DBA barycenter (np.ndarray, length=fixed_length)
        'fixed_length'     — int
        'all_keys'         — attribute key names (inferred from train only)
        'key_types'        — {key: 'numeric'|'category'} (from train only)
        'feature_columns'  — exact column order the model expects
        'val_r2'           — float
        'all_results'      — dict with train/val metrics per model
    """
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2 — BUILD + TRAIN PIPELINE  (train curves only)")
        print("=" * 80)

    if models is None:
        models = {
            'Gradient Boosting': GradientBoostingRegressor,
            'Random Forest':     RandomForestRegressor,
        }

    # ------------------------------------------------------------------
    # 2a. DBA barycenter — train curves only
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[2a] Computing DBA barycenter from {len(train_curves)} train curves "
              f"(fixed_length={fixed_length})...")

    resampled_for_dba = np.array([
        np.interp(
            np.linspace(0, 1, fixed_length),
            np.linspace(0, 1, len(c['original_values'])),
            c['original_values']
        )
        for c in train_curves
    ])[:, :, np.newaxis]

    dba_barycenter  = dtw_barycenter_averaging(resampled_for_dba, barycenter_size=fixed_length)
    reference_curve = dba_barycenter[:, 0]

    if verbose:
        print(f"    Barycenter length: {len(reference_curve)} points")

    # ------------------------------------------------------------------
    # 2b. DTW-align train curves to barycenter  (test curves never touched)
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[2b] Aligning {len(train_curves)} train curves to DBA barycenter...")

    for i, curve in enumerate(train_curves):
        if verbose and i % max(1, len(train_curves) // 10) == 0:
            print(f"    Curve {i + 1}/{len(train_curves)}...")
        curve['resampled_values'] = _align_curve_with_dtw(
            curve['original_values'], reference_curve
        )

    # ------------------------------------------------------------------
    # 2c. Attribute types (train only) + feature matrix (train only)
    # ------------------------------------------------------------------
    all_keys, key_types = _infer_key_types(train_curves)

    if verbose:
        print(f"\n[2c] Attribute columns: {key_types}")

    df_reg = _build_feature_matrix(
        train_curves, all_keys, key_types, fixed_length,
        value_key='resampled_values', include_target=True
    )

    if verbose:
        print(f"     Regression dataset: {len(df_reg)} rows "
              f"({len(train_curves)} curves x {fixed_length} positions)")

    # Dummies defined on training data only. Val instances are a subset of
    # train curves so they share the same category levels — no unseen levels
    # can appear in the val set.
    categorical_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_all = df_reg[['position_idx', 'relative_pos', 'curve_length']].copy()
    X_all = X_all.assign(activity=df_reg['activity'].values)
    for key in all_keys:
        X_all = X_all.assign(**{key: df_reg[key].values})
    X_all = pd.get_dummies(X_all, columns=categorical_cols, drop_first=True)
    y_all = df_reg['y'].copy()

    # Val split by instance_id: an entire curve is either train or val —
    # no point-level leakage between the two sets is possible.
    unique_instances     = df_reg['instance_id'].unique()
    train_inst, val_inst = train_test_split(
        unique_instances, test_size=val_size, random_state=random_state
    )
    train_mask = df_reg['instance_id'].isin(train_inst)
    val_mask   = df_reg['instance_id'].isin(val_inst)

    X_train, X_val = X_all[train_mask].copy(), X_all[val_mask].copy()
    y_train, y_val = y_all[train_mask].copy(), y_all[val_mask].copy()

    feature_columns = X_all.columns.tolist()

    # Scale numeric features (position, curve length, and numeric attributes)
    # using train-only statistics to avoid leakage.
    numeric_feature_cols = ['position_idx', 'relative_pos', 'curve_length'] + [
        k for k in all_keys if key_types[k] == 'numeric'
    ]
    numeric_feature_cols = [c for c in numeric_feature_cols if c in X_train.columns]

    feature_scaler = None
    if numeric_feature_cols:
        X_train[numeric_feature_cols] = X_train[numeric_feature_cols].astype('float64')
        X_val[numeric_feature_cols]   = X_val[numeric_feature_cols].astype('float64')
        feature_scaler = StandardScaler()
        X_train.loc[:, numeric_feature_cols] = feature_scaler.fit_transform(
            X_train[numeric_feature_cols]
        )
        X_val.loc[:, numeric_feature_cols] = feature_scaler.transform(
            X_val[numeric_feature_cols]
        )

    if verbose:
        print(f"\n     Train: {len(train_inst)} curves / {len(X_train)} points")
        print(f"     Val  : {len(val_inst)} curves / {len(X_val)} points")

    # ------------------------------------------------------------------
    # 2d. Train models
    # ------------------------------------------------------------------
    all_results = {}

    for name, model_class in models.items():
        if verbose:
            print(f"\n[2d] Training — {name}...")

        if optimize_hyperparams:
            if not verbose:
                optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial, name=name, model_class=model_class):
                if name == 'Gradient Boosting':
                    params = {
                        'n_estimators':  trial.suggest_int('n_estimators', 50, 300),
                        'max_depth':     trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample':     trial.suggest_float('subsample', 0.5, 1.0),
                        'random_state':  random_state,
                    }
                elif name == 'Random Forest':
                    params = {
                        'n_estimators':      trial.suggest_int('n_estimators', 50, 300),
                        'max_depth':         trial.suggest_int('max_depth', 5, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'random_state':      random_state,
                        'n_jobs':            n_jobs,
                    }
                else:
                    params = {}
                m = model_class(**params)
                m.fit(X_train, y_train)
                return r2_score(y_val, m.predict(X_val))

            sampler = optuna.samplers.TPESampler(seed=random_state)
            study = optuna.create_study(direction='maximize', sampler=sampler)
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            if name in ('Gradient Boosting', 'Random Forest'):
                best_params['random_state'] = random_state
            if name == 'Random Forest':
                best_params['n_jobs'] = n_jobs
            if verbose:
                print(f"     Best params: {best_params}")
            model = model_class(**best_params)

        else:
            if name == 'Gradient Boosting':
                model = model_class(
                    n_estimators=200, max_depth=7, learning_rate=0.1,
                    subsample=0.8, random_state=random_state
                )
            elif name == 'Random Forest':
                model = model_class(
                    n_estimators=200, max_depth=12, min_samples_split=5,
                    random_state=random_state, n_jobs=n_jobs
                )
            else:
                try:
                    model = model_class(random_state=random_state)
                except TypeError:
                    model = model_class()

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred   = model.predict(X_val)
        train_r2   = r2_score(y_train, train_pred)
        val_r2     = r2_score(y_val,   val_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse   = np.sqrt(mean_squared_error(y_val,   val_pred))

        if verbose:
            print(f"     Train  R2={train_r2:.4f}  RMSE={train_rmse:.4f}")
            print(f"     Val    R2={val_r2:.4f}  RMSE={val_rmse:.4f}")

        all_results[name] = {
            'model':      model,
            'train_r2':   train_r2,
            'val_r2':     val_r2,
            'train_rmse': train_rmse,
            'val_rmse':   val_rmse,
        }

    # ------------------------------------------------------------------
    # 2e. Select best model by val R2
    # ------------------------------------------------------------------
    best_name  = max(all_results, key=lambda k: all_results[k]['val_r2'])
    best_model = all_results[best_name]['model']

    if verbose:
        print(f"\n Best model : {best_name}  "
              f"(val R2={all_results[best_name]['val_r2']:.4f})")

    pipeline = {
        'model':           best_model,
        'model_name':      best_name,
        'reference_curve': reference_curve,
        'fixed_length':    fixed_length,
        'all_keys':        all_keys,
        'key_types':       key_types,
        'feature_columns': feature_columns,
        'feature_scaler':  feature_scaler,
        'numeric_feature_cols': numeric_feature_cols,
        'val_r2':          all_results[best_name]['val_r2'],
        'all_results':     all_results,
    }

    return pipeline


# =============================================================================
# STEP 3 — PREDICT ON A SINGLE RAW CURVE
# =============================================================================

def predict_raw_curve(raw_values, activity, attributes, pipeline):
    """
    Predict energy for a single raw, unseen curve using the trained pipeline.

     Steps
     -----
     1. Build canonical features for all reference positions 0..fixed_length-1.
     2. Predict exactly fixed_length values in the canonical (DBA) space.
     3. DTW-align raw curve to reference_curve and decode predictions back to
         raw length using the DTW correspondence (inverse-like mapping).
     4. Return one prediction per raw time step in original energy units.

    Parameters
    ----------
    raw_values  : np.ndarray  — original, unwarped energy curve
    activity    : str
    attributes  : dict
    pipeline    : dict        — from build_and_train_pipeline()

    Returns
    -------
    y_pred : np.ndarray  shape (len(raw_values),), original energy units
    """
    reference_curve = pipeline['reference_curve']
    # Use the trained reference length as the source of truth at inference.
    fixed_length    = len(reference_curve)
    model           = pipeline['model']
    feature_columns = pipeline['feature_columns']
    feature_scaler  = pipeline.get('feature_scaler', None)
    numeric_feature_cols = pipeline.get('numeric_feature_cols', [])
    all_keys        = pipeline['all_keys']
    key_types       = pipeline['key_types']

    # 1) Predict in canonical space (fixed_length points)
    _rel_denom = max(fixed_length - 1, 1)
    rows_ref = []
    for ref_pos in range(fixed_length):
        row = {
            'position_idx': ref_pos,
            'relative_pos': ref_pos / _rel_denom,
            'curve_length': len(raw_values),
            'activity': activity,
        }
        for key in all_keys:
            value = attributes.get(key, None)
            if key_types[key] == 'numeric':
                try:
                    row[key] = float(value) if value is not None else np.nan
                except (ValueError, TypeError):
                    row[key] = np.nan
            else:
                row[key] = str(value) if value is not None else 'None'
        rows_ref.append(row)

    X_ref = pd.DataFrame(rows_ref)
    categorical_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_ref = pd.get_dummies(X_ref, columns=categorical_cols, drop_first=True)

    for col in feature_columns:
        if col not in X_ref.columns:
            X_ref[col] = 0
    X_ref = X_ref[feature_columns]

    if feature_scaler is not None and numeric_feature_cols:
        cols_to_scale = [c for c in numeric_feature_cols if c in X_ref.columns]
        if cols_to_scale:
            X_ref[cols_to_scale] = X_ref[cols_to_scale].astype('float64')
            X_ref.loc[:, cols_to_scale] = feature_scaler.transform(X_ref[cols_to_scale])

    y_ref_pred = model.predict(X_ref)

    # 2) Decode canonical predictions to raw timeline using DTW path
    alignment = dtw(raw_values, reference_curve, keep_internals=True)
    buckets = [[] for _ in range(len(raw_values))]

    for qi, ri in zip(alignment.index1, alignment.index2):
        if 0 <= qi < len(raw_values) and 0 <= ri < fixed_length:
            buckets[qi].append(y_ref_pred[ri])

    y_raw_pred = np.empty(len(raw_values), dtype=float)
    path_pairs = list(zip(alignment.index1, alignment.index2))

    for qi in range(len(raw_values)):
        if buckets[qi]:
            y_raw_pred[qi] = float(np.mean(buckets[qi]))
            continue

        # Robust fallback: use nearest mapped raw index from the DTW path.
        nearest_qi, nearest_ri = min(path_pairs, key=lambda p: abs(p[0] - qi))
        _ = nearest_qi
        y_raw_pred[qi] = float(y_ref_pred[min(max(nearest_ri, 0), fixed_length - 1)])

    return y_raw_pred


# =============================================================================
# INSTANCE STATS — DTW PIPELINE WITH INSTANCE-LEVEL CURVE STATISTICS
#
# The baseline predicts f(position, activity, curve_length) → value.
# For a given activity it learns a single average curve shape and cannot
# distinguish one instance from another.
#
# This variant keeps the exact same DBA + DTW alignment flow but enriches
# the feature row for each position with statistics computed from the
# DTW-aligned curve of that specific instance:
#
#   • curve_mean, curve_std  — overall level and spread of this run
#   • curve_min, curve_max   — range, useful for scaling context
#   • curve_start, curve_end — boundary values that constrain the shape
#
# At train time these come from the DTW-aligned training curve (already
# computed in step 2b — no extra cost).
# At test time the raw curve is DTW-aligned to the reference BEFORE features
# are built, so the model sees the same space it was trained on.
# =============================================================================

def _instance_stats(aligned_curve):
    """Compute per-instance summary statistics from a DTW-aligned curve."""
    v = np.asarray(aligned_curve, dtype=float)
    return {
        'curve_mean':  float(np.mean(v)),
        'curve_std':   float(np.std(v)),
        'curve_min':   float(np.min(v)),
        'curve_max':   float(np.max(v)),
        'curve_start': float(v[0]),
        'curve_end':   float(v[-1]),
    }


def _build_feature_matrix_plus(curves, all_keys, key_types, fixed_length,
                                value_key='resampled_values', include_target=True):
    """
    Like _build_feature_matrix but adds instance-level curve statistics.
    """
    _rel_denom = max(fixed_length - 1, 1)
    rows = []
    for curve in curves:
        stats = _instance_stats(curve[value_key])
        for position_idx in range(fixed_length):
            row = {
                'instance_id':  curve['instance_id'],
                'activity':     curve['activity'],
                'position_idx': position_idx,
                'relative_pos': position_idx / _rel_denom,
                'curve_length': curve['original_length'],
                'curve_mean':   stats['curve_mean'],
                'curve_std':    stats['curve_std'],
                'curve_min':    stats['curve_min'],
                'curve_max':    stats['curve_max'],
                'curve_start':  stats['curve_start'],
                'curve_end':    stats['curve_end'],
            }
            if include_target:
                row['y'] = curve[value_key][position_idx]

            for key in all_keys:
                value = curve['attributes'].get(key, None)
                if key_types[key] == 'numeric':
                    try:
                        row[key] = float(value) if value is not None else np.nan
                    except (ValueError, TypeError):
                        row[key] = np.nan
                else:
                    row[key] = str(value) if value is not None else 'None'
            rows.append(row)

    return pd.DataFrame(rows)


def build_and_train_pipeline_instance_stats(
    train_curves,
    variable,
    fixed_length=100,
    val_size=0.2,
    random_state=42,
    models=None,
    optimize_hyperparams=False,
    n_trials=50,
    verbose=1,
    n_jobs=-1,
):
    """
    Instance Stats — DTW pipeline enriched with instance-level curve statistics.

    Same flow as build_and_train_pipeline(); only _build_feature_matrix_plus
    is used instead of _build_feature_matrix.  Returns the same pipeline dict
    with ``'approach'`` set to ``'instance_stats'``.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("INSTANCE STATS — DTW + INSTANCE CURVE STATISTICS  (train curves only)")
        print("=" * 80)

    if models is None:
        models = {
            'Gradient Boosting': GradientBoostingRegressor,
            'Random Forest':     RandomForestRegressor,
        }

    # 2a. DBA barycenter
    if verbose:
        print(f"\n[2a] Computing DBA barycenter from {len(train_curves)} train curves "
              f"(fixed_length={fixed_length})...")

    resampled_for_dba = np.array([
        np.interp(
            np.linspace(0, 1, fixed_length),
            np.linspace(0, 1, len(c['original_values'])),
            c['original_values']
        )
        for c in train_curves
    ])[:, :, np.newaxis]

    dba_barycenter  = dtw_barycenter_averaging(resampled_for_dba, barycenter_size=fixed_length)
    reference_curve = dba_barycenter[:, 0]

    if verbose:
        print(f"    Barycenter length: {len(reference_curve)} points")

    # 2b. DTW-align train curves
    if verbose:
        print(f"\n[2b] Aligning {len(train_curves)} train curves to DBA barycenter...")

    for i, curve in enumerate(train_curves):
        if verbose and i % max(1, len(train_curves) // 10) == 0:
            print(f"    Curve {i + 1}/{len(train_curves)}...")
        curve['resampled_values'] = _align_curve_with_dtw(
            curve['original_values'], reference_curve
        )

    # 2c. Feature matrix with instance statistics
    all_keys, key_types = _infer_key_types(train_curves)

    if verbose:
        print(f"\n[2c] Attribute columns : {key_types}")
        print("     Extra features     : curve_mean, curve_std, curve_min, "
              "curve_max, curve_start, curve_end")

    df_reg = _build_feature_matrix_plus(
        train_curves, all_keys, key_types, fixed_length,
        value_key='resampled_values', include_target=True
    )

    if verbose:
        print(f"     Regression dataset: {len(df_reg)} rows "
              f"({len(train_curves)} curves x {fixed_length} positions)")

    categorical_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_all = df_reg[['position_idx', 'relative_pos', 'curve_length', 'curve_mean', 'curve_std',
                     'curve_min', 'curve_max', 'curve_start', 'curve_end']].copy()
    X_all = X_all.assign(activity=df_reg['activity'].values)
    for key in all_keys:
        X_all = X_all.assign(**{key: df_reg[key].values})
    X_all = pd.get_dummies(X_all, columns=categorical_cols, drop_first=True)
    y_all = df_reg['y'].copy()

    unique_instances     = df_reg['instance_id'].unique()
    train_inst, val_inst = train_test_split(
        unique_instances, test_size=val_size, random_state=random_state
    )
    train_mask = df_reg['instance_id'].isin(train_inst)
    val_mask   = df_reg['instance_id'].isin(val_inst)

    X_train, X_val = X_all[train_mask].copy(), X_all[val_mask].copy()
    y_train, y_val = y_all[train_mask].copy(), y_all[val_mask].copy()

    feature_columns = X_all.columns.tolist()

    numeric_feature_cols = (
        ['position_idx', 'relative_pos', 'curve_length', 'curve_mean', 'curve_std',
         'curve_min', 'curve_max', 'curve_start', 'curve_end']
        + [k for k in all_keys if key_types[k] == 'numeric']
    )
    numeric_feature_cols = [c for c in numeric_feature_cols if c in X_train.columns]

    feature_scaler = None
    if numeric_feature_cols:
        X_train[numeric_feature_cols] = X_train[numeric_feature_cols].astype('float64')
        X_val[numeric_feature_cols]   = X_val[numeric_feature_cols].astype('float64')
        feature_scaler = StandardScaler()
        X_train.loc[:, numeric_feature_cols] = feature_scaler.fit_transform(
            X_train[numeric_feature_cols]
        )
        X_val.loc[:, numeric_feature_cols] = feature_scaler.transform(
            X_val[numeric_feature_cols]
        )

    if verbose:
        print(f"\n     Train: {len(train_inst)} curves / {len(X_train)} points")
        print(f"     Val  : {len(val_inst)} curves / {len(X_val)} points")

    all_results = {}

    for name, model_class in models.items():
        if verbose:
            print(f"\n[2d] Training — {name}...")

        if optimize_hyperparams:
            if not verbose:
                optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial, name=name, model_class=model_class):
                if name == 'Gradient Boosting':
                    params = {
                        'n_estimators':  trial.suggest_int('n_estimators', 50, 300),
                        'max_depth':     trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample':     trial.suggest_float('subsample', 0.5, 1.0),
                        'random_state':  random_state,
                    }
                elif name == 'Random Forest':
                    params = {
                        'n_estimators':      trial.suggest_int('n_estimators', 50, 300),
                        'max_depth':         trial.suggest_int('max_depth', 5, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'random_state':      random_state,
                        'n_jobs':            n_jobs,
                    }
                else:
                    params = {}
                m = model_class(**params)
                m.fit(X_train, y_train)
                return r2_score(y_val, m.predict(X_val))

            sampler = optuna.samplers.TPESampler(seed=random_state)
            study = optuna.create_study(direction='maximize', sampler=sampler)
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            if name in ('Gradient Boosting', 'Random Forest'):
                best_params['random_state'] = random_state
            if name == 'Random Forest':
                best_params['n_jobs'] = n_jobs
            if verbose:
                print(f"     Best params: {best_params}")
            model = model_class(**best_params)
        else:
            if name == 'Gradient Boosting':
                model = model_class(
                    n_estimators=200, max_depth=7, learning_rate=0.1,
                    subsample=0.8, random_state=random_state
                )
            elif name == 'Random Forest':
                model = model_class(
                    n_estimators=200, max_depth=12, min_samples_split=5,
                    random_state=random_state, n_jobs=n_jobs
                )
            else:
                try:
                    model = model_class(random_state=random_state)
                except TypeError:
                    model = model_class()

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred   = model.predict(X_val)
        train_r2   = r2_score(y_train, train_pred)
        val_r2     = r2_score(y_val,   val_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse   = np.sqrt(mean_squared_error(y_val,   val_pred))

        if verbose:
            print(f"     Train  R2={train_r2:.4f}  RMSE={train_rmse:.4f}")
            print(f"     Val    R2={val_r2:.4f}  RMSE={val_rmse:.4f}")

        all_results[name] = {
            'model':      model,
            'train_r2':   train_r2,
            'val_r2':     val_r2,
            'train_rmse': train_rmse,
            'val_rmse':   val_rmse,
        }

    best_name  = max(all_results, key=lambda k: all_results[k]['val_r2'])
    best_model = all_results[best_name]['model']

    if verbose:
        print(f"\n Best model : {best_name}  "
              f"(val R2={all_results[best_name]['val_r2']:.4f})")

    pipeline = {
        'approach':        'instance_stats',
        'model':           best_model,
        'model_name':      best_name,
        'reference_curve': reference_curve,
        'fixed_length':    fixed_length,
        'all_keys':        all_keys,
        'key_types':       key_types,
        'feature_columns': feature_columns,
        'feature_scaler':  feature_scaler,
        'numeric_feature_cols': numeric_feature_cols,
        'val_r2':          all_results[best_name]['val_r2'],
        'all_results':     all_results,
    }

    return pipeline


def predict_raw_curve_instance_stats(raw_values, activity, attributes, pipeline):
    """
    Predict using the Instance Stats pipeline.

    The raw test curve is DTW-aligned to the reference curve first, then
    instance statistics are computed from that aligned version and used as
    features alongside position_idx.  The canonical predictions are mapped
    back to raw time steps via the same DTW warp path.
    """
    reference_curve      = pipeline['reference_curve']
    fixed_length         = len(reference_curve)
    model                = pipeline['model']
    feature_columns      = pipeline['feature_columns']
    feature_scaler       = pipeline.get('feature_scaler', None)
    numeric_feature_cols = pipeline.get('numeric_feature_cols', [])
    all_keys             = pipeline['all_keys']
    key_types            = pipeline['key_types']

    # Align test curve to reference to get the canonical-space version
    alignment     = dtw(raw_values, reference_curve, keep_internals=True)
    aligned_test  = _align_curve_with_dtw(raw_values, reference_curve)
    stats         = _instance_stats(aligned_test)

    _rel_denom = max(fixed_length - 1, 1)
    rows_ref = []
    for pos in range(fixed_length):
        row = {
            'position_idx': pos,
            'relative_pos': pos / _rel_denom,
            'curve_length': len(raw_values),
            'curve_mean':   stats['curve_mean'],
            'curve_std':    stats['curve_std'],
            'curve_min':    stats['curve_min'],
            'curve_max':    stats['curve_max'],
            'curve_start':  stats['curve_start'],
            'curve_end':    stats['curve_end'],
            'activity':     activity,
        }
        for key in all_keys:
            value = attributes.get(key, None)
            if key_types[key] == 'numeric':
                try:
                    row[key] = float(value) if value is not None else np.nan
                except (ValueError, TypeError):
                    row[key] = np.nan
            else:
                row[key] = str(value) if value is not None else 'None'
        rows_ref.append(row)

    X_ref = pd.DataFrame(rows_ref)
    categorical_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_ref = pd.get_dummies(X_ref, columns=categorical_cols, drop_first=True)

    for col in feature_columns:
        if col not in X_ref.columns:
            X_ref[col] = 0
    X_ref = X_ref[feature_columns]

    if feature_scaler is not None and numeric_feature_cols:
        cols_to_scale = [c for c in numeric_feature_cols if c in X_ref.columns]
        if cols_to_scale:
            X_ref[cols_to_scale] = X_ref[cols_to_scale].astype('float64')
            X_ref.loc[:, cols_to_scale] = feature_scaler.transform(X_ref[cols_to_scale])

    y_ref_pred = model.predict(X_ref)

    # Map canonical predictions back to raw time steps
    buckets    = [[] for _ in range(len(raw_values))]
    path_pairs = list(zip(alignment.index1, alignment.index2))

    for qi, ri in path_pairs:
        if 0 <= qi < len(raw_values) and 0 <= ri < fixed_length:
            buckets[qi].append(y_ref_pred[ri])

    y_raw_pred = np.empty(len(raw_values), dtype=float)
    for qi in range(len(raw_values)):
        if buckets[qi]:
            y_raw_pred[qi] = float(np.mean(buckets[qi]))
        else:
            nearest_qi, nearest_ri = min(path_pairs, key=lambda p: abs(p[0] - qi))
            _ = nearest_qi
            y_raw_pred[qi] = float(y_ref_pred[min(max(nearest_ri, 0), fixed_length - 1)])

    return y_raw_pred


# =============================================================================
# INSTANCE STATS (LEAK-FREE) — TWO-STAGE PIPELINE
#
# Fixes the data leakage in the Instance Stats approach: instead of computing
# curve statistics (mean, std, min, max, start, end) from the actual test curve,
# a first-stage model predicts those statistics from observable metadata only
# (activity, curve_length, process attributes).  The predicted stats are then
# fed into the existing DTW curve-shape model as if they were known.
#
# Stage 1 — Stats Predictor:
#   Input : activity, curve_length, process attributes
#   Output: predicted {curve_mean, curve_std, curve_min, curve_max,
#                       curve_start, curve_end}
#
# Stage 2 — Curve Shape Model (identical to instance_stats):
#   Input : position_idx, curve_length, predicted stats, activity, attributes
#   Output: predicted energy curve
# =============================================================================

_STAT_TARGETS = ['curve_mean', 'curve_std', 'curve_min', 'curve_max',
                 'curve_start', 'curve_end']


def _build_stats_predictor_dataset(curves, all_keys, key_types):
    """
    Build a feature matrix where each row is one curve instance and the targets
    are the six instance statistics.  Only metadata available before seeing the
    curve is used as input.
    """
    rows, targets = [], []
    for curve in curves:
        stats = _instance_stats(curve['resampled_values'])
        row = {
            'curve_length': curve['original_length'],
            'activity':     curve['activity'],
        }
        for key in all_keys:
            value = curve['attributes'].get(key, None)
            if key_types[key] == 'numeric':
                try:
                    row[key] = float(value) if value is not None else np.nan
                except (ValueError, TypeError):
                    row[key] = np.nan
            else:
                row[key] = str(value) if value is not None else 'None'
        rows.append(row)
        targets.append([stats[t] for t in _STAT_TARGETS])
    return pd.DataFrame(rows), np.array(targets)


def build_and_train_pipeline_istats_leakfree(
    train_curves,
    variable,
    fixed_length=100,
    val_size=0.2,
    random_state=42,
    models=None,
    optimize_hyperparams=False,
    n_trials=50,
    verbose=1,
    n_jobs=-1,
):
    """
    Instance Stats (Leak-Free) — two-stage pipeline.

    Stage 1 trains a multi-output regressor to predict the six curve statistics
    from metadata only (no test-curve values used at inference time).
    Stage 2 is identical to ``build_and_train_pipeline_instance_stats`` and uses
    those predicted stats as features.  Returns the same pipeline dict as the
    leaky version with ``'approach'`` set to ``'istats_leakfree'`` and an extra
    ``'stats_predictor'`` key.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("INSTANCE STATS (LEAK-FREE) — TWO-STAGE DTW PIPELINE")
        print("=" * 80)

    if models is None:
        models = {
            'Gradient Boosting': GradientBoostingRegressor,
            'Random Forest':     RandomForestRegressor,
        }

    # ── Stage 0: DBA barycenter + DTW alignment (same as instance_stats) ─────
    if verbose:
        print(f"\n[0] Computing DBA barycenter from {len(train_curves)} train curves "
              f"(fixed_length={fixed_length})...")

    resampled_for_dba = np.array([
        np.interp(
            np.linspace(0, 1, fixed_length),
            np.linspace(0, 1, len(c['original_values'])),
            c['original_values']
        )
        for c in train_curves
    ])[:, :, np.newaxis]

    dba_barycenter  = dtw_barycenter_averaging(resampled_for_dba, barycenter_size=fixed_length)
    reference_curve = dba_barycenter[:, 0]

    for i, curve in enumerate(train_curves):
        if verbose and i % max(1, len(train_curves) // 10) == 0:
            print(f"    Aligning curve {i + 1}/{len(train_curves)}...")
        curve['resampled_values'] = _align_curve_with_dtw(
            curve['original_values'], reference_curve
        )

    all_keys, key_types = _infer_key_types(train_curves)

    # ── Stage 1: train stats predictor ───────────────────────────────────────
    if verbose:
        print(f"\n[1] Training Stage 1 — stats predictor "
              f"(targets: {_STAT_TARGETS})...")

    X_sp, y_sp = _build_stats_predictor_dataset(train_curves, all_keys, key_types)

    categorical_cols_sp = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_sp = pd.get_dummies(X_sp, columns=categorical_cols_sp, drop_first=True)
    sp_feature_columns = X_sp.columns.tolist()

    numeric_sp_cols = ['curve_length'] + [k for k in all_keys if key_types[k] == 'numeric']
    numeric_sp_cols = [c for c in numeric_sp_cols if c in X_sp.columns]

    sp_scaler = None
    if numeric_sp_cols:
        sp_scaler = StandardScaler()
        X_sp = X_sp.copy()
        X_sp.loc[:, numeric_sp_cols] = sp_scaler.fit_transform(X_sp[numeric_sp_cols])

    unique_instances     = [c['instance_id'] for c in train_curves]
    sp_train_idx, sp_val_idx = train_test_split(
        range(len(train_curves)), test_size=val_size, random_state=random_state
    )

    X_sp_train, X_sp_val = X_sp.iloc[sp_train_idx], X_sp.iloc[sp_val_idx]
    y_sp_train, y_sp_val = y_sp[sp_train_idx],       y_sp[sp_val_idx]

    from sklearn.multioutput import MultiOutputRegressor

    sp_results = {}
    for name, model_class in models.items():
        if name == 'Gradient Boosting':
            base = model_class(n_estimators=100, max_depth=5, learning_rate=0.1,
                               random_state=random_state)
        elif name == 'Random Forest':
            base = model_class(n_estimators=100, max_depth=10, random_state=random_state,
                               n_jobs=n_jobs)
        else:
            try:
                base = model_class(random_state=random_state)
            except TypeError:
                base = model_class()

        sp_model = MultiOutputRegressor(base)
        sp_model.fit(X_sp_train, y_sp_train)
        val_r2 = float(np.mean([
            r2_score(y_sp_val[:, i], sp_model.predict(X_sp_val)[:, i])
            for i in range(y_sp.shape[1])
        ]))
        sp_results[name] = {'model': sp_model, 'val_r2': val_r2}
        if verbose:
            print(f"     {name}  mean-val-R²={val_r2:.4f}")

    best_sp_name  = max(sp_results, key=lambda k: sp_results[k]['val_r2'])
    best_sp_model = sp_results[best_sp_name]['model']
    if verbose:
        print(f"  Best stats predictor: {best_sp_name}  "
              f"(mean val R²={sp_results[best_sp_name]['val_r2']:.4f})")

    # ── Stage 2: train curve-shape model (identical to instance_stats) ───────
    if verbose:
        print(f"\n[2] Training Stage 2 — curve shape model...")

    df_reg = _build_feature_matrix_plus(
        train_curves, all_keys, key_types, fixed_length,
        value_key='resampled_values', include_target=True
    )

    categorical_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_all = df_reg[['position_idx', 'relative_pos', 'curve_length', 'curve_mean', 'curve_std',
                     'curve_min', 'curve_max', 'curve_start', 'curve_end']].copy()
    X_all = X_all.assign(activity=df_reg['activity'].values)
    for key in all_keys:
        X_all = X_all.assign(**{key: df_reg[key].values})
    X_all = pd.get_dummies(X_all, columns=categorical_cols, drop_first=True)
    y_all = df_reg['y'].copy()

    train_inst_ids = {train_curves[i]['instance_id'] for i in sp_train_idx}
    train_mask = df_reg['instance_id'].isin(train_inst_ids)
    val_mask   = ~train_mask

    X_train, X_val = X_all[train_mask].copy(), X_all[val_mask].copy()
    y_train, y_val = y_all[train_mask].copy(), y_all[val_mask].copy()

    feature_columns = X_all.columns.tolist()

    numeric_feature_cols = (
        ['position_idx', 'relative_pos', 'curve_length', 'curve_mean', 'curve_std',
         'curve_min', 'curve_max', 'curve_start', 'curve_end']
        + [k for k in all_keys if key_types[k] == 'numeric']
    )
    numeric_feature_cols = [c for c in numeric_feature_cols if c in X_train.columns]

    feature_scaler = None
    if numeric_feature_cols:
        X_train[numeric_feature_cols] = X_train[numeric_feature_cols].astype('float64')
        X_val[numeric_feature_cols]   = X_val[numeric_feature_cols].astype('float64')
        feature_scaler = StandardScaler()
        X_train.loc[:, numeric_feature_cols] = feature_scaler.fit_transform(
            X_train[numeric_feature_cols]
        )
        X_val.loc[:, numeric_feature_cols] = feature_scaler.transform(
            X_val[numeric_feature_cols]
        )

    all_results = {}
    for name, model_class in models.items():
        if verbose:
            print(f"     Training {name}...")
        if name == 'Gradient Boosting':
            model = model_class(n_estimators=200, max_depth=7, learning_rate=0.1,
                                subsample=0.8, random_state=random_state)
        elif name == 'Random Forest':
            model = model_class(n_estimators=200, max_depth=12, min_samples_split=5,
                                random_state=random_state, n_jobs=n_jobs)
        else:
            try:
                model = model_class(random_state=random_state)
            except TypeError:
                model = model_class()

        model.fit(X_train, y_train)
        val_r2   = r2_score(y_val,   model.predict(X_val))
        val_rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
        train_r2   = r2_score(y_train, model.predict(X_train))
        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        if verbose:
            print(f"       Train R²={train_r2:.4f}  RMSE={train_rmse:.4f} | "
                  f"Val R²={val_r2:.4f}  RMSE={val_rmse:.4f}")
        all_results[name] = {
            'model': model, 'train_r2': train_r2, 'val_r2': val_r2,
            'train_rmse': train_rmse, 'val_rmse': val_rmse,
        }

    best_name  = max(all_results, key=lambda k: all_results[k]['val_r2'])
    best_model = all_results[best_name]['model']
    if verbose:
        print(f"\n  Best curve model: {best_name}  "
              f"(val R²={all_results[best_name]['val_r2']:.4f})")

    return {
        'approach':            'istats_leakfree',
        'model':               best_model,
        'model_name':          best_name,
        'reference_curve':     reference_curve,
        'fixed_length':        fixed_length,
        'all_keys':            all_keys,
        'key_types':           key_types,
        'feature_columns':     feature_columns,
        'feature_scaler':      feature_scaler,
        'numeric_feature_cols': numeric_feature_cols,
        'val_r2':              all_results[best_name]['val_r2'],
        'all_results':         all_results,
        # Stage 1
        'stats_predictor':         best_sp_model,
        'sp_feature_columns':      sp_feature_columns,
        'sp_scaler':               sp_scaler,
        'numeric_sp_cols':         numeric_sp_cols,
        'sp_categorical_cols':     categorical_cols_sp,
    }


def predict_raw_curve_istats_leakfree(raw_values, activity, attributes, pipeline):
    """
    Predict using the leak-free two-stage Instance Stats pipeline.

    Stage 1 predicts curve statistics from metadata (no test values used).
    Stage 2 uses those predicted stats as features, identical to the leaky
    version, and maps predictions back to raw time steps via DTW.
    """
    reference_curve      = pipeline['reference_curve']
    fixed_length         = len(reference_curve)
    model                = pipeline['model']
    feature_columns      = pipeline['feature_columns']
    feature_scaler       = pipeline.get('feature_scaler', None)
    numeric_feature_cols = pipeline.get('numeric_feature_cols', [])
    all_keys             = pipeline['all_keys']
    key_types            = pipeline['key_types']

    stats_predictor    = pipeline['stats_predictor']
    sp_feature_columns = pipeline['sp_feature_columns']
    sp_scaler          = pipeline.get('sp_scaler', None)
    numeric_sp_cols    = pipeline.get('numeric_sp_cols', [])
    sp_categorical_cols = pipeline.get('sp_categorical_cols', [])

    # ── Stage 1: predict curve stats from metadata only ──────────────────────
    sp_row = {'curve_length': len(raw_values), 'activity': activity}
    for key in all_keys:
        value = attributes.get(key, None)
        if key_types[key] == 'numeric':
            try:
                sp_row[key] = float(value) if value is not None else np.nan
            except (ValueError, TypeError):
                sp_row[key] = np.nan
        else:
            sp_row[key] = str(value) if value is not None else 'None'

    X_sp = pd.DataFrame([sp_row])
    X_sp = pd.get_dummies(X_sp, columns=sp_categorical_cols, drop_first=True)
    for col in sp_feature_columns:
        if col not in X_sp.columns:
            X_sp[col] = 0
    X_sp = X_sp[sp_feature_columns]

    if sp_scaler is not None and numeric_sp_cols:
        cols_to_scale = [c for c in numeric_sp_cols if c in X_sp.columns]
        if cols_to_scale:
            X_sp[cols_to_scale] = X_sp[cols_to_scale].astype('float64')
            X_sp.loc[:, cols_to_scale] = sp_scaler.transform(X_sp[cols_to_scale])

    predicted_stats_arr = stats_predictor.predict(X_sp)[0]
    predicted_stats = dict(zip(_STAT_TARGETS, predicted_stats_arr))

    # ── Stage 2: build feature rows using predicted stats ────────────────────
    alignment = dtw(raw_values, reference_curve, keep_internals=True)

    _rel_denom = max(fixed_length - 1, 1)
    rows_ref = []
    for pos in range(fixed_length):
        row = {
            'position_idx': pos,
            'relative_pos': pos / _rel_denom,
            'curve_length': len(raw_values),
            'curve_mean':   predicted_stats['curve_mean'],
            'curve_std':    predicted_stats['curve_std'],
            'curve_min':    predicted_stats['curve_min'],
            'curve_max':    predicted_stats['curve_max'],
            'curve_start':  predicted_stats['curve_start'],
            'curve_end':    predicted_stats['curve_end'],
            'activity':     activity,
        }
        for key in all_keys:
            value = attributes.get(key, None)
            if key_types[key] == 'numeric':
                try:
                    row[key] = float(value) if value is not None else np.nan
                except (ValueError, TypeError):
                    row[key] = np.nan
            else:
                row[key] = str(value) if value is not None else 'None'
        rows_ref.append(row)

    X_ref = pd.DataFrame(rows_ref)
    categorical_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_ref = pd.get_dummies(X_ref, columns=categorical_cols, drop_first=True)
    for col in feature_columns:
        if col not in X_ref.columns:
            X_ref[col] = 0
    X_ref = X_ref[feature_columns]

    if feature_scaler is not None and numeric_feature_cols:
        cols_to_scale = [c for c in numeric_feature_cols if c in X_ref.columns]
        if cols_to_scale:
            X_ref[cols_to_scale] = X_ref[cols_to_scale].astype('float64')
            X_ref.loc[:, cols_to_scale] = feature_scaler.transform(X_ref[cols_to_scale])

    y_ref_pred = model.predict(X_ref)

    # Map canonical predictions back to raw time steps via DTW warp path
    buckets    = [[] for _ in range(len(raw_values))]
    path_pairs = list(zip(alignment.index1, alignment.index2))

    for qi, ri in path_pairs:
        if 0 <= qi < len(raw_values) and 0 <= ri < fixed_length:
            buckets[qi].append(y_ref_pred[ri])

    y_raw_pred = np.empty(len(raw_values), dtype=float)
    for qi in range(len(raw_values)):
        if buckets[qi]:
            y_raw_pred[qi] = float(np.mean(buckets[qi]))
        else:
            nearest_qi, nearest_ri = min(path_pairs, key=lambda p: abs(p[0] - qi))
            _ = nearest_qi
            y_raw_pred[qi] = float(y_ref_pred[min(max(nearest_ri, 0), fixed_length - 1)])

    return y_raw_pred


# =============================================================================
# APPROACH 3 — DTW-PHASE AWARE PIPELINE
#
# Same architecture as the baseline (DBA barycenter → DTW alignment → regression)
# but the feature matrix is enriched with phase-aware descriptors computed from
# the DTW-aligned curve at each position:
#
#   • local_slope      — finite-difference derivative, tells the model whether
#                        we are in a rising / falling transient or plateau
#   • local_curvature  — second derivative, distinguishes ramp from inflection
#   • phase_norm       — position / (fixed_length - 1), replaces the raw int
#                        index with a [0, 1] normalised phase coordinate
#   • ref_value        — the DBA barycenter value at this position, giving the
#                        model a shape prior from the training prototype
#   • ref_slope        — slope of the barycenter at this position
#
# At prediction time the same descriptors are derived from the DBA barycenter
# (the only aligned curve available without touching test data), so the model
# sees the same feature space it was trained on.
# =============================================================================

def _phase_features_from_curve(aligned_curve):
    """
    Compute per-position phase descriptors for a DTW-aligned curve.

    Returns a dict of 1-D arrays, each of length len(aligned_curve).
    """
    v = np.asarray(aligned_curve, dtype=float)
    n = len(v)

    slope = np.gradient(v)
    curvature = np.gradient(slope)
    phase_norm = np.linspace(0.0, 1.0, n)

    return {
        'phase_norm':     phase_norm,
        'local_slope':    slope,
        'local_curvature': curvature,
        'ref_value':      v,
        'ref_slope':      slope,   # for reference row: ref_value IS the curve
    }


def _build_feature_matrix_dtw_phase(
    curves, all_keys, key_types, fixed_length,
    reference_curve, value_key='resampled_values', include_target=True
):
    """
    Like _build_feature_matrix but adds phase-aware features derived from the
    DTW-aligned curve and the DBA reference curve.
    """
    ref_feats = _phase_features_from_curve(reference_curve)

    rows = []
    for curve in curves:
        aligned = np.asarray(curve[value_key], dtype=float)
        curve_feats = _phase_features_from_curve(aligned)

        _rel_denom = max(fixed_length - 1, 1)
        for pos in range(fixed_length):
            row = {
                'instance_id':      curve['instance_id'],
                'activity':         curve['activity'],
                'position_idx':     pos,
                'relative_pos':     pos / _rel_denom,
                'curve_length':     curve['original_length'],
                'phase_norm':       curve_feats['phase_norm'][pos],
                'local_slope':      curve_feats['local_slope'][pos],
                'local_curvature':  curve_feats['local_curvature'][pos],
                'ref_value':        ref_feats['ref_value'][pos],
                'ref_slope':        ref_feats['ref_slope'][pos],
            }
            if include_target:
                row['y'] = aligned[pos]

            for key in all_keys:
                value = curve['attributes'].get(key, None)
                if key_types[key] == 'numeric':
                    try:
                        row[key] = float(value) if value is not None else np.nan
                    except (ValueError, TypeError):
                        row[key] = np.nan
                else:
                    row[key] = str(value) if value is not None else 'None'
            rows.append(row)

    return pd.DataFrame(rows)


def build_and_train_pipeline_dtw_phase(
    train_curves,
    variable,
    fixed_length=100,
    val_size=0.2,
    random_state=42,
    models=None,
    optimize_hyperparams=False,
    n_trials=50,
    verbose=1,
    n_jobs=-1,
):
    """
    Approach 3 — DTW-phase-aware pipeline.

    Identical flow to build_and_train_pipeline() but the regression features are
    enriched with phase descriptors (slope, curvature, normalised phase, reference
    curve value/slope) so the model understands *where in the process phase* each
    prediction sits, not just its raw integer index.

    Returns the same pipeline dict as build_and_train_pipeline(), with an extra
    key ``'approach'`` set to ``'dtw_phase'``.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("APPROACH 3 — DTW-PHASE AWARE PIPELINE  (train curves only)")
        print("=" * 80)

    if models is None:
        models = {
            'Gradient Boosting': GradientBoostingRegressor,
            'Random Forest':     RandomForestRegressor,
        }

    # 2a. DBA barycenter — identical to baseline
    if verbose:
        print(f"\n[2a] Computing DBA barycenter from {len(train_curves)} train curves "
              f"(fixed_length={fixed_length})...")

    resampled_for_dba = np.array([
        np.interp(
            np.linspace(0, 1, fixed_length),
            np.linspace(0, 1, len(c['original_values'])),
            c['original_values']
        )
        for c in train_curves
    ])[:, :, np.newaxis]

    dba_barycenter  = dtw_barycenter_averaging(resampled_for_dba, barycenter_size=fixed_length)
    reference_curve = dba_barycenter[:, 0]

    if verbose:
        print(f"    Barycenter length: {len(reference_curve)} points")

    # 2b. DTW-align train curves to barycenter
    if verbose:
        print(f"\n[2b] Aligning {len(train_curves)} train curves to DBA barycenter...")

    for i, curve in enumerate(train_curves):
        if verbose and i % max(1, len(train_curves) // 10) == 0:
            print(f"    Curve {i + 1}/{len(train_curves)}...")
        curve['resampled_values'] = _align_curve_with_dtw(
            curve['original_values'], reference_curve
        )

    # 2c. Attribute types + phase-enriched feature matrix
    all_keys, key_types = _infer_key_types(train_curves)

    if verbose:
        print(f"\n[2c] Attribute columns: {key_types}")
        print("     Phase features: phase_norm, local_slope, local_curvature, "
              "ref_value, ref_slope")

    df_reg = _build_feature_matrix_dtw_phase(
        train_curves, all_keys, key_types, fixed_length,
        reference_curve, value_key='resampled_values', include_target=True
    )

    if verbose:
        print(f"     Regression dataset: {len(df_reg)} rows "
              f"({len(train_curves)} curves x {fixed_length} positions)")

    categorical_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_all = df_reg.drop(columns=['instance_id', 'y'], errors='ignore').copy()
    X_all = pd.get_dummies(X_all, columns=categorical_cols, drop_first=True)
    y_all = df_reg['y'].copy()

    unique_instances     = df_reg['instance_id'].unique()
    train_inst, val_inst = train_test_split(
        unique_instances, test_size=val_size, random_state=random_state
    )
    train_mask = df_reg['instance_id'].isin(train_inst)
    val_mask   = df_reg['instance_id'].isin(val_inst)

    X_train, X_val = X_all[train_mask].copy(), X_all[val_mask].copy()
    y_train, y_val = y_all[train_mask].copy(), y_all[val_mask].copy()

    feature_columns = X_all.columns.tolist()

    numeric_feature_cols = (
        ['position_idx', 'relative_pos', 'curve_length', 'phase_norm',
         'local_slope', 'local_curvature', 'ref_value', 'ref_slope']
        + [k for k in all_keys if key_types[k] == 'numeric']
    )
    numeric_feature_cols = [c for c in numeric_feature_cols if c in X_train.columns]

    feature_scaler = None
    if numeric_feature_cols:
        X_train[numeric_feature_cols] = X_train[numeric_feature_cols].astype('float64')
        X_val[numeric_feature_cols]   = X_val[numeric_feature_cols].astype('float64')
        feature_scaler = StandardScaler()
        X_train.loc[:, numeric_feature_cols] = feature_scaler.fit_transform(
            X_train[numeric_feature_cols]
        )
        X_val.loc[:, numeric_feature_cols] = feature_scaler.transform(
            X_val[numeric_feature_cols]
        )

    if verbose:
        print(f"\n     Train: {len(train_inst)} curves / {len(X_train)} points")
        print(f"     Val  : {len(val_inst)} curves / {len(X_val)} points")

    # 2d. Train models
    all_results = {}

    for name, model_class in models.items():
        if verbose:
            print(f"\n[2d] Training — {name}...")

        if optimize_hyperparams:
            if not verbose:
                optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial, name=name, model_class=model_class):
                if name == 'Gradient Boosting':
                    params = {
                        'n_estimators':  trial.suggest_int('n_estimators', 50, 300),
                        'max_depth':     trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample':     trial.suggest_float('subsample', 0.5, 1.0),
                        'random_state':  random_state,
                    }
                elif name == 'Random Forest':
                    params = {
                        'n_estimators':      trial.suggest_int('n_estimators', 50, 300),
                        'max_depth':         trial.suggest_int('max_depth', 5, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'random_state':      random_state,
                        'n_jobs':            n_jobs,
                    }
                else:
                    params = {}
                m = model_class(**params)
                m.fit(X_train, y_train)
                return r2_score(y_val, m.predict(X_val))

            sampler = optuna.samplers.TPESampler(seed=random_state)
            study = optuna.create_study(direction='maximize', sampler=sampler)
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            if name in ('Gradient Boosting', 'Random Forest'):
                best_params['random_state'] = random_state
            if name == 'Random Forest':
                best_params['n_jobs'] = n_jobs
            if verbose:
                print(f"     Best params: {best_params}")
            model = model_class(**best_params)
        else:
            if name == 'Gradient Boosting':
                model = model_class(
                    n_estimators=200, max_depth=7, learning_rate=0.1,
                    subsample=0.8, random_state=random_state
                )
            elif name == 'Random Forest':
                model = model_class(
                    n_estimators=200, max_depth=12, min_samples_split=5,
                    random_state=random_state, n_jobs=n_jobs
                )
            else:
                try:
                    model = model_class(random_state=random_state)
                except TypeError:
                    model = model_class()

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred   = model.predict(X_val)
        train_r2   = r2_score(y_train, train_pred)
        val_r2     = r2_score(y_val,   val_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse   = np.sqrt(mean_squared_error(y_val,   val_pred))

        if verbose:
            print(f"     Train  R2={train_r2:.4f}  RMSE={train_rmse:.4f}")
            print(f"     Val    R2={val_r2:.4f}  RMSE={val_rmse:.4f}")

        all_results[name] = {
            'model':      model,
            'train_r2':   train_r2,
            'val_r2':     val_r2,
            'train_rmse': train_rmse,
            'val_rmse':   val_rmse,
        }

    # 2e. Best model by val R2
    best_name  = max(all_results, key=lambda k: all_results[k]['val_r2'])
    best_model = all_results[best_name]['model']

    if verbose:
        print(f"\n Best model : {best_name}  "
              f"(val R2={all_results[best_name]['val_r2']:.4f})")

    pipeline = {
        'approach':        'dtw_phase',
        'model':           best_model,
        'model_name':      best_name,
        'reference_curve': reference_curve,
        'fixed_length':    fixed_length,
        'all_keys':        all_keys,
        'key_types':       key_types,
        'feature_columns': feature_columns,
        'feature_scaler':  feature_scaler,
        'numeric_feature_cols': numeric_feature_cols,
        'val_r2':          all_results[best_name]['val_r2'],
        'all_results':     all_results,
    }

    return pipeline


def predict_raw_curve_dtw_phase(raw_values, activity, attributes, pipeline):
    """
    Predict for a single raw curve using the DTW-phase-aware pipeline.

    At inference the phase descriptors are derived from the DBA reference curve
    (the training prototype) evaluated at each of the fixed_length canonical
    positions — not from the test curve itself, which would be data leakage.
    The DTW warp path maps canonical predictions back to raw time steps exactly
    as in the baseline predict_raw_curve().
    """
    reference_curve      = pipeline['reference_curve']
    fixed_length         = len(reference_curve)
    model                = pipeline['model']
    feature_columns      = pipeline['feature_columns']
    feature_scaler       = pipeline.get('feature_scaler', None)
    numeric_feature_cols = pipeline.get('numeric_feature_cols', [])
    all_keys             = pipeline['all_keys']
    key_types            = pipeline['key_types']

    ref_feats = _phase_features_from_curve(reference_curve)

    _rel_denom = max(fixed_length - 1, 1)
    rows_ref = []
    for pos in range(fixed_length):
        row = {
            'position_idx':     pos,
            'relative_pos':     pos / _rel_denom,
            'curve_length':     len(raw_values),
            'activity':         activity,
            'phase_norm':       ref_feats['phase_norm'][pos],
            'local_slope':      ref_feats['local_slope'][pos],
            'local_curvature':  ref_feats['local_curvature'][pos],
            'ref_value':        ref_feats['ref_value'][pos],
            'ref_slope':        ref_feats['ref_slope'][pos],
        }
        for key in all_keys:
            value = attributes.get(key, None)
            if key_types[key] == 'numeric':
                try:
                    row[key] = float(value) if value is not None else np.nan
                except (ValueError, TypeError):
                    row[key] = np.nan
            else:
                row[key] = str(value) if value is not None else 'None'
        rows_ref.append(row)

    X_ref = pd.DataFrame(rows_ref)
    categorical_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_ref = pd.get_dummies(X_ref, columns=categorical_cols, drop_first=True)

    for col in feature_columns:
        if col not in X_ref.columns:
            X_ref[col] = 0
    X_ref = X_ref[feature_columns]

    if feature_scaler is not None and numeric_feature_cols:
        cols_to_scale = [c for c in numeric_feature_cols if c in X_ref.columns]
        if cols_to_scale:
            X_ref[cols_to_scale] = X_ref[cols_to_scale].astype('float64')
            X_ref.loc[:, cols_to_scale] = feature_scaler.transform(X_ref[cols_to_scale])

    y_ref_pred = model.predict(X_ref)

    # Map canonical predictions back to raw time steps via DTW warp path
    alignment = dtw(raw_values, reference_curve, keep_internals=True)
    buckets = [[] for _ in range(len(raw_values))]

    for qi, ri in zip(alignment.index1, alignment.index2):
        if 0 <= qi < len(raw_values) and 0 <= ri < fixed_length:
            buckets[qi].append(y_ref_pred[ri])

    y_raw_pred = np.empty(len(raw_values), dtype=float)
    path_pairs = list(zip(alignment.index1, alignment.index2))

    for qi in range(len(raw_values)):
        if buckets[qi]:
            y_raw_pred[qi] = float(np.mean(buckets[qi]))
            continue
        nearest_qi, nearest_ri = min(path_pairs, key=lambda p: abs(p[0] - qi))
        _ = nearest_qi
        y_raw_pred[qi] = float(y_ref_pred[min(max(nearest_ri, 0), fixed_length - 1)])

    return y_raw_pred


# =============================================================================
# APPROACH 2 — BASIS EXPANSION (B-SPLINE)
#
# Instead of predicting each time step independently, the model learns to
# predict the K coefficients of a B-spline basis expansion of the curve.
# The basis functions enforce smoothness by construction and the output
# generalises to any output length by evaluating the spline at new knots.
#
#   Train:
#     1. Resample every curve to fixed_length (no DTW needed).
#     2. Fit a B-spline basis B of shape (fixed_length × K).
#     3. Project each curve onto the basis: c = pinv(B) @ curve  →  shape (K,).
#     4. Build a multi-output regression dataset (activity, attrs) → c_vector.
#     5. Train a MultiOutputRegressor over the candidate models; pick best by
#        validation R² averaged across all K outputs.
#
#   Predict:
#     1. Predict c_vector for the given (activity, attrs).
#     2. Evaluate B_new @ c_vector at any desired output length.
#        No DTW needed at inference — the spline handles variable lengths.
#
# Why it is better than the baseline:
#   • The model outputs K~20 smooth coefficients, not 100 independent scalars.
#   • Temporal continuity is guaranteed by the basis (no point-level noise).
#   • Inference is O(K) matrix-vector multiply — faster than DTW at test time.
# =============================================================================

from scipy.interpolate import BSpline, make_interp_spline
from sklearn.multioutput import MultiOutputRegressor


def _make_bspline_basis(fixed_length, n_basis):
    """
    Build a B-spline design matrix of shape (fixed_length, n_basis).

    Uses uniform knots on [0, 1] with degree 3 (cubic splines).
    Returns (B, t_train) where t_train is the evaluation grid.
    """
    t = np.linspace(0, 1, fixed_length)
    degree = 3
    n_interior = n_basis - degree - 1
    n_interior = max(n_interior, 0)
    interior_knots = np.linspace(0, 1, n_interior + 2)[1:-1]
    knots = np.concatenate([
        np.zeros(degree + 1),
        interior_knots,
        np.ones(degree + 1),
    ])

    B = np.zeros((fixed_length, n_basis))
    for j in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[j] = 1.0
        try:
            spl = BSpline(knots, coeffs, degree)
            B[:, j] = spl(t)
        except Exception:
            pass

    return B, t, knots, degree


def build_and_train_pipeline_basis(
    train_curves,
    variable,
    fixed_length=100,
    n_basis=20,
    val_size=0.2,
    random_state=42,
    models=None,
    optimize_hyperparams=False,
    n_trials=50,
    verbose=1,
    n_jobs=-1,
):
    """
    Approach 2 — B-spline basis expansion pipeline.

    Returns the same pipeline dict shape as build_and_train_pipeline(), with
    extra keys ``'approach'``, ``'B'`` (basis matrix), ``'knots'``,
    ``'degree'``, ``'n_basis'``.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("APPROACH 2 — B-SPLINE BASIS EXPANSION  (train curves only)")
        print(f"  fixed_length={fixed_length}  n_basis={n_basis}")
        print("=" * 80)

    if models is None:
        models = {
            'Gradient Boosting': GradientBoostingRegressor,
            'Random Forest':     RandomForestRegressor,
        }

    # 1. Resample all train curves to fixed_length
    resampled = np.array([
        np.interp(
            np.linspace(0, 1, fixed_length),
            np.linspace(0, 1, len(c['original_values'])),
            c['original_values']
        )
        for c in train_curves
    ])  # shape (N, fixed_length)

    # 2. Build B-spline basis
    B, t_train, knots, degree = _make_bspline_basis(fixed_length, n_basis)
    B_pinv = np.linalg.pinv(B)  # shape (n_basis, fixed_length)

    if verbose:
        print(f"\n  B-spline basis: {n_basis} basis functions, degree {degree}")
        print(f"  Basis matrix B shape: {B.shape}")

    # 3. Project each curve → coefficients
    coeffs = resampled @ B_pinv.T   # shape (N, n_basis)

    # 4. Build feature matrix (one row per curve, not per time step)
    all_keys, key_types = _infer_key_types(train_curves)

    rows = []
    for i, curve in enumerate(train_curves):
        row = {
            'instance_id':  curve['instance_id'],
            'activity':     curve['activity'],
            'curve_length': curve['original_length'],
        }
        for key in all_keys:
            value = curve['attributes'].get(key, None)
            if key_types[key] == 'numeric':
                try:
                    row[key] = float(value) if value is not None else np.nan
                except (ValueError, TypeError):
                    row[key] = np.nan
            else:
                row[key] = str(value) if value is not None else 'None'
        rows.append(row)

    df_feat = pd.DataFrame(rows)
    categorical_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_all = df_feat.drop(columns=['instance_id'], errors='ignore').copy()
    X_all = pd.get_dummies(X_all, columns=categorical_cols, drop_first=True)
    feature_columns = X_all.columns.tolist()
    Y_all = coeffs   # shape (N, n_basis)

    # Val split by instance
    unique_instances     = df_feat['instance_id'].values
    train_inst, val_inst = train_test_split(
        np.arange(len(unique_instances)), test_size=val_size, random_state=random_state
    )

    X_train, X_val = X_all.iloc[train_inst].copy(), X_all.iloc[val_inst].copy()
    Y_train, Y_val = Y_all[train_inst],              Y_all[val_inst]

    numeric_feature_cols = ['curve_length'] + [k for k in all_keys if key_types[k] == 'numeric']
    numeric_feature_cols = [c for c in numeric_feature_cols if c in X_train.columns]

    feature_scaler = None
    if numeric_feature_cols:
        X_train[numeric_feature_cols] = X_train[numeric_feature_cols].astype('float64')
        X_val[numeric_feature_cols]   = X_val[numeric_feature_cols].astype('float64')
        feature_scaler = StandardScaler()
        X_train.loc[:, numeric_feature_cols] = feature_scaler.fit_transform(
            X_train[numeric_feature_cols]
        )
        X_val.loc[:, numeric_feature_cols] = feature_scaler.transform(
            X_val[numeric_feature_cols]
        )

    if verbose:
        print(f"\n  Train: {len(train_inst)} curves  |  Val: {len(val_inst)} curves")

    # 5. Train models (MultiOutputRegressor wraps single-output estimators)
    all_results = {}

    for name, model_class in models.items():
        if verbose:
            print(f"\n  Training — {name}...")

        if optimize_hyperparams:
            if not verbose:
                optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial, name=name, model_class=model_class):
                if name == 'Gradient Boosting':
                    params = {
                        'n_estimators':  trial.suggest_int('n_estimators', 50, 300),
                        'max_depth':     trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample':     trial.suggest_float('subsample', 0.5, 1.0),
                        'random_state':  random_state,
                    }
                elif name == 'Random Forest':
                    params = {
                        'n_estimators':      trial.suggest_int('n_estimators', 50, 300),
                        'max_depth':         trial.suggest_int('max_depth', 5, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'random_state':      random_state,
                        'n_jobs':            -1,
                    }
                else:
                    params = {}
                base = model_class(**params)
                m = MultiOutputRegressor(base, n_jobs=n_jobs)
                m.fit(X_train, Y_train)
                Y_pred = m.predict(X_val)
                return float(np.mean([r2_score(Y_val[:, k], Y_pred[:, k])
                                      for k in range(n_basis)]))

            sampler = optuna.samplers.TPESampler(seed=random_state)
            study = optuna.create_study(direction='maximize', sampler=sampler)
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            if name in ('Gradient Boosting', 'Random Forest'):
                best_params['random_state'] = random_state
            if name == 'Random Forest':
                best_params['n_jobs'] = n_jobs
            base_model = model_class(**best_params)
        else:
            if name == 'Gradient Boosting':
                base_model = model_class(
                    n_estimators=200, max_depth=7, learning_rate=0.1,
                    subsample=0.8, random_state=random_state
                )
            elif name == 'Random Forest':
                base_model = model_class(
                    n_estimators=200, max_depth=12, min_samples_split=5,
                    random_state=random_state, n_jobs=n_jobs
                )
            else:
                try:
                    base_model = model_class(random_state=random_state)
                except TypeError:
                    base_model = model_class()

        model = MultiOutputRegressor(base_model, n_jobs=n_jobs)
        model.fit(X_train, Y_train)

        Y_train_pred = model.predict(X_train)
        Y_val_pred   = model.predict(X_val)

        train_r2   = float(np.mean([r2_score(Y_train[:, k], Y_train_pred[:, k])
                                    for k in range(n_basis)]))
        val_r2     = float(np.mean([r2_score(Y_val[:, k], Y_val_pred[:, k])
                                    for k in range(n_basis)]))
        train_rmse = float(np.mean([np.sqrt(mean_squared_error(Y_train[:, k], Y_train_pred[:, k]))
                                    for k in range(n_basis)]))
        val_rmse   = float(np.mean([np.sqrt(mean_squared_error(Y_val[:, k], Y_val_pred[:, k]))
                                    for k in range(n_basis)]))

        if verbose:
            print(f"    Train  R2={train_r2:.4f}  RMSE={train_rmse:.4f}  (avg over {n_basis} basis)")
            print(f"    Val    R2={val_r2:.4f}  RMSE={val_rmse:.4f}")

        all_results[name] = {
            'model':      model,
            'train_r2':   train_r2,
            'val_r2':     val_r2,
            'train_rmse': train_rmse,
            'val_rmse':   val_rmse,
        }

    best_name  = max(all_results, key=lambda k: all_results[k]['val_r2'])
    best_model = all_results[best_name]['model']

    if verbose:
        print(f"\n  Best model: {best_name}  (val R2={all_results[best_name]['val_r2']:.4f})")

    pipeline = {
        'approach':        'basis_expansion',
        'model':           best_model,
        'model_name':      best_name,
        'reference_curve': np.mean(resampled, axis=0),  # mean curve as reference visual
        'fixed_length':    fixed_length,
        'n_basis':         n_basis,
        'B':               B,
        'B_pinv':          B_pinv,
        'knots':           knots,
        'degree':          degree,
        'all_keys':        all_keys,
        'key_types':       key_types,
        'feature_columns': feature_columns,
        'feature_scaler':  feature_scaler,
        'numeric_feature_cols': numeric_feature_cols,
        'val_r2':          all_results[best_name]['val_r2'],
        'all_results':     all_results,
    }

    return pipeline


def predict_raw_curve_basis(raw_values, activity, attributes, pipeline):
    """
    Predict for a single raw curve using the B-spline basis pipeline.

    Predicts coefficient vector c, then reconstructs the curve at the desired
    output length by evaluating the spline at len(raw_values) evenly-spaced
    points — no DTW needed at inference.
    """
    model                = pipeline['model']
    feature_columns      = pipeline['feature_columns']
    feature_scaler       = pipeline.get('feature_scaler', None)
    numeric_feature_cols = pipeline.get('numeric_feature_cols', [])
    all_keys             = pipeline['all_keys']
    key_types            = pipeline['key_types']
    n_basis              = pipeline['n_basis']
    knots                = pipeline['knots']
    degree               = pipeline['degree']
    fixed_length         = pipeline['fixed_length']

    row = {
        'curve_length': len(raw_values),
        'activity':     activity,
    }
    for key in all_keys:
        value = attributes.get(key, None)
        if key_types[key] == 'numeric':
            try:
                row[key] = float(value) if value is not None else np.nan
            except (ValueError, TypeError):
                row[key] = np.nan
        else:
            row[key] = str(value) if value is not None else 'None'

    X = pd.DataFrame([row])
    categorical_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]

    if feature_scaler is not None and numeric_feature_cols:
        cols_to_scale = [c for c in numeric_feature_cols if c in X.columns]
        if cols_to_scale:
            X[cols_to_scale] = X[cols_to_scale].astype('float64')
            X.loc[:, cols_to_scale] = feature_scaler.transform(X[cols_to_scale])

    c_pred = model.predict(X)[0]   # shape (n_basis,)

    # Reconstruct at output length by evaluating basis at len(raw_values) points
    t_out = np.linspace(0, 1, len(raw_values))
    B_out = np.zeros((len(raw_values), n_basis))
    for j in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[j] = 1.0
        try:
            spl = BSpline(knots, coeffs, degree)
            B_out[:, j] = spl(t_out)
        except Exception:
            pass

    return B_out @ c_pred


# =============================================================================
# EXTERNAL FACTORS (EXOG) — DTW PIPELINE WITH ef_* TIME-SERIES PREDICTORS
#
# Every existing approach predicts from (position, activity, scalar attributes).
# This variant additionally uses external-factor curves (columns whose name
# starts with 'ef_') as positional features: for each canonical position the
# model sees the resampled value of every ef_ signal at that moment in time.
#
# Training:  ef_ arrays stored in the curve dict (from split_curves with
#            exog_columns=...) are resampled to fixed_length and appended as
#            extra feature columns alongside position_idx.
# Inference: caller passes exog_values={col: np.ndarray} with the actual
#            external-factor readings for the target activity window; they are
#            resampled to fixed_length before building the feature matrix, then
#            the DTW decode maps predictions back to raw length as usual.
# =============================================================================


def _resample_signal(arr, target_length):
    """Resample 1-D array to target_length via linear interpolation."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) == target_length:
        return arr
    if len(arr) < 2:
        return np.full(target_length, arr[0] if len(arr) == 1 else np.nan)
    return np.interp(
        np.linspace(0, 1, target_length),
        np.linspace(0, 1, len(arr)),
        arr,
    )


def build_and_train_pipeline_exog(
    train_curves,
    variable,
    fixed_length=100,
    val_size=0.2,
    random_state=42,
    models=None,
    optimize_hyperparams=False,
    n_trials=50,
    verbose=1,
    n_jobs=-1,
):
    """
    DTW baseline enriched with ef_* external-factor time series as features.

    Requires that each curve dict in train_curves contains 'exog_values' (set
    by split_curves with exog_columns=...).  Curves that lack the key are still
    included but their exog features are filled with NaN.

    Returns a pipeline dict identical to build_and_train_pipeline but with the
    additional keys:
        'exog_cols'  — list[str]  ordered exog column names used at train time
        'approach'   — 'exog'
    """
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2 — BUILD + TRAIN PIPELINE (DTW + External Factors)")
        print("=" * 80)

    if models is None:
        models = {
            'Gradient Boosting': GradientBoostingRegressor,
            'Random Forest':     RandomForestRegressor,
        }

    # Collect all exog column names present across train curves
    exog_cols = sorted({
        col
        for c in train_curves
        for col in c.get('exog_values', {}).keys()
    })

    if verbose:
        print(f"  External-factor columns: {exog_cols}")

    # ── DBA barycenter ────────────────────────────────────────────────────────
    resampled_for_dba = np.array([
        np.interp(
            np.linspace(0, 1, fixed_length),
            np.linspace(0, 1, len(c['original_values'])),
            c['original_values'],
        )
        for c in train_curves
    ])[:, :, np.newaxis]

    dba_barycenter  = dtw_barycenter_averaging(resampled_for_dba, barycenter_size=fixed_length)
    reference_curve = dba_barycenter[:, 0]

    # ── DTW-align train curves ────────────────────────────────────────────────
    for curve in train_curves:
        curve['resampled_values'] = _align_curve_with_dtw(
            curve['original_values'], reference_curve
        )
        # resample exog signals to fixed_length
        curve['_exog_resampled'] = {
            col: _resample_signal(curve['exog_values'][col], fixed_length)
            for col in exog_cols
            if col in curve.get('exog_values', {})
        }

    # ── Attribute key types (train only) ─────────────────────────────────────
    all_keys, key_types = _infer_key_types(train_curves)

    # ── Feature matrix ────────────────────────────────────────────────────────
    _rel_denom = max(fixed_length - 1, 1)
    rows = []
    for curve in train_curves:
        exog_rs = curve.get('_exog_resampled', {})
        for position_idx in range(fixed_length):
            row = {
                'instance_id':  curve['instance_id'],
                'activity':     curve['activity'],
                'position_idx': position_idx,
                'relative_pos': position_idx / _rel_denom,
                'curve_length': curve['original_length'],
                'y':            curve['resampled_values'][position_idx],
            }
            for key in all_keys:
                value = curve['attributes'].get(key, None)
                if key_types[key] == 'numeric':
                    try:
                        row[key] = float(value) if value is not None else np.nan
                    except (ValueError, TypeError):
                        row[key] = np.nan
                else:
                    row[key] = str(value) if value is not None else 'None'
            for col in exog_cols:
                row[col] = float(exog_rs[col][position_idx]) if col in exog_rs else np.nan
            rows.append(row)

    df_reg = pd.DataFrame(rows)

    categorical_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_all = df_reg[['position_idx', 'relative_pos', 'curve_length']].copy()
    X_all = X_all.assign(activity=df_reg['activity'].values)
    for key in all_keys:
        X_all = X_all.assign(**{key: df_reg[key].values})
    for col in exog_cols:
        X_all[col] = df_reg[col].values
    X_all = pd.get_dummies(X_all, columns=categorical_cols, drop_first=True)
    y_all = df_reg['y'].copy()

    unique_instances     = df_reg['instance_id'].unique()
    train_inst, val_inst = train_test_split(
        unique_instances, test_size=val_size, random_state=random_state
    )
    train_mask = df_reg['instance_id'].isin(train_inst)
    val_mask   = df_reg['instance_id'].isin(val_inst)

    X_train, X_val = X_all[train_mask].copy(), X_all[val_mask].copy()
    y_train, y_val = y_all[train_mask].copy(), y_all[val_mask].copy()

    feature_columns = X_all.columns.tolist()

    numeric_feature_cols = ['position_idx', 'relative_pos', 'curve_length'] + \
        [k for k in all_keys if key_types[k] == 'numeric'] + exog_cols
    numeric_feature_cols = [c for c in numeric_feature_cols if c in X_train.columns]

    feature_scaler = None
    if numeric_feature_cols:
        X_train[numeric_feature_cols] = X_train[numeric_feature_cols].astype('float64')
        X_val[numeric_feature_cols]   = X_val[numeric_feature_cols].astype('float64')
        feature_scaler = StandardScaler()
        X_train.loc[:, numeric_feature_cols] = feature_scaler.fit_transform(
            X_train[numeric_feature_cols]
        )
        X_val.loc[:, numeric_feature_cols] = feature_scaler.transform(
            X_val[numeric_feature_cols]
        )

    # ── Train models ──────────────────────────────────────────────────────────
    all_results = {}
    best_model, best_name, best_val_r2 = None, None, -np.inf

    for name, model_class in models.items():
        if verbose:
            print(f"\n  Training {name}...")
        model = model_class()
        model.fit(X_train, y_train)

        train_r2 = float(r2_score(y_train, model.predict(X_train)))
        val_r2   = float(r2_score(y_val,   model.predict(X_val)))
        if verbose:
            print(f"    train R²={train_r2:.4f}  val R²={val_r2:.4f}")

        all_results[name] = {'train_r2': train_r2, 'val_r2': val_r2}
        if val_r2 > best_val_r2:
            best_val_r2  = val_r2
            best_model   = model
            best_name    = name

    if verbose:
        print(f"\n  Best model: {best_name}  val R²={best_val_r2:.4f}")

    return {
        'model':               best_model,
        'model_name':          best_name,
        'reference_curve':     reference_curve,
        'fixed_length':        fixed_length,
        'all_keys':            all_keys,
        'key_types':           key_types,
        'feature_columns':     feature_columns,
        'feature_scaler':      feature_scaler,
        'numeric_feature_cols': numeric_feature_cols,
        'val_r2':              best_val_r2,
        'all_results':         all_results,
        'exog_cols':           exog_cols,
        'approach':            'exog',
    }


def predict_raw_curve_exog(raw_values, activity, attributes, pipeline,
                            exog_values=None):
    """
    Predict for a single raw curve using the external-factors pipeline.

    Parameters
    ----------
    raw_values  : np.ndarray
    activity    : str
    attributes  : dict
    pipeline    : dict  — from build_and_train_pipeline_exog()
    exog_values : dict | None
        {col: np.ndarray} — raw ef_* time series for this activity window,
        same length as raw_values (will be resampled internally).
        Missing columns are filled with 0.
    """
    reference_curve      = pipeline['reference_curve']
    fixed_length         = len(reference_curve)
    model                = pipeline['model']
    feature_columns      = pipeline['feature_columns']
    feature_scaler       = pipeline.get('feature_scaler', None)
    numeric_feature_cols = pipeline.get('numeric_feature_cols', [])
    all_keys             = pipeline['all_keys']
    key_types            = pipeline['key_types']
    exog_cols            = pipeline.get('exog_cols', [])

    if exog_values is None:
        exog_values = {}

    # Resample each exog signal to fixed_length
    exog_rs = {
        col: _resample_signal(exog_values[col], fixed_length)
        if col in exog_values
        else np.zeros(fixed_length)
        for col in exog_cols
    }

    _rel_denom = max(fixed_length - 1, 1)
    rows_ref = []
    for ref_pos in range(fixed_length):
        row = {
            'position_idx': ref_pos,
            'relative_pos': ref_pos / _rel_denom,
            'curve_length': len(raw_values),
            'activity':     activity,
        }
        for key in all_keys:
            value = attributes.get(key, None)
            if key_types[key] == 'numeric':
                try:
                    row[key] = float(value) if value is not None else np.nan
                except (ValueError, TypeError):
                    row[key] = np.nan
            else:
                row[key] = str(value) if value is not None else 'None'
        for col in exog_cols:
            row[col] = float(exog_rs[col][ref_pos])
        rows_ref.append(row)

    X_ref = pd.DataFrame(rows_ref)
    categorical_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_ref = pd.get_dummies(X_ref, columns=categorical_cols, drop_first=True)

    for col in feature_columns:
        if col not in X_ref.columns:
            X_ref[col] = 0
    X_ref = X_ref[feature_columns]

    if feature_scaler is not None and numeric_feature_cols:
        cols_to_scale = [c for c in numeric_feature_cols if c in X_ref.columns]
        if cols_to_scale:
            X_ref[cols_to_scale] = X_ref[cols_to_scale].astype('float64')
            X_ref.loc[:, cols_to_scale] = feature_scaler.transform(X_ref[cols_to_scale])

    y_ref_pred = model.predict(X_ref)

    # DTW decode canonical → raw length
    alignment = dtw(raw_values, reference_curve, keep_internals=True)
    buckets   = [[] for _ in range(len(raw_values))]

    for qi, ri in zip(alignment.index1, alignment.index2):
        if 0 <= qi < len(raw_values) and 0 <= ri < fixed_length:
            buckets[qi].append(y_ref_pred[ri])

    y_raw_pred  = np.empty(len(raw_values), dtype=float)
    path_pairs  = list(zip(alignment.index1, alignment.index2))

    for qi in range(len(raw_values)):
        if buckets[qi]:
            y_raw_pred[qi] = float(np.mean(buckets[qi]))
            continue
        nearest_qi, nearest_ri = min(path_pairs, key=lambda p: abs(p[0] - qi))
        _ = nearest_qi
        y_raw_pred[qi] = float(y_ref_pred[min(max(nearest_ri, 0), fixed_length - 1)])

    return y_raw_pred


# =============================================================================
# AMPLITUDE + SHAPE PIPELINE
#
# Splits the prediction problem into two independent stages:
#
#   Stage A — Amplitude predictor:
#     Input : curve_length, activity, process attributes  (no curve values)
#     Output: predicted curve_mean  (scalar per instance)
#     Model : GradientBoostingRegressor / RandomForest
#
#   Stage B — Shape predictor:
#     Input : relative_pos, curve_length, activity, attributes
#     Output: z-score normalised curve value at each canonical position
#     Model : same family as stage A, trained on shape only
#
#   Inference:
#     1. Predict amplitude (curve_mean) from metadata alone.
#     2. Predict z-score shape from (relative_pos, metadata).
#     3. Multiply predicted shape by predicted amplitude → raw energy curve.
#     4. DTW-decode from canonical to raw length as usual.
#
# Why this helps: the existing approaches try to predict absolute energy values
# directly from positional features, but the energy level varies enormously
# across runs.  Separating amplitude from shape gives each sub-model a much
# simpler, more learnable target.
# =============================================================================


def build_and_train_pipeline_amplitude_shape(
    train_curves,
    variable,
    fixed_length=100,
    val_size=0.2,
    random_state=42,
    models=None,
    optimize_hyperparams=False,
    n_trials=50,
    verbose=1,
    n_jobs=-1,
):
    """
    Amplitude + Shape two-stage pipeline.

    Returns a pipeline dict drop-in compatible with predict_raw_curve_amplitude_shape().
    """
    from sklearn.linear_model import LinearRegression

    if verbose:
        print("\n" + "=" * 80)
        print("AMPLITUDE + SHAPE PIPELINE  (train curves only)")
        print("=" * 80)

    if models is None:
        models = {
            'Gradient Boosting': GradientBoostingRegressor,
            'Random Forest':     RandomForestRegressor,
        }

    # ── DBA barycenter + DTW alignment ───────────────────────────────────────
    resampled_for_dba = np.array([
        np.interp(
            np.linspace(0, 1, fixed_length),
            np.linspace(0, 1, len(c['original_values'])),
            c['original_values'],
        )
        for c in train_curves
    ])[:, :, np.newaxis]

    dba_barycenter  = dtw_barycenter_averaging(resampled_for_dba, barycenter_size=fixed_length)
    reference_curve = dba_barycenter[:, 0]

    for curve in train_curves:
        curve['resampled_values'] = _align_curve_with_dtw(
            curve['original_values'], reference_curve
        )

    all_keys, key_types = _infer_key_types(train_curves)

    # ── Val split by instance (shared across both stages) ────────────────────
    unique_instances     = [c['instance_id'] for c in train_curves]
    train_idx, val_idx   = train_test_split(
        range(len(train_curves)), test_size=val_size, random_state=random_state
    )
    tr_curves = [train_curves[i] for i in train_idx]
    vl_curves = [train_curves[i] for i in val_idx]

    cat_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']

    # ── Helper: build per-instance metadata rows ──────────────────────────────
    def _meta_rows(curves):
        rows = []
        for c in curves:
            row = {'curve_length': c['original_length'], 'activity': c['activity']}
            for key in all_keys:
                v = c['attributes'].get(key, None)
                if key_types[key] == 'numeric':
                    try:
                        row[key] = float(v) if v is not None else np.nan
                    except (ValueError, TypeError):
                        row[key] = np.nan
                else:
                    row[key] = str(v) if v is not None else 'None'
            rows.append(row)
        return pd.DataFrame(rows)

    # ── Stage A: amplitude (curve_mean) predictor ─────────────────────────────
    if verbose:
        print("\n[A] Training amplitude predictor  (target: curve_mean)...")

    amp_tr = np.array([float(np.mean(c['resampled_values'])) for c in tr_curves])
    amp_vl = np.array([float(np.mean(c['resampled_values'])) for c in vl_curves])

    X_amp_tr = _meta_rows(tr_curves)
    X_amp_vl = _meta_rows(vl_curves)
    X_amp_tr = pd.get_dummies(X_amp_tr, columns=cat_cols, drop_first=True)
    X_amp_vl = pd.get_dummies(X_amp_vl, columns=cat_cols, drop_first=True)
    amp_feat_cols = X_amp_tr.columns.tolist()
    for col in amp_feat_cols:
        if col not in X_amp_vl.columns:
            X_amp_vl[col] = 0
    X_amp_vl = X_amp_vl[amp_feat_cols]

    amp_num_cols = ['curve_length'] + [k for k in all_keys if key_types[k] == 'numeric']
    amp_num_cols = [c for c in amp_num_cols if c in X_amp_tr.columns]
    amp_scaler = None
    if amp_num_cols:
        X_amp_tr[amp_num_cols] = X_amp_tr[amp_num_cols].astype('float64')
        X_amp_vl[amp_num_cols] = X_amp_vl[amp_num_cols].astype('float64')
        amp_scaler = StandardScaler()
        X_amp_tr.loc[:, amp_num_cols] = amp_scaler.fit_transform(X_amp_tr[amp_num_cols])
        X_amp_vl.loc[:, amp_num_cols] = amp_scaler.transform(X_amp_vl[amp_num_cols])

    amp_results = {}
    for name, model_class in models.items():
        if name == 'Gradient Boosting':
            m = model_class(n_estimators=200, max_depth=5, learning_rate=0.1,
                            subsample=0.8, random_state=random_state)
        elif name == 'Random Forest':
            m = model_class(n_estimators=200, max_depth=10, random_state=random_state,
                            n_jobs=n_jobs)
        else:
            try:
                m = model_class(random_state=random_state)
            except TypeError:
                m = model_class()
        m.fit(X_amp_tr, amp_tr)
        val_r2 = r2_score(amp_vl, m.predict(X_amp_vl))
        amp_results[name] = {'model': m, 'val_r2': val_r2}
        if verbose:
            print(f"     {name}  val R²={val_r2:.4f}")

    best_amp_name  = max(amp_results, key=lambda k: amp_results[k]['val_r2'])
    best_amp_model = amp_results[best_amp_name]['model']
    if verbose:
        print(f"  Best amplitude model: {best_amp_name}  "
              f"(val R²={amp_results[best_amp_name]['val_r2']:.4f})")

    # ── Stage B: shape predictor on z-score normalised curves ─────────────────
    if verbose:
        print("\n[B] Training shape predictor  (target: z-score normalised curve)...")

    def _zscore(arr):
        mu  = float(np.mean(arr))
        sig = float(np.std(arr)) + 1e-8
        return (arr - mu) / sig, mu, sig

    _rel_denom = max(fixed_length - 1, 1)

    def _shape_rows(curves, include_target=True):
        rows = []
        for c in curves:
            aligned = np.asarray(c['resampled_values'], dtype=float)
            z_curve, _, _ = _zscore(aligned)
            for pos in range(fixed_length):
                row = {
                    'instance_id':  c['instance_id'],
                    'activity':     c['activity'],
                    'position_idx': pos,
                    'relative_pos': pos / _rel_denom,
                    'curve_length': c['original_length'],
                }
                for key in all_keys:
                    v = c['attributes'].get(key, None)
                    if key_types[key] == 'numeric':
                        try:
                            row[key] = float(v) if v is not None else np.nan
                        except (ValueError, TypeError):
                            row[key] = np.nan
                    else:
                        row[key] = str(v) if v is not None else 'None'
                if include_target:
                    row['y_shape'] = float(z_curve[pos])
                rows.append(row)
        return pd.DataFrame(rows)

    df_tr = _shape_rows(tr_curves)
    df_vl = _shape_rows(vl_curves)

    shape_cat_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_sh_tr = df_tr.drop(columns=['instance_id', 'y_shape'], errors='ignore').copy()
    X_sh_vl = df_vl.drop(columns=['instance_id', 'y_shape'], errors='ignore').copy()
    X_sh_tr = pd.get_dummies(X_sh_tr, columns=shape_cat_cols, drop_first=True)
    X_sh_vl = pd.get_dummies(X_sh_vl, columns=shape_cat_cols, drop_first=True)
    shape_feat_cols = X_sh_tr.columns.tolist()
    for col in shape_feat_cols:
        if col not in X_sh_vl.columns:
            X_sh_vl[col] = 0
    X_sh_vl = X_sh_vl[shape_feat_cols]
    y_sh_tr = df_tr['y_shape'].values
    y_sh_vl = df_vl['y_shape'].values

    shape_num_cols = ['position_idx', 'relative_pos', 'curve_length'] + [
        k for k in all_keys if key_types[k] == 'numeric'
    ]
    shape_num_cols = [c for c in shape_num_cols if c in X_sh_tr.columns]
    shape_scaler = None
    if shape_num_cols:
        X_sh_tr[shape_num_cols] = X_sh_tr[shape_num_cols].astype('float64')
        X_sh_vl[shape_num_cols] = X_sh_vl[shape_num_cols].astype('float64')
        shape_scaler = StandardScaler()
        X_sh_tr.loc[:, shape_num_cols] = shape_scaler.fit_transform(X_sh_tr[shape_num_cols])
        X_sh_vl.loc[:, shape_num_cols] = shape_scaler.transform(X_sh_vl[shape_num_cols])

    shape_results = {}
    for name, model_class in models.items():
        if name == 'Gradient Boosting':
            m = model_class(n_estimators=200, max_depth=7, learning_rate=0.1,
                            subsample=0.8, random_state=random_state)
        elif name == 'Random Forest':
            m = model_class(n_estimators=200, max_depth=12, min_samples_split=5,
                            random_state=random_state, n_jobs=n_jobs)
        else:
            try:
                m = model_class(random_state=random_state)
            except TypeError:
                m = model_class()
        m.fit(X_sh_tr, y_sh_tr)
        val_r2 = r2_score(y_sh_vl, m.predict(X_sh_vl))
        shape_results[name] = {'model': m, 'val_r2': val_r2}
        if verbose:
            print(f"     {name}  val R²={val_r2:.4f}")

    best_shape_name  = max(shape_results, key=lambda k: shape_results[k]['val_r2'])
    best_shape_model = shape_results[best_shape_name]['model']
    if verbose:
        print(f"  Best shape model: {best_shape_name}  "
              f"(val R²={shape_results[best_shape_name]['val_r2']:.4f})")

    return {
        'approach':           'amplitude_shape',
        'reference_curve':    reference_curve,
        'fixed_length':       fixed_length,
        'all_keys':           all_keys,
        'key_types':          key_types,
        'cat_cols':           cat_cols,
        # Stage A
        'amp_model':          best_amp_model,
        'amp_model_name':     best_amp_name,
        'amp_feat_cols':      amp_feat_cols,
        'amp_scaler':         amp_scaler,
        'amp_num_cols':       amp_num_cols,
        # Stage B
        'shape_model':        best_shape_model,
        'shape_model_name':   best_shape_name,
        'shape_feat_cols':    shape_feat_cols,
        'shape_scaler':       shape_scaler,
        'shape_num_cols':     shape_num_cols,
        'val_r2':             shape_results[best_shape_name]['val_r2'],
    }


def predict_raw_curve_amplitude_shape(raw_values, activity, attributes, pipeline):
    """
    Predict using the amplitude + shape pipeline.

    Stage A predicts the curve mean (amplitude) from metadata.
    Stage B predicts the z-score normalised shape from (relative_pos, metadata).
    Final prediction = shape * amplitude, DTW-decoded to raw length.
    """
    reference_curve  = pipeline['reference_curve']
    fixed_length     = len(reference_curve)
    all_keys         = pipeline['all_keys']
    key_types        = pipeline['key_types']
    cat_cols         = pipeline['cat_cols']

    amp_model        = pipeline['amp_model']
    amp_feat_cols    = pipeline['amp_feat_cols']
    amp_scaler       = pipeline.get('amp_scaler')
    amp_num_cols     = pipeline.get('amp_num_cols', [])

    shape_model      = pipeline['shape_model']
    shape_feat_cols  = pipeline['shape_feat_cols']
    shape_scaler     = pipeline.get('shape_scaler')
    shape_num_cols   = pipeline.get('shape_num_cols', [])

    def _attr_val(key):
        v = attributes.get(key, None)
        if key_types[key] == 'numeric':
            try:
                return float(v) if v is not None else np.nan
            except (ValueError, TypeError):
                return np.nan
        return str(v) if v is not None else 'None'

    # ── Stage A: predict amplitude ────────────────────────────────────────────
    meta_row = {'curve_length': len(raw_values), 'activity': activity}
    for key in all_keys:
        meta_row[key] = _attr_val(key)

    X_amp = pd.DataFrame([meta_row])
    X_amp = pd.get_dummies(X_amp, columns=cat_cols, drop_first=True)
    for col in amp_feat_cols:
        if col not in X_amp.columns:
            X_amp[col] = 0
    X_amp = X_amp[amp_feat_cols]
    if amp_scaler is not None and amp_num_cols:
        cols = [c for c in amp_num_cols if c in X_amp.columns]
        if cols:
            X_amp[cols] = X_amp[cols].astype('float64')
            X_amp.loc[:, cols] = amp_scaler.transform(X_amp[cols])

    predicted_amplitude = float(amp_model.predict(X_amp)[0])

    # ── Stage B: predict z-score shape ────────────────────────────────────────
    _rel_denom = max(fixed_length - 1, 1)
    rows = []
    for pos in range(fixed_length):
        row = {
            'position_idx': pos,
            'relative_pos': pos / _rel_denom,
            'curve_length': len(raw_values),
            'activity':     activity,
        }
        for key in all_keys:
            row[key] = _attr_val(key)
        rows.append(row)

    X_sh = pd.DataFrame(rows)
    X_sh = pd.get_dummies(X_sh, columns=cat_cols, drop_first=True)
    for col in shape_feat_cols:
        if col not in X_sh.columns:
            X_sh[col] = 0
    X_sh = X_sh[shape_feat_cols]
    if shape_scaler is not None and shape_num_cols:
        cols = [c for c in shape_num_cols if c in X_sh.columns]
        if cols:
            X_sh[cols] = X_sh[cols].astype('float64')
            X_sh.loc[:, cols] = shape_scaler.transform(X_sh[cols])

    y_shape_pred = shape_model.predict(X_sh)   # z-score normalised shape

    # ── Combine: rescale shape by predicted amplitude ─────────────────────────
    # The shape model predicts (value - mean) / std.  We only predicted mean,
    # not std, so we use the shape as-is and simply shift by predicted_amplitude.
    # This recovers: value ≈ z_shape * std_train + predicted_mean.
    # Since std is approximately constant within one combo it's absorbed into
    # the shape model weights; multiplying by predicted_amplitude centres the
    # curve at the right energy level.
    y_canonical = y_shape_pred + predicted_amplitude

    # ── DTW decode to raw length ──────────────────────────────────────────────
    alignment   = dtw(raw_values, reference_curve, keep_internals=True)
    buckets     = [[] for _ in range(len(raw_values))]
    path_pairs  = list(zip(alignment.index1, alignment.index2))

    for qi, ri in path_pairs:
        if 0 <= qi < len(raw_values) and 0 <= ri < fixed_length:
            buckets[qi].append(y_canonical[ri])

    y_raw_pred = np.empty(len(raw_values), dtype=float)
    for qi in range(len(raw_values)):
        if buckets[qi]:
            y_raw_pred[qi] = float(np.mean(buckets[qi]))
        else:
            nearest_qi, nearest_ri = min(path_pairs, key=lambda p: abs(p[0] - qi))
            _ = nearest_qi
            y_raw_pred[qi] = float(y_canonical[min(max(nearest_ri, 0), fixed_length - 1)])

    return y_raw_pred


# =============================================================================
# DTW + SEQ2SEQ  —  LSTM encoder-decoder over canonical DBA space
# =============================================================================
#
# Same DBA barycenter + DTW alignment as the baseline, but the model is an
# LSTM seq2seq that takes the full 100-step context vector as input and
# outputs the full 100-step energy curve in one shot — capturing temporal
# dependencies that a per-row regressor cannot.
#
# Training target:  DTW-aligned curve in canonical space (same as baseline).
# Input sequence :  [phase, curve_length, activity_ohe…, attrs…]  (100 × F)
# Output sequence:  aligned energy values                          (100,)
#
# Decode back to raw length: identical DTW path inversion as baseline.
# =============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _Seq2SeqLSTM(nn.Module):
    """
    Encoder-decoder LSTM.
    Encoder: reads the 100-step input feature sequence.
    Decoder: auto-regressively generates 100 output values conditioned on the
             encoder hidden state. At each step it receives the previous
             predicted value concatenated with the corresponding input features
             (scheduled teacher forcing during training, own predictions at
             inference).
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # decoder input = 1 (prev value) + input_size (context features at that step)
        self.decoder = nn.LSTM(1 + input_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, x, targets=None, teacher_forcing_ratio=0.5):
        """
        x       : (B, T, F)
        targets : (B, T)  — if None, pure auto-regressive
        returns : (B, T)
        """
        B, T, _ = x.shape
        _, (h, c) = self.encoder(x)

        prev = torch.zeros(B, 1, 1, device=x.device)   # (B, 1, 1)
        outputs = []
        for t in range(T):
            dec_in = torch.cat([prev, x[:, t:t+1, :]], dim=-1)  # (B,1,1+F)
            out, (h, c) = self.decoder(dec_in, (h, c))
            pred = self.out_proj(out)                             # (B,1,1)
            outputs.append(pred)
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev = targets[:, t:t+1].unsqueeze(-1)
            else:
                prev = pred.detach()

        return torch.cat(outputs, dim=1).squeeze(-1)             # (B, T)


def _build_seq2seq_input(curves, all_keys, key_types, fixed_length,
                          cat_columns, feature_columns, scaler,
                          numeric_feature_cols, value_key='resampled_values',
                          return_targets=True):
    """
    Build input tensor (N, fixed_length, F) and optional target (N, fixed_length)
    from a list of curve dicts.  Uses the same feature set as the baseline.
    """
    _rel_denom = max(fixed_length - 1, 1)
    X_list, y_list = [], []
    for curve in curves:
        rows = []
        for pos in range(fixed_length):
            row = {
                'position_idx': pos,
                'relative_pos': pos / _rel_denom,
                'curve_length': curve['original_length'],
                'activity':     curve['activity'],
            }
            for key in all_keys:
                v = curve['attributes'].get(key, None)
                if key_types[key] == 'numeric':
                    try:
                        row[key] = float(v) if v is not None else np.nan
                    except (ValueError, TypeError):
                        row[key] = np.nan
                else:
                    row[key] = str(v) if v is not None else 'None'
            rows.append(row)

        df_c = pd.DataFrame(rows)
        df_c = pd.get_dummies(df_c, columns=cat_columns, drop_first=True)
        for col in feature_columns:
            if col not in df_c.columns:
                df_c[col] = 0
        df_c = df_c[feature_columns]

        if scaler is not None and numeric_feature_cols:
            cols_to_scale = [c for c in numeric_feature_cols if c in df_c.columns]
            if cols_to_scale:
                df_c[cols_to_scale] = df_c[cols_to_scale].astype('float64')
                df_c[cols_to_scale] = scaler.transform(df_c[cols_to_scale])

        X_list.append(df_c.values.astype(np.float32))
        if return_targets:
            y_list.append(curve[value_key].astype(np.float32))

    X = torch.tensor(np.array(X_list), dtype=torch.float32)   # (N, T, F)
    if return_targets:
        y = torch.tensor(np.array(y_list), dtype=torch.float32)  # (N, T)
        return X, y
    return X


def build_and_train_pipeline_seq2seq(
    train_curves,
    variable,
    fixed_length=100,
    val_size=0.2,
    random_state=42,
    hidden_size=128,
    num_layers=2,
    dropout=0.1,
    epochs=80,
    batch_size=32,
    lr=1e-3,
    teacher_forcing_ratio=0.5,
    patience=10,
    verbose=1,
):
    """
    DTW + Seq2Seq pipeline.

    Stages
    ------
    2a. DBA barycenter — same as baseline
    2b. DTW alignment  — same as baseline
    2c. Feature matrix — same feature set as baseline (position, curve_length,
                         activity, attrs), shaped as sequences
    2d. LSTM encoder-decoder trained with teacher forcing
    2e. Best epoch selected by val loss (MSE in canonical space)

    Returns
    -------
    pipeline : dict  (drop-in compatible with predict_raw_curve_seq2seq)
    """
    from sklearn.model_selection import train_test_split as _tts

    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2 — BUILD + TRAIN SEQ2SEQ PIPELINE  (train curves only)")
        print("=" * 80)

    # ------------------------------------------------------------------
    # 2a. DBA barycenter
    # ------------------------------------------------------------------
    resampled_for_dba = np.array([
        np.interp(
            np.linspace(0, 1, fixed_length),
            np.linspace(0, 1, len(c['original_values'])),
            c['original_values']
        )
        for c in train_curves
    ])[:, :, np.newaxis]

    dba_barycenter  = dtw_barycenter_averaging(resampled_for_dba, barycenter_size=fixed_length)
    reference_curve = dba_barycenter[:, 0]

    if verbose:
        print(f"    Barycenter length: {len(reference_curve)}")

    # ------------------------------------------------------------------
    # 2b. DTW-align train curves
    # ------------------------------------------------------------------
    for i, curve in enumerate(train_curves):
        if verbose and i % max(1, len(train_curves) // 10) == 0:
            print(f"    Aligning {i+1}/{len(train_curves)}...")
        curve['resampled_values'] = _align_curve_with_dtw(
            curve['original_values'], reference_curve
        )

    # ------------------------------------------------------------------
    # 2c. Feature setup (same as baseline — fit on train only)
    # ------------------------------------------------------------------
    all_keys, key_types = _infer_key_types(train_curves)
    cat_columns = ['activity'] + [k for k in all_keys if key_types[k] == 'category']

    # Build a dummy flat df just to determine feature columns + fit scaler
    _df_ref = _build_feature_matrix(
        train_curves, all_keys, key_types, fixed_length,
        value_key='resampled_values', include_target=False
    )
    _df_ref = _df_ref.drop(columns=['instance_id'], errors='ignore')
    _df_ohe = pd.get_dummies(_df_ref, columns=cat_columns, drop_first=True)
    feature_columns = _df_ohe.columns.tolist()

    numeric_feature_cols = ['position_idx', 'relative_pos', 'curve_length'] + [
        k for k in all_keys if key_types[k] == 'numeric'
    ]
    numeric_feature_cols = [c for c in numeric_feature_cols if c in _df_ohe.columns]

    scaler = StandardScaler()
    _df_ohe[numeric_feature_cols] = scaler.fit_transform(_df_ohe[numeric_feature_cols])

    # ------------------------------------------------------------------
    # 2d. Build tensors + val split by instance
    # ------------------------------------------------------------------
    unique_instances = list({c['instance_id'] for c in train_curves})
    train_inst, val_inst = _tts(unique_instances, test_size=val_size,
                                random_state=random_state)
    tr_curves = [c for c in train_curves if c['instance_id'] in set(train_inst)]
    vl_curves = [c for c in train_curves if c['instance_id'] in set(val_inst)]

    X_tr, y_tr = _build_seq2seq_input(tr_curves, all_keys, key_types, fixed_length,
                                       cat_columns, feature_columns, scaler,
                                       numeric_feature_cols)
    X_vl, y_vl = _build_seq2seq_input(vl_curves, all_keys, key_types, fixed_length,
                                       cat_columns, feature_columns, scaler,
                                       numeric_feature_cols)

    if verbose:
        print(f"    Train: {len(tr_curves)} curves | Val: {len(vl_curves)} curves")

    # Normalise targets with train statistics (avoids magnitude dominance)
    y_mean = float(y_tr.mean())
    y_std  = float(y_tr.std()) + 1e-8
    y_tr_n = (y_tr - y_mean) / y_std
    y_vl_n = (y_vl - y_mean) / y_std

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X_tr.shape[-1]

    model = _Seq2SeqLSTM(input_size, hidden_size=hidden_size,
                         num_layers=num_layers, dropout=dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(X_tr, y_tr_n), batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    best_state    = None
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model(xb, targets=yb, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_vl.to(device), targets=None, teacher_forcing_ratio=0.0)
            val_loss = criterion(val_pred, y_vl_n.to(device)).item()

        if verbose and epoch % 10 == 0:
            print(f"    Epoch {epoch:4d}/{epochs}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    if verbose:
        print(f"    Best val_loss={best_val_loss:.5f}")

    return {
        'approach':            'seq2seq',
        'model':               model,
        'reference_curve':     reference_curve,
        'fixed_length':        fixed_length,
        'all_keys':            all_keys,
        'key_types':           key_types,
        'cat_columns':         cat_columns,
        'feature_columns':     feature_columns,
        'scaler':              scaler,
        'numeric_feature_cols': numeric_feature_cols,
        'y_mean':              y_mean,
        'y_std':               y_std,
        'device':              device,
        'val_loss':            best_val_loss,
    }


def predict_raw_curve_seq2seq(raw_values, activity, attributes, pipeline):
    """
    Predict a single raw curve using the trained Seq2Seq pipeline.

    1. Build 100-step feature sequence (same as baseline).
    2. Run the LSTM decoder (no teacher forcing) → 100 canonical predictions.
    3. DTW-decode back to raw length using the same path inversion as baseline.
    """
    reference_curve      = pipeline['reference_curve']
    fixed_length         = len(reference_curve)
    model                = pipeline['model']
    all_keys             = pipeline['all_keys']
    key_types            = pipeline['key_types']
    cat_columns          = pipeline['cat_columns']
    feature_columns      = pipeline['feature_columns']
    scaler               = pipeline['scaler']
    numeric_feature_cols = pipeline['numeric_feature_cols']
    y_mean               = pipeline['y_mean']
    y_std                = pipeline['y_std']
    device               = pipeline['device']

    # Build feature sequence
    _rel_denom = max(fixed_length - 1, 1)
    rows = []
    for pos in range(fixed_length):
        row = {
            'position_idx': pos,
            'relative_pos': pos / _rel_denom,
            'curve_length': len(raw_values),
            'activity':     activity,
        }
        for key in all_keys:
            v = attributes.get(key, None)
            if key_types[key] == 'numeric':
                try:
                    row[key] = float(v) if v is not None else np.nan
                except (ValueError, TypeError):
                    row[key] = np.nan
            else:
                row[key] = str(v) if v is not None else 'None'
        rows.append(row)

    df_seq = pd.DataFrame(rows)
    df_seq = pd.get_dummies(df_seq, columns=cat_columns, drop_first=True)
    for col in feature_columns:
        if col not in df_seq.columns:
            df_seq[col] = 0
    df_seq = df_seq[feature_columns]

    if scaler is not None and numeric_feature_cols:
        cols_to_scale = [c for c in numeric_feature_cols if c in df_seq.columns]
        if cols_to_scale:
            df_seq[cols_to_scale] = df_seq[cols_to_scale].astype('float64')
            df_seq[cols_to_scale] = scaler.transform(df_seq[cols_to_scale])

    X = torch.tensor(df_seq.values.astype(np.float32)).unsqueeze(0).to(device)  # (1,T,F)

    model.eval()
    with torch.no_grad():
        y_norm = model(X, targets=None, teacher_forcing_ratio=0.0)  # (1, T)

    y_ref_pred = (y_norm.squeeze(0).cpu().numpy() * y_std + y_mean)  # (T,)

    # DTW path decode — identical to baseline predict_raw_curve
    alignment  = dtw(raw_values, reference_curve, keep_internals=True)
    buckets    = [[] for _ in range(len(raw_values))]
    path_pairs = list(zip(alignment.index1, alignment.index2))

    for qi, ri in path_pairs:
        if 0 <= qi < len(raw_values) and 0 <= ri < fixed_length:
            buckets[qi].append(y_ref_pred[ri])

    y_raw_pred = np.empty(len(raw_values), dtype=float)
    for qi in range(len(raw_values)):
        if buckets[qi]:
            y_raw_pred[qi] = float(np.mean(buckets[qi]))
        else:
            nearest_qi, nearest_ri = min(path_pairs, key=lambda p: abs(p[0] - qi))
            _ = nearest_qi
            y_raw_pred[qi] = float(y_ref_pred[min(max(nearest_ri, 0), fixed_length - 1)])

    return y_raw_pred


# =============================================================================
# SEQ2SEQ ONLY  —  no DBA, no DTW alignment, no DTW decode
# =============================================================================
# Target: raw curve linearly resampled to fixed_length.
# Input : same positional feature sequence (phase, curve_length, activity, attrs).
# Decode: linear resample of the 100 predictions back to raw length.
# =============================================================================

def build_and_train_pipeline_seq2seq_only(
    train_curves,
    variable,
    fixed_length=100,
    val_size=0.2,
    random_state=42,
    hidden_size=128,
    num_layers=2,
    dropout=0.1,
    epochs=80,
    batch_size=32,
    lr=1e-3,
    teacher_forcing_ratio=0.5,
    patience=10,
    verbose=1,
):
    """
    Pure Seq2Seq pipeline — no DTW anywhere.

    Target  : raw curve linearly resampled to fixed_length (no DBA, no alignment).
    Input   : [phase, curve_length, activity_ohe, attrs] sequence (fixed_length × F).
    Decode  : linear resample of fixed_length predictions back to raw length.
    """
    from sklearn.model_selection import train_test_split as _tts

    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2 — BUILD + TRAIN SEQ2SEQ-ONLY PIPELINE  (no DTW)")
        print("=" * 80)

    # Resample each curve to fixed_length as training target (no DTW)
    for curve in train_curves:
        curve['resampled_values'] = np.interp(
            np.linspace(0, 1, fixed_length),
            np.linspace(0, 1, len(curve['original_values'])),
            curve['original_values'],
        ).astype(np.float32)

    # Feature setup — same as baseline, fit on train only
    all_keys, key_types = _infer_key_types(train_curves)
    cat_columns = ['activity'] + [k for k in all_keys if key_types[k] == 'category']

    _df_ref = _build_feature_matrix(
        train_curves, all_keys, key_types, fixed_length,
        value_key='resampled_values', include_target=False,
    )
    _df_ref = _df_ref.drop(columns=['instance_id'], errors='ignore')
    _df_ohe = pd.get_dummies(_df_ref, columns=cat_columns, drop_first=True)
    feature_columns = _df_ohe.columns.tolist()

    numeric_feature_cols = ['position_idx', 'relative_pos', 'curve_length'] + [
        k for k in all_keys if key_types[k] == 'numeric'
    ]
    numeric_feature_cols = [c for c in numeric_feature_cols if c in _df_ohe.columns]

    scaler = StandardScaler()
    _df_ohe[numeric_feature_cols] = scaler.fit_transform(_df_ohe[numeric_feature_cols])

    # Val split by instance
    unique_instances = list({c['instance_id'] for c in train_curves})
    train_inst, val_inst = _tts(unique_instances, test_size=val_size, random_state=random_state)
    tr_curves = [c for c in train_curves if c['instance_id'] in set(train_inst)]
    vl_curves = [c for c in train_curves if c['instance_id'] in set(val_inst)]

    X_tr, y_tr = _build_seq2seq_input(tr_curves, all_keys, key_types, fixed_length,
                                       cat_columns, feature_columns, scaler,
                                       numeric_feature_cols)
    X_vl, y_vl = _build_seq2seq_input(vl_curves, all_keys, key_types, fixed_length,
                                       cat_columns, feature_columns, scaler,
                                       numeric_feature_cols)

    if verbose:
        print(f"    Train: {len(tr_curves)} curves | Val: {len(vl_curves)} curves")

    y_mean = float(y_tr.mean())
    y_std  = float(y_tr.std()) + 1e-8
    y_tr_n = (y_tr - y_mean) / y_std
    y_vl_n = (y_vl - y_mean) / y_std

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X_tr.shape[-1]
    model      = _Seq2SeqLSTM(input_size, hidden_size=hidden_size,
                               num_layers=num_layers, dropout=dropout).to(device)
    optimiser  = torch.optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.MSELoss()
    loader     = DataLoader(TensorDataset(X_tr, y_tr_n), batch_size=batch_size, shuffle=True)

    best_val_loss, best_state, no_improve = float('inf'), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model(xb, targets=yb, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_vl.to(device), targets=None, teacher_forcing_ratio=0.0)
            val_loss = criterion(val_pred, y_vl_n.to(device)).item()

        if verbose and epoch % 10 == 0:
            print(f"    Epoch {epoch:4d}/{epochs}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    if verbose:
        print(f"    Best val_loss={best_val_loss:.5f}")

    return {
        'approach':             'seq2seq_only',
        'model':                model,
        'fixed_length':         fixed_length,
        'all_keys':             all_keys,
        'key_types':            key_types,
        'cat_columns':          cat_columns,
        'feature_columns':      feature_columns,
        'scaler':               scaler,
        'numeric_feature_cols': numeric_feature_cols,
        'y_mean':               y_mean,
        'y_std':                y_std,
        'device':               device,
        'val_loss':             best_val_loss,
    }


def predict_raw_curve_seq2seq_only(raw_values, activity, attributes, pipeline):
    """
    Predict using the pure Seq2Seq pipeline (no DTW).
    Builds feature sequence → LSTM decoder → linear resample to raw length.
    """
    fixed_length         = pipeline['fixed_length']
    model                = pipeline['model']
    all_keys             = pipeline['all_keys']
    key_types            = pipeline['key_types']
    cat_columns          = pipeline['cat_columns']
    feature_columns      = pipeline['feature_columns']
    scaler               = pipeline['scaler']
    numeric_feature_cols = pipeline['numeric_feature_cols']
    y_mean               = pipeline['y_mean']
    y_std                = pipeline['y_std']
    device               = pipeline['device']

    _rel_denom = max(fixed_length - 1, 1)
    rows = []
    for pos in range(fixed_length):
        row = {'position_idx': pos, 'relative_pos': pos / _rel_denom,
               'curve_length': len(raw_values), 'activity': activity}
        for key in all_keys:
            v = attributes.get(key, None)
            if key_types[key] == 'numeric':
                try:
                    row[key] = float(v) if v is not None else np.nan
                except (ValueError, TypeError):
                    row[key] = np.nan
            else:
                row[key] = str(v) if v is not None else 'None'
        rows.append(row)

    df_seq = pd.DataFrame(rows)
    df_seq = pd.get_dummies(df_seq, columns=cat_columns, drop_first=True)
    for col in feature_columns:
        if col not in df_seq.columns:
            df_seq[col] = 0
    df_seq = df_seq[feature_columns]

    if scaler is not None and numeric_feature_cols:
        cols_to_scale = [c for c in numeric_feature_cols if c in df_seq.columns]
        if cols_to_scale:
            df_seq[cols_to_scale] = df_seq[cols_to_scale].astype('float64')
            df_seq[cols_to_scale] = scaler.transform(df_seq[cols_to_scale])

    X = torch.tensor(df_seq.values.astype(np.float32)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        y_norm = model(X, targets=None, teacher_forcing_ratio=0.0)

    y_canonical = y_norm.squeeze(0).cpu().numpy() * y_std + y_mean  # (fixed_length,)

    # Linear resample back to raw length — no DTW
    y_raw_pred = np.interp(
        np.linspace(0, 1, len(raw_values)),
        np.linspace(0, 1, fixed_length),
        y_canonical,
    )
    return y_raw_pred


# =============================================================================
# DTW + SEQ2SEQ + EXT. FACTORS  —  same as DTW+Seq2Seq but ef_* appended to input
# =============================================================================
# Each position's feature vector gains one value per ef_ column (resampled to
# fixed_length).  Everything else — DBA, DTW alignment, DTW decode — is identical
# to the standard DTW+Seq2Seq pipeline.
# =============================================================================

def _build_seq2seq_exog_input(curves, all_keys, key_types, fixed_length,
                               cat_columns, feature_columns, scaler,
                               numeric_feature_cols, exog_cols,
                               value_key='resampled_values', return_targets=True):
    """
    Like _build_seq2seq_input but appends resampled ef_* values to each position.
    exog_cols : list of ef_ column names in order.
    """
    X_list, y_list = [], []
    for curve in curves:
        # Resample each exog signal to fixed_length
        exog_rs = {}
        for col in exog_cols:
            raw_sig = np.asarray(curve.get('exog_values', {}).get(col, []), dtype=float)
            if len(raw_sig) < 2:
                exog_rs[col] = np.zeros(fixed_length, dtype=np.float32)
            else:
                exog_rs[col] = np.interp(
                    np.linspace(0, 1, fixed_length),
                    np.linspace(0, 1, len(raw_sig)),
                    raw_sig,
                ).astype(np.float32)

        rows = []
        for pos in range(fixed_length):
            row = {
                'position_idx': pos,
                'curve_length': curve['original_length'],
                'activity':     curve['activity'],
            }
            for key in all_keys:
                v = curve['attributes'].get(key, None)
                if key_types[key] == 'numeric':
                    try:
                        row[key] = float(v) if v is not None else np.nan
                    except (ValueError, TypeError):
                        row[key] = np.nan
                else:
                    row[key] = str(v) if v is not None else 'None'
            for col in exog_cols:
                row[col] = float(exog_rs[col][pos])
            rows.append(row)

        df_c = pd.DataFrame(rows)
        df_c = pd.get_dummies(df_c, columns=cat_columns, drop_first=True)
        for col in feature_columns:
            if col not in df_c.columns:
                df_c[col] = 0
        df_c = df_c[feature_columns]

        if scaler is not None and numeric_feature_cols:
            cols_to_scale = [c for c in numeric_feature_cols if c in df_c.columns]
            if cols_to_scale:
                df_c[cols_to_scale] = df_c[cols_to_scale].astype('float64')
                df_c[cols_to_scale] = scaler.transform(df_c[cols_to_scale])

        X_list.append(df_c.values.astype(np.float32))
        if return_targets:
            y_list.append(curve[value_key].astype(np.float32))

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    if return_targets:
        y = torch.tensor(np.array(y_list), dtype=torch.float32)
        return X, y
    return X


def build_and_train_pipeline_seq2seq_exog(
    train_curves,
    variable,
    fixed_length=100,
    val_size=0.2,
    random_state=42,
    hidden_size=128,
    num_layers=2,
    dropout=0.1,
    epochs=80,
    batch_size=32,
    lr=1e-3,
    teacher_forcing_ratio=0.5,
    patience=10,
    verbose=1,
    n_jobs=-1,
):
    """
    DTW + Seq2Seq + External Factors pipeline.

    Identical to build_and_train_pipeline_seq2seq except each position's input
    vector is extended with resampled ef_* values read from curve['exog_values'].
    DBA barycenter, DTW alignment and DTW decode are all unchanged.
    """
    from sklearn.model_selection import train_test_split as _tts

    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2 — BUILD + TRAIN DTW + SEQ2SEQ + EXT. FACTORS PIPELINE")
        print("=" * 80)

    # Detect exog cols from the first curve that has them
    exog_cols = []
    for c in train_curves:
        ev = c.get('exog_values', {})
        if ev:
            exog_cols = sorted(ev.keys())
            break

    if verbose:
        print(f"    External factor columns: {exog_cols}")

    # DBA barycenter
    resampled_for_dba = np.array([
        np.interp(
            np.linspace(0, 1, fixed_length),
            np.linspace(0, 1, len(c['original_values'])),
            c['original_values'],
        )
        for c in train_curves
    ])[:, :, np.newaxis]

    dba_barycenter  = dtw_barycenter_averaging(resampled_for_dba, barycenter_size=fixed_length)
    reference_curve = dba_barycenter[:, 0]

    if verbose:
        print(f"    Barycenter length: {len(reference_curve)}")

    # DTW-align train curves
    for i, curve in enumerate(train_curves):
        if verbose and i % max(1, len(train_curves) // 10) == 0:
            print(f"    Aligning {i+1}/{len(train_curves)}...")
        curve['resampled_values'] = _align_curve_with_dtw(
            curve['original_values'], reference_curve
        )

    # Feature setup — base features + exog cols, fit scaler on train only
    all_keys, key_types = _infer_key_types(train_curves)
    cat_columns = ['activity'] + [k for k in all_keys if key_types[k] == 'category']

    _df_ref = _build_feature_matrix(
        train_curves, all_keys, key_types, fixed_length,
        value_key='resampled_values', include_target=False,
    )
    _df_ref = _df_ref.drop(columns=['instance_id'], errors='ignore')
    _df_ohe = pd.get_dummies(_df_ref, columns=cat_columns, drop_first=True)
    # Add exog columns (filled with zeros for scaler fitting — values added per-curve at train time)
    for col in exog_cols:
        _df_ohe[col] = 0.0
    feature_columns = _df_ohe.columns.tolist()

    numeric_feature_cols = ['position_idx', 'relative_pos', 'curve_length'] + [
        k for k in all_keys if key_types[k] == 'numeric'
    ] + exog_cols
    numeric_feature_cols = [c for c in numeric_feature_cols if c in _df_ohe.columns]

    scaler = StandardScaler()
    _df_ohe[numeric_feature_cols] = scaler.fit_transform(_df_ohe[numeric_feature_cols])

    # Val split by instance
    unique_instances = list({c['instance_id'] for c in train_curves})
    train_inst, val_inst = _tts(unique_instances, test_size=val_size, random_state=random_state)
    tr_curves = [c for c in train_curves if c['instance_id'] in set(train_inst)]
    vl_curves = [c for c in train_curves if c['instance_id'] in set(val_inst)]

    X_tr, y_tr = _build_seq2seq_exog_input(tr_curves, all_keys, key_types, fixed_length,
                                            cat_columns, feature_columns, scaler,
                                            numeric_feature_cols, exog_cols)
    X_vl, y_vl = _build_seq2seq_exog_input(vl_curves, all_keys, key_types, fixed_length,
                                            cat_columns, feature_columns, scaler,
                                            numeric_feature_cols, exog_cols)

    if verbose:
        print(f"    Train: {len(tr_curves)} curves | Val: {len(vl_curves)} curves | Input size: {X_tr.shape[-1]}")

    y_mean = float(y_tr.mean())
    y_std  = float(y_tr.std()) + 1e-8
    y_tr_n = (y_tr - y_mean) / y_std
    y_vl_n = (y_vl - y_mean) / y_std

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X_tr.shape[-1]
    model      = _Seq2SeqLSTM(input_size, hidden_size=hidden_size,
                               num_layers=num_layers, dropout=dropout).to(device)
    optimiser  = torch.optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.MSELoss()
    loader     = DataLoader(TensorDataset(X_tr, y_tr_n), batch_size=batch_size, shuffle=True)

    best_val_loss, best_state, no_improve = float('inf'), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model(xb, targets=yb, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_vl.to(device), targets=None, teacher_forcing_ratio=0.0)
            val_loss = criterion(val_pred, y_vl_n.to(device)).item()

        if verbose and epoch % 10 == 0:
            print(f"    Epoch {epoch:4d}/{epochs}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    if verbose:
        print(f"    Best val_loss={best_val_loss:.5f}")

    return {
        'approach':             'seq2seq_exog',
        'model':                model,
        'reference_curve':      reference_curve,
        'fixed_length':         fixed_length,
        'all_keys':             all_keys,
        'key_types':            key_types,
        'cat_columns':          cat_columns,
        'feature_columns':      feature_columns,
        'scaler':               scaler,
        'numeric_feature_cols': numeric_feature_cols,
        'exog_cols':            exog_cols,
        'y_mean':               y_mean,
        'y_std':                y_std,
        'device':               device,
        'val_loss':             best_val_loss,
    }


def predict_raw_curve_seq2seq_exog(raw_values, activity, attributes, pipeline,
                                    exog_values=None):
    """
    Predict using DTW + Seq2Seq + External Factors.
    exog_values : dict {col: np.ndarray} of raw-length ef_ signals (or empty).
    """
    reference_curve      = pipeline['reference_curve']
    fixed_length         = len(reference_curve)
    model                = pipeline['model']
    all_keys             = pipeline['all_keys']
    key_types            = pipeline['key_types']
    cat_columns          = pipeline['cat_columns']
    feature_columns      = pipeline['feature_columns']
    scaler               = pipeline['scaler']
    numeric_feature_cols = pipeline['numeric_feature_cols']
    exog_cols            = pipeline.get('exog_cols', [])
    y_mean               = pipeline['y_mean']
    y_std                = pipeline['y_std']
    device               = pipeline['device']

    if exog_values is None:
        exog_values = {}

    # Resample ef_ signals to fixed_length
    exog_rs = {}
    for col in exog_cols:
        raw_sig = np.asarray(exog_values.get(col, []), dtype=float)
        if len(raw_sig) < 2:
            exog_rs[col] = np.zeros(fixed_length, dtype=np.float32)
        else:
            exog_rs[col] = np.interp(
                np.linspace(0, 1, fixed_length),
                np.linspace(0, 1, len(raw_sig)),
                raw_sig,
            ).astype(np.float32)

    _rel_denom = max(fixed_length - 1, 1)
    rows = []
    for pos in range(fixed_length):
        row = {'position_idx': pos, 'relative_pos': pos / _rel_denom,
               'curve_length': len(raw_values), 'activity': activity}
        for key in all_keys:
            v = attributes.get(key, None)
            if key_types[key] == 'numeric':
                try:
                    row[key] = float(v) if v is not None else np.nan
                except (ValueError, TypeError):
                    row[key] = np.nan
            else:
                row[key] = str(v) if v is not None else 'None'
        for col in exog_cols:
            row[col] = float(exog_rs[col][pos])
        rows.append(row)

    df_seq = pd.DataFrame(rows)
    df_seq = pd.get_dummies(df_seq, columns=cat_columns, drop_first=True)
    for col in feature_columns:
        if col not in df_seq.columns:
            df_seq[col] = 0
    df_seq = df_seq[feature_columns]

    if scaler is not None and numeric_feature_cols:
        cols_to_scale = [c for c in numeric_feature_cols if c in df_seq.columns]
        if cols_to_scale:
            df_seq[cols_to_scale] = df_seq[cols_to_scale].astype('float64')
            df_seq[cols_to_scale] = scaler.transform(df_seq[cols_to_scale])

    X = torch.tensor(df_seq.values.astype(np.float32)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        y_norm = model(X, targets=None, teacher_forcing_ratio=0.0)

    y_ref_pred = y_norm.squeeze(0).cpu().numpy() * y_std + y_mean

    # DTW decode — same as baseline
    alignment  = dtw(raw_values, reference_curve, keep_internals=True)
    buckets    = [[] for _ in range(len(raw_values))]
    path_pairs = list(zip(alignment.index1, alignment.index2))

    for qi, ri in path_pairs:
        if 0 <= qi < len(raw_values) and 0 <= ri < fixed_length:
            buckets[qi].append(y_ref_pred[ri])

    y_raw_pred = np.empty(len(raw_values), dtype=float)
    for qi in range(len(raw_values)):
        if buckets[qi]:
            y_raw_pred[qi] = float(np.mean(buckets[qi]))
        else:
            nearest_qi, nearest_ri = min(path_pairs, key=lambda p: abs(p[0] - qi))
            _ = nearest_qi
            y_raw_pred[qi] = float(y_ref_pred[min(max(nearest_ri, 0), fixed_length - 1)])

    return y_raw_pred


# =============================================================================
# STEP 4 — EVALUATE ON RAW TEST CURVES
# =============================================================================

def _dispatch_predict(raw_values, curve, pipeline):
    """Route prediction to the right function based on pipeline['approach']."""
    approach = pipeline.get('approach', 'baseline')
    act, attrs = curve['activity'], curve['attributes']
    if approach == 'exog':
        return predict_raw_curve_exog(raw_values, act, attrs, pipeline,
                                      exog_values=curve.get('exog_values', {}))
    if approach == 'instance_stats':
        return predict_raw_curve_instance_stats(raw_values, act, attrs, pipeline)
    if approach == 'dtw_phase':
        return predict_raw_curve_dtw_phase(raw_values, act, attrs, pipeline)
    if approach == 'basis_expansion':
        return predict_raw_curve_basis(raw_values, act, attrs, pipeline)
    if approach == 'seq2seq':
        return predict_raw_curve_seq2seq(raw_values, act, attrs, pipeline)
    if approach == 'seq2seq_only':
        return predict_raw_curve_seq2seq_only(raw_values, act, attrs, pipeline)
    if approach == 'seq2seq_exog':
        return predict_raw_curve_seq2seq_exog(raw_values, act, attrs, pipeline,
                                              exog_values=curve.get('exog_values', {}))
    if approach == 'amplitude_shape':
        return predict_raw_curve_amplitude_shape(raw_values, act, attrs, pipeline)
    return predict_raw_curve(raw_values, act, attrs, pipeline)


def _train_energy_pipeline_worker(sensor, activity, obj, df_train, n_jobs=1):
    """
    Top-level (picklable) worker for parallel pipeline training.
    Trains one pipeline for a single (sensor, activity, object) combination.
    Must be a module-level function so joblib/loky can pickle it.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor

    curves, _ = split_curves(
        df_train,
        variable=sensor,
        activities=[activity],
        objects=[obj],
        test_size=0.0,
        verbose=0,
    )
    if len(curves) < 5:
        return sensor, activity, obj, None
    pipeline = build_and_train_pipeline(
        curves,
        variable=sensor,
        fixed_length=100,
        val_size=0.2,
        models={
            'Linear Regression': LinearRegression,
            'Gradient Boosting': GradientBoostingRegressor,
        },
        optimize_hyperparams=False,
        verbose=0,
        n_jobs=n_jobs,
    )
    return sensor, activity, obj, pipeline


def _train_curve_only_worker(sensor, activity, obj, df_train, approaches, ef_cols,
                             fixed_length=100, val_size=0.2):
    """
    Top-level picklable worker for RUN_CURVE_ONLY_EVALUATION parallel training.
    Trains all sklearn-based approaches for one (sensor, activity, object) combo.
    Seq2seq approaches are excluded — they use PyTorch and must stay sequential.

    Returns dict keyed by approach name, each value is the raw pipeline dict,
    plus 'sensor'/'activity'/'object' keys for reassembly.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor

    _SKLEARN_APPROACHES = {'baseline', 'instance_stats', 'istats_leakfree',
                           'dtw_phase', 'basis', 'exog', 'amplitude_shape'}
    _active = [a for a in approaches if a in _SKLEARN_APPROACHES]

    curves, _ = split_curves(
        df_train,
        variable=sensor,
        activities=[activity],
        objects=[obj],
        test_size=0.0,
        verbose=0,
        exog_columns=ef_cols,
    )
    if len(curves) < 5:
        return {'sensor': sensor, 'activity': activity, 'object': obj, 'skipped': True}

    models = {
        'Linear Regression': LinearRegression,
        'Gradient Boosting': GradientBoostingRegressor,
    }
    result = {'sensor': sensor, 'activity': activity, 'object': obj, 'skipped': False}

    if 'baseline' in _active:
        result['baseline'] = build_and_train_pipeline(
            curves, variable=sensor, fixed_length=fixed_length,
            val_size=val_size, models=models,
            optimize_hyperparams=False, verbose=0, n_jobs=1,
        )
    if 'instance_stats' in _active:
        result['instance_stats'] = build_and_train_pipeline_instance_stats(
            curves, variable=sensor, fixed_length=fixed_length,
            val_size=val_size, models=models,
            optimize_hyperparams=False, verbose=0, n_jobs=1,
        )
    if 'istats_leakfree' in _active:
        result['istats_leakfree'] = build_and_train_pipeline_istats_leakfree(
            curves, variable=sensor, fixed_length=fixed_length,
            val_size=val_size, models=models,
            optimize_hyperparams=False, verbose=0, n_jobs=1,
        )
    if 'dtw_phase' in _active:
        result['dtw_phase'] = build_and_train_pipeline_dtw_phase(
            curves, variable=sensor, fixed_length=fixed_length,
            val_size=val_size, models=models,
            optimize_hyperparams=False, verbose=0, n_jobs=1,
        )
    if 'basis' in _active:
        result['basis'] = build_and_train_pipeline_basis(
            curves, variable=sensor, fixed_length=fixed_length,
            val_size=val_size, models=models,
            optimize_hyperparams=False, verbose=0, n_jobs=1,
        )
    if 'exog' in _active and ef_cols:
        result['exog'] = build_and_train_pipeline_exog(
            curves, variable=sensor, fixed_length=fixed_length,
            val_size=val_size, models=models,
            optimize_hyperparams=False, verbose=0, n_jobs=1,
        )
    if 'amplitude_shape' in _active:
        result['amplitude_shape'] = build_and_train_pipeline_amplitude_shape(
            curves, variable=sensor, fixed_length=fixed_length,
            val_size=val_size, models=models,
            optimize_hyperparams=False, verbose=0, n_jobs=1,
        )
    return result


def _train_seq2seq_worker(sensor, activity, obj, df_train, approaches, ef_cols,
                          hidden_size=128, num_layers=2, dropout=0.1,
                          epochs=80, batch_size=32, lr=1e-3,
                          teacher_forcing_ratio=0.5, patience=10,
                          fixed_length=100, val_size=0.2):
    """
    Top-level picklable worker for parallel seq2seq training.
    Trains all seq2seq variants for one (sensor, activity, object) combo —
    identical scoping to _train_curve_only_worker so each model sees only
    homogeneous curves from that single combo.
    Returns dict with sensor/activity/object + results keyed by approach name.
    """
    import torch
    torch.set_num_threads(1)  # prevent OpenMP/MKL thread-pool contention across workers
    _SEQ2SEQ = {'seq2seq', 'seq2seq_only', 'seq2seq_exog'}
    _active = [a for a in approaches if a in _SEQ2SEQ]
    if not _active:
        return {'sensor': sensor, 'activity': activity, 'object': obj, 'skipped': True}

    curves, _ = split_curves(
        df_train,
        variable=sensor,
        activities=[activity],
        objects=[obj],
        test_size=0.0,
        verbose=0,
        exog_columns=ef_cols,
    )
    if len(curves) < 5:
        return {'sensor': sensor, 'activity': activity, 'object': obj, 'skipped': True}

    result = {'sensor': sensor, 'activity': activity, 'object': obj, 'skipped': False}

    if 'seq2seq' in _active:
        result['seq2seq'] = build_and_train_pipeline_seq2seq(
            curves, variable=sensor, fixed_length=fixed_length,
            val_size=val_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, epochs=epochs, batch_size=batch_size, lr=lr,
            teacher_forcing_ratio=teacher_forcing_ratio, patience=patience,
            verbose=False,
        )

    if 'seq2seq_only' in _active:
        result['seq2seq_only'] = build_and_train_pipeline_seq2seq_only(
            curves, variable=sensor, fixed_length=fixed_length,
            val_size=val_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, epochs=epochs, batch_size=batch_size, lr=lr,
            teacher_forcing_ratio=teacher_forcing_ratio, patience=patience,
            verbose=False,
        )

    if 'seq2seq_exog' in _active and ef_cols:
        result['seq2seq_exog'] = build_and_train_pipeline_seq2seq_exog(
            curves, variable=sensor, fixed_length=fixed_length,
            val_size=val_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, epochs=epochs, batch_size=batch_size, lr=lr,
            teacher_forcing_ratio=teacher_forcing_ratio, patience=patience,
            verbose=False,
        )

    return result


def evaluate_pipeline_on_test(test_curves, pipeline, max_plot_curves=6, verbose=1, save_dir=None):
    """
    Evaluate the trained pipeline on completely unseen, raw test curves.
    Ground truth = raw_values (never warped, never seen during training).
    Predictions  = predict_raw_curve() output (original energy units).

    Plots are shown in a grid with 3 columns and as many rows as necessary.
    """

    if verbose:
        print("\n" + "=" * 80)
        print("STEP 4 — EVALUATION ON RAW TEST CURVES")
        print("  Ground truth : raw_values — never warped, never seen in training")
        print("  Predictions  : original energy units (DTW for positioning only)")
        print("=" * 80)

    per_curve_metrics = []
    all_true, all_pred = [], []

    for curve in test_curves:
        raw_values = curve['original_values']
        y_pred = _dispatch_predict(raw_values, curve, pipeline)

        mae  = mean_absolute_error(raw_values, y_pred)
        rmse = np.sqrt(mean_squared_error(raw_values, y_pred))
        r2   = r2_score(raw_values, y_pred)
        _denom = np.sum(np.abs(raw_values))
        wape = np.sum(np.abs(raw_values - y_pred)) / _denom * 100 if _denom != 0 else np.nan

        per_curve_metrics.append({
            'instance_id': curve['instance_id'],
            'activity':    curve['activity'],
            'n_points':    len(raw_values),
            'MAE':         mae,
            'RMSE':        rmse,
            'WAPE (%)':    wape,
            'R2':          r2,
        })

        all_true.extend(raw_values.tolist())
        all_pred.extend(y_pred.tolist())

    # Aggregate metrics
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    agg_metrics = {
        'MAE':      mean_absolute_error(all_true, all_pred),
        'RMSE':     np.sqrt(mean_squared_error(all_true, all_pred)),
        'WAPE (%)': np.sum(np.abs(all_true - all_pred)) / np.sum(np.abs(all_true)) * 100,
        'R2':       r2_score(all_true, all_pred),
    }

    metrics_df = pd.DataFrame(per_curve_metrics)

    if verbose:
        print("\nPer-curve metrics:")
        print(metrics_df.to_string(index=False, float_format='%.4f'))

    print("\nAggregate metrics across all test curves:")
    for k, v in agg_metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    # ------------------------------------------------------------------
    # GRID PLOTTING (3 columns)
    # ------------------------------------------------------------------
    import os as _os
    import itertools as _itertools

    _sensor_slug = (
        pipeline.get('variable_name', 'sensor')
        .replace('/', '-').replace(' ', '_')[:80]
    )
    _approach = pipeline.get('approach', 'baseline')

    # When saving to disk: plot every curve grouped by activity.
    # When only displaying interactively: respect max_plot_curves.
    if save_dir is not None:
        _os.makedirs(save_dir, exist_ok=True)
        # Group curves by activity and save one figure per (activity) group
        from collections import defaultdict as _dd
        _by_activity = _dd(list)
        for _c in test_curves:
            _by_activity[_c['activity']].append(_c)

        for _act, _act_curves in _by_activity.items():
            _act_slug = _act.replace('/', '-').replace(' ', '_')[:60]
            _n = len(_act_curves)
            _n_cols = 3
            _n_rows = int(np.ceil(_n / _n_cols))
            _fig, _axes = plt.subplots(_n_rows, _n_cols,
                                       figsize=(14, 4.5 * max(_n_rows, 1)))
            _axes = np.array(_axes).reshape(-1)
            for _ax, _curve in zip(_axes[:_n], _act_curves):
                _rv = _curve['original_values']
                _yp = _dispatch_predict(_rv, _curve, pipeline)
                _r2  = r2_score(_rv, _yp)
                _rms = np.sqrt(mean_squared_error(_rv, _yp))
                _ax.plot(_rv, label='Actual', color='steelblue', linewidth=2)
                _ax.plot(_yp, label='Predicted', color='tomato',
                         linewidth=2, linestyle='--')
                _ax.set_title(
                    f"ID {_curve['instance_id']} | {_act}\n"
                    f"R²={_r2:.3f}  RMSE={_rms:.4f}", fontsize=9)
                _ax.set_xlabel("Time step")
                _ax.set_ylabel("Energy")
                _ax.grid(True, alpha=0.3)
                _ax.legend(fontsize=8)
            for _ax in _axes[_n:]:
                _ax.set_visible(False)
            _fig.suptitle(
                f"{_approach} | {_sensor_slug} | {_act}",
                fontsize=12, fontweight='bold')
            plt.tight_layout()
            _save_path = _os.path.join(
                save_dir, f"{_approach}__{_sensor_slug}__{_act_slug}.png")
            plt.savefig(_save_path, dpi=150, bbox_inches='tight')
            plt.close(_fig)

    if verbose:
        n_plot = min(max_plot_curves, len(test_curves))
        n_cols = 3
        n_rows = int(np.ceil(n_plot / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(14, 4.5 * n_rows)
        )
        fig.suptitle(
            f"ENERGY CURVE EVALUATION (TEST SET)\nSensor: {pipeline.get('variable_name', 'Unknown')}",
            fontsize=18, fontweight='bold', color='navy', y=0.98)
        plt.subplots_adjust(top=0.9)
        axes = np.array(axes).reshape(-1)

        for ax, curve in zip(axes[:n_plot], test_curves[:n_plot]):
            raw_values = curve['original_values']
            y_pred = _dispatch_predict(raw_values, curve, pipeline)
            r2_val   = r2_score(raw_values, y_pred)
            rmse_val = np.sqrt(mean_squared_error(raw_values, y_pred))
            ax.plot(raw_values, label='Actual (raw)', color='steelblue', linewidth=2)
            ax.plot(y_pred, label='Predicted', color='tomato',
                    linewidth=2, linestyle='--')
            ax.set_title(
                f"ID {curve['instance_id']} | {curve['activity']}\n"
                f"R2={r2_val:.3f}  RMSE={rmse_val:.4f}", fontsize=9)
            ax.set_xlabel("Time step")
            ax.set_ylabel("Energy")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        for ax in axes[n_plot:]:
            ax.set_visible(False)

        plt.suptitle(
            f"Test Evaluation: Raw Predicted vs Raw Actual\nSensor: {pipeline.get('variable_name', 'Unknown')}",
            fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    return metrics_df, agg_metrics

# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import optuna  # Assuming optuna is installed

def train_position_based_regression(df_expanded, variable, activities, fixed_length=100, test_size=0.2, random_state=42, 
                                   models=None, optimize_hyperparams=False, n_trials=50):
    """
    Train position-based regression models for energy curve prediction.
    
    Parameters:
    -----------
    df_expanded : pd.DataFrame
        The expanded dataframe with energy data
    variable : str
        The energy variable column name
    activities : list
        List of activity names to include
    fixed_length : int
        Fixed length to resample curves to
    test_size : float
        Proportion of data for testing
    random_state : int
        Random state for reproducibility
    models : dict, optional
        Dictionary of model names to model classes (e.g., {'Gradient Boosting': GradientBoostingRegressor})
        If None, defaults to GradientBoostingRegressor and RandomForestRegressor
    optimize_hyperparams : bool, optional
        Whether to perform hyperparameter optimization with Optuna (default: False)
    n_trials : int, optional
        Number of Optuna trials for optimization (default: 50)
    
    Returns:
    --------
    dict : Results containing models, best model, and metadata
    """
    print("="*80)
    print("POSITION-BASED REGRESSION - NO ERROR COMPOUNDING!")
    print("="*80)
    
    # Default models if not provided
    if models is None:
        models = {
            'Gradient Boosting': GradientBoostingRegressor,
            'Random Forest': RandomForestRegressor
        }
    
    # Prepare data
    df_plot = df_expanded.copy()
    # Select necessary columns including object_attributes_log
    df_plot = df_plot[['case_id_log', 'activity_log', 'timestamp_start_log', 'datetime_energy', variable, 'object_attributes_log']]
    df_plot = df_plot[df_plot['activity_log'].isin(activities)]
    df_plot['timestamp_start'] = pd.to_datetime(df_plot['timestamp_start_log'])
    df_plot['datetime_energy'] = pd.to_datetime(df_plot['datetime_energy'])
    df_plot = df_plot.sort_values(['case_id_log', 'timestamp_start_log', 'datetime_energy'])
    
    # Create instance IDs
    df_plot['activity_instance_id'] = (
        df_plot.groupby(['case_id_log', 'activity_log', 'timestamp_start_log']).ngroup()
    )
    
    # Extract curves and attributes
    curves_data = []
    for instance_id, group in df_plot.groupby('activity_instance_id'):
        group = group.sort_values('datetime_energy').reset_index(drop=True)
        values = group[variable].dropna().values
        
        if len(values) >= 5:
            # Get attributes from the first row (assuming same per instance)
            attributes = group['object_attributes_log'].iloc[0] if not group['object_attributes_log'].empty else {}
            
            curves_data.append({
                'instance_id': instance_id,
                'activity': group['activity_log'].iloc[0],
                'original_values': values,
                'original_length': len(values),
                'attributes': attributes
            })
    
    print(f"\nResampling all curves to fixed length: {fixed_length} points")
    
    # Resample all curves to fixed length
    for curve in curves_data:
        x_old = np.linspace(0, 1, len(curve['original_values']))
        x_new = np.linspace(0, 1, fixed_length)
        curve['resampled_values'] = np.interp(x_new, x_old, curve['original_values'])
    
    print(f"Total curves: {len(curves_data)}")
    
    # Determine attribute columns and types
    if curves_data:
        all_keys = set()
        for curve in curves_data:
            all_keys.update(curve['attributes'].keys())
        all_keys = sorted(all_keys)
        
        # Check convertibility for each key
        key_types = {}
        for key in all_keys:
            is_numeric = True
            for curve in curves_data:
                if key in curve['attributes']:
                    try:
                        float(curve['attributes'][key])
                    except (ValueError, TypeError):
                        is_numeric = False
                        break
            key_types[key] = 'numeric' if is_numeric else 'category'
        
        print(f"Attribute columns: {key_types}")
    else:
        all_keys = []
        key_types = {}
    
    # Build position-based regression dataset
    regression_data = []
    
    for curve in curves_data:
        for position_idx in range(fixed_length):
            # Position as fraction (0 to 1)
            position_fraction = position_idx / (fixed_length - 1) if fixed_length > 1 else 0
            
            row = {
                'instance_id': curve['instance_id'],
                'activity': curve['activity'],
                'position_idx': position_idx,
                'position_fraction': position_fraction,
                'curve_length': curve['original_length'],
                'y': curve['resampled_values'][position_idx]
            }
            
            # Add attributes
            for key in all_keys:
                value = curve['attributes'].get(key, None)
                if key_types[key] == 'numeric':
                    try:
                        row[key] = float(value) if value is not None else np.nan
                    except (ValueError, TypeError):
                        row[key] = np.nan
                else:
                    row[key] = str(value) if value is not None else 'None'
            
            regression_data.append(row)
    
    df_regression = pd.DataFrame(regression_data)
    
    print(f"Regression dataset: {len(df_regression)} samples")
    print(f"  {len(curves_data)} curves × {fixed_length} points")
    
    # Prepare features
    feature_cols = ['position_idx', 'position_fraction', 'curve_length']
    X = df_regression[feature_cols].copy()
    y = df_regression['y'].copy()
    
    # Add activity
    X['activity'] = df_regression['activity']
    
    # Add attributes
    categorical_cols = ['activity']
    for key in all_keys:
        if key_types[key] == 'numeric':
            X[key] = df_regression[key]
        else:
            X[key] = df_regression[key]
            categorical_cols.append(key)
    
    # One-hot encode categorical columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Split by COMPLETE CURVES
    unique_instances = df_regression['instance_id'].unique()
    train_instances, test_instances = train_test_split(
        unique_instances, test_size=test_size, random_state=random_state
    )
    
    train_mask = df_regression['instance_id'].isin(train_instances)
    test_mask = df_regression['instance_id'].isin(test_instances)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"\nTrain/Test split:")
    print(f"  Train: {len(train_instances)} curves, {len(X_train)} points")
    print(f"  Test: {len(test_instances)} curves, {len(X_test)} points")
    
    # Train models
    results_position = {}
    
    for name, model_class in models.items():
        print(f"\nTraining {name}...")
        
        if optimize_hyperparams:
            # Define objective function for Optuna
            def objective(trial):
                if name == 'Gradient Boosting':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'random_state': random_state
                    }
                elif name == 'Random Forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 5, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'random_state': random_state,
                        'n_jobs': n_jobs
                    }
                else:
                    # For other models, no hyperparameter optimization defined
                    params = {}

                model = model_class(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                return r2_score(y_test, y_pred)
            
            # Run Optuna study
            sampler = optuna.samplers.TPESampler(seed=random_state)
            study = optuna.create_study(direction='maximize', sampler=sampler)
            study.optimize(objective, n_trials=n_trials)
            
            # Get best parameters
            best_params = study.best_params
            if name == 'Gradient Boosting':
                best_params['random_state'] = random_state
            elif name == 'Random Forest':
                best_params['random_state'] = random_state
                best_params['n_jobs'] = n_jobs
            
            print(f"  Best params: {best_params}")
            model = model_class(**best_params)
        else:
            # Use default parameters
            if name == 'Gradient Boosting':
                model = model_class(
                    n_estimators=200, max_depth=7, learning_rate=0.1,
                    subsample=0.8, random_state=random_state
                )
            elif name == 'Random Forest':
                model = model_class(
                    n_estimators=200, max_depth=12, min_samples_split=5,
                    random_state=random_state, n_jobs=n_jobs
                )
            else:
                # For other models, try with random_state, if not supported, without
                try:
                    model = model_class(random_state=random_state)
                except TypeError:
                    model = model_class()
        
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        results_position[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
        
        print(f"  Train R²={train_r2:.4f}, RMSE={train_rmse:.2f}")
        print(f"  Test R²={test_r2:.4f}, RMSE={test_rmse:.2f}")
    
    # Select best model
    best_model_name_pos = max(results_position.keys(), key=lambda k: results_position[k]['test_r2'])
    best_model_pos = results_position[best_model_name_pos]['model']
    
    print(f"\n✓ Best model: {best_model_name_pos}")
    print(f"  Test R² = {results_position[best_model_name_pos]['test_r2']:.4f}")
    
    # Store for later use
    fixed_curve_length = fixed_length
    feature_columns_pos = X.columns.tolist()
    
    return {
        'results': results_position,
        'best_model': best_model_pos,
        'best_model_name': best_model_name_pos,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_columns': feature_columns_pos,
        'fixed_length': fixed_curve_length,
        'curves_data': curves_data,
        'test_instances': test_instances
    }

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate comprehensive metrics: MAE, RMSE, WAPE, R²
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model for print
    
    Returns:
    --------
    dict : Metrics dictionary
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # WAPE (Weighted Absolute Percentage Error)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'WAPE (%)': wape,
        'R²': r2
    }

def print_metrics_table(results_dict, y_test, X_test):
    """
    print a table of metrics for all models in results_dict
    
    Parameters:
    -----------
    results_dict : dict
        Results from training function
    y_test : array-like
        Test true values
    X_test : pd.DataFrame
        Test features
    """
    metrics_list = []
    
    for model_name, model_info in results_dict['results'].items():
        model = model_info['model']
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred, model_name)
        metrics_list.append(metrics)
    
    # Create DataFrame for table
    metrics_df = pd.DataFrame(metrics_list)
    
    # print table
    print("\n" + "="*80)
    print("MODEL PERFORMANCE METRICS")
    print("="*80)
    print(metrics_df.to_string(index=False, float_format='%.4f'))
    
    # Highlight best model
    best_row = metrics_df.loc[metrics_df['R²'].idxmax()]
    print(f"\n✓ Best model: {best_row['Model']} (R² = {best_row['R²']:.4f})")
    
    return metrics_df

# %%
