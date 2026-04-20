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


def extract_energy_modifiers(
    df_expanded,
    sensors,
    activity_col='activity_log',
    duration_model_class=None,
    duration_model_params=None,
    transition_model_class=None,
    transition_model_params=None,
    min_transitions=30,
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
    from sklearn.linear_model import Lasso, LogisticRegression

    if duration_model_class is None:
        duration_model_class = Lasso
    if duration_model_params is None:
        duration_model_params = {'alpha': 0.1}
    if transition_model_class is None:
        transition_model_class = LogisticRegression
    if transition_model_params is None:
        transition_model_params = {'penalty': 'l2', 'C': 1.0, 'max_iter': 1000}

    print("\n" + "=" * 70)
    print("ENERGY MODIFIER EXTRACTION")
    print("=" * 70)
    print(f"  Sensors          : {sensors}")
    print(f"  Duration model   : {duration_model_class.__name__}({duration_model_params})")
    print(f"  Transition model : {transition_model_class.__name__}({transition_model_params})")

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

    # Compute next_activity as the next row's activity within each (case, object)
    # We rely on the order already embedded inside df_expanded
    group_cols = ['case_id_log', 'object_log']
    available = [c for c in group_cols if c in df_expanded.columns]
    if available:
        df_sorted = df_expanded.dropna(subset=[activity_col]).sort_values(
            available + ['timestamp_start_log']
        ).copy()
        df_sorted['_next_activity'] = (
            df_sorted.groupby(available)[activity_col].shift(-1).fillna('__END__')
        )
        # Map instance → next_activity via timestamp_start_log
        ts_to_next = dict(zip(
            df_sorted['timestamp_start_log'].astype(str),
            df_sorted['_next_activity'].astype(str)
        ))
        # Attach next_activity to df_recs — use a best-effort join
        # (df_recs was built per instance so we re-derive from df_expanded)
        df_next = (
            df_sorted[[activity_col, 'timestamp_start_log', '_next_activity']]
            .drop_duplicates()
            .copy()
        )
        df_next.columns = ['activity', 'timestamp_start_log', 'next_activity']
        # We can't directly join df_recs to df_next without an instance key,
        # so rebuild with next_activity included.
        df_recs = _build_energy_state_matrix_with_next(
            df_expanded, sensors, activity_col, timestamp_start_col, datetime_energy_col
        )
        energy_state_columns_check = []
        for s in sensors:
            energy_state_columns_check += [f'{s}_mean', f'{s}_end', f'{s}_std']
        energy_state_columns = energy_state_columns_check
    else:
        df_recs['next_activity'] = '__END__'

    if df_recs.empty:
        print("  WARNING: no valid instances with next_activity — returning empty modifiers.")
        return {}, {}, energy_state_columns

    print(f"  Total instances  : {len(df_recs)}")

    # ── Per-activity modifiers ────────────────────────────────────────────
    energy_duration_modifiers = {}
    energy_transition_modifiers = {}

    for activity, grp_act in df_recs.groupby('activity'):
        X = grp_act[energy_state_columns].values
        n = len(grp_act)

        # Training-time column means — attached to every model so the
        # simulation can synthesise a fallback energy state without a
        # live prediction pipeline.
        train_feature_mean = dict(zip(energy_state_columns, X.mean(axis=0)))

        # ── Duration modifier ─────────────────────────────────────────
        mean_dur = grp_act['duration'].mean()
        if mean_dur > 0 and n >= 5:
            y_dur = np.log(grp_act['duration'].values / mean_dur)
            try:
                mdl = duration_model_class(**duration_model_params)
                mdl.fit(X, y_dur)
                mdl._mean_duration        = mean_dur
                mdl._train_feature_mean   = train_feature_mean  # ← fallback state
                energy_duration_modifiers[str(activity)] = mdl
                print(f"  [{activity}] duration modifier fitted  (n={n})")
            except Exception as exc:
                print(f"  [{activity}] duration modifier FAILED: {exc}")

        # ── Transition modifier ────────────────────────────────────────
        y_tr = grp_act['next_activity'].values
        n_classes = len(set(y_tr))
        if n >= min_transitions and n_classes >= 2:
            try:
                clf = transition_model_class(**transition_model_params)
                clf.fit(X, y_tr)
                clf._train_feature_mean   = train_feature_mean  # ← fallback state
                energy_transition_modifiers[str(activity)] = clf
                print(f"  [{activity}] transition modifier fitted (n={n}, classes={list(set(y_tr))})")
            except Exception as exc:
                print(f"  [{activity}] transition modifier FAILED: {exc}")
        else:
            reason = (f"n={n} < {min_transitions}" if n < min_transitions
                      else f"only {n_classes} class")
            print(f"  [{activity}] transition modifier SKIPPED ({reason})")

    print(f"\n  Duration modifiers  : {len(energy_duration_modifiers)} activities")
    print(f"  Transition modifiers: {len(energy_transition_modifiers)} activities")
    print(f"  Energy state cols   : {energy_state_columns}")

    return energy_duration_modifiers, energy_transition_modifiers, energy_state_columns


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
                 test_size=0.15, random_state=42, verbose=1):
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

    Returns
    -------
    train_curves : list[dict]
    test_curves  : list[dict]
        Each dict: 'instance_id', 'activity', 'original_values' (np.ndarray),
                   'original_length', 'attributes'
    """
    if verbose:
        print("=" * 80)
        print("STEP 1 — RAW TRAIN/TEST SPLIT (before any preprocessing)")
        print("=" * 80)

    df = df_expanded.copy()
    df = df[['case_id_log', 'activity_log', 'object_log', 'timestamp_start_log',
             'datetime_energy', variable, 'object_attributes_log']]
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
        values = group[variable].dropna().values
        if len(values) >= 5:
            attributes = (
                group['object_attributes_log'].iloc[0]
                if not group['object_attributes_log'].empty else {}
            )
            curves_data.append({
                'instance_id':     instance_id,
                'activity':        group['activity_log'].iloc[0],
                'original_values': values,
                'original_length': len(values),
                'attributes':      attributes,
            })

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
    rows = []
    for curve in curves:
        for position_idx in range(fixed_length):
            row = {
                'instance_id':       curve['instance_id'],
                'activity':          curve['activity'],
                'position_idx':      position_idx,
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
    X_all = df_reg[['position_idx', 'curve_length']].copy()
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
    numeric_feature_cols = ['position_idx', 'curve_length'] + [
        k for k in all_keys if key_types[k] == 'numeric'
    ]
    numeric_feature_cols = [c for c in numeric_feature_cols if c in X_train.columns]

    feature_scaler = None
    if numeric_feature_cols:
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
                        'n_jobs':            -1,
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
                best_params['n_jobs'] = -1
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
                    random_state=random_state, n_jobs=-1
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
    rows_ref = []
    for ref_pos in range(fixed_length):
        row = {
            'position_idx': ref_pos,
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
# STEP 4 — EVALUATE ON RAW TEST CURVES
# =============================================================================

def evaluate_pipeline_on_test(test_curves, pipeline, max_plot_curves=6, verbose=1):
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
        y_pred = predict_raw_curve(
            raw_values, curve['activity'], curve['attributes'], pipeline
        )

        mae  = mean_absolute_error(raw_values, y_pred)
        rmse = np.sqrt(mean_squared_error(raw_values, y_pred))
        r2   = r2_score(raw_values, y_pred)
        wape = np.sum(np.abs(raw_values - y_pred)) / np.sum(np.abs(raw_values)) * 100

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
    if verbose:
        n_plot = min(max_plot_curves, len(test_curves))

        n_cols = 3
        n_rows = int(np.ceil(n_plot / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(14, 4.5 * n_rows) # Increased height for suptitle room
        )

        # Very prominent main overall title for the figure
        fig.suptitle(f"ENERGY CURVE EVALUATION (TEST SET)\nSensor: {pipeline.get('variable_name', 'Unknown')}", 
                     fontsize=18, fontweight='bold', color='navy', y=0.98)
        
        plt.subplots_adjust(top=0.9) # Make room for suptitle

        axes = np.array(axes).reshape(-1)  # flatten safely

        for ax, curve in zip(axes[:n_plot], test_curves[:n_plot]):
            raw_values = curve['original_values']
            y_pred = predict_raw_curve(
                raw_values, curve['activity'], curve['attributes'], pipeline
            )

            r2_val   = r2_score(raw_values, y_pred)
            rmse_val = np.sqrt(mean_squared_error(raw_values, y_pred))

            ax.plot(raw_values, label='Actual (raw)', color='steelblue', linewidth=2)
            ax.plot(y_pred, label='Predicted', color='tomato',
                    linewidth=2, linestyle='--')

            ax.set_title(
                f"ID {curve['instance_id']} | {curve['activity']}\n"
                f"R2={r2_val:.3f}  RMSE={rmse_val:.4f}",
                fontsize=9
            )
            ax.set_xlabel("Time step")
            ax.set_ylabel("Energy")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Hide unused subplots
        for ax in axes[n_plot:]:
            ax.set_visible(False)

        plt.suptitle(
            "Test Evaluation: Raw Predicted vs Raw Actual (original energy units)",
            fontsize=14, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        plt.savefig("tmp.png"); plt.close()

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
                        'n_jobs': -1
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
                best_params['n_jobs'] = -1
            
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
                    random_state=random_state, n_jobs=-1
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
