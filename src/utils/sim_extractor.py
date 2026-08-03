from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import warnings

# ---------------------------------------------------------------------------
# Reproducibility — one seed for the whole pipeline
# ---------------------------------------------------------------------------
import hashlib as _hashlib
import os as _os_seed
import random as _random_seed

#: Master seed. Override for a whole run with PIPELINE_RANDOM_SEED (01_pipeline.py
#: sets it in the child environment). Everything stochastic in this codebase
#: derives from it, so a run is a pure function of (data, config, this seed).
GLOBAL_RANDOM_SEED = int(_os_seed.environ.get('PIPELINE_RANDOM_SEED', '42'))


def stable_seed(*parts, base=None):
    """
    Deterministic 31-bit seed derived from a stable key, e.g. a
    (sensor, activity, object) triple.

    Uses blake2b rather than hash(): Python randomises str hashing per process
    unless PYTHONHASHSEED is fixed, so hash()-derived seeds would silently
    differ between runs — exactly the failure this function exists to prevent.

    Deriving each parallel task's seed from its own key (instead of letting it
    inherit whatever RNG state the worker process happens to hold) makes the
    result independent of worker count and of the order in which the pool
    happens to schedule tasks.
    """
    base = GLOBAL_RANDOM_SEED if base is None else int(base)
    key = '|'.join(str(p) for p in parts).encode('utf-8')
    digest = _hashlib.blake2b(key, digest_size=8).digest()
    return (int.from_bytes(digest, 'big') ^ base) % (2 ** 31 - 1)


def set_global_seeds(seed=None, deterministic_torch=True):
    """
    Seed every RNG this codebase can reach: python `random`, numpy's global
    RNG, and (when torch is importable) CPU + all CUDA devices.

    NOTE on PYTHONHASHSEED: it only takes effect at interpreter start, so
    setting it here would be useless — 01_pipeline.py exports it for the child
    process instead. Without it, iteration order of str sets can vary between
    runs even though every RNG is seeded.

    Returns the seed actually applied.
    """
    seed = GLOBAL_RANDOM_SEED if seed is None else int(seed)
    _random_seed.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # cuDNN picks convolution algorithms by benchmarking otherwise,
            # which makes even a seeded run vary between invocations.
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return seed

# ---------------------------------------------------------------------------
# pm4py imports (used when mining_algorithm != 'manual')
# ---------------------------------------------------------------------------
import pm4py
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

# ---------------------------------------------------------------------------
# WIP / resource-occupation load profile
# ---------------------------------------------------------------------------

class LoadProfile:
    """
    Precomputed workload snapshot built from an event log (real or simulated).

    Exposes two lookups, both defined as "how many *other* things were
    active at instant t":
      - ``wip_at(t)``          — case-level Work-In-Progress: cases started
                                  but not yet finished.
      - ``ro_at(t, resource)`` — Resource-Occupation: activity instances
                                  running on the same ``resource`` (the
                                  ``object`` column, e.g. a machine id).

    ``df`` must have columns: case_id, activity, timestamp_start,
    timestamp_end, plus a resource column (``resource_col``, default
    ``'object'`` — the true machine/resource id; NOT the same as
    ``ProcessSimulation``'s synthetic per-group ``object`` field on
    *simulated* logs, which is why ``resource_col`` is configurable).
    An internal per-row id is used so a row's own instance can be
    excluded from its own WIP/RO count (``exclude_case_id`` /
    ``exclude_instance_id``).
    """

    def __init__(self, df: pd.DataFrame, resource_col: str = 'object'):
        d = df.reset_index(drop=True).copy()
        if '_row_uid' not in d.columns:
            d['_row_uid'] = np.arange(len(d))
        # Every caller computes its query `t` via pandas Timestamp.timestamp()
        # (columns are always datetime64[ns]), which — like .astype('int64') —
        # treats naive timestamps as UTC. Keep the same convention here.
        d['_start_ts'] = d['timestamp_start'].astype('int64') / 1e9
        d['_end_ts']   = d['timestamp_end'].astype('int64') / 1e9

        case_g = d.groupby('case_id').agg(
            _start_ts=('_start_ts', 'min'), _end_ts=('_end_ts', 'max')
        )
        self._case_ids    = case_g.index.to_numpy()
        self._case_starts = case_g['_start_ts'].to_numpy()
        self._case_ends   = case_g['_end_ts'].to_numpy()

        # Some event logs (e.g. process_4) don't carry a resource/machine
        # column at all. Without one, per-resource occupancy is undefined —
        # fall back to a per-row unique id so ro_at() always resolves to 0
        # rather than raising.
        if resource_col in d.columns:
            self._res_names = d[resource_col].to_numpy()
        else:
            self._res_names = d['_row_uid'].astype(str).to_numpy()
        self._res_starts   = d['_start_ts'].to_numpy()
        self._res_ends     = d['_end_ts'].to_numpy()
        self._res_instance = d['_row_uid'].to_numpy()

    def wip_at(self, t, exclude_case_id=None) -> float:
        t = float(t)
        active = (self._case_starts <= t) & (self._case_ends > t)
        if exclude_case_id is not None:
            active &= (self._case_ids != exclude_case_id)
        return float(active.sum())

    def ro_at(self, t, resource, exclude_instance_id=None) -> float:
        if resource is None:
            return 0.0
        t = float(t)
        mask = (
            (self._res_names == resource)
            & (self._res_starts <= t)
            & (self._res_ends > t)
        )
        if exclude_instance_id is not None:
            mask &= (self._res_instance != exclude_instance_id)
        return float(mask.sum())


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

# ML model acceptance thresholds.
# Duration: ML is used only when CV MAE is at least this fraction below the
#           mean-prediction baseline (e.g. 0.80 = must beat baseline by ≥20%).
# Transition: ML is *kept* (and blended) when CV balanced-accuracy exceeds the
#             majority-class baseline by at least this absolute margin.
# _TR_BLEND_FULL: lift over baseline at which the simulator uses fully-ML
#             transition probabilities (linear interpolation below this).
_DUR_ACCEPTANCE_RATIO = 0.95   # require ≥5% MAE improvement over baseline
_TR_ACCEPTANCE_MARGIN = 0.0    # accept any balanced-accuracy lift over baseline
_TR_BLEND_FULL        = 0.05   # ≥5 pp lift → α = 1.0 (pure ML routing)

# Minimum samples an activity instance must have to become a curve. SHARED by
# split_curves and split_curves_with_prev_activity so every approach is trained
# and scored on exactly the same set of curves.
# Until 2026-07-24 the two disagreed — 2 and 5 respectively — so curves of 2-4
# samples existed for the plain approaches and not for the *_external ones
# (experiment_964: 30,080 vs 26,000 TEST curves), and the approach columns of
# the curve comparison were medians over different populations. Five is the
# stricter of the two and the more defensible floor: a 2-point "curve" carries
# no shape for a DBA/DTW pipeline to align.
MIN_CURVE_SAMPLES = 5


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
                    heuristic_params=None, ilp_variant_coverage=1.0):
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
    ilp_variant_coverage : float
        For ILP: fraction of traces to retain (by most-frequent variants first).
        1.0 = keep all variants; 0.8 = keep variants covering 80% of traces.

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
        _log_to_mine = sub_log
        cov = float(ilp_variant_coverage)
        if cov < 1.0 and len(sub_log) > 0:
            _filtered = pm4py.filter_variants_by_coverage_percentage(sub_log, cov)
            if len(_filtered) > 0:
                _log_to_mine = _filtered
        net, im, fm = pm4py.discover_petri_net_ilp(_log_to_mine)
    else:
        raise ValueError(f"Unknown mining algorithm: {algorithm}")

    net = _repair_and_join_artifacts(net)

    return net, im, fm


def _repair_and_join_artifacts(net):
    """
    Fix a miner→Petri-net conversion artifact seen with the Heuristics
    Miner: a labelled transition occasionally ends up wired with in-arcs
    from MULTIPLE places, so firing it requires tokens in ALL of them
    simultaneously (an AND-join) — but those places are actually
    alternative predecessors of the same activity (an XOR choice: "reached
    from A, or from B"), not a genuine concurrent split. Since nothing else
    in the net produces tokens into both places at once, the transition
    becomes almost unreachable during token-game replay even though
    token-based-replay (which uses forced/logged moves to push through
    fitness gaps) still assigns it a legitimate-looking positive stochastic
    weight. Net effect: a real, frequent activity silently drops out of
    every simulated case.

    Confirmed directly: an activity occurring in 71% of real training
    cases, with stochastic weight 21 (comparable to siblings that DO fire),
    was enabled 0 times across 200 replayed simulated cases purely because
    its two "predecessor" places are never simultaneously marked.

    Detects the artifact by checking whether any *other* transition in the
    net has an AND-split feeding tokens into ALL of the candidate
    transition's input places at once (a genuine synchronization). If no
    such split exists, the multi-input transition is split into one
    single-input transition per original input place (same label, same
    outputs) — correctly modelling "reachable from any of these places"
    instead of "needs all of them at once".
    """
    from pm4py.objects.petri_net.utils import petri_utils

    suspects = [t for t in list(net.transitions)
               if t.label is not None and len(t.in_arcs) > 1]

    for t in suspects:
        in_places = [a.source for a in t.in_arcs]

        has_and_split = any(
            other is not t and all(p in {a.target for a in other.out_arcs} for p in in_places)
            for other in net.transitions
        )
        if has_and_split:
            continue  # genuine synchronization -- leave it alone

        out_targets = [(a.target, a.weight) for a in t.out_arcs]
        label = t.label
        base_name = t.name

        petri_utils.remove_transition(net, t)

        for i, p in enumerate(in_places):
            new_t = petri_utils.add_transition(net, name=f"{base_name}__split{i}", label=label)
            petri_utils.add_arc_from_to(p, new_t, net)
            for target, weight in out_targets:
                petri_utils.add_arc_from_to(new_t, target, net, weight=weight)

    return net


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
        'noise_threshold':       float(noise_threshold),
        'heuristic_params':      dict(heuristic_params or {}),
        'ilp_variant_coverage':  1.0,
    }

    if not optimize_mining_hyperparams or algorithm not in ('inductive', 'heuristic', 'ilp'):
        return [base]

    search_space = mining_search_space or {}
    candidates = []

    if algorithm == 'inductive':
        default_grid = [0.05, 0.10, 0.20, 0.30, 0.40]
        noise_grid = search_space.get('inductive_noise_thresholds', default_grid)
        for n in noise_grid:
            n = float(max(0.0, min(1.0, n)))
            candidates.append({'noise_threshold': n, 'heuristic_params': {}, 'ilp_variant_coverage': 1.0})

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
                               'heuristic_params': dict(params or {}),
                               'ilp_variant_coverage': 1.0})

    elif algorithm == 'ilp':
        default_grid = [0.80, 0.90, 0.95, 1.0]
        cov_grid = search_space.get('ilp_variant_coverages', default_grid)
        for cov in cov_grid:
            cov = float(max(0.0, min(1.0, cov)))
            candidates.append({'noise_threshold': 0.0, 'heuristic_params': {}, 'ilp_variant_coverage': cov})

    if not candidates:
        candidates = [base]

    # Deduplicate candidates while preserving order.
    seen = set()
    unique = []
    for cand in candidates:
        key = (
            round(float(cand.get('noise_threshold', 0.0)), 6),
            tuple(sorted((cand.get('heuristic_params') or {}).items())),
            round(float(cand.get('ilp_variant_coverage', 1.0)), 6),
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
    # Typical (median) case length is what the budget-mode over-generation
    # guard needs: max_case_length is a single outlier case and is far too
    # permissive as a bound (measured: budget modes emitting 2.8x the real
    # mean activity count while still sitting well under max).
    _case_lengths = []

    for case_id, case_df in case_sorted.items():
        activities_in_case = case_df['activity'].tolist()
        if not activities_in_case:
            continue
        max_case_length = max(max_case_length, len(activities_in_case))
        _case_lengths.append(len(activities_in_case))

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

    # Median, not mean: case-length distributions here are right-skewed (a few
    # very long cases), so the mean sits above the typical case and would make
    # the budget-mode length guard too permissive.
    median_case_length = float(np.median(_case_lengths)) if _case_lengths else 0.0

    return decision_weights, max_case_length, median_case_length


def _compute_activity_repeat_counts(case_sorted):
    """
    For every activity label that appears in at least one case, collect the
    per-case occurrence count (including 0 for cases where it never fires).

    Returns dict[activity_label] -> list[int], one entry per case -- the raw
    empirical distribution used to sample a per-case repeat quota at
    simulation time (see ProcessSimulation._sample_activity_caps). This keeps
    the Petri-net token game's loop lengths grounded in real per-case repeat
    behaviour instead of a memoryless per-step frequency draw, which has a
    geometric tail and can occasionally run away well past anything ever
    observed in training (the mechanism behind the max_steps safety-net
    firing on a small fraction of cases).
    """
    per_case_counts = []
    all_activities = set()
    for _, case_df in case_sorted.items():
        counts = Counter(case_df['activity'].tolist())
        per_case_counts.append(counts)
        all_activities.update(counts.keys())

    repeat_counts = {act: [] for act in all_activities}
    for counts in per_case_counts:
        for act in all_activities:
            repeat_counts[act].append(counts.get(act, 0))
    return repeat_counts


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
                              higher_level_activity, case_sorted,
                              load_profile=None):
    """
    For every activity in *group*, compute duration stats and collect
    per-instance raw rows for ML training.

    This is shared between 'manual' and pm4py modes — the duration
    fitting and raw-row collection are independent of the process model.

    Parameters
    ----------
    load_profile : LoadProfile or None
        When provided, each raw row also gets ``wip`` (case-level
        Work-In-Progress) and ``ro`` (Resource-Occupation of the row's
        own ``resource_id``) computed at the activity's start time,
        excluding the row's own case/instance. Also adds ``waiting_time``
        — the gap since the previous activity in the same case ended
        (0 for the first activity of a case).

    Returns
    -------
    duration_info : dict  {activity: {duration, duration_std, dist_name,
                           dist_params, n_events, resource_id}}
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

        # ── Resource (machine/object) this activity runs on ───────────
        # Real historical frequency of every resource this activity actually
        # ran on, e.g. {'autoclave_1': 0.55, 'autoclave_2': 0.45} -- used to
        # SAMPLE a resource per simulated instance (see utils/simulation.py's
        # _log_event) instead of hardcoding whichever resource happened to
        # be most frequent for every single simulated case. Hardcoding the
        # mode meant ~all cases that historically ran on the minority
        # resource would be logged against the wrong one by construction.
        # 'resource_id' (the mode) is kept as a backward-compatible fallback
        # for any consumer that doesn't sample from resource_weights.
        resource_id = None
        resource_weights = {}
        if 'object' in activity_data.columns and len(activity_data) > 0:
            _counts = activity_data['object'].value_counts(normalize=True)
            if len(_counts) > 0:
                resource_weights = _counts.to_dict()
                resource_id = _counts.index[0]

        duration_info[activity] = {
            'duration':         duration_median,
            'duration_std':     duration_std,
            'dist_name':        dist_name,
            'dist_params':      dist_params,
            'n_events':         n_events,
            'resource_id':      resource_id,
            'resource_weights': resource_weights,
        }

        # ── Raw rows for ML training ──────────────────────────────────
        # Also accumulate this activity's real waiting-time samples (gap
        # before it starts) so the base petri_net simulation can reproduce
        # idle time between activities instead of butting them back-to-back
        # (see MODEL_BASE_IDLE in utils/simulation.py).
        _waiting_samples = []
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
                waiting_time = 0.0
                if idx >= 1:
                    prev_row = case_acts.iloc[idx - 1]
                    prev_act_1 = prev_row['activity']
                    prev_dur_1 = (
                        (prev_row['timestamp_end'] - prev_row['timestamp_start'])
                        .total_seconds() / 60
                    )
                    # Waiting time: gap since the previous activity in this
                    # case ended (a proxy for queueing/contention delay).
                    gap = (
                        (row_here['timestamp_start'] - prev_row['timestamp_end'])
                        .total_seconds() / 60
                    )
                    waiting_time = max(0.0, gap)
                _waiting_samples.append(waiting_time)
                if idx >= 2:
                    prev_row2 = case_acts.iloc[idx - 2]
                    prev_act_2 = prev_row2['activity']
                    prev_dur_2 = (
                        (prev_row2['timestamp_end'] - prev_row2['timestamp_start'])
                        .total_seconds() / 60
                    )

                # ── WIP / resource-occupation at this activity's start ──
                resource_id = row_here.get('object', None)
                wip, ro = 0.0, 0.0
                if load_profile is not None:
                    t = row_here['timestamp_start'].timestamp()
                    wip = load_profile.wip_at(t, exclude_case_id=case_id)
                    ro = load_profile.ro_at(
                        t, resource_id,
                        exclude_instance_id=row_here.get('_row_uid'),
                    )

                raw_rows.append({
                    'case_id':               case_id,
                    'activity':              activity,
                    'object':                object_name,
                    'object_type':           object_type,
                    'higher_level_activity': higher_level_activity,
                    'duration':              inst_duration,
                    'waiting_time':          waiting_time,
                    'wip':                   wip,
                    'ro':                    ro,
                    'resource_id':           resource_id,
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

        # ── Per-activity waiting-time (idle) empirical sample ─────────
        # Store the REAL waiting-time samples (bounded reservoir) so the base
        # petri_net sim can *bootstrap* an idle gap — i.e. draw an actual
        # observed value at random. Real waiting times are heavily zero-
        # inflated with a long tail; a parametric normal(median,std) badly
        # over-injects idle for such distributions (confirmed: process_2 got
        # 45% simulated idle vs 0.6% real). Bootstrapping reproduces the
        # zero-mass and the tail exactly. waiting_cap is kept only as a hard
        # safety clip. waiting_median stays for reporting/diagnostics.
        if _waiting_samples:
            _ws = np.asarray(_waiting_samples, dtype=float)
            _ws = np.clip(_ws[np.isfinite(_ws)], 0.0, None)
            # Bounded, deterministic reservoir so a high-frequency activity
            # doesn't bloat the stats; preserves the empirical distribution.
            _RESERVOIR_N = 1000
            if len(_ws) > _RESERVOIR_N:
                _ws = np.random.RandomState(0).choice(_ws, size=_RESERVOIR_N, replace=False)
            duration_info[activity]['waiting_samples'] = _ws.tolist()
            duration_info[activity]['waiting_median']  = float(np.median(_ws))
            duration_info[activity]['waiting_cap']     = float(np.percentile(_ws, 99.9)) if len(_ws) >= 10 else float(_ws.max())
        else:
            duration_info[activity]['waiting_samples'] = []
            duration_info[activity]['waiting_median']  = 0.0
            duration_info[activity]['waiting_cap']     = 0.0

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
            'resource_id':           d.get('resource_id'),
            'resource_weights':      d.get('resource_weights', {}),
            # Real idle-gap samples before this activity, bootstrapped by the
            # base sim to reproduce idle time (see MODEL_BASE_IDLE).
            'waiting_samples':       d.get('waiting_samples', []),
            'waiting_median':        d.get('waiting_median', 0.0),
            'waiting_cap':           d.get('waiting_cap', 0.0),
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
        cand_cov  = cand.get('ilp_variant_coverage', 1.0)
        print(
            f"    Mining Petri net with '{algorithm}' "
            f"(candidate {idx}/{len(candidates)}): "
            f"noise={cand_noise}, heur={cand_heur}, ilp_cov={cand_cov}"
        )

        net_i, im_i, fm_i = _mine_petri_net(
            sub_log,
            algorithm,
            noise_threshold=cand_noise,
            heuristic_params=cand_heur,
            ilp_variant_coverage=cand_cov,
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
    decision_weights, max_case_length, median_case_length = _compute_decision_point_weights(
        net, im, fm, case_sorted
    )
    n_dp = len(decision_weights)
    n_end = sum(1 for dp in decision_weights.values() if '__END__' in dp)
    print(f"    Decision-point weights: {n_dp} decision points, "
          f"{n_end} with __END__ probability")
    print(f"    Max case length in training: {max_case_length}")

    # ── Per-activity repeat-count distribution (for simulation-time
    #    loop-quota sampling — see ProcessSimulation._sample_activity_caps) ──
    activity_repeat_counts = _compute_activity_repeat_counts(case_sorted)

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
            'resource_id':           d.get('resource_id'),
            'resource_weights':      d.get('resource_weights', {}),
            # Real idle-gap samples before this activity, bootstrapped by the
            # base sim to reproduce idle time (see MODEL_BASE_IDLE).
            'waiting_samples':       d.get('waiting_samples', []),
            'waiting_median':        d.get('waiting_median', 0.0),
            'waiting_cap':           d.get('waiting_cap', 0.0),
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
            'ilp_variant_coverage': best['cand'].get('ilp_variant_coverage', 1.0),
            'optimization_metrics': best['eval'],
        },
        'bigram_transitions': bigram_transitions,
        'activity_count_transitions': activity_count_transitions,
        'decision_weights': decision_weights,
        'max_case_length': max_case_length,
        'median_case_length': median_case_length,
        'activity_repeat_counts': activity_repeat_counts,
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

    # Ensure higher_level_activity exists; if missing treat the whole log as one group
    if 'higher_level_activity' not in df.columns:
        df = df.copy()
        df['higher_level_activity'] = 'process'

    # Stable global row id (survives groupby/sort/reset_index) so the WIP/RO
    # load profile can exclude a row's own case/instance from its own count.
    df = df.copy()
    df['_row_uid'] = np.arange(len(df))

    # Ground-truth workload profile computed once over the whole log — used
    # to attach 'wip'/'ro' features to every raw training row.
    load_profile = LoadProfile(df)

    # Group by higher_level_activity only — one Petri net per process group
    grouped = df.groupby('higher_level_activity')

    for higher_level_activity, group in grouped:
        object_type = group['object_type'].iloc[0] if 'object_type' in group.columns else 'unknown'
        object_name = higher_level_activity  # synthetic key: whole process represented by its group name
        n_objects = group['object'].nunique() if 'object' in group.columns else 'N/A'
        print(f"\nProcessing: {higher_level_activity} ({n_objects} objects, one whole-process net)")
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
            group, object_name, object_type, higher_level_activity, case_sorted,
            load_profile=load_profile,
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
    """Reduce a 1-D sensor curve to 6 scalar features."""
    start = float(curve[0])
    end   = float(curve[-1])
    return {
        f'{sensor_name}_mean':     float(np.mean(curve)),
        f'{sensor_name}_end':      end,
        f'{sensor_name}_std':      float(np.std(curve)),
        f'{sensor_name}_start':    start,
        f'{sensor_name}_delta':    end - start,
        f'{sensor_name}_integral': float(np.trapz(curve) / max(len(curve), 1)),
    }


# %%
# Modelling the curves

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from dtw import dtw
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


def relevant_objects_for_sensor(df, sensor_col, objects, object_col='object_log'):
    """
    Subset of `objects` for which `sensor_col` actually carries signal
    (not identically zero/NaN) in `df`.

    In multi-object processes (e.g. process_1: destillation, bottling,
    autoclaving_1/2/3, ...), most sensor columns are wired to exactly one
    object and are constant-zero for every other object's rows. Falls back
    to `objects` unchanged if the sensor shows no signal anywhere (avoids
    silently training on nothing).
    """
    if object_col not in df.columns:
        return list(objects)
    relevant = []
    for o in objects:
        vals = df.loc[df[object_col] == o, sensor_col]
        if vals.notna().any() and (vals.fillna(0) != 0).any():
            relevant.append(o)
    return relevant if relevant else list(objects)


def build_sensor_activity_object_combos(df, sensors, activities, objects,
                                         object_col='object_log', activity_col='activity_log'):
    """
    (sensor, activity, object) combos to train/evaluate, restricted per-sensor
    to the objects where that sensor actually carries signal (see
    relevant_objects_for_sensor), and per-object to the activities that
    actually occur for that object — instead of the blind full cross
    product of every sensor x every activity x every object, which wastes
    compute and trains degenerate always-zero pipelines for sensor/object
    pairs that have nothing to do with each other.
    """
    has_object_col = object_col in df.columns
    combos = []
    for s in sensors:
        rel_objects = relevant_objects_for_sensor(df, s, objects, object_col) if has_object_col else objects
        for o in rel_objects:
            if has_object_col:
                acts_for_o = df.loc[df[object_col] == o, activity_col].dropna().unique().tolist()
                acts_for_o = [a for a in acts_for_o if a in activities]
            else:
                acts_for_o = list(activities)
            for a in acts_for_o:
                combos.append((s, a, o))
    return combos


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
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]
    if 'object_log' not in df.columns:
        df['object_log'] = '_all_'
    if 'object_attributes_log' not in df.columns:
        df['object_attributes_log'] = [{} for _ in range(len(df))]
    # '_all_' sentinel means no object dimension — match every row
    if '_all_' in objects:
        df['object_log'] = df['object_log'].fillna('_all_')
    else:
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
        if len(values) >= MIN_CURVE_SAMPLES:
            _raw_attrs = (
                group['object_attributes_log'].iloc[0]
                if not group['object_attributes_log'].empty else {}
            )
            attributes = dict(_raw_attrs) if isinstance(_raw_attrs, dict) else {}
            try:
                _ts = pd.to_datetime(group['timestamp_start_log'].iloc[0])
                if not pd.isnull(_ts):
                    attributes['hour_of_day'] = float(_ts.hour)
                    attributes['day_of_week']  = float(_ts.dayofweek)
            except Exception:
                pass
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
# STEP 1 (variant) — SPLIT CURVES WITH PREVIOUS-ACTIVITY CONTEXT
# =============================================================================

def split_curves_with_prev_activity(df_expanded, variable, activities, objects,
                                    test_size=0.15, random_state=42, verbose=1,
                                    exog_columns=None, include_prev_energy=False):
    """
    Like split_curves() but enriches each curve's 'attributes' with context from
    the immediately preceding activity instance in the same case.

    Default (include_prev_energy=False) adds two features:
        prev_act_name         — name of the previous activity (str; 'none' if first)
        prev_act_duration_min — wall-clock minutes the previous activity ran
                                (timestamp_end_log − timestamp_start_log; 0.0 if
                                no predecessor)

    Both are process-log information: known at activity-start time from the
    event log alone, with no meter reading involved, so they are available in a
    pure what-if simulation where no energy has been observed yet. Combined with
    the ef_* external factors (weather, hour_of_day, day_of_week, …) passed via
    exog_columns, this is the "ML + Ext. Factors" approach as reported.

    include_prev_energy=True additionally adds lagged statistics of the TARGET
    SENSOR's own measured energy during the previous activity:
        prev_act_mean / prev_act_std / prev_act_max / prev_act_end / prev_act_length

    Those are NOT reportable as "external factors". They require a live meter
    feed for the predecessor, which a simulation does not have, and evaluating
    them against the real test series is teacher forcing — optimistic relative
    to a free-running rollout. Kept only for the explicit autoregressive
    experiment (_run_curve_eval_autoregressive_prev_act), which feeds them from
    PREDICTED predecessors. Do not switch this on for paper numbers.

    Also accepts exog_columns exactly like split_curves().
    """
    if verbose:
        print("=" * 80)
        print("STEP 1 — RAW TRAIN/TEST SPLIT + PREVIOUS-ACTIVITY CONTEXT")
        print("=" * 80)

    exog_cols_present = [c for c in (exog_columns or []) if c in df_expanded.columns]

    # Normalise object_log: inject '_all_' sentinel for processes with no object dimension
    if 'object_log' not in df_expanded.columns or ('_all_' in objects and df_expanded['object_log'].isna().all()):
        df_expanded = df_expanded.copy()
        if 'object_log' not in df_expanded.columns:
            df_expanded['object_log'] = '_all_'
        else:
            df_expanded['object_log'] = df_expanded['object_log'].fillna('_all_')

    # ── Build per-case activity timeline from the full df_expanded ────────────
    # Groups every (case, activity, object, ts_start) instance and pre-computes
    # stats on the target variable. Covers ALL activities, not just the target ones,
    # so a target curve can look back at any predecessor in the same case.
    _grp_cols = ['case_id_log', 'activity_log', 'object_log', 'timestamp_start_log']
    _case_timelines = {}   # case_id -> [(ts_start, stats_dict), ...] sorted by ts_start

    # Only the energy-conditioned variant needs the sensor column; the
    # name-only default is built from the event log alone.
    _use_energy = bool(include_prev_energy) and variable in df_expanded.columns
    _has_end    = 'timestamp_end_log' in df_expanded.columns
    if _use_energy or not include_prev_energy:
        _tl_cols = (_grp_cols + ['datetime_energy']
                    + (['timestamp_end_log'] if _has_end else [])
                    + ([variable] if _use_energy else []))
        _tl = df_expanded[[c for c in _tl_cols if c in df_expanded.columns]].copy()
        _tl['timestamp_start_log'] = pd.to_datetime(_tl['timestamp_start_log'])
        _tl['datetime_energy']     = pd.to_datetime(_tl['datetime_energy'])
        if _has_end:
            _tl['timestamp_end_log'] = pd.to_datetime(_tl['timestamp_end_log'])
        _tl = _tl.sort_values(['case_id_log', 'timestamp_start_log', 'datetime_energy'])

        for (_case_id, _act, _obj, _ts_start), _grp in _tl.groupby(_grp_cols):
            _ts    = pd.to_datetime(_ts_start)
            # Wall-clock duration of the predecessor — a log fact (start/end
            # timestamps), not a meter reading. Falls back to the span of the
            # instance's own timestamps when the log has no end column.
            if _has_end:
                _end = _grp['timestamp_end_log'].max()
            else:
                _end = _grp['datetime_energy'].max()
            _dur = (_end - _ts).total_seconds() / 60.0 if pd.notnull(_end) and pd.notnull(_ts) else np.nan
            _stats = {
                'prev_act_name':         _act,
                'prev_act_duration_min': float(_dur) if np.isfinite(_dur) and _dur >= 0 else 0.0,
            }
            if _use_energy:
                _vals = np.asarray(_grp[variable].dropna().values, dtype=float)
                # Use 0.0 as numeric sentinel when stats are undefined; the
                # categorical prev_act_name flag carries the "no predecessor"
                # signal so models trained without NaN-handling still work.
                _stats.update({
                    'prev_act_mean':   float(np.mean(_vals))  if len(_vals) >= 1 else 0.0,
                    'prev_act_std':    float(np.std(_vals))   if len(_vals) >= 2 else 0.0,
                    'prev_act_max':    float(np.max(_vals))   if len(_vals) >= 1 else 0.0,
                    'prev_act_end':    float(_vals[-1])       if len(_vals) >= 1 else 0.0,
                    'prev_act_length': float(len(_vals)),
                })
            _case_timelines.setdefault(_case_id, []).append((_ts, _stats))

    for _cid in _case_timelines:
        _case_timelines[_cid].sort(key=lambda x: x[0])

    def _prev_stats(case_id, ts_start):
        """Return stats of the activity that started immediately before ts_start."""
        prev = None
        for t, s in _case_timelines.get(case_id, []):
            if t < ts_start:
                prev = s
            else:
                break
        if prev is None:
            _none = {'prev_act_name': 'none', 'prev_act_duration_min': 0.0}
            if include_prev_energy:
                _none.update({
                    'prev_act_mean':   0.0,
                    'prev_act_std':    0.0,
                    'prev_act_max':    0.0,
                    'prev_act_end':    0.0,
                    'prev_act_length': 0.0,
                })
            return _none
        return dict(prev)

    # ── Extract target curves — identical filtering to split_curves ────────────
    df = df_expanded.copy()
    keep_cols = ['case_id_log', 'activity_log', 'object_log', 'timestamp_start_log',
                 'datetime_energy', variable, 'object_attributes_log'] + exog_cols_present
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]
    if 'object_log' not in df.columns:
        df['object_log'] = '_all_'
    if 'object_attributes_log' not in df.columns:
        df['object_attributes_log'] = [{} for _ in range(len(df))]
    # '_all_' sentinel means no object dimension — match every row
    if '_all_' in objects:
        df['object_log'] = df['object_log'].fillna('_all_')
    else:
        df = df[df['object_log'].isin(objects)]
    df = df[df['activity_log'].isin(activities)]
    df['timestamp_start_log'] = pd.to_datetime(df['timestamp_start_log'])
    df['datetime_energy']     = pd.to_datetime(df['datetime_energy'])
    df = df.sort_values(['case_id_log', 'timestamp_start_log', 'datetime_energy'])

    df['activity_instance_id'] = df.groupby(
        ['case_id_log', 'object_log', 'activity_log', 'timestamp_start_log']
    ).ngroup()

    curves_data = []
    for instance_id, group in df.groupby('activity_instance_id'):
        group  = group.sort_values('datetime_energy').reset_index(drop=True)
        values = np.asarray(group[variable].dropna().values).squeeze()
        if values.ndim != 1 or len(values) < MIN_CURVE_SAMPLES:
            continue

        case_id  = group['case_id_log'].iloc[0]
        ts_start = pd.to_datetime(group['timestamp_start_log'].iloc[0])

        _raw_attrs = (group['object_attributes_log'].iloc[0]
                      if 'object_attributes_log' in group.columns
                      and not group['object_attributes_log'].empty else {})
        attributes = dict(_raw_attrs) if isinstance(_raw_attrs, dict) else {}
        try:
            if not pd.isnull(ts_start):
                attributes['hour_of_day'] = float(ts_start.hour)
                attributes['day_of_week']  = float(ts_start.dayofweek)
        except Exception:
            pass

        attributes.update(_prev_stats(case_id, ts_start))

        entry = {
            'instance_id':     instance_id,
            'activity':        group['activity_log'].iloc[0],
            'original_values': values,
            'original_length': len(values),
            'attributes':      attributes,
        }
        if exog_cols_present:
            entry['exog_values'] = {col: group[col].values for col in exog_cols_present}
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

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _align_curve_with_dtw(query, reference, min_length_ratio=0.2, return_counts=False):
    """
    DTW-align a single query curve onto the reference grid (train only).

    return_counts=True additionally returns, per canonical position, how many raw
    query points the DTW path folded into it. That number is the position's real
    weight in raw space: a canonical point built from 40 raw samples drives 40
    residuals when the prediction is decoded back, one built from a single sample
    drives one — yet the regression treats every canonical row as one equally
    weighted observation. The counts are what the *_wcounts / *_wmetric approaches
    feed to sample_weight to close that gap. Default False keeps the original
    single-array return for every existing caller.

    Falls back to plain linear interpolation when the query is much shorter
    than the reference. DTW's monotonic path has to hold a handful of query
    points constant across most of the reference's index range when the
    query is this short (length 11 onto a length-197 reference collapses to
    essentially ONE repeated value for ~190 of the 197 positions) -- a
    degenerate step function that, used directly as a regression target,
    teaches the model contradictory locally-flat values at whatever position
    each such curve happens to land on. Root-caused 2026-07-22 on
    process_4_1's main activity (instance durations 2..1440 min,
    fixed_length~197 -- curves this short are common there). Linear
    interpolation gives a smooth, honest (if under-resolved) target instead.
    """
    if len(query) < min_length_ratio * len(reference):
        lin = np.interp(
            np.linspace(0, 1, len(reference)),
            np.linspace(0, 1, len(query)),
            query,
        )
        if return_counts:
            # Linear-interp fallback: the query is spread evenly over the grid,
            # so every canonical position represents the same len(query)/len(ref)
            # share of raw samples. Uniform weights, normalised the same way as
            # the DTW branch so the two are on one scale.
            return lin, np.full(len(reference), len(query) / len(reference))
        return lin
    alignment = dtw(query, reference, keep_internals=True)
    aligned = np.zeros(len(reference))
    counts  = np.zeros(len(reference))
    for qi, ri in zip(alignment.index1, alignment.index2):
        aligned[ri] += query[qi]
        counts[ri]  += 1
    counts  = np.where(counts == 0, 1, counts)
    aligned /= counts
    if return_counts:
        return aligned, counts
    return aligned


class ExternalFactorWindows:
    """
    Mean of each ef_* series over a forward window [t, t + w), shared by the
    duration-model training code and the simulator so both compute the feature
    the same way.

    Why a *fixed* per-activity window rather than the activity's real duration:
    the window length would otherwise be the regression target itself, leaking
    it into the feature — and at generation time the duration is exactly what is
    being predicted, so the true window is not knowable. Using the activity's
    median duration as w keeps the feature identical in training and simulation
    and leak-free in both.

    ef_* are smooth interpolated exogenous series (weather), so a window mean is
    well behaved even where samples are sparse; windows containing no samples
    fall back to the column's global mean.
    """

    def __init__(self, expanded_df, ef_cols, time_col='datetime_energy'):
        self.ef_cols = [c for c in ef_cols if c in expanded_df.columns]
        self.means = {}
        self._t = np.empty(0)
        self._cum = {}
        if not self.ef_cols:
            return
        d = (expanded_df[[time_col] + self.ef_cols]
             .dropna(subset=[time_col])
             .sort_values(time_col))
        if d.empty:
            return
        self._t = d[time_col].astype('int64').to_numpy() / 1e9      # epoch seconds
        for c in self.ef_cols:
            v = pd.to_numeric(d[c], errors='coerce').ffill().bfill()
            self.means[c] = float(v.mean()) if v.notna().any() else 0.0
            arr = v.fillna(self.means[c]).to_numpy(dtype=float)
            self._cum[c] = np.concatenate([[0.0], np.cumsum(arr)])

    @property
    def feature_names(self):
        """Column names this produces, in a stable order."""
        return [f'feat_{c}' for c in self.ef_cols]

    def window_means(self, start_epoch, window_seconds):
        """{feat_<ef_col>: mean over [start, start+window)}; global mean if empty."""
        if not self.ef_cols or self._t.size == 0:
            return {}
        w = max(float(window_seconds or 0.0), 0.0)
        i0 = np.searchsorted(self._t, float(start_epoch), 'left')
        i1 = np.searchsorted(self._t, float(start_epoch) + w, 'right')
        n = i1 - i0
        if n <= 0:
            return {f'feat_{c}': self.means[c] for c in self.ef_cols}
        return {f'feat_{c}': (self._cum[c][i1] - self._cum[c][i0]) / n
                for c in self.ef_cols}


def _make_curve_models():
    """
    Candidate regressors competed for every curve approach, per
    (sensor, activity, object). The best is kept by validation MAE, so adding a
    candidate costs training time but cannot make the selection worse.

    Single source of truth — imported by 02_modelling.py for reporting and used by
    _train_curve_only_worker for training. Every name here needs a branch in
    _curve_model_trial_params, or Optuna spends all its trials on one default
    configuration.

    Ridge rather than plain OLS is the regularised member: the curve design
    matrix is one-hot expanded over activity plus the categorical attributes,
    so it is high-dimensional and collinear. LinearRegression is kept alongside
    it as the unpenalised reference. XGBoost is the gradient-boosting member
    (plain GradientBoostingRegressor was dropped as redundant with it).
    """
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor
    all_models = {
        'Linear Regression': LinearRegression,
        'Ridge':             Ridge,
        'Random Forest':     RandomForestRegressor,
        'XGBoost':           XGBRegressor,
        'Hist Gradient Boosting': HistGradientBoostingRegressor,
        'MLP':               MLPRegressor,
    }

    # Subset via PIPELINE_CURVE_MODELS (comma-separated), set by 01_pipeline.py's
    # 'curve_models' key. Accepts the display names above or the short aliases
    # below, case-insensitively. Unset -> all candidates compete.
    requested = _os.environ.get('PIPELINE_CURVE_MODELS')
    if not requested:
        return all_models

    aliases = {'linear': 'Linear Regression', 'linear_regression': 'Linear Regression',
               'ols': 'Linear Regression', 'ridge': 'Ridge',
               'rf': 'Random Forest', 'random_forest': 'Random Forest',
               'xgb': 'XGBoost', 'xgboost': 'XGBoost',
               'hgb': 'Hist Gradient Boosting', 'histgb': 'Hist Gradient Boosting',
               'hist_gradient_boosting': 'Hist Gradient Boosting',
               'mlp': 'MLP', 'ffnn': 'MLP'}
    canonical = {k.lower(): k for k in all_models}

    picked, unknown = [], []
    for raw in requested.split(','):
        key = raw.strip().lower().replace(' ', '_')
        name = canonical.get(raw.strip().lower()) or aliases.get(key)
        if name is None:
            unknown.append(raw.strip())
        elif name not in picked:
            picked.append(name)

    if unknown:
        print(f"[sim_extractor] WARNING: PIPELINE_CURVE_MODELS has unknown names "
              f"{unknown} — valid: {list(all_models)} (or aliases {sorted(aliases)}). Ignoring those.")
    if not picked:
        print(f"[sim_extractor] WARNING: PIPELINE_CURVE_MODELS={requested!r} selected no "
              f"valid model — falling back to all candidates.")
        return all_models
    return {name: all_models[name] for name in picked}


def _curve_model_trial_params(trial, name, random_state, n_jobs=1):
    """
    Optuna trial params for one curve regressor: the searched hyper-parameters
    plus the fixed absolute-error loss, so every trial is scored on the same
    objective the metrics use. See CURVE_MODEL_LOSS.
    """
    params = _curve_model_search_space(trial, name, random_state, n_jobs)
    params.update(_curve_model_loss_kwargs(name))
    return params


def _curve_model_search_space(trial, name, random_state, n_jobs=1):
    """
    Optuna search space for one curve regressor, keyed by its display name in
    _CURVE_MODELS (02_modelling.py).

    Centralised because the same chain was duplicated across every
    build_and_train_pipeline_* variant: a model missing from here silently falls
    through to `{}`, which trains it at library defaults for *every* trial — the
    search burns n_trials evaluating one identical configuration. Add a model to
    _CURVE_MODELS and it must be added here too.
    """
    if name == 'Gradient Boosting':
        return {
            'n_estimators':  trial.suggest_int('n_estimators', 50, 300),
            'max_depth':     trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample':     trial.suggest_float('subsample', 0.5, 1.0),
            'random_state':  random_state,
        }
    if name == 'Random Forest':
        return {
            'n_estimators':      trial.suggest_int('n_estimators', 50, 300),
            'max_depth':         trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'random_state':      random_state,
            'n_jobs':            n_jobs,
        }
    if name == 'XGBoost':
        return {
            'n_estimators':     trial.suggest_int('n_estimators', 50, 300),
            'max_depth':        trial.suggest_int('max_depth', 2, 8),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'random_state':     random_state,
            'n_jobs':           n_jobs,
            'verbosity':        0,
        }
    if name == 'Hist Gradient Boosting':
        # Histogram-based GBM: far faster than GradientBoostingRegressor at
        # these row counts (one row per canonical position per curve) and
        # regularised via leaf count + L2 rather than tree depth.
        return {
            'max_iter':          trial.suggest_int('max_iter', 100, 500),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_leaf_nodes':    trial.suggest_int('max_leaf_nodes', 15, 127),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 5, 50),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-6, 1.0, log=True),
            'random_state':      random_state,
        }
    if name == 'Ridge':
        # The curve feature matrix is one-hot expanded (activity + categorical
        # attributes), so it is high-dimensional and collinear — the penalty
        # matters far more here than for the tree models.
        return {
            'alpha':        trial.suggest_float('alpha', 1e-3, 1e3, log=True),
            'random_state': random_state,
        }
    if name == 'MLP':
        # hidden_layer_sizes is suggested as a string key rather than a tuple:
        # Optuna only stores str/int/float/bool categoricals cleanly, and
        # _curve_model_finalise_params converts it back to a tuple.
        arch = trial.suggest_categorical('hidden_layers', ['50', '100', '50_50', '100_50'])
        return {
            'hidden_layer_sizes': tuple(int(x) for x in arch.split('_')),
            'activation':         trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha':              trial.suggest_float('alpha', 1e-6, 1e-1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
            'max_iter':           500,
            'random_state':       random_state,
        }
    # 'Linear Regression' lands here on purpose: OLS has no hyperparameters, so
    # its trials are all identical. Harmless (the fit is cheap) but it means
    # n_trials is effectively wasted on it — unlike an unlisted model, which
    # would be an oversight.
    return {}


# ── Training loss for the curve regressors ───────────────────────────────────
# Every reported curve metric is L1 (sMAE and WAPE are both mean-absolute), and
# model selection uses validation MAE — but the regressors ran at library
# defaults, i.e. squared error, which estimates the conditional MEAN. On spiky
# energy curves the mean is a poor L1 predictor, which is why the plain
# 'median_activity_sensor' baseline (a conditional MEDIAN) beat the learned
# 'ml_only'. Fitting absolute error makes the models estimate the conditional
# MEDIAN instead, so a learned model becomes a strict generalisation of that
# baseline: same estimator when the features say nothing, better when they do.
# Combined with the residual target it also makes "do no harm" the default —
# predicting a zero residual reproduces the median anchor exactly.
#
# 'squared_error' restores the previous behaviour for ablation.
CURVE_MODEL_LOSS = _os_seed.environ.get('PIPELINE_CURVE_LOSS', 'absolute_error').lower()

_CURVE_L1_KWARGS = {
    'Gradient Boosting':      {'loss': 'absolute_error'},
    'Hist Gradient Boosting': {'loss': 'absolute_error'},
    'XGBoost':                {'objective': 'reg:absoluteerror'},
    # LinearRegression / Ridge / MLP have no absolute-error option in sklearn;
    # they keep their default objective and are simply not median estimators.
}

# Random Forest is deliberately NOT in the table above. criterion='absolute_error'
# makes sklearn's split search O(n log n) instead of O(n) with a large constant —
# measured at >7 min for a single (sensor, activity, object) leaf here, against
# ~800 leaves in a full run. Leaving RF on squared error costs nothing in
# correctness: every candidate is selected by validation MAE, so RF simply
# competes as a mean-estimator and loses to the L1 models where that matters.
# Set PIPELINE_CURVE_L1_RANDOM_FOREST=true to include it anyway (expect a very
# slow run), or drop 'Random Forest' from _CURVE_MODELS to skip it entirely.
CURVE_L1_RANDOM_FOREST = _os_seed.environ.get(
    'PIPELINE_CURVE_L1_RANDOM_FOREST', 'false').lower() == 'true'
if CURVE_L1_RANDOM_FOREST:
    _CURVE_L1_KWARGS['Random Forest'] = {'criterion': 'absolute_error'}


def _fit_with_weight(model, X, y, sample_weight):
    """
    model.fit with optional sample weights.

    Every candidate in _make_curve_models accepts sample_weight, but a custom
    model_class passed by a caller might not — fall back to an unweighted fit
    rather than killing the whole leaf, and say so, because silently dropping
    the weights would make a *_wcounts / *_wmetric run look like it had been
    applied when it had not.
    """
    if sample_weight is None:
        return model.fit(X, y)
    try:
        return model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        warnings.warn(f"{type(model).__name__} does not accept sample_weight — "
                      f"fitting UNWEIGHTED; this leaf is not row-weighted.",
                      RuntimeWarning)
        return model.fit(X, y)


def _curve_model_loss_kwargs(name, loss=None):
    """
    Constructor kwargs that switch one curve regressor to absolute-error loss.
    Empty dict for 'squared_error' (library defaults) and for models with no L1
    option. See CURVE_MODEL_LOSS.
    """
    loss = CURVE_MODEL_LOSS if loss is None else str(loss).lower()
    if loss in ('squared_error', 'l2', 'mse'):
        return {}
    return dict(_CURVE_L1_KWARGS.get(name, {}))


def _curve_model_finalise_params(name, best_params, random_state, n_jobs=1):
    """
    Re-attach the non-searched constructor kwargs Optuna does not return in
    study.best_params (it only reports suggested values) — including the
    absolute-error loss, which is fixed rather than searched.
    """
    best_params = dict(best_params)
    best_params.update(_curve_model_loss_kwargs(name))
    if name == 'MLP':
        # 'hidden_layers' is the suggest-name, not an MLPRegressor kwarg.
        arch = best_params.pop('hidden_layers', '100')
        best_params['hidden_layer_sizes'] = tuple(int(x) for x in arch.split('_'))
        best_params['max_iter'] = 500
    # LinearRegression takes no random_state, so it is deliberately absent here.
    if name in ('Gradient Boosting', 'Random Forest', 'XGBoost',
                'Hist Gradient Boosting', 'Ridge', 'MLP'):
        best_params['random_state'] = random_state
    if name in ('Random Forest', 'XGBoost'):
        best_params['n_jobs'] = n_jobs
    if name == 'XGBoost':
        best_params['verbosity'] = 0
    return best_params


def _robust_dtw_barycenter(resampled_curves, barycenter_size, n_iterations=3,
                            trim_fraction=0.1):
    """
    Duration-robust drop-in replacement for tslearn's dtw_barycenter_averaging.

    DBA refines its barycenter by repeatedly DTW-aligning every curve to the
    current estimate and MEAN-averaging the aligned points. That mean is not
    robust when instance durations vary wildly relative to `barycenter_size`
    (e.g. a 2-minute curve linearly stretched onto a 200-point canonical grid
    injects implausible values at nearly every position). A handful of such
    curves is enough to drag the whole barycenter shape off the true signal
    -- root-caused 2026-07-22 on process_4_1's main activity
    (durations 2..1440 min), where DBA produced an artificial ~30% dip that a
    plain coordinate-wise median of the same curves did not show, and every
    approach sharing this barycenter (ml_dtw, ml_external, seq2seq*) inherited
    the distortion and lost to the naive median-curve baseline as a result.

    Same DTW re-alignment DBA does, but each round aggregates with a trimmed
    mean instead of a mean, so shape alignment still improves iteration over
    iteration while a minority of badly-resampled curves can no longer pull
    the estimate off the true shape.

    Parameters
    ----------
    resampled_curves : np.ndarray, shape (n_curves, barycenter_size, 1)
                        curves already linearly resampled onto the canonical
                        grid -- the same input dtw_barycenter_averaging takes.
    barycenter_size   : int  (unused directly; kept for call-site symmetry
                        with dtw_barycenter_averaging -- length is implicit
                        in resampled_curves' second dimension)
    n_iterations      : int    -- DTW re-alignment rounds (DBA-style refinement)
    trim_fraction     : float  -- fraction trimmed from EACH tail before
                        averaging at every position (0.1 -> 20% trimmed mean)

    Returns
    -------
    barycenter : np.ndarray, shape (barycenter_size, 1) -- same shape as
                 tslearn's dtw_barycenter_averaging, so every call site can
                 swap the two without touching downstream code.
    """
    flat = resampled_curves[:, :, 0]              # (n_curves, barycenter_size)
    reference = np.median(flat, axis=0)            # robust starting point

    for _ in range(n_iterations):
        aligned = np.array([_align_curve_with_dtw(curve, reference) for curve in flat])
        reference = scipy_stats.trim_mean(aligned, trim_fraction, axis=0)

    return reference[:, np.newaxis]


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


def _stamp_train_sensor_median(result, curves):
    """
    Attach the leaf's training median of the target sensor to EVERY pipeline in
    `result`, including approaches that don't use it as a model feature
    (baseline/median/seq2seq). Training curves only — usable as a level anchor,
    a sanity check, or a fallback when a pipeline cannot predict.
    Never overwrites a value a builder already set.
    """
    raw = [np.asarray(c['original_values'], dtype=float) for c in curves
           if len(np.asarray(c['original_values'])) > 0]
    leaf_median = float(np.median(np.concatenate(raw))) if raw else 0.0
    for _p in result.values():
        if isinstance(_p, dict):
            _p.setdefault('train_sensor_median', leaf_median)
    return result


def _training_median_stats(train_curves, fixed_length, value_key='resampled_values'):
    """
    Training-only median statistics for one leaf pipeline (sensor × activity × object).

    Returns (median_curve, sensor_median):
        median_curve  — np.ndarray[fixed_length], pointwise median of the
                        DTW-aligned TRAINING curves. This is the typical shape
                        AND level of the sensor at each canonical position, so a
                        model conditioned on it only has to learn the deviation.
        sensor_median — scalar median over every training sample of the sensor.

    Computed from train_curves only (same set the DBA barycenter comes from), so
    it carries no test information. At inference the stored array is replayed
    positionwise — nothing is recomputed from the data being predicted.
    """
    mats = [
        np.asarray(c[value_key], dtype=float)
        for c in train_curves
        if value_key in c and len(np.asarray(c[value_key])) == fixed_length
    ]
    median_curve = (np.median(np.vstack(mats), axis=0) if mats
                    else np.zeros(fixed_length, dtype=float))
    raw = [np.asarray(c['original_values'], dtype=float) for c in train_curves
           if len(np.asarray(c['original_values'])) > 0]
    sensor_median = float(np.median(np.concatenate(raw))) if raw else 0.0
    return median_curve, sensor_median


# Feature names injected from the previous activity's training energy level.
PREV_ACT_ENERGY_FEATURES = ('prev_act_energy_median',
                            'prev_act_energy_end',
                            'prev_act_energy_max')


def build_prev_activity_energy_map(df_expanded, variables):
    """
    Per-(sensor, activity) TRAINING energy level, for use as previous-activity
    context.

    'ML + Ext. Factors' tells the model WHICH activity ran before the one being
    predicted (prev_act_name, one-hot) but nothing about that activity's energy
    behaviour, so the model has to infer "the predecessor was a high-power step"
    from the categorical alone — which it can only do for predecessors seen often
    enough in this leaf's training curves, and not at all for a predecessor that
    never appears there (an unseen level silently collapses into the
    drop_first=True reference category at inference). This map turns that sparse
    categorical into dense numbers.

    Must be built from the TRAINING frame only and stored in the pipeline: the
    lookup at predict time is then (test-time-known activity name) x (train-only
    level), so nothing about the test series enters the features. This is what
    makes it different from the include_prev_energy=True lagged features, which
    need a live meter feed for the predecessor and are not simulation-available.

    Parameters
    ----------
    df_expanded : pd.DataFrame  — training rows, ALL activities (not just the
                  target ones) so a target curve can look back at any predecessor
    variables   : list[str]     — sensor columns to build the map for

    Returns
    -------
    {sensor: {activity: {feature: float}, '__global__': {feature: float}}}
        '__global__' is the across-activity fallback used when the predecessor's
        activity never appears in training (and for 'none', i.e. first activity
        in a case).
    """
    _vars = [v for v in variables if v in df_expanded.columns]
    if not _vars:
        return {}

    df = df_expanded
    if 'object_log' not in df.columns:
        df = df.copy()
        df['object_log'] = '_all_'

    _inst_keys = ['case_id_log', 'activity_log', 'object_log', 'timestamp_start_log']
    if any(k not in df.columns for k in _inst_keys):
        return {}

    # dict.fromkeys de-duplicates while preserving order: a sensor column can
    # coincide with a key column, and df[[dup, dup]] breaks sort_values.
    cols = list(dict.fromkeys(
        _inst_keys + _vars + (['datetime_energy'] if 'datetime_energy' in df.columns else [])))
    _vars = [v for v in _vars if v not in _inst_keys and v != 'datetime_energy']
    if not _vars:
        return {}
    df = df[cols].copy()
    df['object_log'] = df['object_log'].fillna('_all_')
    if 'datetime_energy' in df.columns:
        # 'last' below means "value at the END of the activity", so the rows have
        # to be in chronological order within each instance.
        df = df.sort_values(_inst_keys + ['datetime_energy'])

    # One row per activity instance, then the median instance per activity.
    # 'last' = value at the END of the activity (rows were sorted above); pandas
    # groupby aggregations skip NaN, so sparse sensors still yield a level.
    _aggs = ('mean', 'last', 'max')
    per_instance = df.groupby(_inst_keys, sort=False)[_vars].agg(list(_aggs))
    # Flatten the MultiIndex columns — reset_index() would pad the key columns
    # with an empty second level and make flat/tuple selection inconsistent.
    per_instance.columns = [f'{var}||{agg}' for var, agg in per_instance.columns]
    per_instance = per_instance.reset_index()

    out = {}
    for var in _vars:
        stats = {}
        _cols = {feat: f'{var}||{agg}'
                 for feat, agg in zip(PREV_ACT_ENERGY_FEATURES, _aggs)}
        _sub = per_instance[['activity_log'] + list(_cols.values())].dropna(
            subset=list(_cols.values()), how='all')
        if _sub.empty:
            continue
        for act, grp in _sub.groupby('activity_log', sort=False):
            stats[str(act)] = {
                feat: float(grp[col].median())
                for feat, col in _cols.items()
                if pd.notna(grp[col].median())
            }
        _global = {feat: float(_sub[col].median())
                   for feat, col in _cols.items()
                   if pd.notna(_sub[col].median())}
        # Drop activities the sensor never recorded during — they would otherwise
        # be a NaN row that the model cannot distinguish from "no predecessor".
        stats = {a: s for a, s in stats.items() if len(s) == len(PREV_ACT_ENERGY_FEATURES)}
        if _global:
            stats['__global__'] = _global
        if stats:
            out[var] = stats
    return out


def _inject_prev_act_energy(attributes, energy_map):
    """
    Add PREV_ACT_ENERGY_FEATURES to `attributes` by looking `prev_act_name` up in
    `energy_map` (one sensor's sub-map from build_prev_activity_energy_map).

    Returns a NEW dict — never mutates the caller's attributes, so a test curve
    can be predicted by several pipelines without them contaminating each other.
    No-op when the map is empty, which keeps pipelines trained before this
    feature existed working unchanged.
    """
    if not energy_map:
        return attributes
    attrs = dict(attributes)
    prev = str(attrs.get('prev_act_name', 'none'))
    # 'none' (first activity in the case) has no predecessor level — fall back to
    # the across-activity median rather than 0, which would read as "the previous
    # step drew no power" and is a different, wrong claim.
    stats = energy_map.get(prev) or energy_map.get('__global__') or {}
    for feat in PREV_ACT_ENERGY_FEATURES:
        if feat in stats:
            attrs[feat] = float(stats[feat])
    return attrs


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
    fixed_length=None,
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
    2e. Best model      — selected by validation MAE

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
        'val_mae'          — float
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

    if fixed_length is None:
        fixed_length = max(2, int(round(np.median([len(c['original_values']) for c in train_curves]))))
    resampled_for_dba = np.array([
        np.interp(
            np.linspace(0, 1, fixed_length),
            np.linspace(0, 1, len(c['original_values'])),
            c['original_values']
        )
        for c in train_curves
    ])[:, :, np.newaxis]

    dba_barycenter  = _robust_dtw_barycenter(resampled_for_dba, barycenter_size=fixed_length)
    # Restore the on/off duty cycle DBA averages away (see
    # _zero_calibrate_barycenter); no-op for non-intermittent sensors.
    reference_curve = _zero_calibrate_barycenter(dba_barycenter[:, 0], resampled_for_dba)

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

    # Training-median anchor: the sensor's typical level at this canonical
    # position, from train curves only (see _training_median_stats).
    train_median_curve, train_sensor_median = _training_median_stats(train_curves, fixed_length)
    df_reg['train_median_at_pos'] = train_median_curve[df_reg['position_idx'].to_numpy()]

    # Dummies defined on training data only. Val instances are a subset of
    # train curves so they share the same category levels — no unseen levels
    # can appear in the val set.
    categorical_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_all = df_reg[['position_idx', 'relative_pos', 'curve_length', 'train_median_at_pos']].copy()
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

    # Scale numeric features (position, curve length, training-median anchor,
    # and numeric attributes) using train-only statistics to avoid leakage.
    numeric_feature_cols = ['position_idx', 'relative_pos', 'curve_length',
                            'train_median_at_pos'] + [
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
                params = _curve_model_trial_params(trial, name, random_state, n_jobs)
                m = model_class(**params)
                m.fit(X_train, y_train)
                return -mean_absolute_error(y_val, m.predict(X_val))

            sampler = optuna.samplers.TPESampler(seed=random_state)
            study = optuna.create_study(direction='maximize', sampler=sampler)
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            best_params = _curve_model_finalise_params(name, best_params, random_state, n_jobs)
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
                # Untuned path: the loss is not searched, so bake it in here too
                # or this branch silently stays on squared error.
                _lk = _curve_model_loss_kwargs(name)
                try:
                    model = model_class(random_state=random_state, **_lk)
                except TypeError:
                    try:
                        model = model_class(**_lk)
                    except TypeError:
                        model = model_class()

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred   = model.predict(X_val)
        train_mae  = mean_absolute_error(y_train, train_pred)
        val_mae    = mean_absolute_error(y_val,   val_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse   = np.sqrt(mean_squared_error(y_val,   val_pred))

        if verbose:
            print(f"     Train  MAE={train_mae:.4f}  RMSE={train_rmse:.4f}")
            print(f"     Val    MAE={val_mae:.4f}  RMSE={val_rmse:.4f}")

        all_results[name] = {
            'model':      model,
            'train_mae':  train_mae,
            'val_mae':    val_mae,
            'train_rmse': train_rmse,
            'val_rmse':   val_rmse,
        }

    # ------------------------------------------------------------------
    # 2e. Select best model by val MAE
    # ------------------------------------------------------------------
    best_name  = min(all_results, key=lambda k: all_results[k]['val_mae'])
    best_model = all_results[best_name]['model']

    if verbose:
        print(f"\n Best model : {best_name}  "
              f"(val MAE={all_results[best_name]['val_mae']:.4f})")

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
        'train_median_curve':  train_median_curve,
        'train_sensor_median': train_sensor_median,
        'val_mae':         all_results[best_name]['val_mae'],
        'all_results':     all_results,
        # Explicit tag (previously relied on _dispatch_predict's implicit
        # 'baseline' default, which collided with the *actual* trivial
        # baseline once one was added -- see build_and_train_pipeline_median).
        'approach':        'ml_dtw',
    }

    # 2f. Median floor — keep the winner only if it beats this leaf's own median

    # Realism-based re-selection — see CURVE_SELECT_BY_REALISM. Scored through
    # this pipeline's OWN predictor on the held-out curves, so the decode and the
    # real (never DTW-aligned) values are what decide the winner.
    if CURVE_SELECT_BY_REALISM:
        _vc = [c for c in train_curves if c['instance_id'] in set(val_inst)]
        _select_by_realism(all_results, pipeline, _vc,
                           lambda rv, c: predict_raw_curve(rv, c['activity'], c['attributes'], pipeline),
                           best_name, verbose=verbose, label=' ml_dtw')
    # curve on the same held-out instances. See _attach_median_floor.
    _attach_median_floor(
        pipeline, train_curves, val_inst,
        lambda rv, c: predict_raw_curve(rv, c['activity'], c['attributes'], pipeline),
        label=f' {variable}', verbose=verbose,
    )

    return pipeline


def build_and_train_pipeline_median(train_curves, variable, fixed_length=None,
                                    verbose=1, **_ignored_hp_kwargs):
    """
    A single MEDIAN LEVEL across a set of training instances -- one number, not
    a profile. Every value of every training curve is pooled into one vector and
    the median of that vector is the estimate; at predict time it is emitted as a
    horizontal line for the target duration. No DBA, no DTW alignment, no
    regression model, no attribute conditioning, and deliberately no shape: this
    is the naive floor, so it must not have a ramp-up/plateau/ramp-down of its
    own to be credited for.

    (It used to take a POINTWISE median of length-normalised curves, which is a
    full profile -- a shape-carrying estimator masquerading as a floor.)

    Used for BOTH naive-floor approaches, which differ only in which curves
    are pooled before the median is taken (the caller sets the 'approach' tag):
      * 'median_activity_sensor' ("Median per Activity & Sensor") -- called by
        _train_curve_only_worker with just this (sensor, activity, object)'s
        curves, so one level per (sensor, activity, object).
      * 'baseline' ("Baseline") -- called from 02_modelling.py's curve-only
        section with EVERY curve of the sensor pooled across all its
        activities/objects, so one level per sensor (the coarser floor).

    **_ignored_hp_kwargs absorbs val_size/models/optimize_hyperparams/
    n_trials/n_jobs so it can be called with the exact same signature as
    every other approach builder in _train_curve_only_worker, even though
    none of them apply here (there is no model to validate or tune).

    Returns a minimal pipeline dict: {'median_level', 'reference_curve',
    'fixed_length', 'approach'}. 'median_level' is the estimate; the flat
    'reference_curve' of length `fixed_length` is kept only because consumers
    read it (02_modelling.py's pipeline dicts, simulation.py's presence check,
    _build_gap_fill_curves). The default 'approach' tag is 'baseline' (the
    per-sensor use); the per-combo worker overrides it to
    'median_activity_sensor'.
    """
    if fixed_length is None:
        fixed_length = max(2, int(round(np.median([len(c['original_values']) for c in train_curves]))))
    vals = np.concatenate([
        np.asarray(c['original_values'], dtype=float).ravel()
        for c in train_curves
    ]) if train_curves else np.array([], dtype=float)
    vals = vals[np.isfinite(vals)]
    level = float(np.median(vals)) if vals.size else 0.0

    if verbose:
        print(f"  [baseline/median] {len(train_curves)} curves -> "
              f"median level={level:.4f} ({vals.size} pooled values), "
              f"fixed_length={fixed_length}")

    return {
        'median_level':    level,
        'reference_curve': np.full(fixed_length, level, dtype=float),
        'fixed_length':    fixed_length,
        'approach':        'baseline',
    }


# =============================================================================
# PER-INSTANCE FEATURE FRAME + DTW HELPERS (shared by the step-DTW approach)
# =============================================================================
# _exemplar_feature_frame, WARP_KNOTS, _dtw_distance_matrix and
# _warp_knots_from_path originate from the retired exemplar/exemplar_dtw/
# ml_cluster_dtw family but are live dependencies of
# build_and_train_pipeline_step_dtw (medoid selection, segment targets) —
# do not remove them with the rest of that family.

def _exemplar_feature_frame(curves, columns=None, exog_cols=None):
    """
    One row per curve: its attributes, its length, and one summary per ef_*
    external factor, one-hot encoded. `columns` replays the training layout so
    predict-time frames line up.

    Each ef_* series is reduced to its MEAN over the activity's own window
    (feat_<col>). The other approaches feed ef_* in per canonical position, which
    only makes sense when the target is a per-position value; here both targets
    are per-instance (which shape, what level), so the feature has to be
    per-instance too. A window mean is the natural summary — these are smooth
    interpolated weather-style series, so it loses very little.
    """
    exog_cols = list(exog_cols or [])
    rows = []
    for c in curves:
        r = {str(k): v for k, v in (c.get('attributes') or {}).items()
             if not str(k).startswith('_')}
        r['curve_length'] = len(c['original_values'])
        ev = c.get('exog_values') or {}
        for col in exog_cols:
            vals = np.asarray(ev.get(col, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            r[f'feat_{col}'] = float(vals.mean()) if vals.size else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)
    df = pd.get_dummies(df)
    if columns is not None:
        df = df.reindex(columns=columns, fill_value=0)
    # HistGradientBoosting* handle NaN natively, so an ef_* series that is empty
    # for one instance stays missing rather than being imputed to a wrong level.
    return df.astype(float)


WARP_KNOTS = (0.25, 0.50, 0.75)     # interior knots of gamma; 0 and 1 are pinned


def _dtw_distance_matrix(shapes, max_curves=60, random_state=42):
    """
    Symmetric pairwise DTW distance matrix over shape-normalised curves.

    O(n^2) DTW calls, so above `max_curves` a deterministic random subset is used
    for the clustering and the rest are assigned to the resulting medoids
    afterwards. Without the cap a single 300-curve leaf would dominate the run.
    Returns (D, idx) where idx are the rows of `shapes` that D refers to.
    """
    n = len(shapes)
    idx = np.arange(n)
    if n > max_curves:
        idx = np.sort(np.random.default_rng(random_state).choice(n, max_curves,
                                                                 replace=False))
    sub = shapes[idx]
    m = len(sub)
    D = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            try:
                d = float(dtw(sub[i], sub[j], distance_only=True).normalizedDistance)
            except Exception:
                d = float(np.mean(np.abs(sub[i] - sub[j])))
            D[i, j] = D[j, i] = d
    return D, idx


def _warp_knots_from_path(query, reference, knots=WARP_KNOTS):
    """
    DTW-align `query` to `reference` and reduce the path to gamma evaluated at
    `knots`: gamma(u) = the normalised REFERENCE position matched to normalised
    QUERY position u. Identity (gamma(u) = u) means the curve runs in step with
    the medoid; gamma(0.5) < 0.5 means it reaches the medoid's midpoint late.
    Falls back to the identity when the alignment cannot be computed.
    """
    try:
        al = dtw(query, reference, keep_internals=True)
        qi = np.asarray(al.index1, dtype=float)
        ri = np.asarray(al.index2, dtype=float)
    except Exception:
        return np.array(knots, dtype=float)
    if len(qi) < 2 or qi.max() <= 0 or ri.max() <= 0:
        return np.array(knots, dtype=float)
    # Several reference indices can match one query index; average them so the
    # mapping is a function, then read it at the knots.
    qn, rn = qi / qi.max(), ri / ri.max()
    order = np.argsort(qn, kind='stable')
    qn, rn = qn[order], rn[order]
    uq, inv = np.unique(qn, return_inverse=True)
    ur = np.zeros(len(uq)); cnt = np.zeros(len(uq))
    np.add.at(ur, inv, rn); np.add.at(cnt, inv, 1.0)
    ur /= np.where(cnt == 0, 1.0, cnt)
    return np.clip(np.interp(np.asarray(knots, dtype=float), uq, ur), 0.0, 1.0)


# =============================================================================
# MEDIAN FLOOR — no leaf ships a model that loses to its own median level
# =============================================================================
#
# 'median_activity_sensor' (build_and_train_pipeline_median) is the naive floor:
# the median of every value of one (sensor, activity, object)'s training curves,
# predicted as a flat line for the target duration. It beat several learned approaches
# outright, and that is a real finding about the data rather than a bug — on
# many leaves the attributes carry no usable signal about the curve, so a model
# fitted on them spends its capacity on noise and lands above the median.
#
# The per-leaf bake-off cannot catch this on its own: it ranks the six
# regressors against EACH OTHER by validation MAE, so the winner of that field
# can still be worse than predicting the median. This makes the median a final
# candidate that the model has to beat, decided per leaf on the same held-out
# curves the regressors were selected on. A learned approach therefore becomes a
# strict improvement on the floor by construction: never worse, better wherever
# the features actually say something.
#
# The comparison is END TO END in raw energy units — each candidate's own decode
# included, predictions clipped exactly as _clip_physical clips them at
# evaluation time — because that is the quantity sMAE and WAPE score. Comparing
# in canonical space would not be valid: the approaches do not share one
# (ml_dtw's is DTW-aligned, ml_only's is linearly resampled), and the decode is
# precisely where an approach gains or loses against the floor. Errors are
# averaged per curve, not pooled over points, so long instances do not outvote
# short ones — the same weighting the per-instance metrics use.
#
# OFF by default (changed 2026-07-26). The floor accepts a method only if it
# beats the leaf median on POINTWISE error, so it silently deletes any method
# that trades pointwise accuracy for curve realism — measured at 33 of 40 leaves
# for 'exemplar', i.e. on 83% of leaves that method was throwing away its real
# curve and emitting the smooth median instead, which is the exact output it
# exists to avoid. With it off, every approach reports what it actually predicts.
#
# Set PIPELINE_CURVE_MEDIAN_FLOOR=true (or curve_median_floor=True in
# 01_pipeline.py) to switch it back on. PIPELINE_CURVE_MEDIAN_FLOOR_RATIO raises the
# bar: the model is kept only when its held-out MAE is below RATIO x the floor's,
# so 1.0 = "strictly better" and 0.95 = "better by at least 5%", the rule
# _DUR_ACCEPTANCE_RATIO already applies to the duration models.
#
# If you turn it on for a run that includes 'exemplar'/'exemplar_dtw', know that
# it applies to them too and will neutralise them — the acceptance test would
# need a realism term for that to be a fair comparison.
CURVE_MEDIAN_FLOOR = _os_seed.environ.get(
    'PIPELINE_CURVE_MEDIAN_FLOOR', 'false').lower() == 'true'
CURVE_MEDIAN_FLOOR_RATIO = float(_os_seed.environ.get(
    'PIPELINE_CURVE_MEDIAN_FLOOR_RATIO', '1.0'))


def _pointwise_median_curve(curves, fixed_length):
    """
    Pointwise median of `curves` linearly resampled onto `fixed_length` points —
    a median PROFILE (ramp-up/plateau/ramp-down preserved).

    This is no longer what the naive floor does (see _median_floor_reference);
    it survives only as the 'reference_curve' the step-DTW builder exports for
    downstream consumers that want a representative shape per leaf
    (_build_gap_fill_curves, simulation.py's presence check).

    NOT pipeline['train_median_curve']: that one is the median of the
    DTW-ALIGNED curves and lives in canonical space as a model feature.
    """
    resampled = np.array([
        np.interp(np.linspace(0, 1, fixed_length),
                  np.linspace(0, 1, len(c['original_values'])),
                  np.asarray(c['original_values'], dtype=float))
        for c in curves
    ])
    return np.median(resampled, axis=0)


def _median_floor_reference(curves, fixed_length):
    """
    The floor as a LEVEL: every value of every curve in `curves` pooled into one
    vector, non-finite dropped, median taken, emitted as a flat array of
    `fixed_length` points.

    Mirrors build_and_train_pipeline_median exactly, so a model is measured
    against the floor that actually ships as 'median_activity_sensor' rather
    than a lookalike. It used to take a pointwise median (a full profile), which
    stopped being the shipped floor when that estimator became level-only.

    Flat by construction — the length exists only so the array-shaped consumers
    keep working; _resample_median_floor re-emits it at any length.
    """
    vals = np.concatenate([
        np.asarray(c['original_values'], dtype=float).ravel() for c in curves
    ]) if len(curves) else np.array([], dtype=float)
    vals = vals[np.isfinite(vals)]
    level = float(np.median(vals)) if vals.size else 0.0
    return np.full(max(2, int(fixed_length)), level, dtype=float)


def _resample_median_floor(floor_curve, n):
    """The floor's prediction for a curve of `n` points — the constant line
    predict_raw_curve_median emits, kept in one place so the fallback and the
    standalone baseline cannot drift apart."""
    floor_curve = np.asarray(floor_curve, dtype=float)
    n = max(2, int(n))
    level = float(np.median(floor_curve)) if floor_curve.size else 0.0
    return np.full(n, level, dtype=float)


# ── Realism-based model selection ────────────────────────────────────────────
# Which candidate regressor wins its leaf is normally decided by validation MAE
# computed on the CANONICAL rows — a warped, position-averaged surrogate, per row,
# in raw units. What actually gets reported is a per-CURVE error on the RAW
# timeline after the decode. Three mismatches at once: space, unit and metric.
#
# With CURVE_SELECT_BY_REALISM on, the winner is instead chosen by the same
# realism score the results notebooks report: each candidate is run through its
# REAL predictor (decode included) on the held-out validation curves, its
# per-curve features are compared with the REAL curves' features, and the
# normalised absolute errors are averaged into one 'Overall'. Lower is better.
#
#     err_f = |f(pred) - f(real)| / mean|f(real)|      over this leaf's val curves
#     Overall = mean over the selected features
#
# Off (default) keeps the historical val-MAE selection, so runs stay comparable.
CURVE_SELECT_BY_REALISM = _os_seed.environ.get(
    'PIPELINE_CURVE_SELECT_BY_REALISM', 'false').lower() == 'true'

# Which per-curve features go into that average. Change this and the selection
# objective changes with it — e.g. drop 'sum' to stop rewarding total-energy
# accuracy, or add 'roughness' to punish smoothed-out dynamics.
CURVE_SELECTION_METRICS = tuple(
    m.strip().lower() for m in _os_seed.environ.get(
        'PIPELINE_CURVE_SELECTION_METRICS', 'sum,max,mean,std').split(',') if m.strip())


def _curve_feature(v, name):
    """One per-curve scalar, matching the notebooks' definitions exactly."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    if name == 'sum':       return float(np.sum(v))
    if name == 'max':       return float(np.max(v))
    if name == 'mean':      return float(np.mean(v))
    if name == 'std':       return float(np.std(v))
    if name == 'roughness': return float(np.abs(np.diff(v)).mean()) if v.size >= 2 else np.nan
    if name == 'acf1':
        # Guard the two SLICES, not v itself: a curve that is flat except for its
        # first or last sample has nonzero overall std but a constant slice, which
        # makes corrcoef divide 0/0 and emit a RuntimeWarning for the same nan.
        if v.size < 3:
            return np.nan
        a, b = v[:-1], v[1:]
        if a.std() < 1e-12 or b.std() < 1e-12:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])
    raise ValueError(f'unknown curve feature {name!r}')


def _realism_overall(val_curves, predict_fn, metrics=None, verbose=0, label=''):
    """
    'Overall' realism error of a predictor on held-out curves — the selection
    counterpart of the notebooks' scorecard.

    predict_fn : callable(raw_values, curve) -> np.ndarray, the pipeline's REAL
                 predictor, so the decode is part of what is scored.

    Scores against curve['original_values'] — the untouched real curve, never the
    DTW-aligned target. Scale is this leaf's own mean|f(real)|, the direct analogue
    of the notebooks' per-(process, sensor) scale.

    Returns (overall, {feature: err}); overall is NaN when nothing is scoreable,
    which the callers treat as "fall back to val MAE" rather than as a win.
    """
    metrics = list(metrics if metrics is not None else CURVE_SELECTION_METRICS)
    reals, preds = [], []
    for c in val_curves:
        raw = np.asarray(c['original_values'], dtype=float)
        if len(raw) < 2:
            continue
        try:
            y = np.asarray(predict_fn(raw, c), dtype=float)
        except Exception:
            return np.nan, {}          # a predictor that raises cannot be ranked
        if len(y) != len(raw) or not np.all(np.isfinite(y)):
            return np.nan, {}
        reals.append(raw); preds.append(y)
    if not reals:
        return np.nan, {}

    per_feature = {}
    for f in metrics:
        rv = np.array([_curve_feature(r, f) for r in reals], dtype=float)
        pv = np.array([_curve_feature(p, f) for p in preds], dtype=float)
        ok = np.isfinite(rv) & np.isfinite(pv)
        if not ok.any():
            continue
        scale = float(np.nanmean(np.abs(rv[ok])))
        # Degenerate-scale guard, as in the notebooks: a feature the real curves
        # have essentially none of carries no information and would explode.
        if not np.isfinite(scale) or scale < 1e-6:
            continue
        per_feature[f] = float(np.mean(np.abs(pv[ok] - rv[ok])) / scale)
    if not per_feature:
        return np.nan, {}
    overall = float(np.mean(list(per_feature.values())))
    if verbose:
        print(f"  [realism-select{label}] Overall={overall:.4f}  "
              + '  '.join(f'{k}={v:.4f}' for k, v in per_feature.items()))
    return overall, per_feature


def _select_by_realism(all_results, pipeline, val_curves, predict_fn,
                       fallback_name, verbose=0, label=''):
    """
    Re-pick the winning candidate by realism Overall instead of val MAE.

    `pipeline` is mutated in place: pipeline['model'] is swapped to each candidate
    and `predict_fn` must READ pipeline['model'] at call time (close over the dict,
    not over a model), so every candidate is scored through the real predict path.

    Falls back to `fallback_name` — the val-MAE winner — whenever nothing is
    scoreable, so this can never leave a leaf without a model. The chosen scores
    are recorded on the pipeline for later inspection.
    """
    scores = {}
    for name, res in all_results.items():
        m = res.get('model')
        if m is None:
            continue
        pipeline['model'] = m
        ov, per = _realism_overall(val_curves, predict_fn, verbose=0, label=label)
        all_results[name]['realism_overall'] = ov
        all_results[name]['realism_per_feature'] = per
        if np.isfinite(ov):
            scores[name] = ov
    if not scores:
        pipeline['model'] = all_results[fallback_name]['model']
        pipeline['model_name'] = fallback_name
        pipeline['selection_criterion'] = 'val_mae (realism unscoreable)'
        return fallback_name
    best = min(scores, key=scores.get)
    pipeline['model'] = all_results[best]['model']
    pipeline['model_name'] = best
    pipeline['selection_criterion'] = 'realism_overall'
    pipeline['selection_metrics'] = list(CURVE_SELECTION_METRICS)
    pipeline['realism_overall'] = scores[best]
    pipeline['realism_per_feature'] = all_results[best].get('realism_per_feature', {})
    if verbose:
        _alt = f' (val-MAE would have picked {fallback_name})' if best != fallback_name else ''
        print(f"  [realism-select{label}] {best} Overall={scores[best]:.4f}{_alt}")
    return best


def _attach_median_floor(pipeline, train_curves, val_instance_ids, predict_fn,
                         label='', verbose=0):
    """
    Decide on held-out curves whether this leaf's trained model really beats its
    own median curve, and store the floor so predict time can fall back to it
    when it does not. See CURVE_MEDIAN_FLOOR.

    Parameters
    ----------
    pipeline         : dict     — the finished pipeline, mutated in place
    train_curves     : list[dict] — every curve the builder was handed
    val_instance_ids : iterable — the builder's OWN validation split, so the
                       model is judged on curves it did not fit and the floor is
                       built without them; anything else would hand one of the
                       two candidates an advantage the other does not have
    predict_fn       : callable(raw_values, curve) -> np.ndarray — the
                       pipeline's real predictor closed over the finished dict,
                       so the decode step is part of what gets scored

    Adds to `pipeline`: 'median_floor_curve', 'median_floor_mae',
    'model_floor_mae' and 'median_floor_active' (True = the model lost and the
    floor is what will be predicted).
    """
    pipeline['median_floor_active'] = False
    if not CURVE_MEDIAN_FLOOR:
        return pipeline

    fixed_length = pipeline.get('fixed_length')
    val_ids    = set(val_instance_ids)
    fit_curves = [c for c in train_curves if c['instance_id'] not in val_ids]
    val_curves = [c for c in train_curves if c['instance_id'] in val_ids]
    if not fixed_length or not fit_curves or not val_curves:
        # No held-out curves to judge on — keep the model rather than guess.
        return pipeline

    floor_curve = _median_floor_reference(fit_curves, fixed_length)

    model_err, floor_err = [], []
    for c in val_curves:
        raw = np.asarray(c['original_values'], dtype=float)
        if len(raw) < 2:
            continue
        try:
            y_model = np.asarray(predict_fn(raw, c), dtype=float)
        except Exception as exc:
            # A predictor that raises on its own validation curves cannot be
            # scored, and would raise the same way in the simulation. The floor
            # always predicts, so fall back to it instead of shipping the model.
            if verbose:
                print(f"  [median-floor{label}] predictor raised {exc!r} — using the floor")
            pipeline.update({'median_floor_curve':  floor_curve,
                             'median_floor_active': True,
                             'median_floor_mae':    float('nan'),
                             'model_floor_mae':     float('nan')})
            return pipeline
        if len(y_model) != len(raw):
            y_model = np.interp(np.linspace(0, 1, len(raw)),
                                np.linspace(0, 1, len(y_model)), y_model)
        y_model = _clip_physical(y_model)
        y_floor = _clip_physical(_resample_median_floor(floor_curve, len(raw)))
        model_err.append(float(np.abs(y_model - raw).mean()))
        floor_err.append(float(np.abs(y_floor - raw).mean()))

    if not model_err:
        return pipeline

    model_mae = float(np.mean(model_err))
    floor_mae = float(np.mean(floor_err))
    active    = not (model_mae < CURVE_MEDIAN_FLOOR_RATIO * floor_mae)

    pipeline.update({
        'median_floor_curve':  floor_curve,
        'median_floor_mae':    floor_mae,
        'model_floor_mae':     model_mae,
        'median_floor_active': active,
    })
    if verbose:
        _kept = 'FLOOR (median level)' if active else f"model ({pipeline.get('model_name')})"
        print(f"  [median-floor{label}] held-out MAE: model={model_mae:.4f}  "
              f"floor={floor_mae:.4f} -> keeping {_kept}")
    return pipeline


def _median_floor_prediction(raw_values, attributes, pipeline):
    """
    The floor's prediction when this leaf fell back to it, otherwise None.

    Called first thing by every learned predictor rather than once inside
    _dispatch_predict, because that router is only one of the paths into them —
    02_modelling.py binds predict_raw_curve* into predict_fn lambdas directly, and
    predict_curve_for_instance calls those lambdas. Hooking the predictors
    themselves is what makes the fallback hold in the simulation as well as in
    the curve-only evaluation. A no-op unless _attach_median_floor decided the
    model lost. See CURVE_MEDIAN_FLOOR.
    """
    if not pipeline.get('median_floor_active'):
        return None
    floor_curve = pipeline.get('median_floor_curve')
    if floor_curve is None or len(floor_curve) == 0:
        return None
    n = (attributes or {}).get('_pred_curve_length', len(raw_values))
    return _resample_median_floor(floor_curve, n)


# =============================================================================
# STEP 3 — PREDICT ON A SINGLE RAW CURVE
# =============================================================================

# ── Canonical -> raw decode, and the shape-blind switch ──────────────────────
# The DTW approaches decode a canonical (barycenter-space) prediction back onto
# the test curve's timeline by DTW-warping it against THAT CURVE'S OWN VALUES.
# The non-DTW approaches (predict_raw_curve_median / _ml_only / _seq2seq_only)
# stretch their prediction with a plain linear resample and therefore use only
# len(raw_values). So by default the two families do not receive the same
# test-time information: the DTW ones are handed the observed shape to align to,
# which cancels exactly the phase error that pointwise metrics (sMAE, WAPE)
# punish hardest, while the others must get the timing right unaided.
#
# This is ON by default: every approach decodes by linear resample, so the only
# thing any of them takes from the test curve is its LENGTH. DTW stays a
# training-time representation choice (align the training curves to a barycenter
# so the per-position target has lower variance) and the comparison is
# like-for-like.
#
# Default flipped 2026-07-25. The shape-aware decode is not merely unfair, it is
# unavailable where the pipeline is actually used: ProcessSimulation predicts a
# curve for an activity instance it has just GENERATED, from an activity, a
# predicted duration and attributes. There is no measured curve to align to, so
# a decode that needs one cannot run at simulation time at all — it only ever
# worked in the curve-only evaluation, which is what made the two settings
# disagree about which approach is best. Blind is the honest default; set
# PIPELINE_DTW_SHAPE_BLIND_DECODE=false to reproduce the old runs (experiment_969
# and earlier were all shape-AWARE, so their curve tables are not comparable to
# anything produced after this date).
DTW_DECODE_SHAPE_BLIND = _os_seed.environ.get(
    'PIPELINE_DTW_SHAPE_BLIND_DECODE', 'true').lower() == 'true'


def _decode_canonical_to_raw(y_ref_pred, raw_values, reference_curve, fixed_length,
                             shape_blind=None):
    """
    Map a canonical-space prediction (fixed_length points) onto the raw timeline
    (len(raw_values) points).

    shape_blind=True (default): linear resample using only len(raw_values) —
    identical to the decode the non-DTW predictors already use, and blind to the
    test curve's values. This is the only decode available at simulation time.

    shape_blind=False: DTW-warp against raw_values. Points of the raw curve that
    the path maps to several canonical positions get the mean of them; unmapped
    points fall back to the nearest position on the path. Reproduces runs up to
    and including experiment_969. See DTW_DECODE_SHAPE_BLIND.

    None -> the module default, so a run can be flipped from the pipeline config
    without touching any predictor.
    """
    if shape_blind is None:
        shape_blind = DTW_DECODE_SHAPE_BLIND

    y_ref_pred = np.asarray(y_ref_pred, dtype=float)
    n = len(raw_values)

    if shape_blind:
        if fixed_length < 2 or n < 1:
            return np.full(max(n, 0), float(y_ref_pred[0]) if len(y_ref_pred) else 0.0)
        return np.interp(np.linspace(0, 1, n),
                         np.linspace(0, 1, fixed_length), y_ref_pred)

    alignment  = dtw(raw_values, reference_curve, keep_internals=True)
    path_pairs = list(zip(alignment.index1, alignment.index2))
    buckets    = [[] for _ in range(n)]
    for qi, ri in path_pairs:
        if 0 <= qi < n and 0 <= ri < fixed_length:
            buckets[qi].append(y_ref_pred[ri])

    y_raw_pred = np.empty(n, dtype=float)
    for qi in range(n):
        if buckets[qi]:
            y_raw_pred[qi] = float(np.mean(buckets[qi]))
            continue
        # Robust fallback: nearest mapped raw index from the DTW path.
        _, nearest_ri = min(path_pairs, key=lambda p: abs(p[0] - qi))
        y_raw_pred[qi] = float(y_ref_pred[min(max(nearest_ri, 0), fixed_length - 1)])
    return y_raw_pred


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
    _floor = _median_floor_prediction(raw_values, attributes, pipeline)
    if _floor is not None:
        return _floor

    reference_curve = pipeline['reference_curve']
    # Use the trained reference length as the source of truth at inference.
    fixed_length    = len(reference_curve)
    model           = pipeline['model']
    feature_columns = pipeline['feature_columns']
    feature_scaler  = pipeline.get('feature_scaler', None)
    numeric_feature_cols = pipeline.get('numeric_feature_cols', [])
    all_keys        = pipeline['all_keys']
    key_types       = pipeline['key_types']
    # Training-median anchor, replayed positionwise from the trained pipeline.
    # Absent on pipelines trained before this feature existed — then the column
    # is simply not built, and it is not in feature_columns either.
    train_median_curve = pipeline.get('train_median_curve', None)

    # 1) Predict in canonical space (fixed_length points)
    _rel_denom = max(fixed_length - 1, 1)
    rows_ref = []
    for ref_pos in range(fixed_length):
        row = {
            'position_idx': ref_pos,
            'relative_pos': ref_pos / _rel_denom,
            'curve_length': attributes.get('_pred_curve_length', len(raw_values)),
            'activity': activity,
        }
        if train_median_curve is not None:
            row['train_median_at_pos'] = float(train_median_curve[ref_pos])
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

    # Fill any remaining NaN (missing object_attributes at inference time) with 0.
    # After scaling, 0 == training mean for numeric features; safe fallback.
    X_ref = X_ref.fillna(0)

    y_ref_pred = model.predict(X_ref)

    # 2) Decode canonical predictions to raw timeline using DTW path
    return _decode_canonical_to_raw(y_ref_pred, raw_values, reference_curve,
                                    fixed_length)


def predict_raw_curve_median(raw_values, activity, attributes, pipeline):
    """
    Predict the stored median LEVEL (pipeline['median_level'], from
    build_and_train_pipeline_median) as one horizontal line for the target
    duration -- no model, no DTW warp, no resampled profile, no attribute
    conditioning. The true naive floor: a level, not a shape.

    Only the LENGTH is instance-specific, and it comes from exactly where it
    always did: attributes['_pred_curve_length'] when the caller supplies a
    predicted duration, else len(raw_values).
    """
    level = pipeline.get('median_level')
    if level is None:
        # Pipelines persisted before the level-only rewrite carry the flat
        # reference curve but not the scalar; its median is the same number.
        level = float(np.median(np.asarray(pipeline['reference_curve'], dtype=float)))
    n = attributes.get('_pred_curve_length', len(raw_values)) if attributes else len(raw_values)
    n = max(2, int(n))
    return np.full(n, float(level), dtype=float)


# =============================================================================
# ML LINEAR — ML MODEL WITHOUT ANY DTW
#
# Drop the DBA barycenter and DTW alignment steps entirely.
# Training target: raw curve linearly resampled to fixed_length.
# Decode      : linear resample of the fixed_length predictions back to raw.
# Everything else (feature matrix, model selection) is identical to baseline.
# =============================================================================

def build_and_train_pipeline_ml_only(
    train_curves,
    variable,
    fixed_length=None,
    val_size=0.2,
    random_state=42,
    models=None,
    optimize_hyperparams=False,
    n_trials=50,
    verbose=1,
    n_jobs=-1,
):
    """
    ML pipeline — no DBA, no DTW anywhere.

    Training target for each position i in 0..fixed_length-1 is the value of
    the raw curve linearly resampled to fixed_length at position i.
    At decode time the fixed_length predictions are linearly resampled back to
    the original curve length (no DTW path used).
    """
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2 — BUILD + TRAIN ML-LINEAR PIPELINE  (no DTW)")
        print("=" * 80)

    if models is None:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        models = {
            'Gradient Boosting': GradientBoostingRegressor,
            'Random Forest':     RandomForestRegressor,
        }

    if fixed_length is None:
        fixed_length = max(2, int(round(np.median([len(c['original_values']) for c in train_curves]))))
    # Linearly resample each curve to fixed_length — no DBA, no DTW
    for curve in train_curves:
        orig = np.asarray(curve['original_values'], dtype=float)
        curve['resampled_values'] = np.interp(
            np.linspace(0, 1, fixed_length),
            np.linspace(0, 1, len(orig)),
            orig,
        )

    all_keys, key_types = _infer_key_types(train_curves)

    if verbose:
        print(f"\n[2a] Attribute columns: {key_types}")

    df_reg = _build_feature_matrix(
        train_curves, all_keys, key_types, fixed_length,
        value_key='resampled_values', include_target=True,
    )

    if verbose:
        print(f"     Regression dataset: {len(df_reg)} rows "
              f"({len(train_curves)} curves × {fixed_length} positions)")

    # Training-median anchor (train curves only, see _training_median_stats).
    train_median_curve, train_sensor_median = _training_median_stats(train_curves, fixed_length)
    df_reg['train_median_at_pos'] = train_median_curve[df_reg['position_idx'].to_numpy()]

    categorical_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_all = df_reg[['position_idx', 'relative_pos', 'curve_length',
                    'train_median_at_pos']].copy()
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

    numeric_feature_cols = ['position_idx', 'relative_pos', 'curve_length',
                            'train_median_at_pos'] + [
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

    all_results = {}
    for name, model_class in models.items():
        if verbose:
            print(f"\n[2b] Training — {name}...")
        if optimize_hyperparams:
            if not verbose:
                optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial, name=name, model_class=model_class):
                params = _curve_model_trial_params(trial, name, random_state, n_jobs)
                m = model_class(**params)
                m.fit(X_train, y_train)
                return -mean_absolute_error(y_val, m.predict(X_val))

            sampler = optuna.samplers.TPESampler(seed=random_state)
            study = optuna.create_study(direction='maximize', sampler=sampler)
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            best_params = _curve_model_finalise_params(name, best_params, random_state, n_jobs)
            if verbose:
                print(f"     Best params: {best_params}")
            model = model_class(**best_params)
        else:
            if name == 'Gradient Boosting':
                model = model_class(
                    n_estimators=200, max_depth=7, learning_rate=0.1,
                    subsample=0.8, random_state=random_state,
                )
            elif name == 'Random Forest':
                model = model_class(
                    n_estimators=200, max_depth=12, min_samples_split=5,
                    random_state=random_state, n_jobs=n_jobs,
                )
            else:
                # Untuned path: the loss is not searched, so bake it in here too
                # or this branch silently stays on squared error.
                _lk = _curve_model_loss_kwargs(name)
                try:
                    model = model_class(random_state=random_state, **_lk)
                except TypeError:
                    try:
                        model = model_class(**_lk)
                    except TypeError:
                        model = model_class()

        model.fit(X_train, y_train)
        train_mae  = mean_absolute_error(y_train, model.predict(X_train))
        val_mae    = mean_absolute_error(y_val,   model.predict(X_val))
        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        val_rmse   = np.sqrt(mean_squared_error(y_val,   model.predict(X_val)))
        if verbose:
            print(f"     Train  MAE={train_mae:.4f}  RMSE={train_rmse:.4f}")
            print(f"     Val    MAE={val_mae:.4f}  RMSE={val_rmse:.4f}")
        all_results[name] = {
            'model': model, 'train_mae': train_mae, 'val_mae': val_mae,
            'train_rmse': train_rmse, 'val_rmse': val_rmse,
        }

    best_name  = min(all_results, key=lambda k: all_results[k]['val_mae'])
    best_model = all_results[best_name]['model']
    if verbose:
        print(f"\n Best model : {best_name}  "
              f"(val MAE={all_results[best_name]['val_mae']:.4f})")

    pipeline = {
        'approach':             'ml_only',
        'model':                best_model,
        'model_name':           best_name,
        'fixed_length':         fixed_length,
        'all_keys':             all_keys,
        'key_types':            key_types,
        'feature_columns':      feature_columns,
        'feature_scaler':       feature_scaler,
        'numeric_feature_cols': numeric_feature_cols,
        'train_median_curve':   train_median_curve,
        'train_sensor_median':  train_sensor_median,
        'val_mae':              all_results[best_name]['val_mae'],
        'all_results':          all_results,
    }


    # Realism-based re-selection — see CURVE_SELECT_BY_REALISM. Scored through
    # this pipeline's OWN predictor on the held-out curves, so the decode and the
    # real (never DTW-aligned) values are what decide the winner.
    if CURVE_SELECT_BY_REALISM:
        _vc = [c for c in train_curves if c['instance_id'] in set(val_inst)]
        _select_by_realism(all_results, pipeline, _vc,
                           lambda rv, c: predict_raw_curve_ml_only(rv, c['activity'], c['attributes'], pipeline),
                           best_name, verbose=verbose, label=' ml_only')
    # Median floor — see _attach_median_floor.
    _attach_median_floor(
        pipeline, train_curves, val_inst,
        lambda rv, c: predict_raw_curve_ml_only(rv, c['activity'], c['attributes'], pipeline),
        label=f' {variable}', verbose=verbose,
    )

    return pipeline


def predict_raw_curve_ml_only(raw_values, activity, attributes, pipeline):
    """
    Predict using the ML-linear pipeline (no DTW anywhere).
    Builds canonical feature matrix → ML predict → linear resample to raw length.
    """
    _floor = _median_floor_prediction(raw_values, attributes, pipeline)
    if _floor is not None:
        return _floor

    fixed_length         = pipeline['fixed_length']
    model                = pipeline['model']
    feature_columns      = pipeline['feature_columns']
    feature_scaler       = pipeline.get('feature_scaler', None)
    numeric_feature_cols = pipeline.get('numeric_feature_cols', [])
    all_keys             = pipeline['all_keys']
    key_types            = pipeline['key_types']
    # Training-median anchor, replayed positionwise (None on older pipelines).
    train_median_curve   = pipeline.get('train_median_curve', None)

    _rel_denom = max(fixed_length - 1, 1)
    rows = []
    for ref_pos in range(fixed_length):
        row = {
            'position_idx': ref_pos,
            'relative_pos': ref_pos / _rel_denom,
            'curve_length': attributes.get('_pred_curve_length', len(raw_values)),
            'activity':     activity,
        }
        if train_median_curve is not None:
            row['train_median_at_pos'] = float(train_median_curve[ref_pos])
        for key in all_keys:
            value = attributes.get(key, None)
            if key_types[key] == 'numeric':
                try:
                    row[key] = float(value) if value is not None else np.nan
                except (ValueError, TypeError):
                    row[key] = np.nan
            else:
                row[key] = str(value) if value is not None else 'None'
        rows.append(row)

    categorical_cols = ['activity'] + [k for k in all_keys if key_types[k] == 'category']
    X_ref = pd.DataFrame(rows)
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

    X_ref = X_ref.fillna(0)
    y_ref_pred = model.predict(X_ref)

    # Linear resample canonical predictions back to raw length — no DTW
    return np.interp(
        np.linspace(0, 1, len(raw_values)),
        np.linspace(0, 1, fixed_length),
        y_ref_pred,
    )




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


# Fit the exog models on (y - train_median_at_pos) instead of y, adding the
# anchor back at predict time. Set False to ablate — see the `residual_target`
# note in build_and_train_pipeline_exog.
EXOG_RESIDUAL_TARGET = True


def build_and_train_pipeline_exog(
    train_curves,
    variable,
    fixed_length=None,
    val_size=0.2,
    random_state=42,
    models=None,
    optimize_hyperparams=False,
    n_trials=50,
    verbose=1,
    n_jobs=-1,
    residual_target=None,
    sample_weight_mode=None,
):
    """
    DTW baseline enriched with ef_* external-factor time series as features.

    Requires that each curve dict in train_curves contains 'exog_values' (set
    by split_curves with exog_columns=...).  Curves that lack the key are still
    included but their exog features are filled with NaN.

    residual_target : bool | None
        Fit on (y - train_median_at_pos) and add the anchor back in
        predict_raw_curve_exog, instead of fitting y directly. None -> the
        EXOG_RESIDUAL_TARGET module default.

        The anchor is a per-position constant shared by target and prediction, so
        |y - pred| is unchanged by the shift: train/val MAE and the Optuna
        objective keep their original scale and stay comparable to the pipelines
        that fit y directly. What changes is what the model has to represent.
        Fitting y, a tree can only reach levels it saw in training — a test curve
        above the training range gets capped, and the level has to be
        reconstructed by partitioning on train_median_at_pos. Fitting the
        residual makes the level exact and unbounded by construction, and leaves
        the model to spend its capacity on the deviation. train_median_at_pos
        stays in the feature set: it still tells the model where on the curve it
        is, which is what modulates how large a deviation to expect.

    sample_weight_mode : None | 'counts' | 'counts_sigma'
        Reweights the canonical training rows so the fitted objective is a closer
        surrogate for the metric actually reported, which is computed on the RAW
        timeline after decoding. See _align_curve_with_dtw(return_counts=True).

        None (default)  — every canonical row weighs 1, the original behaviour.
        'counts'        — weigh each row by how many raw samples the DTW path
                          folded into that canonical position. Without this, a
                          position summarising 40 raw points and one summarising
                          a single point count equally in the loss, while in raw
                          space they drive 40 residuals and 1.
        'counts_sigma'  — additionally divide by the curve's own std, because
                          sMAE is L1 on residuals divided by exactly that. Curves
                          already contribute equally (each gives fixed_length
                          rows), so this is the remaining term. The std is the
                          TRAINING curve's own, so nothing leaks.

        This narrows the train/eval gap but does not close it: the decode's
        bucket-mean is a nonlinear step the model still never sees. ml_rawspace
        is the version with no gap at all.

    Returns a pipeline dict identical to build_and_train_pipeline but with the
    additional keys:
        'exog_cols'          — list[str]  ordered exog column names used at train time
        'residual_target'    — bool       whether 'model' predicts y or y - anchor
        'sample_weight_mode' — as passed
        'val_mae'            — weighted when sample_weight_mode is set (it is the
                               selection criterion, so it must match the fit)
        'val_mae_unweighted' — always plain MAE, so runs stay comparable
        'approach'           — 'ml_exog'
    """
    if sample_weight_mode not in (None, 'counts', 'counts_sigma'):
        raise ValueError(f"sample_weight_mode must be None|'counts'|'counts_sigma', "
                         f"got {sample_weight_mode!r}")
    if residual_target is None:
        residual_target = EXOG_RESIDUAL_TARGET

    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2 — BUILD + TRAIN PIPELINE (DTW + External Factors)")
        print(f"  target: {'residual from training-median anchor' if residual_target else 'raw level'}")
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
    if fixed_length is None:
        fixed_length = max(2, int(round(np.median([len(c['original_values']) for c in train_curves]))))
    resampled_for_dba = np.array([
        np.interp(
            np.linspace(0, 1, fixed_length),
            np.linspace(0, 1, len(c['original_values'])),
            c['original_values'],
        )
        for c in train_curves
    ])[:, :, np.newaxis]

    dba_barycenter  = _robust_dtw_barycenter(resampled_for_dba, barycenter_size=fixed_length)
    # Restore the on/off duty cycle DBA averages away (see
    # _zero_calibrate_barycenter); no-op for non-intermittent sensors.
    reference_curve = _zero_calibrate_barycenter(dba_barycenter[:, 0], resampled_for_dba)

    # ── DTW-align train curves ────────────────────────────────────────────────
    for curve in train_curves:
        if sample_weight_mode:
            curve['resampled_values'], _cnts = _align_curve_with_dtw(
                curve['original_values'], reference_curve, return_counts=True
            )
            _w = np.asarray(_cnts, dtype=float)
            if sample_weight_mode == 'counts_sigma':
                # sMAE divides residuals by the curve's own std, so the matching
                # row weight is counts/sigma. Guard the flat-curve case, which
                # the evaluator scores as NaN anyway.
                _sd = float(np.std(np.asarray(curve['original_values'], dtype=float)))
                _w = _w / _sd if _sd > 1e-10 else _w
            curve['_row_weight'] = _w
        else:
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

    # Training-median anchor: sensor's typical level at each canonical
    # position, train curves only (see _training_median_stats).
    train_median_curve, train_sensor_median = _training_median_stats(train_curves, fixed_length)

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
                'train_median_at_pos': float(train_median_curve[position_idx]),
                'y':            curve['resampled_values'][position_idx],
                '_w':           (float(curve['_row_weight'][position_idx])
                                 if sample_weight_mode else 1.0),
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
    X_all = df_reg[['position_idx', 'relative_pos', 'curve_length',
                    'train_median_at_pos']].copy()
    X_all = X_all.assign(activity=df_reg['activity'].values)
    for key in all_keys:
        X_all = X_all.assign(**{key: df_reg[key].values})
    for col in exog_cols:
        X_all[col] = df_reg[col].values
    X_all = pd.get_dummies(X_all, columns=categorical_cols, drop_first=True)
    # Residual target: subtract the same per-position anchor that predict adds
    # back. Shifting both sides by a constant leaves every MAE below unchanged.
    y_all = (df_reg['y'] - df_reg['train_median_at_pos']).copy() if residual_target \
            else df_reg['y'].copy()

    unique_instances     = df_reg['instance_id'].unique()
    train_inst, val_inst = train_test_split(
        unique_instances, test_size=val_size, random_state=random_state
    )
    train_mask = df_reg['instance_id'].isin(train_inst)
    val_mask   = df_reg['instance_id'].isin(val_inst)

    X_train, X_val = X_all[train_mask].copy(), X_all[val_mask].copy()
    y_train, y_val = y_all[train_mask].copy(), y_all[val_mask].copy()

    # Row weights follow the same split. None when unweighted, so every fit /
    # score call below can pass them through unconditionally.
    if sample_weight_mode:
        w_train = df_reg.loc[train_mask, '_w'].to_numpy(dtype=float)
        w_val   = df_reg.loc[val_mask,   '_w'].to_numpy(dtype=float)
    else:
        w_train = w_val = None

    feature_columns = X_all.columns.tolist()

    numeric_feature_cols = ['position_idx', 'relative_pos', 'curve_length',
                            'train_median_at_pos'] + \
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
    best_model, best_name, best_val_mae = None, None, np.inf
    best_val_mae_unw = np.nan

    for name, model_class in models.items():
        if verbose:
            print(f"\n  Training {name}...")

        if optimize_hyperparams:
            if not verbose:
                optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial, name=name, model_class=model_class):
                params = _curve_model_trial_params(trial, name, random_state, n_jobs)
                m = model_class(**params)
                _fit_with_weight(m, X_train, y_train, w_train)
                # Search the SAME weighted objective the model is fitted on, or
                # tuning optimises a different target than training.
                return -mean_absolute_error(y_val, m.predict(X_val),
                                            sample_weight=w_val)

            sampler = optuna.samplers.TPESampler(seed=random_state)
            study = optuna.create_study(direction='maximize', sampler=sampler)
            study.optimize(objective, n_trials=n_trials)
            best_params = _curve_model_finalise_params(
                name, study.best_params, random_state, n_jobs
            )
            if verbose:
                print(f"    Best params: {best_params}")
            model = model_class(**best_params)
        else:
            # Untuned path: the loss is not searched, so bake it in here too or
            # this branch silently stays on squared error.
            _lk = _curve_model_loss_kwargs(name)
            try:
                model = model_class(random_state=random_state, **_lk)
            except TypeError:
                try:
                    model = model_class(**_lk)
                except TypeError:
                    model = model_class()
        _fit_with_weight(model, X_train, y_train, w_train)

        train_mae = float(mean_absolute_error(y_train, model.predict(X_train),
                                              sample_weight=w_train))
        # Weighted when weighting is on: this is the SELECTION criterion, so it
        # has to match the fit. The unweighted value is kept alongside so runs
        # with and without weighting stay comparable.
        val_mae   = float(mean_absolute_error(y_val, model.predict(X_val),
                                              sample_weight=w_val))
        val_mae_unw = float(mean_absolute_error(y_val, model.predict(X_val)))
        if verbose:
            print(f"    train MAE={train_mae:.4f}  val MAE={val_mae:.4f}")

        all_results[name] = {'train_mae': train_mae, 'val_mae': val_mae,
                             'val_mae_unweighted': val_mae_unw,
                             'model': model}   # kept so realism re-selection can rank it
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_mae_unw = val_mae_unw
            best_model   = model
            best_name    = name

    if verbose:
        print(f"\n  Best model: {best_name}  val MAE={best_val_mae:.4f}")

    pipeline = {
        'model':               best_model,
        'model_name':          best_name,
        'reference_curve':     reference_curve,
        'fixed_length':        fixed_length,
        'all_keys':            all_keys,
        'key_types':           key_types,
        'feature_columns':     feature_columns,
        'feature_scaler':      feature_scaler,
        'numeric_feature_cols': numeric_feature_cols,
        'train_median_curve':  train_median_curve,
        'train_sensor_median': train_sensor_median,
        'val_mae':             best_val_mae,
        'all_results':         all_results,
        'exog_cols':           exog_cols,
        'residual_target':     bool(residual_target),
        'sample_weight_mode':  sample_weight_mode,
        'val_mae_unweighted':  best_val_mae_unw,
        'approach':            'ml_exog',
    }


    # Realism-based re-selection — see CURVE_SELECT_BY_REALISM. Scored through
    # this pipeline's OWN predictor on the held-out curves, so the decode and the
    # real (never DTW-aligned) values are what decide the winner.
    if CURVE_SELECT_BY_REALISM:
        _vc = [c for c in train_curves if c['instance_id'] in set(val_inst)]
        _select_by_realism(all_results, pipeline, _vc,
                           lambda rv, c: predict_raw_curve_exog(rv, c['activity'], c['attributes'],
                                                pipeline, exog_values=c.get('exog_values', {})),
                           best_name, verbose=verbose, label=' exog')
    # Median floor — see _attach_median_floor. Scored through
    # predict_raw_curve_exog rather than the ml_external wrapper because the
    # prev-activity energy features are already in these curves' attributes
    # (build_and_train_pipeline_exog_prev_activity injects them before calling
    # this builder), so the wrapper's re-derivation would be a no-op here.
    _attach_median_floor(
        pipeline, train_curves, val_inst,
        lambda rv, c: predict_raw_curve_exog(rv, c['activity'], c['attributes'],
                                             pipeline,
                                             exog_values=c.get('exog_values', {})),
        label=f' {variable}', verbose=verbose,
    )

    return pipeline


def _exog_canonical_prediction(raw_values, activity, attributes, pipeline,
                               exog_values=None):
    """
    The exog pipeline's prediction in CANONICAL (barycenter) space —
    `fixed_length` points, before any decode onto the raw timeline.

    Split out of predict_raw_curve_exog so a caller can substitute its own decode
    for the built-in one. ml_cluster_dtw does exactly that: it replaces the
    uniform resample with a warp predicted from the attributes, so the canonical
    curve has to be reachable without being decoded first. The only thing read
    from `raw_values` here is its length, via the 'curve_length' feature.

    Does NOT consult the median floor — that is a decision about the finished
    raw-space prediction, so it stays in the callers.
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
    # Training-median anchor, replayed positionwise (None on older pipelines).
    train_median_curve   = pipeline.get('train_median_curve', None)

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
            'curve_length': attributes.get('_pred_curve_length', len(raw_values)),
            'activity':     activity,
        }
        if train_median_curve is not None:
            row['train_median_at_pos'] = float(train_median_curve[ref_pos])
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

    # Fill any remaining NaN (missing object_attributes at inference time) with 0.
    # After scaling, 0 == training mean for numeric features; safe fallback.
    X_ref = X_ref.fillna(0)

    y_ref_pred = model.predict(X_ref)

    # Residual target: the model predicted the deviation from the training-median
    # anchor, so put the level back before decoding. Both are in canonical
    # (barycenter) space, hence positionwise. Pipelines fitted before this flag
    # existed don't carry the key and fall through unchanged.
    if pipeline.get('residual_target') and train_median_curve is not None:
        y_ref_pred = y_ref_pred + np.asarray(train_median_curve, dtype=float)

    return y_ref_pred


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
    _floor = _median_floor_prediction(raw_values, attributes, pipeline)
    if _floor is not None:
        return _floor

    y_ref_pred = _exog_canonical_prediction(raw_values, activity, attributes,
                                            pipeline, exog_values=exog_values)

    # DTW decode canonical → raw length
    reference_curve = pipeline['reference_curve']
    return _decode_canonical_to_raw(y_ref_pred, raw_values, reference_curve,
                                    len(reference_curve))


# =============================================================================
# DTW + EXTERNAL FACTORS + PREVIOUS-ACTIVITY CONTEXT PIPELINE
# =============================================================================

def build_and_train_pipeline_exog_prev_activity(
    train_curves,
    variable,
    fixed_length=None,
    val_size=0.2,
    random_state=42,
    models=None,
    optimize_hyperparams=False,
    n_trials=50,
    verbose=1,
    n_jobs=-1,
    prev_act_energy_map=None,
    residual_target=None,
    sample_weight_mode=None,
):
    """
    DTW + external factors + previous-activity NAME and training energy level.

    The prev-activity context must already be present in each curve's
    'attributes' dict — populated by split_curves_with_prev_activity(), which by
    default contributes only the categorical prev_act_name. It flows through the
    standard attribute path in build_and_train_pipeline_exog(), so no changes to
    the feature-matrix logic are needed.

    prev_act_energy_map : dict | None
        One sensor's sub-map from build_prev_activity_energy_map(), built on the
        TRAINING frame across ALL activities. When given, PREV_ACT_ENERGY_FEATURES
        are derived from prev_act_name and injected into every training curve's
        attributes here, and re-derived from the stored map at predict time — so
        the model gets the predecessor's typical energy level, not just its name.
        The map is train-only and the lookup key is an event-log fact known at
        activity-start time, so this stays simulation-available and leak-free,
        unlike the include_prev_energy lagged features. None -> feature omitted,
        which is the ablation.

    The lagged-energy features (prev_act_mean/std/max/end/length) are only
    present when the splitter was called with include_prev_energy=True — off for
    reported runs, see that function's docstring.
    """
    # Inject before _infer_key_types runs inside the builder, so the new keys are
    # typed and scaled like any other numeric attribute. Curves are copied rather
    # than mutated — callers reuse the same list for other approaches.
    if prev_act_energy_map:
        train_curves = [dict(_c, attributes=_inject_prev_act_energy(
            _c['attributes'], prev_act_energy_map)) for _c in train_curves]

    pipeline = build_and_train_pipeline_exog(
        train_curves, variable,
        fixed_length=fixed_length,
        val_size=val_size,
        random_state=random_state,
        models=models,
        optimize_hyperparams=optimize_hyperparams,
        n_trials=n_trials,
        verbose=verbose,
        n_jobs=n_jobs,
        residual_target=residual_target,
        sample_weight_mode=sample_weight_mode,
    )
    pipeline['approach'] = 'ml_external'
    pipeline['prev_act_energy_map'] = prev_act_energy_map or {}

    # Per-activity training-curve medians — used as first-of-case defaults
    # during autoregressive test-time rollout (no real predecessor available).
    _by_act = {}
    for _c in train_curves:
        _v = np.asarray(_c['original_values'], dtype=float)
        if len(_v) == 0:
            continue
        _a = _c['activity']
        _by_act.setdefault(_a, {'m': [], 's': [], 'mx': [], 'e': [], 'l': []})
        _by_act[_a]['m'].append(float(np.mean(_v)))
        _by_act[_a]['s'].append(float(np.std(_v))  if len(_v) >= 2 else 0.0)
        _by_act[_a]['mx'].append(float(np.max(_v)))
        _by_act[_a]['e'].append(float(_v[-1]))
        _by_act[_a]['l'].append(float(len(_v)))

    pipeline['first_of_case_defaults'] = {
        _a: {
            'prev_act_name':   'none',
            'prev_act_mean':   float(np.median(_d['m'])),
            'prev_act_std':    float(np.median(_d['s'])),
            'prev_act_max':    float(np.median(_d['mx'])),
            'prev_act_end':    float(np.median(_d['e'])),
            'prev_act_length': float(np.median(_d['l'])),
        }
        for _a, _d in _by_act.items()
    }
    return pipeline


def predict_raw_curve_exog_prev_activity(raw_values, activity, attributes, pipeline,
                                         exog_values=None):
    """
    Predict using the exog + previous-activity pipeline.

    The prev-activity features live in 'attributes' and are handled transparently
    by the base exog predict function. The only work done here is re-deriving
    PREV_ACT_ENERGY_FEATURES from the pipeline's stored TRAIN-only map, keyed on
    the test curve's prev_act_name — the caller never has to supply them, and
    they cannot pick up test-set levels. No-op for pipelines trained without a
    map (and for the ml_dtw pipeline substituted into thin combos, which routes
    elsewhere via _dispatch_predict anyway).
    """
    attributes = _inject_prev_act_energy(attributes,
                                         pipeline.get('prev_act_energy_map'))
    return predict_raw_curve_exog(raw_values, activity, attributes, pipeline,
                                  exog_values=exog_values or {})


# =============================================================================
# STEP DTW — the leaf's STEP STRUCTURE is the regression target
# =============================================================================
#
# ml_cluster_dtw kept the canonical-space pointwise regression and tried to fix
# its smoothness from the outside (a finer partition, a predicted warp). Run 982
# measured that this does not work: its roughness/std ratios landed on top of
# ml_external's (0.25/0.46 vs 0.21/0.32 median, against ~0.50/0.59 for the real
# exemplar), because a conditional mean PER POSITION stays smooth no matter how
# the space is partitioned — the members' step edges never line up exactly, and
# averaging across a misaligned edge always produces a ramp.
#
# This approach changes the TARGET instead of the partition. A process-step load
# profile is, to first order, a sequence of segments: hold a level for a while,
# then move to the next. So the leaf is parameterised as M segments, and the
# regression predicts per instance the segment DURATIONS (fractions of the
# activity) and the segment LEVELS (raw units) — 2M numbers instead of
# fixed_length positions. The curve is RECONSTRUCTED from those parameters, so a
# step edge sits exactly where the duration models put it and is sharp by
# construction. Averaging still happens, but in PARAMETER space (a typical
# duration, a typical level), where it is harmless — not in value space, where
# it is what erases the steps.
#
# Where the segments come from:
#   1. the leaf's DTW medoid — a real measured curve — is segmented ONCE by
#      change-point detection: exact dynamic programming under an L2 cost, the
#      number of segments picked by a BIC-style penalty, so a featureless leaf
#      comes back with one segment instead of an arbitrary staircase;
#   2. every training curve is DTW-aligned to the medoid and the medoid's
#      breakpoints are carried through the alignment onto the curve's OWN
#      timeline — the same gamma-through-the-path mechanic as exemplar_dtw's
#      warp, evaluated at the breakpoints instead of at fixed knots;
#   3. that gives every training curve the same M segments in its own time, and
#      with them the per-curve targets: M duration fractions and M raw levels.
#
# Every duration/level model is validated against its constant fallback (the
# training mean/median of that parameter), the same do-no-harm rule the
# exemplar/warp models follow — an unpredictable parameter degrades to the
# leaf's typical value, never to noise.
#
# Within a segment the output is filled from the medoid's own samples for that
# segment, rescaled to the predicted level (segment_fill='medoid'), so ramps and
# duty cycling INSIDE a step survive; 'flat' emits the predicted level as a
# constant, which makes every output value purely model-generated and the curve
# an actual step function. Either way the defining property holds: WHERE each
# step is and HOW HIGH it sits is predicted per instance, never replayed.
#
# One deliberate non-feature: no shape classifier. Hard-routing whole instances
# between per-cluster models is what gave ml_cluster_dtw its catastrophic tail
# (q99 sMAE 51.7 vs ml_external's 45.3; a wrong route lands on a model fitted to
# a different profile). Here every instance flows through the same segment
# models, so the worst failure is a mis-sized segment — local and bounded.
#
# 'ml_step_dtw_smooth' (gain_mode='smooth') is the SAME pipeline with one change
# in reconstruction: the medoid fill is continuous across segment boundaries by
# construction (consecutive segments read contiguous medoid samples), so the
# only jump the step output has is the per-segment gain lv[m]/mu applied
# piecewise-CONSTANT. The smooth variant evaluates those gains at segment
# midpoints and interpolates them linearly across the curve instead, so a
# breakpoint that lands inside a ramp (experiment_992's simulated steam
# declines) no longer terraces it. Deliberately a separate approach name, not a
# flag on ml_step_dtw: the step texture is part of what earned 984's realism
# ratios, so the two must land side by side in every results table.

STEP_DTW_MAX_SEGMENTS = 8

# 'medoid' | 'flat' — see the section comment. 'medoid' is the default because
# within-step texture (decay ramps, duty cycling) is real signal the flat fill
# throws away; switch to 'flat' for the fully-model-generated ablation.
STEP_DTW_SEGMENT_FILL = _os_seed.environ.get(
    'PIPELINE_STEP_DTW_SEGMENT_FILL', 'medoid').strip().lower()

# Do-no-harm fallback gate. A leaf whose signal is not step-shaped (smooth
# ramps: process_4_2's temperature/mass-flow sensors in experiment_984) pays
# for the step template with a heavy pointwise tail — 22/351 leaves came out
# >2 sMAE worse than ml_external there. When the builder is handed the leaf's
# already-trained ml_external pipeline, it compares the two on ITS OWN held-out
# validation curves and routes the leaf to the fallback iff the step pipeline
# loses CLEARLY: val MAE > ratio x the fallback's. The margin is wide on
# purpose — a smooth conditional median beats a textured curve slightly on
# pointwise error even where the steps are real (the median-floor lesson), and
# a near-tie must go to the step model or the realism gain is deleted leaf by
# leaf. Raise the ratio towards infinity to disable the gate.
STEP_DTW_FALLBACK_RATIO = float(_os_seed.environ.get(
    'PIPELINE_STEP_DTW_FALLBACK_RATIO', '1.25'))


def _stepdtw_segment_medoid(shape, max_segments=STEP_DTW_MAX_SEGMENTS, min_seg=None):
    """
    Interior breakpoints (fractions in (0,1)) of the best piecewise-constant
    approximation of `shape` — exact DP under an L2 cost, one pass per segment
    count, the count chosen by a BIC-style penalty. Empty array = one segment.

    Returns (breakpoints, explained): `explained` is the fraction of the
    one-segment residual the chosen segmentation removes (0 when it stays at
    one segment). A diagnostic, not a gate: a staircase approximates even a
    smooth ramp well, so this cannot by itself tell steps from ramps — the
    fallback decision below is made on held-out prediction error instead.
    """
    y = np.asarray(shape, dtype=float)
    y = np.where(np.isfinite(y), y, 0.0)
    L = len(y)
    if min_seg is None:
        # Short enough to catch a brief step, long enough that a single spike
        # does not become its own segment.
        min_seg = max(2, L // 25)
    m_max = int(max(1, min(int(max_segments), L // min_seg)))
    if m_max <= 1 or L < 2 * min_seg:
        return np.array([], dtype=float), 0.0

    c1 = np.concatenate([[0.0], np.cumsum(y)])
    c2 = np.concatenate([[0.0], np.cumsum(y * y)])

    INF = np.inf
    cost = np.full((m_max + 1, L + 1), INF)
    back = np.zeros((m_max + 1, L + 1), dtype=int)
    cost[0, 0] = 0.0
    for m in range(1, m_max + 1):
        for j in range(m * min_seg, L + 1):
            i = np.arange((m - 1) * min_seg, j - min_seg + 1)
            n_ = j - i
            s = c1[j] - c1[i]
            tot = cost[m - 1, i] + (c2[j] - c2[i]) - s * s / n_
            a = int(np.argmin(tot))
            cost[m, j], back[m, j] = tot[a], i[a]

    # Each extra segment buys residual sum of squares but costs two parameters
    # (a breakpoint and a level). max() guards the log when the fit is exact.
    rss = cost[1:, L]
    ms = np.arange(1, m_max + 1)
    bic = L * np.log(np.maximum(rss, 1e-12) / L) + 2.0 * ms * np.log(L)
    best_m = int(ms[int(np.argmin(bic))])

    cuts = []
    j = L
    for m in range(best_m, 0, -1):
        i = int(back[m, j])
        if 0 < i < L:
            cuts.append(i)
        j = i
    explained = float(np.clip(1.0 - rss[best_m - 1] / max(rss[0], 1e-12), 0.0, 1.0))
    return np.array(sorted(c / L for c in cuts), dtype=float), explained


def _stepdtw_targets(shapes, medoid_row, breaks, train_curves):
    """
    Carry the medoid's breakpoints onto every training curve through DTW and
    read off the per-curve targets. Returns (durations, levels): durations are
    fractions of the curve summing to 1 per row (a segment the curve does not
    have comes out ~0, which is informative, not an error); levels are means of
    the RAW values inside each segment, in raw units.
    """
    M = len(breaks) + 1
    n_curves = len(train_curves)
    durations = np.zeros((n_curves, M))
    levels = np.zeros((n_curves, M))
    med_shape = shapes[medoid_row]
    for r in range(n_curves):
        if M > 1:
            if r == medoid_row:
                pos = np.asarray(breaks, dtype=float).copy()
            else:
                pos = _warp_knots_from_path(med_shape, shapes[r],
                                            knots=tuple(breaks))
            pos = np.clip(np.maximum.accumulate(np.asarray(pos, dtype=float)),
                          0.0, 1.0)
        else:
            pos = np.array([], dtype=float)
        edges = np.concatenate([[0.0], pos, [1.0]])
        durations[r] = np.diff(edges)

        v = np.asarray(train_curves[r]['original_values'], dtype=float)
        Lr = len(v)
        for m in range(M):
            i0 = min(int(np.floor(edges[m] * Lr)), Lr - 1)
            i1 = min(max(int(np.ceil(edges[m + 1] * Lr)), i0 + 1), Lr)
            if i1 <= i0:
                i0, i1 = Lr - 1, Lr
            with np.errstate(invalid='ignore'):
                levels[r, m] = float(np.nanmean(v[i0:i1]))
    # A curve of NaNs inside one segment must not poison the whole bank.
    if not np.all(np.isfinite(levels)):
        _fb = float(np.nanmedian(levels)) if np.any(np.isfinite(levels)) else 0.0
        levels = np.where(np.isfinite(levels), levels, _fb)
    return durations, levels


def build_and_train_pipeline_step_dtw(
    train_curves,
    variable,
    fixed_length=None,
    val_size=0.2,
    random_state=42,
    models=None,
    optimize_hyperparams=False,
    n_trials=50,
    verbose=1,
    n_jobs=1,
    prev_act_energy_map=None,
    max_segments=STEP_DTW_MAX_SEGMENTS,
    segment_fill=None,
    gain_mode=None,
    fallback_pipeline=None,
    **_ignored_hp_kwargs,
):
    """
    Segment-parameter regression via DTW correspondence — see the section
    comment. There are 2M models, one per segment duration and one per segment
    level.

    `models` and `optimize_hyperparams` / `n_trials` ARE used here, unlike the
    exemplar builders: `models` is the candidate family dict the sklearn
    approaches compete over, and optimize_hyperparams runs the same Optuna
    search per segment. It fires only when a real held-out split exists
    (`_tunable` below) — with too few curves the search would select on its own
    training rows, so the fixed fit is used instead. Both arrive from
    CURVE_OPTIMIZE_HYPERPARAMS / CURVE_N_OPTUNA_TRIALS via
    _train_curve_only_worker.

    gain_mode : 'step' (default) applies each segment's level gain piecewise-
    constant — sharp edges, the original ml_step_dtw; 'smooth' interpolates the
    gains between segment midpoints, which removes the boundary jumps and is
    published under the approach name 'ml_step_dtw_smooth'. Training, targets
    and models are identical; only reconstruction differs, and the fallback
    gate below scores whichever reconstruction this pipeline will actually use.

    fallback_pipeline : the leaf's already-trained ml_external pipeline, or
    None. When given, the assembled step pipeline is compared against it on
    this builder's own held-out validation curves and the leaf is routed to the
    fallback iff the step pipeline loses by more than STEP_DTW_FALLBACK_RATIO —
    see that constant for why the margin is wide. Both builders receive the
    same curve list, the same val_size and the same random_state, so their
    train/validation partitions coincide and neither side is scored on curves
    it fitted.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    # Same prev-activity energy injection as ml_external / ml_cluster_dtw, so
    # the features are identical and the comparison isolates the target change.
    if prev_act_energy_map:
        train_curves = [dict(_c, attributes=_inject_prev_act_energy(
            _c['attributes'], prev_act_energy_map)) for _c in train_curves]

    if fixed_length is None:
        fixed_length = max(2, int(round(np.median(
            [len(c['original_values']) for c in train_curves]))))

    resampled = np.array([
        np.interp(np.linspace(0, 1, fixed_length),
                  np.linspace(0, 1, len(c['original_values'])),
                  np.asarray(c['original_values'], dtype=float))
        for c in train_curves
    ])
    scale = np.mean(resampled, axis=1, keepdims=True)
    shapes = resampled / np.where(np.abs(scale) < 1e-12, 1.0, scale)

    # ── 1. The leaf's DTW medoid — the curve the segmentation is read from ──
    if len(train_curves) > 2:
        D, sub_idx = _dtw_distance_matrix(shapes, random_state=random_state)
        medoid_row = int(sub_idx[int(D.sum(axis=1).argmin())])
    else:
        medoid_row = 0

    # ── 2. Segment the medoid, 3. carry the breakpoints onto every curve ────
    breaks, seg_explained = _stepdtw_segment_medoid(shapes[medoid_row],
                                                    max_segments=max_segments)
    n_segments = len(breaks) + 1
    durations, levels = _stepdtw_targets(shapes, medoid_row, breaks, train_curves)

    # ── 4. One L1 gradient-boosting model per parameter, do-no-harm kept ────
    exog_cols = sorted({col for c in train_curves
                        for col in (c.get('exog_values') or {}).keys()})
    X = _exemplar_feature_frame(train_curves, exog_cols=exog_cols)
    feature_columns = list(X.columns)

    n = len(train_curves)
    idx = np.arange(n)
    if 0 < val_size < 1 and n >= 10:
        tr_i, vl_i = train_test_split(idx, test_size=val_size, random_state=random_state)
    else:
        tr_i = vl_i = idx

    const_durations = durations.mean(axis=0)
    const_levels = np.median(levels, axis=0)

    # A real held-out split is required to TUNE on: when val_size collapses
    # (tr_i is vl_i, i.e. too few curves) the search would be selecting on its own
    # training rows, so the historical fixed fit is used instead.
    _tunable = bool(optimize_hyperparams) and not np.array_equal(tr_i, vl_i)

    # The candidate families, same dict the sklearn approaches compete. Passed in
    # by the worker as `models` (already narrowed by PIPELINE_CURVE_MODELS); the
    # historical single-family behaviour is the fallback for direct callers.
    _families = dict(models) if models else {
        'Hist Gradient Boosting': HistGradientBoostingRegressor}

    def _fit_one(name, model_class, y):
        """Fit one family to one segment target, tuned if there is a split to tune on."""
        if _tunable:
            if not verbose:
                optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                m = model_class(**_curve_model_trial_params(trial, name,
                                                            random_state, n_jobs))
                m.fit(X.values[tr_i], y[tr_i])
                return -float(np.mean(np.abs(m.predict(X.values[vl_i]) - y[vl_i])))

            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=random_state))
            study.optimize(objective, n_trials=n_trials)
            m = model_class(**_curve_model_finalise_params(name, study.best_params,
                                                           random_state, n_jobs))
        else:
            # Untuned: bake the absolute-error loss in here too, or this branch
            # silently trains on squared error while the metrics are L1.
            _lk = _curve_model_loss_kwargs(name)
            try:
                m = model_class(random_state=random_state, **_lk)
            except TypeError:
                try:
                    m = model_class(**_lk)
                except TypeError:
                    m = model_class()
        m.fit(X.values[tr_i], y[tr_i])
        return m

    def _fit_segment_model(y):
        """Best family for one segment target (a duration or a level), by held-out MAE.

        Every family in `models` competes, exactly as they do for ml_external and
        ml_only. Before this the step model was locked to a fixed 200-iteration
        HGB while the incumbent picked its best of six AND tuned it, so any gap
        between them confounded the segment reparameterisation with the model
        choice. Returns (model, name, val_mae); (None, ...) if nothing fitted."""
        best, best_name, best_mae = None, None, np.inf
        for _name, _cls in _families.items():
            try:
                _m = _fit_one(_name, _cls, y)
                _mae = float(np.mean(np.abs(_m.predict(X.values[vl_i]) - y[vl_i])))
            except Exception:
                continue
            if _mae < best_mae:
                best, best_name, best_mae = _m, _name, _mae
        return best, best_name, best_mae

    def _fit_bank(targets, const):
        bank, names = [], []
        for j in range(targets.shape[1]):
            m_, nm_ = None, None
            if feature_columns and len(tr_i) >= 8:
                _m, _nm, _mae = _fit_segment_model(targets[:, j])
                # Do-no-harm, unchanged: the winning family still has to beat the
                # segment's own constant on the held-out curves to be kept at all.
                if _m is not None and _mae < float(np.mean(np.abs(const[j] - targets[vl_i, j]))):
                    m_, nm_ = _m, _nm
            bank.append(m_)
            names.append(nm_)
        return bank, names

    duration_models, duration_model_names = (
        _fit_bank(durations, const_durations) if n_segments > 1
        else ([None] * n_segments, [None] * n_segments))
    level_models, level_model_names = _fit_bank(levels, const_levels)

    _gain = (gain_mode or 'step').strip().lower()
    _tag = 'ml_step_dtw_smooth' if _gain == 'smooth' else 'ml_step_dtw'

    if verbose:
        _won = sorted({x for x in (duration_model_names + level_model_names) if x})
        print(f"  [{_tag}] {n} curves -> {n_segments} segment(s); "
              f"{len(_families)} famil(y/ies)"
              f"{f' x Optuna {n_trials} trials' if _tunable else ''} per segment; "
              f"duration models kept {sum(m is not None for m in duration_models)}/{n_segments}, "
              f"level models kept {sum(m is not None for m in level_models)}/{n_segments}"
              + (f'; won: {", ".join(_won)}' if _won else ''))

    pipeline = {
        'approach':          _tag,
        # Pointwise median, NOT the naive floor's level: step-DTW never predicts
        # from this (it reconstructs from medoid_values/break_fracs/the segment
        # models), but _build_gap_fill_curves and simulation.py read leaf
        # reference curves for a representative SHAPE. Was
        # _median_floor_reference until that helper went level-only.
        'reference_curve':   _pointwise_median_curve(train_curves, fixed_length),
        'fixed_length':      fixed_length,
        'medoid_values':     np.asarray(train_curves[medoid_row]['original_values'],
                                        dtype=float),
        'break_fracs':       np.asarray(breaks, dtype=float),
        'n_segments':        n_segments,
        'const_durations':   const_durations,
        'const_levels':      const_levels,
        'duration_models':   duration_models,
        'level_models':      level_models,
        'segment_fill':      (segment_fill or STEP_DTW_SEGMENT_FILL),
        'gain_mode':         _gain,
        'feature_columns':   feature_columns,
        'exog_cols':         exog_cols,
        'prev_act_energy_map': prev_act_energy_map or {},
        'model_name':        (f'{n_segments} segments x durations+levels'
                              + (f' [{len(_families)} families'
                                 + (f', tuned {n_trials}]' if _tunable else ']'))),
        'duration_model_names': duration_model_names,
        'level_model_names':    level_model_names,
        'segment_explained': seg_explained,
        'fallback_active':   False,
    }

    # ── Do-no-harm fallback gate — see STEP_DTW_FALLBACK_RATIO ──────────────
    # Scored through both REAL predictors on the held-out curves, decode and
    # floor semantics included, exactly like _attach_median_floor scores.
    if fallback_pipeline is not None and len(vl_i) > 0:
        _step_err, _fb_err = [], []
        for _i in vl_i:
            _c = train_curves[int(_i)]
            _raw = np.asarray(_c['original_values'], dtype=float)
            if len(_raw) < 2:
                continue
            try:
                _ys = np.asarray(predict_raw_curve_step_dtw(
                    _raw, _c['activity'], _c['attributes'], pipeline,
                    exog_values=_c.get('exog_values', {})), dtype=float)
                _yf = np.asarray(predict_raw_curve_exog_prev_activity(
                    _raw, _c['activity'], _c['attributes'], fallback_pipeline,
                    exog_values=_c.get('exog_values', {})), dtype=float)
            except Exception:
                _step_err, _fb_err = [], []   # unscoreable -> keep the step model
                break
            for _y, _acc in ((_ys, _step_err), (_yf, _fb_err)):
                if len(_y) != len(_raw):
                    _y = np.interp(np.linspace(0, 1, len(_raw)),
                                   np.linspace(0, 1, len(_y)), _y)
                _acc.append(float(np.abs(_clip_physical(_y) - _raw).mean()))
        if _step_err and _fb_err:
            _step_mae = float(np.mean(_step_err))
            _fb_mae = float(np.mean(_fb_err))
            _fired = _step_mae > STEP_DTW_FALLBACK_RATIO * _fb_mae
            pipeline['fallback_step_mae'] = _step_mae
            pipeline['fallback_ml_external_mae'] = _fb_mae
            pipeline['fallback_active'] = _fired
            if _fired:
                # Stored only when it fires, so a leaf that keeps the step
                # model does not drag a second full pipeline through pickling.
                pipeline['fallback_pipeline'] = fallback_pipeline
                pipeline['model_name'] += ' -> ml_external fallback'
            if verbose:
                print(f"  [{_tag}] fallback gate: step val MAE={_step_mae:.4f} vs "
                      f"ml_external {_fb_mae:.4f} (ratio {STEP_DTW_FALLBACK_RATIO}, "
                      f"explained={seg_explained:.2f}) -> "
                      f"{'ML_EXTERNAL FALLBACK' if _fired else 'step model kept'}")

    _attach_median_floor(
        pipeline, train_curves,
        [train_curves[i]['instance_id'] for i in vl_i],
        lambda rv, c: predict_raw_curve_step_dtw(
            rv, c['activity'], c['attributes'], pipeline,
            exog_values=c.get('exog_values', {})),
        label=f' {variable} ({_tag})', verbose=verbose,
    )
    return pipeline


def predict_raw_curve_step_dtw(raw_values, activity, attributes, pipeline,
                               exog_values=None):
    """
    Predict the segment durations and levels from the features and reassemble
    the curve. Reads only the feature vector and the target length from the
    instance — the test curve's values are never touched, so this runs
    unchanged inside the simulation.
    """
    _floor = _median_floor_prediction(raw_values, attributes, pipeline)
    if _floor is not None:
        return _floor

    # Leaf routed to ml_external by the do-no-harm gate — the step template
    # lost clearly on this leaf's held-out curves (see STEP_DTW_FALLBACK_RATIO).
    if pipeline.get('fallback_active') and pipeline.get('fallback_pipeline') is not None:
        return predict_raw_curve_exog_prev_activity(
            raw_values, activity, attributes, pipeline['fallback_pipeline'],
            exog_values=exog_values or {})

    attributes = _inject_prev_act_energy(attributes,
                                         pipeline.get('prev_act_energy_map'))
    attributes = attributes or {}
    n = max(2, int(attributes.get('_pred_curve_length', len(raw_values))))

    curve = {'attributes': attributes, 'original_values': np.zeros(n),
             'exog_values': exog_values or {}}
    X = _exemplar_feature_frame([curve], pipeline['feature_columns'],
                                exog_cols=pipeline.get('exog_cols')).values

    n_segments = int(pipeline['n_segments'])
    d = np.asarray(pipeline['const_durations'], dtype=float).copy()
    for j, m in enumerate(pipeline.get('duration_models') or []):
        if m is None:
            continue
        try:
            d[j] = float(m.predict(X)[0])
        except Exception:
            pass
    d = np.clip(d, 0.0, None)
    d = d / d.sum() if d.sum() > 0 else np.full(n_segments, 1.0 / n_segments)

    lv = np.asarray(pipeline['const_levels'], dtype=float).copy()
    for j, m in enumerate(pipeline.get('level_models') or []):
        if m is None:
            continue
        try:
            lv[j] = float(m.predict(X)[0])
        except Exception:
            pass
    lv = np.where(np.isfinite(lv), lv, 0.0)

    # A predicted duration of ~0 simply skips that segment for this instance;
    # monotone cumulative edges pinned to [0, n] guarantee full coverage.
    edges = np.round(np.cumsum(np.concatenate([[0.0], d])) * n).astype(int)
    edges[0], edges[-1] = 0, n
    edges = np.maximum.accumulate(edges)

    med = np.asarray(pipeline.get('medoid_values', []), dtype=float)
    med_edges = np.concatenate([[0.0],
                                np.asarray(pipeline.get('break_fracs', []), dtype=float),
                                [1.0]])
    Lm = len(med)
    fill = pipeline.get('segment_fill', 'medoid')

    # gain_mode='smooth' (ml_step_dtw_smooth): two continuity repairs over the
    # step fill, and BOTH are needed (experiment_993's simulated heat declines
    # stayed terraced with the gain repair alone; a 995 attempt at per-segment
    # resampling with smoothed gains re-terraced them — the medoid is too coarse
    # for ANY per-segment replay when instances run longer than it):
    #   1. the medoid is read through ONE piecewise-linear time warp (output
    #      edge fracs -> medoid edge fracs) and interpolated continuously over
    #      the whole curve. Per-segment resampling — the step fill — breaks
    #      down at this data's granularity: a decline segment holds only 1-2
    #      medoid samples, so stretching it renders a flat tread while the
    #      medoid's real per-sample decline lands as a cliff at the boundary;
    #   2. the per-segment level gains are evaluated at segment midpoints and
    #      interpolated linearly (flat beyond the first/last midpoint —
    #      np.interp's clamping) instead of applied piecewise-constant.
    # Slope kinks at the warp knots remain on purpose: they are the predicted
    # durations doing their job, not reconstruction artifacts. Real steps in
    # the medoid survive (steep but continuous); only the fill's jumps go.
    # Levels are safe: 994 (warp) vs 995 (per-segment) steam Wasserstein value
    # medians were 73.3 vs 74.6 — the warp is not trading level accuracy.
    if pipeline.get('gain_mode', 'step') == 'smooth':
        if fill == 'medoid' and Lm >= 2:
            xs = np.asarray(edges, dtype=float) / n
            ys = med_edges
            # Collapse zero-length output segments: their medoid chunk is
            # dropped from the warp, same as the step fill's skip.
            _keep = np.r_[xs[1:] != xs[:-1], True]
            pos = np.interp((np.arange(n) + 0.5) / n, xs[_keep], ys[_keep])
            base = np.interp(pos, np.linspace(0.0, 1.0, Lm), med)
        else:
            # Flat fill (or a degenerate medoid): base 1, so the interpolated
            # gains below ARE the curve — a polyline through the step levels.
            base = np.ones(n, dtype=float)
        mids, gains = [], []
        for m in range(n_segments):
            a, b = int(edges[m]), int(edges[m + 1])
            if b <= a:
                continue
            mu = float(np.mean(base[a:b]))
            if abs(mu) > 1e-12:
                mids.append((a + b - 1) / 2.0)
                gains.append(lv[m] / mu)
        if not mids:
            return base
        return base * np.interp(np.arange(n, dtype=float), mids, gains)

    y = np.empty(n, dtype=float)
    for m in range(n_segments):
        a, b = int(edges[m]), int(edges[m + 1])
        if b <= a:
            continue
        if fill == 'medoid' and Lm >= 1:
            i0 = min(int(np.floor(med_edges[m] * Lm)), Lm - 1)
            i1 = min(max(int(np.ceil(med_edges[m + 1] * Lm)), i0 + 1), Lm)
            seg = med[i0:i1]
            if b - a > len(seg):
                # Upsampling: index selection would REPEAT samples, which turns
                # the medoid's ramps into staircase treads (experiment_988's
                # simulated heat declines). Interpolating within one real curve
                # smooths nothing across instances, so it is safe here.
                seg = np.interp(np.linspace(0.0, len(seg) - 1.0, b - a),
                                np.arange(len(seg), dtype=float), seg)
            else:
                # Downsampling: index selection, same rule as the exemplar
                # family, so the medoid's within-segment texture survives.
                seg = seg[np.clip(np.round(np.linspace(0, len(seg) - 1, b - a)).astype(int),
                                  0, len(seg) - 1)]
            mu = float(np.mean(seg))
            y[a:b] = seg * (lv[m] / mu) if abs(mu) > 1e-12 else lv[m]
        else:
            y[a:b] = lv[m]
    return y


# =============================================================================

import os as _os
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


class _Seq2SeqTransformer(nn.Module):
    """
    Transformer counterpart of _Seq2SeqLSTM, with an identical call signature so
    both are drop-in interchangeable inside the seq2seq family (and behind the
    same 'seq2seq*' approach names — the cell type is an internal choice, not a
    new approach).

    Encoder-only and NON-autoregressive: the input features at every timestep
    (position, curve_length, activity, attributes, exog) are known up front, so
    there is nothing to condition auto-regressively on. Predicting the whole
    canonical curve in one pass removes the LSTM's exposure bias (train with
    teacher forcing, infer on its own predictions) and lets attention run in
    parallel over timesteps instead of stepping T times.

    `targets` / `teacher_forcing_ratio` are accepted and ignored — they exist
    only so the training loop can call either cell identically.
    """
    def __init__(self, input_size, seq_len, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        # nn.MultiheadAttention requires d_model % nhead == 0.
        nhead = next((h for h in (8, 4, 2, 1) if hidden_size % h == 0), 1)
        self.seq_len    = seq_len
        self.nhead      = nhead
        self.input_proj = nn.Linear(input_size, hidden_size)
        # Learned positional embedding, sized to this pipeline's fixed_length so
        # hundreds of persisted per-combo models stay small.
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        nn.init.normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead,
            dim_feedforward=4 * hidden_size, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        # enable_nested_tensor=False: incompatible with norm_first, and it only
        # warns per instantiation — noisy across hundreds of per-combo models.
        self.encoder  = nn.TransformerEncoder(layer, num_layers=num_layers,
                                              enable_nested_tensor=False)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, x, targets=None, teacher_forcing_ratio=0.5):
        """
        x       : (B, T, F)
        targets : ignored (see class docstring)
        returns : (B, T)
        """
        B, T, _ = x.shape
        if T > self.seq_len:
            raise ValueError(
                f"_Seq2SeqTransformer got T={T} > seq_len={self.seq_len}; the "
                f"positional embedding is sized to the pipeline's fixed_length."
            )
        h = self.input_proj(x) + self.pos_emb[:, :T, :]
        h = self.encoder(h)
        return self.out_proj(h).squeeze(-1)                       # (B, T)


# Which recurrent/attention cells compete inside every seq2seq* approach. Both
# are trained per (sensor, activity, object) combo and the one with the lower
# validation loss is kept — same "train both, keep the winner" rule the duration
# models use. Override with PIPELINE_SEQ2SEQ_CELLS=lstm (or =transformer) to
# train only one, e.g. to reproduce a pre-transformer run or to halve the cost.
_SEQ2SEQ_KNOWN_CELLS = ('lstm', 'transformer')


def _resolve_seq2seq_cells(spec=None):
    """
    Parse PIPELINE_SEQ2SEQ_CELLS (set by 01_pipeline.py's 'seq2seq_cells' key).

    Validated here rather than at training time: an unrecognised cell used to
    survive into _fit_seq2seq_with_selection, where every candidate is skipped
    and the run dies with 'no seq2seq cell trained successfully' after the
    curve stage has already finished.
    """
    raw = _os.environ.get('PIPELINE_SEQ2SEQ_CELLS') if spec is None else spec
    if not raw or not raw.strip():
        return _SEQ2SEQ_KNOWN_CELLS
    picked, unknown = [], []
    for c in raw.split(','):
        c = c.strip().lower()
        if not c:
            continue
        if c in _SEQ2SEQ_KNOWN_CELLS:
            if c not in picked:
                picked.append(c)
        else:
            unknown.append(c)
    if unknown:
        print(f"[sim_extractor] WARNING: PIPELINE_SEQ2SEQ_CELLS has unknown cells "
              f"{unknown} — valid: {list(_SEQ2SEQ_KNOWN_CELLS)}. Ignoring those.")
    if not picked:
        print(f"[sim_extractor] WARNING: PIPELINE_SEQ2SEQ_CELLS={raw!r} selected no valid "
              f"cell — falling back to {list(_SEQ2SEQ_KNOWN_CELLS)}.")
        return _SEQ2SEQ_KNOWN_CELLS
    return tuple(picked)


_SEQ2SEQ_CELL_TYPES = _resolve_seq2seq_cells()


def _seq2seq_device():
    """
    Device for seq2seq training and inference — CPU on purpose.

    These models are trained in a pool of *forked* workers
    (_train_seq2seq_worker), and a fork of a parent that has already touched the
    CUDA driver can never initialize CUDA — see the fork-poisoning note there.

    CUDA would still be the wrong choice even from a clean fork: the trained
    nn.Module is pickled back to the parent, so a CUDA model would initialize
    CUDA in the *parent* and poison the worker pools of every later process.
    And it buys little — measured 16-wide on 2-layer/128-unit cells over ~100
    steps, GPU came out only ~1.4x ahead of 16 single-threaded CPU workers,
    because models this small leave the GPU launch-bound and contended.
    """
    return torch.device('cpu')


def _fit_seq2seq_with_selection(
    X_tr, y_tr_n, X_vl, y_vl_n, device, input_size, seq_len,
    hidden_size, num_layers, dropout, epochs, batch_size, lr,
    teacher_forcing_ratio, patience, verbose, cell_types=None,
    random_state=None,
):
    """
    Train one model per cell type in `cell_types` on identical data/splits and
    return the one with the lowest validation loss.

    Returns
    -------
    (model, best_val_loss, best_cell, val_loss_by_cell)
        model            — the winning nn.Module, in eval mode, best-epoch weights
        best_val_loss    — its validation MSE in canonical (normalised) space
        best_cell        — 'lstm' | 'transformer'
        val_loss_by_cell — {cell: val_loss} for every candidate that trained
    """
    cell_types = tuple(cell_types) if cell_types else _SEQ2SEQ_CELL_TYPES
    criterion  = nn.MSELoss()

    # Reproducibility: an explicit generator for the shuffling DataLoader (it
    # otherwise draws from torch's global RNG, so batch order depended on
    # whatever else had consumed randomness first).
    _seed = GLOBAL_RANDOM_SEED if random_state is None else int(random_state)
    _loader_gen = torch.Generator()
    _loader_gen.manual_seed(_seed)
    loader = DataLoader(TensorDataset(X_tr, y_tr_n), batch_size=batch_size,
                        shuffle=True, generator=_loader_gen)

    results = {}
    for cell in cell_types:
        # Re-seed per cell so 'lstm' and 'transformer' start from the same
        # stream: which cell wins must not depend on which trained first.
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
        _loader_gen.manual_seed(_seed)
        if cell == 'lstm':
            model = _Seq2SeqLSTM(input_size, hidden_size=hidden_size,
                                 num_layers=num_layers, dropout=dropout).to(device)
        elif cell == 'transformer':
            model = _Seq2SeqTransformer(input_size, seq_len=seq_len,
                                        hidden_size=hidden_size,
                                        num_layers=num_layers, dropout=dropout).to(device)
        else:
            print(f"    [WARN] unknown seq2seq cell '{cell}' — skipped.")
            continue

        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
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
                print(f"    [{cell}] Epoch {epoch:4d}/{epochs}  val_loss={val_loss:.5f}")

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"    [{cell}] Early stopping at epoch {epoch}")
                    break

        if best_state is None:      # never improved on inf → no usable model
            continue
        model.load_state_dict(best_state)
        model.eval()
        results[cell] = (model, best_val_loss)

    if not results:
        raise RuntimeError(f"no seq2seq cell trained successfully (tried {cell_types})")

    best_cell = min(results, key=lambda c: results[c][1])
    model, best_val_loss = results[best_cell]
    val_loss_by_cell = {c: v for c, (_, v) in results.items()}

    if verbose:
        _scores = '  '.join(f'{c}={v:.5f}' for c, v in val_loss_by_cell.items())
        print(f"    Cell selection: {_scores}  ->  {best_cell}")

    return model, best_val_loss, best_cell, val_loss_by_cell


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
    fixed_length=None,
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
    if fixed_length is None:
        fixed_length = max(2, int(round(np.median([len(c['original_values']) for c in train_curves]))))
    resampled_for_dba = np.array([
        np.interp(
            np.linspace(0, 1, fixed_length),
            np.linspace(0, 1, len(c['original_values'])),
            c['original_values']
        )
        for c in train_curves
    ])[:, :, np.newaxis]

    dba_barycenter  = _robust_dtw_barycenter(resampled_for_dba, barycenter_size=fixed_length)
    # Restore the on/off duty cycle DBA averages away (see
    # _zero_calibrate_barycenter); no-op for non-intermittent sensors.
    reference_curve = _zero_calibrate_barycenter(dba_barycenter[:, 0], resampled_for_dba)

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

    device = _seq2seq_device()
    input_size = X_tr.shape[-1]

    model, best_val_loss, best_cell, val_loss_by_cell = _fit_seq2seq_with_selection(
        X_tr, y_tr_n, X_vl, y_vl_n, device, input_size, fixed_length,
        hidden_size, num_layers, dropout, epochs, batch_size, lr,
        teacher_forcing_ratio, patience, verbose,
        random_state=random_state,
    )

    if verbose:
        print(f"    Best val_loss={best_val_loss:.5f}  (cell={best_cell})")

    pipeline = {
        'approach':            'seq2seq',
'cell_type':           best_cell,
'val_loss_by_cell':    val_loss_by_cell,
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

    # Median floor — see _attach_median_floor. Applied to the seq2seq approaches
    # too, so the results table compares like with like: every learned approach
    # is held to the same floor, none is protected while another is not.
    _attach_median_floor(
        pipeline, train_curves, val_inst,
        lambda rv, c: predict_raw_curve_seq2seq(rv, c['activity'], c['attributes'], pipeline),
        label=f' {variable}', verbose=verbose,
    )

    return pipeline


def predict_raw_curve_seq2seq(raw_values, activity, attributes, pipeline):
    """
    Predict a single raw curve using the trained Seq2Seq pipeline.

    1. Build 100-step feature sequence (same as baseline).
    2. Run the LSTM decoder (no teacher forcing) → 100 canonical predictions.
    3. DTW-decode back to raw length using the same path inversion as baseline.
    """
    _floor = _median_floor_prediction(raw_values, attributes, pipeline)
    if _floor is not None:
        return _floor

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
            'curve_length': attributes.get('_pred_curve_length', len(raw_values)),
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

    # Missing-attribute robustness -- see the identical fillna(0) in the other
    # seq2seq* predict functions for the full explanation: any trained
    # feature absent from `attributes` (e.g. hour_of_day/day_of_week) is NaN
    # here and would silently propagate to an all-NaN prediction otherwise.
    df_seq = df_seq.fillna(0)

    X = torch.tensor(df_seq.values.astype(np.float32)).unsqueeze(0).to(device)  # (1,T,F)

    model.eval()
    with torch.no_grad():
        y_norm = model(X, targets=None, teacher_forcing_ratio=0.0)  # (1, T)

    y_ref_pred = (y_norm.squeeze(0).cpu().numpy() * y_std + y_mean)  # (T,)

    # DTW path decode — identical to baseline predict_raw_curve
    return _decode_canonical_to_raw(y_ref_pred, raw_values, reference_curve,
                                    fixed_length)


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
    fixed_length=None,
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

    if fixed_length is None:
        fixed_length = max(2, int(round(np.median([len(c['original_values']) for c in train_curves]))))
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

    device     = _seq2seq_device()
    input_size = X_tr.shape[-1]
    model, best_val_loss, best_cell, val_loss_by_cell = _fit_seq2seq_with_selection(
        X_tr, y_tr_n, X_vl, y_vl_n, device, input_size, fixed_length,
        hidden_size, num_layers, dropout, epochs, batch_size, lr,
        teacher_forcing_ratio, patience, verbose,
        random_state=random_state,
    )

    if verbose:
        print(f"    Best val_loss={best_val_loss:.5f}  (cell={best_cell})")

    pipeline = {
        'approach':             'seq2seq_only',
'cell_type':           best_cell,
'val_loss_by_cell':    val_loss_by_cell,
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

    # Median floor — see _attach_median_floor.
    _attach_median_floor(
        pipeline, train_curves, val_inst,
        lambda rv, c: predict_raw_curve_seq2seq_only(rv, c['activity'], c['attributes'], pipeline),
        label=f' {variable}', verbose=verbose,
    )

    return pipeline


def predict_raw_curve_seq2seq_only(raw_values, activity, attributes, pipeline):
    """
    Predict using the pure Seq2Seq pipeline (no DTW).
    Builds feature sequence → LSTM decoder → linear resample to raw length.
    """
    _floor = _median_floor_prediction(raw_values, attributes, pipeline)
    if _floor is not None:
        return _floor

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
               'curve_length': attributes.get('_pred_curve_length', len(raw_values)), 'activity': activity}
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

    # Missing-attribute robustness: any trained feature absent from `attributes`
    # (e.g. hour_of_day/day_of_week, injected only at TRAIN time by
    # split_curves*/split_curves_with_prev_activity and never present in
    # production object_attributes) is NaN at this point and would silently
    # propagate through the scaler into the torch tensor, producing an
    # all-NaN prediction with no error -- confirmed as the actual cause of
    # every seq2seq* failure in the real complete-curve pipeline (previously
    # misattributed to needing autoregressive rollout). 0 == training mean
    # after scaling, mirroring predict_raw_curve's X_ref.fillna(0) for the
    # sklearn/DTW family.
    df_seq = df_seq.fillna(0)

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
# SEQ2SEQ IOM  —  Woerrlein & Strassburger's "Iterating over Metrics" baseline
# =============================================================================
# Faithful port of the competing method (SNE 34(4) 2024, DOI 10.11128/sne.34.tn.10713)
# rather than of its architecture alone. What makes IOM a method is not the
# encoder-decoder -- 'seq2seq_only' already has that -- but WHERE DTW enters:
#
#   seq2seq       DTW builds the training TARGET (curves warped onto a barycenter)
#   seq2seq_only  no DTW at all; select by validation MSE against the own target
#   seq2seq_iom   train on vanilla/ambiguous targets like seq2seq_only, but use a
#                 softDTW barycenter as the SCORING REFERENCE, and select the
#                 model by generating full curves at fixed checkpoints and
#                 comparing them to that reference (their Fig. 4, steps 3-6)
#
# So IOM sits between the two variants this repo already had, and neither of them
# is it. Their argument is that a per-timestep softmax/pointwise loss cannot see
# a whole time series, and that ambiguous data (identical labels, non-identical
# curves) has no single ground-truth curve to select against -- the DTW average
# supplies one.
#
# Deliberate deviations, all disclosed:
#
#  1. Continuous regression under MSE instead of their discretised values +
#     softmax + categorical cross-entropy. This makes the baseline STRONGER, so
#     it is the safe direction to deviate in: 'ml_step_dtw' beating a regression
#     IOM is a stronger claim than beating a classification IOM.
#  2. Equal training budget. They train 1000 epochs with no early stopping; here
#     IOM gets the same `epochs` every other seq2seq approach gets (default 80),
#     checkpointed every SEQ2SEQ_IOM_EPOCHS_PER_ITER epochs. Their 1000 epochs
#     were tuned for 10k synthetic sequence pairs, not for per-leaf curve counts
#     in the dozens, and an equal budget is what makes the comparison fair. IOM
#     does keep their no-early-stopping rule: the whole point is that a late
#     iteration may score better than the best-so-far under the metric.
#  3. Sigma length is DEGENERATE here and is reported, not relied on -- see
#     _iom_sigma_length.
#
# Everything else -- feature sequences, cells, decode, median floor -- is shared
# with 'seq2seq_only', so the gap between the two is attributable to the
# selection rule alone.
#
# KNOWN PROPERTY OF THE METHOD (not of this port). IOM scores every generated
# curve against ONE reference per leaf -- the analogue of their one reference per
# NC-code variant. That is sound only while the leaf is unimodal. Measured here
# on synthetic leaves, 40 epochs, checkpoints every 10:
#
#   homogeneous leaf (one shape, noise+length vary -- their setting):
#       iom_mse falls 0.123 -> 0.004 alongside pointwise 0.139 -> 0.013;
#       IOM selects the fully-trained checkpoint, same as pointwise would.
#   heterogeneous leaf (attributes genuinely differentiate curves within it):
#       iom_mse 0.374 at epoch 10 then 0.93 for the rest, while pointwise keeps
#       falling 0.493 -> 0.021; IOM selects the BARELY-TRAINED epoch-10 model.
#
# The second case is not a defect in the port -- a single-reference metric
# rewards predicting the average, so it penalises exactly the within-leaf
# conditioning this repo's approaches add. Their model conditions only on the
# variant label, so within a variant it has nothing to differentiate and the
# situation cannot arise for them. Expect IOM to look worst on the leaves where
# attributes carry the most signal; that is the method's own limitation and is
# the honest thing for the comparison to show.
# =============================================================================

# Epochs per IOM iteration. Their paper folds 10 epochs into one iteration for
# storage reasons and scores only the last epoch of each; the same figure here
# turns the default 80-epoch budget into 8 scored checkpoints.
SEQ2SEQ_IOM_EPOCHS_PER_ITER = 10

# Build the scoring reference with tslearn's softDTW barycenter (their [11],
# Cuturi & Blondel) rather than this repo's _robust_dtw_barycenter. The reference
# is the one part of IOM that is genuinely theirs, so it is built their way; the
# repo's duration-robust variant and its _zero_calibrate_barycenter duty-cycle
# fix are deliberately NOT applied, because editorialising the baseline's own
# reference would make any gap against it unattributable. Set False to swap in
# the repo barycenter and measure what that choice alone is worth.
SEQ2SEQ_IOM_SOFTDTW = True


def _iom_reference_curve(resampled_curves, fixed_length, verbose=0):
    """
    The DTW reference time series IOM scores generated curves against (their
    step 2). softDTW barycenter over the train curves, resampled to a common
    grid -- computed from TRAIN curves only, like every other reference here.

    resampled_curves : (n_curves, fixed_length, 1) -- same input
                       _robust_dtw_barycenter takes.

    Returns (fixed_length,) float array.
    """
    if SEQ2SEQ_IOM_SOFTDTW:
        try:
            from tslearn.barycenters import softdtw_barycenter
            # gamma is softDTW's smoothing: it is a squared-distance scale, so it
            # has to follow the magnitude of these curves (energy readings run to
            # thousands, where gamma=1.0 would degenerate to hard DTW and lose the
            # error tolerance that is the whole reason they chose softDTW).
            _scale = float(np.var(resampled_curves[:, :, 0])) or 1.0
            bary = softdtw_barycenter(resampled_curves, gamma=0.1 * _scale,
                                      max_iter=50, tol=1e-3)
            return np.asarray(bary, dtype=float).reshape(-1)
        except Exception as exc:
            # tslearn's L-BFGS can fail to converge on degenerate leaves (all-flat
            # curves, n=5). Falling back keeps the leaf trainable; it is logged
            # because it silently changes which barycenter the baseline was scored
            # against.
            if verbose:
                print(f"    [WARN] softdtw_barycenter failed ({exc}) -- "
                      f"falling back to _robust_dtw_barycenter.")
    return _robust_dtw_barycenter(resampled_curves,
                                  barycenter_size=fixed_length)[:, 0]


def _iom_sigma_length(pred, reference_n):
    """
    Their second metric: len(generated) / len(reference), optimum 1.

    DEGENERATE IN THIS PORT, on purpose. Their decoder emits an end-of-sequence
    symbol, so the generated series owns its length and sigma varies across
    iterations (their Fig. 10 legends show predicted lengths differing from the
    reference). Here every curve is generated on the fixed canonical grid and the
    RAW length comes from the simulation's duration model downstream
    (_pred_curve_length), never from the network -- so this ratio is 1.0 by
    construction and their two-step selection collapses to its MSE half.

    Computed and returned anyway for two reasons: it is logged, so the constant
    is visible in the results instead of being a silent omission; and the
    selection rule below is written as their full two-key rule, so adding a stop
    head later makes sigma bind without touching the selection code.
    """
    return float(pred.shape[1]) / float(reference_n.shape[0])


def _fit_seq2seq_iom(
    X_tr, y_tr_n, X_vl, y_vl_n, reference_n, device, input_size, seq_len,
    hidden_size, num_layers, dropout, epochs, batch_size, lr,
    teacher_forcing_ratio, verbose, epochs_per_iteration=None, cell_types=None,
    random_state=None,
):
    """
    IOM training loop: train in fixed-length iterations, and at the end of each
    one GENERATE full curves and score them against the DTW reference, instead of
    scoring per-timestep predictions against each instance's own target the way
    _fit_seq2seq_with_selection does.

    reference_n : (seq_len,) tensor -- the DTW reference in the SAME normalised
                  canonical space as y_tr_n/y_vl_n.

    Selection follows their two-step rule: among all checkpoints take those with
    sigma length closest to 1, and among those the lowest MSE. Checkpoints are
    pooled across cells, so the cell competition rides along on the same rule.

    No early stopping -- their method deliberately trains the full budget,
    because a later iteration may score better under the metric than an earlier
    one that had a lower pointwise loss.

    Returns
    -------
    (model, best_iom_mse, best_cell, val_loss_by_cell, history)
        history — list of per-checkpoint dicts, saved on the pipeline so the
                  metric trajectory of their Fig. 9 can be replotted.
    """
    cell_types = tuple(cell_types) if cell_types else _SEQ2SEQ_CELL_TYPES
    if epochs_per_iteration is None:
        epochs_per_iteration = SEQ2SEQ_IOM_EPOCHS_PER_ITER
    epochs_per_iteration = max(1, int(epochs_per_iteration))
    n_iterations = max(1, int(np.ceil(epochs / epochs_per_iteration)))

    criterion   = nn.MSELoss()
    reference_n = reference_n.to(device)

    _seed = GLOBAL_RANDOM_SEED if random_state is None else int(random_state)
    _loader_gen = torch.Generator()
    _loader_gen.manual_seed(_seed)
    loader = DataLoader(TensorDataset(X_tr, y_tr_n), batch_size=batch_size,
                        shuffle=True, generator=_loader_gen)

    X_vl_d, y_vl_d = X_vl.to(device), y_vl_n.to(device)

    results, history = {}, []
    for cell in cell_types:
        # Re-seed per cell exactly as _fit_seq2seq_with_selection does, so which
        # cell wins cannot depend on which trained first.
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
        _loader_gen.manual_seed(_seed)
        if cell == 'lstm':
            model = _Seq2SeqLSTM(input_size, hidden_size=hidden_size,
                                 num_layers=num_layers, dropout=dropout).to(device)
        elif cell == 'transformer':
            model = _Seq2SeqTransformer(input_size, seq_len=seq_len,
                                        hidden_size=hidden_size,
                                        num_layers=num_layers, dropout=dropout).to(device)
        else:
            print(f"    [WARN] unknown seq2seq cell '{cell}' — skipped.")
            continue

        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        # Running best under their rule. The key is the full two-key tuple so a
        # future stop head makes sigma bind with no change here.
        best_key, best_state, best_record = None, None, None

        for iteration in range(1, n_iterations + 1):
            model.train()
            for _ in range(epochs_per_iteration):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimiser.zero_grad()
                    pred = model(xb, targets=yb,
                                 teacher_forcing_ratio=teacher_forcing_ratio)
                    loss = criterion(pred, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimiser.step()

            # --- their step 4: generate, then compare to the DTW reference ----
            model.eval()
            with torch.no_grad():
                gen = model(X_vl_d, targets=None, teacher_forcing_ratio=0.0)
                # Mean over curves of the per-curve MSE against the reference --
                # their "aggregate the mean of applied metrics over all time
                # series samples". NOT the pointwise loss against each curve's
                # own target: that is the metric IOM exists to replace.
                iom_mse = float(((gen - reference_n.unsqueeze(0)) ** 2)
                                .mean(dim=1).mean())
                sigma   = _iom_sigma_length(gen, reference_n)
                # Kept only as a diagnostic, so the IOM checkpoint can be read
                # against what the ordinary selection rule would have said.
                pointwise = float(criterion(gen, y_vl_d))

            record = {'cell': cell, 'iteration': iteration,
                      'epoch': iteration * epochs_per_iteration,
                      'iom_mse': iom_mse, 'sigma_length': sigma,
                      'pointwise_val_loss': pointwise}
            history.append(record)
            if verbose:
                print(f"    [{cell}] iter {iteration:3d}/{n_iterations} "
                      f"(epoch {record['epoch']:4d})  iom_mse={iom_mse:.5f}  "
                      f"sigma={sigma:.3f}  pointwise={pointwise:.5f}")

            key = (abs(sigma - 1.0), iom_mse)
            if best_key is None or key < best_key:
                best_key    = key
                best_record = record
                best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is None:
            continue
        model.load_state_dict(best_state)
        model.eval()
        results[cell] = (model, best_key, best_record)

    if not results:
        raise RuntimeError(f"no seq2seq IOM cell trained successfully (tried {cell_types})")

    best_cell = min(results, key=lambda c: results[c][1])
    model, best_key, best_record = results[best_cell]
    val_loss_by_cell = {c: rec['iom_mse'] for c, (_, _, rec) in results.items()}

    if verbose:
        _scores = '  '.join(f'{c}={v:.5f}' for c, v in val_loss_by_cell.items())
        print(f"    IOM cell selection: {_scores}  ->  {best_cell} "
              f"(iteration {best_record['iteration']}, epoch {best_record['epoch']})")

    return model, best_record['iom_mse'], best_cell, val_loss_by_cell, history, best_record


def build_and_train_pipeline_seq2seq_iom(
    train_curves,
    variable,
    fixed_length=None,
    val_size=0.2,
    random_state=42,
    hidden_size=128,
    num_layers=2,
    dropout=0.1,
    epochs=80,
    batch_size=32,
    lr=1e-3,
    teacher_forcing_ratio=0.5,
    epochs_per_iteration=None,
    verbose=1,
):
    """
    Seq2Seq with Iterating-over-Metrics selection (Woerrlein & Strassburger).

    Target  : raw curve linearly resampled to fixed_length -- the ambiguous
              "vanilla" data their paper trains on. DTW never touches it.
    Input   : [phase, curve_length, activity_ohe, attrs] sequence, as seq2seq_only.
    Select  : softDTW barycenter as reference; generate + score every
              SEQ2SEQ_IOM_EPOCHS_PER_ITER epochs; pick by (|sigma-1|, MSE).
    Decode  : linear resample back to raw length -- identical to seq2seq_only.

    Differs from build_and_train_pipeline_seq2seq_only in the selection rule and
    nothing else, by design.
    """
    from sklearn.model_selection import train_test_split as _tts

    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2 — BUILD + TRAIN SEQ2SEQ-IOM PIPELINE  (vanilla targets, DTW-scored)")
        print("=" * 80)

    if fixed_length is None:
        fixed_length = max(2, int(round(np.median([len(c['original_values']) for c in train_curves]))))

    # Training target: plain resample, NO DTW alignment (their step 1 -- the
    # model is deliberately shown the ambiguity).
    for curve in train_curves:
        curve['resampled_values'] = np.interp(
            np.linspace(0, 1, fixed_length),
            np.linspace(0, 1, len(curve['original_values'])),
            curve['original_values'],
        ).astype(np.float32)

    # Their step 2: the DTW reference, from the same synthesised/observed set the
    # training uses. Scoring only -- it is never a target.
    resampled_for_dtw = np.array([c['resampled_values'] for c in train_curves],
                                 dtype=float)[:, :, np.newaxis]
    reference_curve = _iom_reference_curve(resampled_for_dtw, fixed_length,
                                           verbose=verbose)
    if verbose:
        print(f"    IOM reference length: {len(reference_curve)} "
              f"({'softDTW' if SEQ2SEQ_IOM_SOFTDTW else 'robust DBA'})")

    # Feature setup — same as seq2seq_only, fit on train only
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
    # The reference has to live in the same normalised space as the generated
    # curves, or the IOM metric would compare raw units against z-scores.
    reference_n = torch.tensor((np.asarray(reference_curve, dtype=np.float32) - y_mean) / y_std,
                               dtype=torch.float32)

    device     = _seq2seq_device()
    input_size = X_tr.shape[-1]
    (model, best_iom_mse, best_cell,
     val_loss_by_cell, iom_history, best_record) = _fit_seq2seq_iom(
        X_tr, y_tr_n, X_vl, y_vl_n, reference_n, device, input_size, fixed_length,
        hidden_size, num_layers, dropout, epochs, batch_size, lr,
        teacher_forcing_ratio, verbose,
        epochs_per_iteration=epochs_per_iteration,
        random_state=random_state,
    )

    if verbose:
        print(f"    Best iom_mse={best_iom_mse:.5f}  (cell={best_cell}, "
              f"epoch={best_record['epoch']})")

    pipeline = {
        'approach':             'seq2seq_iom',
        'cell_type':            best_cell,
        'val_loss_by_cell':     val_loss_by_cell,
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
        # Scoring reference only -- predict never reads it (no DTW decode here).
        # Stored so the selected model can be plotted against what selected it.
        'reference_curve':      np.asarray(reference_curve, dtype=float),
        # val_loss is the IOM metric, NOT the pointwise validation MSE the other
        # seq2seq approaches report under the same key. Same key so the shared
        # logging/report code works; the two are not comparable across approaches,
        # hence 'val_loss_pointwise' alongside it for anyone who needs the
        # like-for-like number.
        'val_loss':             best_iom_mse,
        'val_loss_pointwise':   best_record['pointwise_val_loss'],
        'iom_sigma_length':     best_record['sigma_length'],
        'iom_iteration':        best_record['iteration'],
        'iom_epoch':            best_record['epoch'],
        'iom_epochs_per_iteration': (epochs_per_iteration or SEQ2SEQ_IOM_EPOCHS_PER_ITER),
        # Full metric trajectory -- their Fig. 9 is replottable from this.
        'iom_history':          iom_history,
    }

    # Median floor — see _attach_median_floor. Applied exactly as to every other
    # learned approach, so none is protected while another is not.
    _attach_median_floor(
        pipeline, train_curves, val_inst,
        lambda rv, c: predict_raw_curve_seq2seq_iom(rv, c['activity'], c['attributes'], pipeline),
        label=f' {variable} (iom)', verbose=verbose,
    )

    return pipeline


def predict_raw_curve_seq2seq_iom(raw_values, activity, attributes, pipeline):
    """
    Predict with the IOM pipeline. Generation and decode are identical to
    'seq2seq_only' -- IOM changes which weights got selected, not how the
    selected model is run -- so this delegates rather than duplicating, keeping
    the two approaches guaranteed-identical at inference.
    """
    return predict_raw_curve_seq2seq_only(raw_values, activity, attributes, pipeline)


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
    fixed_length=None,
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
    if fixed_length is None:
        fixed_length = max(2, int(round(np.median([len(c['original_values']) for c in train_curves]))))
    resampled_for_dba = np.array([
        np.interp(
            np.linspace(0, 1, fixed_length),
            np.linspace(0, 1, len(c['original_values'])),
            c['original_values'],
        )
        for c in train_curves
    ])[:, :, np.newaxis]

    dba_barycenter  = _robust_dtw_barycenter(resampled_for_dba, barycenter_size=fixed_length)
    # Restore the on/off duty cycle DBA averages away (see
    # _zero_calibrate_barycenter); no-op for non-intermittent sensors.
    reference_curve = _zero_calibrate_barycenter(dba_barycenter[:, 0], resampled_for_dba)

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

    device     = _seq2seq_device()
    input_size = X_tr.shape[-1]
    model, best_val_loss, best_cell, val_loss_by_cell = _fit_seq2seq_with_selection(
        X_tr, y_tr_n, X_vl, y_vl_n, device, input_size, fixed_length,
        hidden_size, num_layers, dropout, epochs, batch_size, lr,
        teacher_forcing_ratio, patience, verbose,
        random_state=random_state,
    )

    if verbose:
        print(f"    Best val_loss={best_val_loss:.5f}  (cell={best_cell})")

    pipeline = {
        'approach':             'seq2seq_exog',
'cell_type':           best_cell,
'val_loss_by_cell':    val_loss_by_cell,
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

    # Median floor — see _attach_median_floor.
    _attach_median_floor(
        pipeline, train_curves, val_inst,
        lambda rv, c: predict_raw_curve_seq2seq_exog(rv, c['activity'], c['attributes'],
                                                     pipeline,
                                                     exog_values=c.get('exog_values', {})),
        label=f' {variable}', verbose=verbose,
    )

    return pipeline


def predict_raw_curve_seq2seq_exog(raw_values, activity, attributes, pipeline,
                                    exog_values=None):
    """
    Predict using DTW + Seq2Seq + External Factors.
    exog_values : dict {col: np.ndarray} of raw-length ef_ signals (or empty).
    """
    _floor = _median_floor_prediction(raw_values, attributes, pipeline)
    if _floor is not None:
        return _floor

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
               'curve_length': attributes.get('_pred_curve_length', len(raw_values)), 'activity': activity}
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

    # Missing-attribute robustness: any trained feature absent from `attributes`
    # (e.g. hour_of_day/day_of_week, injected only at TRAIN time by
    # split_curves*/split_curves_with_prev_activity and never present in
    # production object_attributes) is NaN at this point and would silently
    # propagate through the scaler into the torch tensor, producing an
    # all-NaN prediction with no error -- confirmed as the actual cause of
    # every seq2seq* failure in the real complete-curve pipeline (previously
    # misattributed to needing autoregressive rollout). 0 == training mean
    # after scaling, mirroring predict_raw_curve's X_ref.fillna(0) for the
    # sklearn/DTW family.
    df_seq = df_seq.fillna(0)

    X = torch.tensor(df_seq.values.astype(np.float32)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        y_norm = model(X, targets=None, teacher_forcing_ratio=0.0)

    y_ref_pred = y_norm.squeeze(0).cpu().numpy() * y_std + y_mean

    # DTW decode — same as baseline
    return _decode_canonical_to_raw(y_ref_pred, raw_values, reference_curve,
                                    fixed_length)


def build_and_train_pipeline_seq2seq_external(
    train_curves,
    variable,
    fixed_length=None,
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
    DTW + Seq2Seq + External Factors + previous-activity NAME.

    prev_act_* features must already be in each curve's 'attributes' dict —
    populated by split_curves_with_prev_activity(), which by default contributes
    only the categorical prev_act_name. They flow through
    build_and_train_pipeline_seq2seq_exog unchanged because _infer_key_types
    picks them up from the attributes dict.

    Lagged-energy features appear only with include_prev_energy=True on the
    splitter — off for reported runs, see that function's docstring.
    """
    pipeline = build_and_train_pipeline_seq2seq_exog(
        train_curves, variable,
        fixed_length=fixed_length,
        val_size=val_size,
        random_state=random_state,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        teacher_forcing_ratio=teacher_forcing_ratio,
        patience=patience,
        verbose=verbose,
    )
    pipeline['approach'] = 'seq2seq_external'

    # Per-activity training-curve medians for first-of-case autoregressive defaults
    _by_act = {}
    for _c in train_curves:
        _v = np.asarray(_c['original_values'], dtype=float)
        if len(_v) == 0:
            continue
        _a = _c['activity']
        _by_act.setdefault(_a, {'m': [], 's': [], 'mx': [], 'e': [], 'l': []})
        _by_act[_a]['m'].append(float(np.mean(_v)))
        _by_act[_a]['s'].append(float(np.std(_v))  if len(_v) >= 2 else 0.0)
        _by_act[_a]['mx'].append(float(np.max(_v)))
        _by_act[_a]['e'].append(float(_v[-1]))
        _by_act[_a]['l'].append(float(len(_v)))

    pipeline['first_of_case_defaults'] = {
        _a: {
            'prev_act_name':   'none',
            'prev_act_mean':   float(np.median(_d['m'])),
            'prev_act_std':    float(np.median(_d['s'])),
            'prev_act_max':    float(np.median(_d['mx'])),
            'prev_act_end':    float(np.median(_d['e'])),
            'prev_act_length': float(np.median(_d['l'])),
        }
        for _a, _d in _by_act.items()
    }
    return pipeline


def predict_raw_curve_seq2seq_external(raw_values, activity, attributes, pipeline,
                                             exog_values=None):
    """Thin alias — prev_act_* features live in attributes, handled by seq2seq_exog path."""
    return predict_raw_curve_seq2seq_exog(raw_values, activity, attributes, pipeline,
                                          exog_values=exog_values or {})


# =============================================================================
# STEP 4 — EVALUATE ON RAW TEST CURVES
# =============================================================================

def _dispatch_predict(raw_values, curve, pipeline):
    """Route prediction to the right function based on pipeline['approach'].

    'baseline' vs 'ml_dtw': previously 'baseline' had no explicit
    branch here and silently fell through to predict_raw_curve (the full
    DBA+DTW+regression pipeline) via this function's own default -- i.e. the
    approach NAMED 'baseline' was actually the strong reference, while the
    table LABEL "Baseline" pointed at a completely separate, unregistered
    flat-mean predictor ('mean_baseline', computed inline in 02_modelling.py).
    Both are now explicit and distinct: 'baseline' = the true naive floor
    (build_and_train_pipeline_median), 'ml_dtw' = the former 'baseline'
    (build_and_train_pipeline, now explicitly tagged).
    """
    approach = pipeline.get('approach', 'ml_dtw')
    act, attrs = curve['activity'], curve['attributes']
    if approach in ('baseline', 'median_activity_sensor'):
        # Both are the median-LEVEL predictor (one horizontal line); they differ
        # only in how the values were pooled before the median: 'baseline' = one
        # level per sensor (all activities pooled), 'median_activity_sensor' =
        # one level per (sensor, activity, object).
        return predict_raw_curve_median(raw_values, act, attrs, pipeline)
    if approach == 'ml_dtw':
        return predict_raw_curve(raw_values, act, attrs, pipeline)
    if approach in ('ml_external', 'ml_external_wcounts', 'ml_external_wmetric'):
        # The variants differ only in how the canonical model was FITTED
        # (row weights), which is handled inside the shared predictor.
        return predict_raw_curve_exog_prev_activity(raw_values, act, attrs, pipeline,
                                                    exog_values=curve.get('exog_values', {}))
    if approach in ('ml_step_dtw', 'ml_step_dtw_smooth'):
        # Same predictor: the pipeline's stored gain_mode decides whether the
        # per-segment gains are applied piecewise-constant or interpolated.
        return predict_raw_curve_step_dtw(raw_values, act, attrs, pipeline,
                                          exog_values=curve.get('exog_values', {}))
    if approach == 'seq2seq':
        return predict_raw_curve_seq2seq(raw_values, act, attrs, pipeline)
    if approach == 'seq2seq_only':
        return predict_raw_curve_seq2seq_only(raw_values, act, attrs, pipeline)
    if approach == 'seq2seq_iom':
        return predict_raw_curve_seq2seq_iom(raw_values, act, attrs, pipeline)
    if approach == 'seq2seq_external':
        return predict_raw_curve_seq2seq_external(raw_values, act, attrs, pipeline,
                                                       exog_values=curve.get('exog_values', {}))
    if approach == 'ml_only':
        return predict_raw_curve_ml_only(raw_values, act, attrs, pipeline)
    return predict_raw_curve(raw_values, act, attrs, pipeline)


def _train_curve_only_worker(sensor, activity, obj, df_train, approaches, ef_cols,
                             fixed_length=None, val_size=0.2,
                             optimize_hyperparams=False, n_trials=50,
                             prev_act_energy_map=None):
    """
    Top-level picklable worker for RUN_CURVE_ONLY_EVALUATION parallel training.
    Trains all sklearn-based approaches for one (sensor, activity, object) combo.
    Seq2seq approaches are excluded — they are trained by _train_seq2seq_worker
    in its own process pool.

    prev_act_energy_map is THIS sensor's sub-map from
    build_prev_activity_energy_map() — built once per process in 02_modelling.py
    rather than here, because it spans every activity while this worker only sees
    one, and recomputing it per combo would repeat the same groupby for every
    (activity, object) of the sensor.

    Returns dict keyed by approach name, each value is the raw pipeline dict,
    plus 'sensor'/'activity'/'object' keys for reassembly and '_timings', the
    per-approach training seconds for this leaf (CPU time of this worker, so
    they sum across leaves to far more than the pool's wall clock — 02_modelling
    records both; see its RUNTIME PROFILING section).
    """
    import time as _wtime
    # Reproducibility: seed from this combo's own key, not from whatever RNG
    # state this pool worker inherited. Same combo -> same models, regardless of
    # worker count or the order the pool scheduled tasks in.
    _seed = stable_seed('curve_only', sensor, activity, obj)
    set_global_seeds(_seed)

    # NOTE: 'baseline' (median per SENSOR, pooled over activities) is NOT here —
    # it can't be trained by a per-(sensor,activity,object) worker; it is built
    # separately in 02_modelling.py's curve-only section. This worker trains the
    # per-combo median under 'median_activity_sensor'.
    _SKLEARN_APPROACHES = {'median_activity_sensor', 'ml_dtw', 'ml_external', 'ml_only',
                           'ml_step_dtw', 'ml_step_dtw_smooth',
                           # train/eval-gap variants — see the block below
                           'ml_external_wcounts', 'ml_external_wmetric'}
    _active = [a for a in approaches if a in _SKLEARN_APPROACHES]

    _t_extract = _wtime.perf_counter()
    curves, _ = split_curves(
        df_train,
        variable=sensor,
        activities=[activity],
        objects=[obj],
        test_size=0.0,
        verbose=0,
        exog_columns=ef_cols,
    )
    _timings = {'_curve_extraction': _wtime.perf_counter() - _t_extract}
    if len(curves) < 5:
        return {'sensor': sensor, 'activity': activity, 'object': obj, 'skipped': True,
                '_timings': _timings}

    models = _make_curve_models()

    # One BLAS thread per worker. The pool already saturates the machine with
    # one process per (sensor, activity, object), but OpenBLAS/OpenMP default to
    # one thread per core *inside each* of them — 16 workers x 16 threads on 16
    # cores. The tree models are safe (n_jobs=1 is passed explicitly); the
    # BLAS-bound candidates, MLP above all, oversubscribe badly without this.
    # Applied AFTER _make_curve_models() on purpose: threadpool_limits only
    # governs the pools already loaded when it runs, and that call is what
    # lazily imports sklearn.neural_network / xgboost, each registering its own.
    # Not a `with` block, so the limit persists across every task this pooled
    # worker process handles rather than just the first.
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    result = {'sensor': sensor, 'activity': activity, 'object': obj, 'skipped': False}

    _hp_kwargs = dict(optimize_hyperparams=optimize_hyperparams, n_trials=n_trials)

    if 'median_activity_sensor' in _active:
        # "Median per Activity & Sensor": median training curve for THIS
        # (sensor, activity, object), no model. (The coarser 'baseline' —
        # one median per sensor — is built in 02_modelling.py, not here.)
        _t_a = _wtime.perf_counter()
        _mas_pipe = build_and_train_pipeline_median(
            curves, variable=sensor, fixed_length=fixed_length, verbose=0,
        )
        _mas_pipe['approach'] = 'median_activity_sensor'
        result['median_activity_sensor'] = _mas_pipe
        _timings['median_activity_sensor'] = _wtime.perf_counter() - _t_a
    if 'ml_dtw' in _active:
        # The former 'baseline': DBA barycenter + DTW alignment + regression.
        _t_a = _wtime.perf_counter()
        result['ml_dtw'] = build_and_train_pipeline(
            curves, variable=sensor, fixed_length=fixed_length,
            val_size=val_size, models=models, verbose=0, n_jobs=1,
            random_state=_seed, **_hp_kwargs,
        )
        _timings['ml_dtw'] = _wtime.perf_counter() - _t_a
    if 'ml_external' in _active:
        # Previous-activity NAME + its TRAINING energy level (both event-log-keyed
        # facts) + ef_* external factors. No lagged meter readings — see
        # split_curves_with_prev_activity / build_prev_activity_energy_map.
        _t_a = _wtime.perf_counter()
        _prev_curves, _ = split_curves_with_prev_activity(
            df_train,
            variable=sensor,
            activities=[activity],
            objects=[obj],
            test_size=0.0,
            verbose=0,
            exog_columns=ef_cols,
            include_prev_energy=False,
        )
        if len(_prev_curves) >= 5:
            result['ml_external'] = build_and_train_pipeline_exog_prev_activity(
                _prev_curves, variable=sensor,
                fixed_length=fixed_length, val_size=val_size,
                models=models, verbose=0, n_jobs=1,
                random_state=_seed,
                prev_act_energy_map=prev_act_energy_map,
                **_hp_kwargs,
            )
            _timings['ml_external'] = _wtime.perf_counter() - _t_a
        elif 'ml_dtw' in result:
            # Fewer than 5 curves survive the stricter prev-activity-context
            # extraction (a separate, narrower filter than the len(curves)<5
            # check above that already passed for 'baseline'/'ml_dtw')
            # -- previously this silently left 'ml_external' unset
            # with no log line at all, which meant predict_curve_for_instance
            # would return None for every instance of this (sensor, activity,
            # object) and leave an unexplained gap in the complete-curve/
            # schedule-profile curves. Fall back to 'ml_dtw' (already
            # trained above, just without previous-activity context) rather
            # than the trivial 'baseline' median -- ml_external's
            # reassembly always calls predict_raw_curve_exog_prev_activity on
            # whatever ends up here, and only ml_dtw's pipeline dict
            # (model/feature_columns/etc.) has the keys that function needs.
            print(f"  [WARN] ml_external: only {len(_prev_curves)} curve(s) with "
                  f"previous-activity context for {sensor}|{activity}|{obj} (need >=5) -- "
                  f"falling back to the ml_dtw pipeline for this combo.")
            result['ml_external'] = result['ml_dtw']
        else:
            print(f"  [WARN] ml_external: only {len(_prev_curves)} curve(s) with "
                  f"previous-activity context for {sensor}|{activity}|{obj} (need >=5), and "
                  f"no ml_dtw pipeline available either -- no prediction possible for this combo.")
    # ── Train/eval-gap variants ──────────────────────────────────────────────
    # All four reuse the ml_external curve set (same features, same prev-activity
    # context), so any difference against ml_external is attributable to the one
    # thing each changes. They are additive: the originals above are untouched.
    _GAP_VARIANTS = ('ml_external_wcounts', 'ml_external_wmetric',
                     # Not train/eval-gap variants, but they need the same
                     # prev-activity curve set so their only difference against
                     # ml_external is the segment-target reparameterisation.
                     'ml_step_dtw',
                     # Identical training to ml_step_dtw; only the reconstruction
                     # differs (interpolated gains), so the gap between the two
                     # prices the smoothing alone.
                     'ml_step_dtw_smooth')
    if any(v in _active for v in _GAP_VARIANTS):
        _t_a = _wtime.perf_counter()
        _pv, _ = split_curves_with_prev_activity(
            df_train, variable=sensor, activities=[activity], objects=[obj],
            test_size=0.0, verbose=0, exog_columns=ef_cols,
            include_prev_energy=False,
        )
        _timings['_curve_extraction_prev_activity'] = _wtime.perf_counter() - _t_a
        _common = dict(fixed_length=fixed_length, val_size=val_size, models=models,
                       verbose=0, n_jobs=1, random_state=_seed,
                       prev_act_energy_map=prev_act_energy_map)
        if len(_pv) >= 5:
            for _v in _GAP_VARIANTS:
                if _v not in _active:
                    continue
                _t_a = _wtime.perf_counter()
                try:
                    if _v in ('ml_step_dtw', 'ml_step_dtw_smooth'):
                        # The leaf's ml_external (trained above when active) is
                        # handed in as the do-no-harm fallback; None when this
                        # run doesn't train ml_external, which disables the gate.
                        result[_v] = build_and_train_pipeline_step_dtw(
                            [dict(c) for c in _pv], variable=sensor,
                            gain_mode=('smooth' if _v == 'ml_step_dtw_smooth'
                                       else 'step'),
                            fallback_pipeline=result.get('ml_external'),
                            **_common, **_hp_kwargs)
                    else:
                        _mode = ('counts' if _v == 'ml_external_wcounts'
                                 else 'counts_sigma')
                        _p = build_and_train_pipeline_exog_prev_activity(
                            [dict(c) for c in _pv], variable=sensor,
                            sample_weight_mode=_mode, **_common, **_hp_kwargs)
                        _p['approach'] = _v
                        result[_v] = _p
                except Exception as _e:
                    # Never let a new variant take the whole leaf down with it —
                    # the established approaches for this combo must still land.
                    print(f"  [WARN] {_v} failed for {sensor}|{activity}|{obj}: "
                          f"{type(_e).__name__}: {_e}")
                finally:
                    _timings[_v] = _wtime.perf_counter() - _t_a
        else:
            print(f"  [WARN] {'/'.join(v for v in _GAP_VARIANTS if v in _active)}: only "
                  f"{len(_pv)} curve(s) with previous-activity context for "
                  f"{sensor}|{activity}|{obj} (need >=5) -- skipped. No ml_dtw "
                  f"fallback here on purpose: substituting a different approach "
                  f"would silently contaminate the comparison these variants exist for.")

    if 'ml_only' in _active:
        _t_a = _wtime.perf_counter()
        result['ml_only'] = build_and_train_pipeline_ml_only(
            curves, variable=sensor, fixed_length=fixed_length,
            val_size=val_size, models=models, verbose=0, n_jobs=1,
            random_state=_seed, **_hp_kwargs,
        )
        _timings['ml_only'] = _wtime.perf_counter() - _t_a
    # Stamped BEFORE '_timings' is attached: _stamp_train_sensor_median writes
    # into every dict value it finds, and the timings dict is not a pipeline.
    result = _stamp_train_sensor_median(result, curves)
    result['_timings'] = _timings
    return result


# Keeps the trap's file object referenced for the life of the worker. If it were
# garbage-collected the fd would close and faulthandler would write into a closed
# descriptor, which is exactly when we need it to work.
_STALL_TRAP_FILES = []


def _pool_stall_trap_initializer(label='pool'):
    """
    ProcessPoolExecutor `initializer` that makes this worker dumpable on demand:
    SIGUSR1 writes every thread's stack to
    <PIPELINE_STALL_TRACE_DIR or $TMPDIR>/<label>_<pid>.trace.

    Runs at worker startup, before any task argument is unpickled, so it also
    covers a hang that happens while receiving the (large) df_train argument —
    the seq2seq workers wedged at ~4s of CPU, which is about what unpickling that
    frame costs, so the trap has to be armed before then to be worth anything.

    Diagnostic only; it changes nothing about how the worker runs. It exists
    because yama ptrace_scope=1 on this host blocks py-spy and gdb without root,
    so a wedged worker cannot be inspected from outside.
    """
    import faulthandler, signal, tempfile, os as _os
    try:
        _dir = _os.environ.get('PIPELINE_STALL_TRACE_DIR') or tempfile.gettempdir()
        _os.makedirs(_dir, exist_ok=True)
        _f = open(_os.path.join(_dir, f'{label}_{_os.getpid()}.trace'), 'w')
        _f.write(f'pid={_os.getpid()} ppid={_os.getppid()} label={label}\n')
        _f.flush()
        faulthandler.register(signal.SIGUSR1, file=_f, all_threads=True, chain=False)
        _STALL_TRAP_FILES.append(_f)
    except Exception:
        pass   # a missing trap must never be the reason a run fails


def _train_seq2seq_worker(sensor, activity, obj, df_train, approaches, ef_cols,
                          hidden_size=128, num_layers=2, dropout=0.1,
                          epochs=80, batch_size=32, lr=1e-3,
                          teacher_forcing_ratio=0.5, patience=10,
                          fixed_length=None, val_size=0.2):
    """
    Top-level picklable worker for parallel seq2seq training.
    Trains all seq2seq variants for one (sensor, activity, object) combo —
    identical scoping to _train_curve_only_worker so each model sees only
    homogeneous curves from that single combo.
    Returns dict with sensor/activity/object + results keyed by approach name.
    """
    import os as _os, torch
    # Make this worker genuinely CPU-only before anything asks torch about CUDA.
    # We are forked from a parent that has already called torch.cuda.is_available()
    # (02_modelling.py's startup set_global_seeds), which PyTorch documents as
    # fork-poisoning: cuInit has run pre-fork, so every CUDA call here fails.
    # Hiding the GPUs is not enough on its own — the default availability probe
    # goes through the CUDA runtime and still answers True-but-unusable, and Adam's
    # _cuda_graph_capture_health_check then dies with "CUDA error: initialization
    # error" even though the model sits on CPU. PYTORCH_NVML_BASED_CUDA_CHECK
    # switches that probe to the NVML path, which reads both env vars at call time
    # and correctly answers False. Root-caused 2026-07-24: experiment_967 lost all
    # 390 seq2seq workers to "Cannot re-initialize CUDA in forked subprocess".
    _os.environ['CUDA_VISIBLE_DEVICES'] = ''
    _os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
    torch.cuda._cached_device_count = None   # drop the count inherited from the fork parent
    torch.set_num_threads(1)  # prevent OpenMP/MKL thread-pool contention across workers
    # Per-combo seed (see _train_curve_only_worker) — covers torch weight init,
    # dropout masks and DataLoader shuffling in this process.
    _s2s_seed = stable_seed('seq2seq', sensor, activity, obj)
    set_global_seeds(_s2s_seed)
    # Same for the BLAS pool numpy uses while building the feature sequences —
    # torch's setting does not cover it.
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    import time as _wtime
    _SEQ2SEQ = {'seq2seq', 'seq2seq_only', 'seq2seq_external', 'seq2seq_iom'}
    _active = [a for a in approaches if a in _SEQ2SEQ]
    if not _active:
        return {'sensor': sensor, 'activity': activity, 'object': obj, 'skipped': True}

    _t_extract = _wtime.perf_counter()
    curves, _ = split_curves(
        df_train,
        variable=sensor,
        activities=[activity],
        objects=[obj],
        test_size=0.0,
        verbose=0,
        exog_columns=ef_cols,
    )
    # Per-approach training seconds for this leaf, same contract as
    # _train_curve_only_worker's '_timings' (CPU time inside one pool worker).
    _timings = {'_curve_extraction': _wtime.perf_counter() - _t_extract}
    if len(curves) < 5:
        return {'sensor': sensor, 'activity': activity, 'object': obj, 'skipped': True,
                '_timings': _timings}

    result = {'sensor': sensor, 'activity': activity, 'object': obj, 'skipped': False}

    if 'seq2seq' in _active:
        _t_a = _wtime.perf_counter()
        result['seq2seq'] = build_and_train_pipeline_seq2seq(
            curves, variable=sensor, fixed_length=fixed_length, random_state=_s2s_seed,
            val_size=val_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, epochs=epochs, batch_size=batch_size, lr=lr,
            teacher_forcing_ratio=teacher_forcing_ratio, patience=patience,
            verbose=False,
        )

        _timings['seq2seq'] = _wtime.perf_counter() - _t_a

    if 'seq2seq_only' in _active:
        _t_a = _wtime.perf_counter()
        result['seq2seq_only'] = build_and_train_pipeline_seq2seq_only(
            curves, variable=sensor, fixed_length=fixed_length, random_state=_s2s_seed,
            val_size=val_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, epochs=epochs, batch_size=batch_size, lr=lr,
            teacher_forcing_ratio=teacher_forcing_ratio, patience=patience,
            verbose=False,
        )

        _timings['seq2seq_only'] = _wtime.perf_counter() - _t_a

    if 'seq2seq_iom' in _active:
        _t_a = _wtime.perf_counter()
        # No `patience`: IOM trains the full budget by design (see the section
        # header) -- early stopping is the rule it replaces, not one it inherits.
        result['seq2seq_iom'] = build_and_train_pipeline_seq2seq_iom(
            curves, variable=sensor, fixed_length=fixed_length, random_state=_s2s_seed,
            val_size=val_size, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout, epochs=epochs, batch_size=batch_size, lr=lr,
            teacher_forcing_ratio=teacher_forcing_ratio,
            verbose=False,
        )

        _timings['seq2seq_iom'] = _wtime.perf_counter() - _t_a

    if 'seq2seq_external' in _active:
        _t_a = _wtime.perf_counter()
        # Same as ml_external: previous-activity NAME + ef_* only, no lagged energy.
        _prev_curves, _ = split_curves_with_prev_activity(
            df_train,
            variable=sensor,
            activities=[activity],
            objects=[obj],
            test_size=0.0,
            verbose=0,
            exog_columns=ef_cols,
            include_prev_energy=False,
        )
        if len(_prev_curves) >= 5:
            result['seq2seq_external'] = build_and_train_pipeline_seq2seq_external(
                _prev_curves, variable=sensor, fixed_length=fixed_length, random_state=_s2s_seed,
                val_size=val_size, hidden_size=hidden_size, num_layers=num_layers,
                dropout=dropout, epochs=epochs, batch_size=batch_size, lr=lr,
                teacher_forcing_ratio=teacher_forcing_ratio, patience=patience,
                verbose=False,
            )
        elif 'seq2seq' in result:
            # Same silent-skip issue as ml_external above -- fall back
            # to the plain seq2seq pipeline (no previous-activity context)
            # instead of leaving this (sensor, activity, object) with no
            # prediction and no explanation in the logs.
            print(f"  [WARN] seq2seq_external: only {len(_prev_curves)} curve(s) with "
                  f"previous-activity context for {sensor}|{activity}|{obj} (need >=5) -- "
                  f"falling back to the plain seq2seq pipeline for this combo.")
            result['seq2seq_external'] = result['seq2seq']
        else:
            print(f"  [WARN] seq2seq_external: only {len(_prev_curves)} curve(s) with "
                  f"previous-activity context for {sensor}|{activity}|{obj} (need >=5), and "
                  f"no plain seq2seq pipeline available either -- no prediction possible for this combo.")
        _timings['seq2seq_external'] = _wtime.perf_counter() - _t_a

    # Stamped BEFORE '_timings' is attached — see _train_curve_only_worker.
    result = _stamp_train_sensor_median(result, curves)
    result['_timings'] = _timings
    return result


# Persist the full y_true / y_pred arrays on every scored curve, so metrics
# nobody has thought of yet can be computed from the saved results instead of by
# re-running the pipeline.
#
# ON by default (changed 2026-07-26). Sized against the last full run: 586k rows
# across 10 approaches, median 16 points per curve but a mean of 100 and a max of
# 3,432, so ~117M values -> roughly 0.2-0.3 GB of compressed parquet against the
# 42 MB the metrics-only file takes. That is a real cost but a bounded one, and
# it is the difference between choosing a metric later and re-running everything
# to get it. Set save_curve_values=False in 01_pipeline.py (or
# PIPELINE_SAVE_CURVE_VALUES=false) if a particular run needs the small file.
#
# Stored as float32: these are sensor readings whose own precision is far coarser
# than 7 significant digits, and it halves the file for no loss that any metric
# here can detect.
#
# y_true is emitted here on every scored row, but 02_modelling.py's export SPLITS it
# out before writing: every approach is scored on the same curves, so the real
# array is identical across them and storing it per-row would duplicate roughly
# half the file once per approach. It lands in real_test_curves.parquet keyed on
# (Process, Sensor, Activity, Instance, Split); y_pred stays on the per-approach
# rows. The results notebooks re-join it automatically.
CURVE_EVAL_SAVE_VALUES = _os_seed.environ.get(
    'PIPELINE_SAVE_CURVE_VALUES', 'true').lower() == 'true'


def _curve_shape_stats(y_true, y_pred):
    """
    Per-curve statistics of the true and predicted curves, for metrics that
    pointwise error cannot express. Written on every scored curve.

    std_*    spread — std_pred/std_real is the flatness measure; the averaging
             approaches sit near 0.1, meaning they emit a tenth of the real
             variation
    acf1_*   lag-1 autocorrelation — how smooth the curve is in time
    rough_*  mean |v[t] - v[t-1]| — jaggedness / switching intensity. It catches
             the OPPOSITE failure to std_*: a prediction can reach the right
             standard deviation by adding high-frequency noise instead of
             reproducing real dynamics, and only this separates "right
             amplitude, right structure" from "right amplitude, wrong texture".
             Comparable between the two curves because the prediction is always
             emitted at the real curve's own length.
    max_*    peak, which sizes equipment and tariffs and is the first thing an
             averaged curve loses

    Deliberately NOT here: a duty-cycle / zero-fraction statistic. The obvious
    definition (share of exactly-zero samples) is degenerate for every sensor
    that never switches fully off — the temp_* and Druck_* channels among the
    modelled sensors never read 0, so it would be 0 for real AND predicted and
    the ratio undefined. 05_results_complete_energy_profile.ipynb measures duty
    cycle properly, relative to each curve's own range; use that one.
    """
    def _acf1(x):
        # Guard the two SLICES, not x itself: a curve that is flat except for its
        # first or last sample has nonzero overall std but a constant slice, which
        # makes corrcoef divide 0/0 and emit a RuntimeWarning for the same nan.
        x = np.asarray(x, dtype=float)
        if len(x) < 3:
            return np.nan
        a, b = x[:-1], x[1:]
        if a.std() < 1e-12 or b.std() < 1e-12:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    out = {}
    for tag, arr in (('real', np.asarray(y_true, dtype=float)),
                     ('pred', np.asarray(y_pred, dtype=float))):
        out[f'mean_{tag}']  = float(arr.mean()) if arr.size else np.nan
        out[f'std_{tag}']   = float(arr.std()) if arr.size else np.nan
        out[f'max_{tag}']   = float(arr.max()) if arr.size else np.nan
        out[f'acf1_{tag}']  = _acf1(arr)
        out[f'rough_{tag}'] = (float(np.abs(np.diff(arr)).mean())
                               if arr.size >= 2 else np.nan)
    return out


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
        _denom = np.sum(np.abs(raw_values))
        wape = np.sum(np.abs(raw_values - y_pred)) / _denom * 100 if _denom != 0 else np.nan

        _mu  = raw_values.mean()
        _sig = raw_values.std()
        # NaN when curve is essentially constant (CV < 1%) — z-scoring a flat
        # signal produces astronomically large sMAE/sRMSE that are meaningless.
        _cv_ok = _sig > 1e-10 and (_mu == 0 or (_sig / abs(_mu)) > 0.01)
        if _cv_ok:
            _zt = (raw_values - _mu) / _sig
            _zp = (y_pred    - _mu) / _sig
            smae  = float(np.mean(np.abs(_zt - _zp)))
            srmse = float(np.sqrt(np.mean((_zt - _zp) ** 2)))
        else:
            smae = srmse = np.nan

        _rec = {
            'instance_id': curve['instance_id'],
            'activity':    curve['activity'],
            'n_points':    len(raw_values),
            'MAE':         mae,
            'RMSE':        rmse,
            'WAPE (%)':    wape,
            'sMAE':        smae,
            'sRMSE':       srmse,
        }
        # Shape/texture statistics of BOTH curves, always. The five metrics
        # above are all pointwise, so none of them can distinguish a usable load
        # profile from a flat line through the middle of one -- and once the run
        # is over they cannot be recomputed, because the values are gone. These
        # sufficient statistics are what the realism measures are built from
        # (flatness = std_pred/std_real, texture = rough_pred/rough_real,
        # smoothness = acf1_pred - acf1_real), and they cost 10 floats a curve.
        # Stored RAW, never as ratios: aggregating the two sides separately and
        # dividing at the end avoids both the near-zero denominators and the
        # asymmetry of a per-curve ratio (0.5 and 2.0 are both "factor 2 wrong"
        # but sit at different distances from 1.0, which under-penalises the
        # flatness these metrics exist to expose).
        _rec.update(_curve_shape_stats(raw_values, y_pred))
        if CURVE_EVAL_SAVE_VALUES:
            # Full arrays, so ANY future metric can be computed offline without
            # re-running the pipeline. Off by default -- this multiplies the
            # results file by roughly n_points per row.
            _rec['y_true'] = np.asarray(raw_values, dtype=np.float32)
            _rec['y_pred'] = np.asarray(y_pred, dtype=np.float32)
        per_curve_metrics.append(_rec)

        all_true.extend(raw_values.tolist())
        all_pred.extend(y_pred.tolist())

    # Aggregate metrics
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    agg_metrics = {
        'MAE':      mean_absolute_error(all_true, all_pred),
        'RMSE':     np.sqrt(mean_squared_error(all_true, all_pred)),
        'WAPE (%)': np.sum(np.abs(all_true - all_pred)) / np.sum(np.abs(all_true)) * 100,
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
                _mae = mean_absolute_error(_rv, _yp)
                _rms = np.sqrt(mean_squared_error(_rv, _yp))
                _ax.plot(_rv, label='Actual', color='steelblue', linewidth=2)
                _ax.plot(_yp, label='Predicted', color='tomato',
                         linewidth=2, linestyle='--')
                _ax.set_title(
                    f"ID {_curve['instance_id']} | {_act}\n"
                    f"MAE={_mae:.4f}  RMSE={_rms:.4f}", fontsize=9)
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
            mae_val  = mean_absolute_error(raw_values, y_pred)
            rmse_val = np.sqrt(mean_squared_error(raw_values, y_pred))
            ax.plot(raw_values, label='Actual (raw)', color='steelblue', linewidth=2)
            ax.plot(y_pred, label='Predicted', color='tomato',
                    linewidth=2, linestyle='--')
            ax.set_title(
                f"ID {curve['instance_id']} | {curve['activity']}\n"
                f"MAE={mae_val:.4f}  RMSE={rmse_val:.4f}", fontsize=9)
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


# ---------------------------------------------------------------------------
# Energy-distribution evaluation — no curve/instance matching required.
#
# Instead of pairing a specific simulated activity instance with a specific
# real one (which breaks down once cases start/end at different times, or a
# stochastic sim produces a different number of activities per case than
# reality), this reduces every instance's curve to a summary statistic and
# compares the *distributions* of that statistic between real and simulated,
# pooled at two granularities:
#   - per (activity, sensor)  — pools across all cases
#   - per sensor              — pools per-case totals across all cases
# ---------------------------------------------------------------------------

ZERO_CALIBRATION_MIN_FRACTION = 0.05


def _zero_calibrate_barycenter(reference_curve, resampled_train):
    """
    Restore the duty cycle (on/off intermittency) that DBA averages away.

    Many of these sensors are intermittent: the equipment cycles on and off
    within an activity, so the REAL curves sit at exactly zero most of the
    time (measured: 80-87% zeros on processes 1/2/3). Averaging destroys
    that -- the "off" blocks land at different phases in different cases, so
    at nearly every position at least one case is on and the barycenter is
    non-zero almost everywhere (measured: DBA reproduces only 26% zeros
    against a real 87%, and a plain mean reproduces 0%). The barycenter is a
    good estimate of the CONDITIONAL MEAN but an implausible SAMPLE, and the
    complete-profile evaluation is distributional, so it is scored as a
    sample.

    Fix: keep the DBA shape, then snap its lowest-q fraction to exactly zero,
    where q is the zero-fraction of the TRAINING curves (no test information
    is used). Measured on 17 zero-inflated sensors, held out:
        DBA            W1 0.118, zeros 26.0%, RMSE 1.006
        DBA+zero-snap  W1 0.093, zeros 82.7%, RMSE 0.998   (real zeros 86.7%)
    i.e. 21% better on W1 -- the headline complete-profile metric -- while
    also fixing the duty cycle, and neutral on RMSE. Wins W1 on 16/17
    sensors. Note a MEDIAN barycenter also fixes the zeros but doubles W1
    (0.232), so it is not the right trade here.

    No-ops for sensors whose training curves are not meaningfully
    zero-inflated (< ZERO_CALIBRATION_MIN_FRACTION), leaving continuously
    running sensors exactly as before.
    """
    ref = np.asarray(reference_curve, dtype=float)
    train = np.asarray(resampled_train, dtype=float)
    if train.size == 0 or ref.size == 0:
        return ref
    zero_fraction = float(np.mean(np.abs(train) < 1e-9))
    if zero_fraction < ZERO_CALIBRATION_MIN_FRACTION:
        return ref
    threshold = float(np.quantile(ref, min(zero_fraction, 1.0)))
    calibrated = ref.copy()
    calibrated[calibrated <= threshold] = 0.0
    return calibrated


def _clip_physical(curve):
    """
    Clip a predicted sensor curve to the physically valid range.

    Every sensor modelled here measures a non-negative quantity (energy /
    power demand, mass or volume flow, temperature in deg C above the
    process floor, concentration), and the REAL data confirms it: 0.00%
    negative readings on processes 1/2/3/5 and <0.1% (sensor noise around
    zero) on 4_1/4_2. The curve regressors, however, are unconstrained
    (GradientBoosting / linear DTW decode) and freely predict below zero --
    measured at ~23-24% of all predicted points on processes 2 and 3, down
    to -512 kW of "cooling water demand".

    Those values are physically impossible, and negative mass in a predicted
    curve corrupts any downstream distributional comparison directly.
    Clipping is applied to EVERY predictor, so it is a correctness fix, not
    a thumb on the scale for any one method.
    """
    return np.clip(np.asarray(curve, dtype=float), 0, None)


def build_exog_lookup(df_expanded, ef_cols=None, time_col='datetime_energy'):
    """
    Time-indexed table of the ef_* external factors, for use at simulation time.

    Weather is exogenous: its value at a timestamp is the same for every case,
    object and sensor, and it is known for any date without predicting anything.
    So one table indexed by timestamp is all a simulated activity needs to get
    the real external conditions over its own window.

    Returns a DataFrame indexed by timestamp (sorted, one row per timestamp,
    duplicates averaged), or None when no ef_* columns are present.
    """
    if df_expanded is None or len(df_expanded) == 0 or time_col not in df_expanded.columns:
        return None
    cols = [c for c in (ef_cols or [c for c in df_expanded.columns if c.startswith('ef_')])
            if c in df_expanded.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df_expanded[c])]
    if not cols:
        return None
    tbl = df_expanded[[time_col] + cols].copy()
    tbl[time_col] = pd.to_datetime(tbl[time_col])
    tbl = tbl.dropna(subset=[time_col]).groupby(time_col, as_index=True)[cols].mean()
    return tbl.sort_index()


def _exog_series_for_window(exog_lookup, exog_cols, ts_start, ts_end, n_ts):
    """
    Per-timestep external factors over one activity's window: the actual value
    of each ef_* column at each of the n_ts steps between ts_start and ts_end,
    read off the time-indexed lookup with nearest-timestamp matching (weather is
    typically hourly, curve steps are finer).

    Returns {col: np.ndarray(n_ts)}, or {} when it cannot be built.
    """
    if exog_lookup is None or exog_lookup.empty or not exog_cols:
        return {}
    if ts_start is None or ts_end is None or pd.isnull(ts_start) or pd.isnull(ts_end):
        return {}
    cols = [c for c in exog_cols if c in exog_lookup.columns]
    if not cols:
        return {}
    ts_start, ts_end = pd.to_datetime(ts_start), pd.to_datetime(ts_end)
    if ts_end < ts_start:
        ts_end = ts_start
    # A tz-aware simulated log against a tz-naive weather table (or vice versa)
    # makes the index union below raise; normalise both to naive wall time.
    _idx_tz = getattr(exog_lookup.index, 'tz', None)
    if (_idx_tz is not None) != (ts_start.tz is not None):
        if _idx_tz is not None:
            exog_lookup = exog_lookup.copy()
            exog_lookup.index = exog_lookup.index.tz_localize(None)
        else:
            ts_start, ts_end = ts_start.tz_localize(None), ts_end.tz_localize(None)
    grid = pd.date_range(ts_start, ts_end, periods=max(int(n_ts), 1))
    vals = exog_lookup[cols].reindex(exog_lookup.index.union(grid)).interpolate(
        method='time', limit_direction='both').reindex(grid)
    return {c: np.asarray(vals[c].to_numpy(), dtype=float) for c in cols
            if not vals[c].isna().all()}


def predict_curve_for_instance(activity, object_name, duration_minutes,
                               object_attributes, energy_pipelines, sensor,
                               activity_exog_means=None,
                               temporal_resolution_minutes=15.0,
                               exog_lookup=None, ts_start=None, ts_end=None):
    """
    Predict one sensor's curve for a single (activity, object) instance of a
    given duration, using a trained `energy_pipelines` dict — the same
    prediction mechanism ProcessSimulation's energy-aware modes use
    internally (see utils/simulation.py's "Update energy state" step), exposed
    standalone so it can be applied *after the fact* to the output of any
    simulation mode, not just petri_net_energy_*/petri_net_energy_direct*.

    All returned curves are clipped at zero (see _clip_physical): these
    sensors measure demand / flow / temperature, which are physically
    non-negative, but the underlying regressors are unconstrained and do
    emit negative values (up to ~24% of predicted points on some processes),
    which is indefensible and corrupts every downstream distributional
    comparison.

    Returns None when no pipeline exists for (sensor, activity, object).
    """
    obj_map = energy_pipelines.get(sensor, {}).get(activity, {})
    ep = obj_map.get(object_name) or (next(iter(obj_map.values())) if obj_map else None)
    if ep is None:
        return None
    ref_curve = ep.get('reference_curve')
    if ref_curve is None or len(ref_curve) == 0:
        return None

    predict_fn = ep.get('predict_fn')
    # exog_cols lives at the top level for the energy-aware modes' pipelines,
    # but is nested under 'full_pipeline' for Curve-Only Evaluation pipelines
    # (baseline, ml_external, etc. all wrap the trained pipeline dict
    # under 'full_pipeline' when reassembled — see 02_modelling.py).
    exog_cols = ep.get('exog_cols') or ep.get('full_pipeline', {}).get('exog_cols')

    n_ts = max(2, round(duration_minutes / temporal_resolution_minutes))

    # External factors over THIS activity's own window, one value per timestep,
    # read from the real exogenous series at the simulated timestamps. This is
    # what the pipelines were trained on (ef_* resampled per canonical
    # position), so it keeps train and simulation inputs the same shape.
    exog_vals = {}
    if exog_cols:
        exog_vals = _exog_series_for_window(exog_lookup, exog_cols, ts_start, ts_end, n_ts)
    if not exog_vals and exog_cols and activity_exog_means:
        # Fallback only: per-activity training mean, broadcast flat across the
        # curve. Loses all within-window and seasonal variation — used when no
        # exog_lookup/timestamps were supplied by the caller.
        act_means = activity_exog_means.get(activity, {})
        exog_vals = {
            col: np.array([v]) for col, v in act_means.items()
            if col in exog_cols
        }

    input_curve = np.interp(
        np.linspace(0, 1, n_ts),
        np.linspace(0, 1, len(ref_curve)),
        ref_curve,
    )
    if predict_fn is None:
        return _clip_physical(input_curve)

    # energy_pipelines is populated by several different producers with
    # different predict_fn signatures: the energy-aware simulation modes use
    # keyword args (raw_values=, activity=, object_attributes=, exog=);
    # Curve-Only Evaluation's plain approaches (baseline, ml_dtw, ml_only)
    # use positional (rv, act, attrs) with no exog param at all; its
    # exog-aware approaches (ml_external, seq2seq_external) use
    # positional (rv, act, attrs, exog=None). Try each in turn — a plain
    # 3-arg positional call as the last resort would silently succeed on an
    # exog-aware predict_fn by using its exog=None default, silently
    # dropping the external-factor values, so the exog-carrying attempt
    # must come before the exog-less one.
    exog_arg = exog_vals if exog_vals else None
    attempts = (
        lambda: predict_fn(raw_values=input_curve, activity=activity,
                           object_attributes=object_attributes, exog=exog_arg),
        lambda: predict_fn(input_curve, activity, object_attributes, exog_arg),
        lambda: predict_fn(input_curve, activity, object_attributes),
    )
    last_exc = None
    for attempt in attempts:
        try:
            return _clip_physical(attempt())
        except TypeError as exc:
            last_exc = exc
    raise last_exc


def _inject_schedule_idle_min(simulated_df, case_col='case_id',
                              activity_col='activity', object_col='object',
                              resource_col='resource_id'):
    """Recompute `idle_min` from the SIMULATED schedule and write it into each row's
    object_attributes.

    The generator stamps one idle_min per autoclave cycle / station visit — how long
    that resource stood idle before it — and the holding duty keys off it, so the
    level models are trained on it. A simulated log, though, inherits
    object_attributes from the production plan, which carries only the case's FIRST
    value: every simulated event of every case then arrives with the same number (the
    batch-stagger gap), and the feature has no variance left to predict from.

    Rebuilt here from the timestamps the simulation itself produced, exactly as
    prev_act_name / prev_act_duration_min are: per resource, the gap between a visit's
    start and the previous visit's end, held constant across that visit's events. A
    visit starts when the case changes or the activity is a 'prepare' — the same two
    boundaries the generator uses, since a rework restarts at 'prepare' within one case.

    A run whose training data has no idle_min is unaffected: the extra key is dropped
    when the feature frame is reindexed onto the training layout.
    """
    if simulated_df.empty or 'object_attributes' not in simulated_df.columns:
        return simulated_df
    # Which column names the RESOURCE. A simulated log carries the process-level
    # object ('sterilization_process') in `object` and the actual machine in
    # `resource_id`; a real log names the machine in `object` and has no
    # resource_id. Grouping by the wrong one puts every case on one resource and
    # every gap collapses to zero.
    res_col = (resource_col
               if resource_col in simulated_df.columns
               and simulated_df[resource_col].nunique(dropna=True) > 1
               else object_col)
    if res_col not in simulated_df.columns:
        return simulated_df
    df = simulated_df.copy()
    idle = np.zeros(len(df), dtype=float)
    pos = {ix: k for k, ix in enumerate(df.index)}
    origin = df['timestamp_start'].min()
    for _, g in df.groupby(df[res_col], sort=False):
        g = g.sort_values('timestamp_start')
        prev_end, prev_case, gap = None, None, 0.0
        for ix, r in g.iterrows():
            if (prev_case is None or r[case_col] != prev_case
                    or str(r[activity_col]).endswith('prepare')):
                ref = prev_end if prev_end is not None else origin
                gap = max(0.0, (r['timestamp_start'] - ref).total_seconds() / 60.0)
            idle[pos[ix]] = gap
            prev_end = (r['timestamp_end'] if prev_end is None
                        else max(prev_end, r['timestamp_end']))
            prev_case = r[case_col]
    df['object_attributes'] = [
        {**(a if isinstance(a, dict) else {}), 'idle_min': float(v)}
        for a, v in zip(df['object_attributes'], idle)
    ]
    return df


def _extract_real_case_curves_all_sensors(real_case_df, sensors,
                                          time_col='datetime_energy', start_col='timestamp_start_log'):
    """
    Build every sensor's (relative_time_minutes, value) curve for one real
    case in a single pass — sorts the case's rows by timestamp once and
    reuses that ordering for every sensor, instead of re-sorting per sensor.

    Returns {sensor: (times, values)}, only for sensors with usable data.
    """
    if real_case_df.empty:
        return {}
    df = real_case_df.sort_values(time_col)
    case_start = pd.to_datetime(df[start_col]).min()
    t_all = (pd.to_datetime(df[time_col]) - case_start).dt.total_seconds().values / 60.0

    result = {}
    for sensor in sensors:
        if sensor not in df.columns:
            continue
        mask = df[sensor].notna().values
        if not mask.any():
            continue
        result[sensor] = (t_all[mask], df[sensor].values[mask].astype(float))
    return result


# A leaf pipeline exists only for (sensor, activity, object) combinations with
# enough training curves (>=5). When one is missing, the case assembler used to
# emit NOTHING for that activity's interval, so the assembled case curve had a
# literal hole: no samples at all while an activity was running. Measured on
# experiment_1's budget mode, 81.5% of (case, sensor) curves had at least one
# such hole and process_4_1 lost a median 28.1% of every case span -- always to a
# running activity, never to idle time. That silently depressed every
# complete-profile metric (a plain sum over samples loses the missing minutes
# outright) and it did so ONLY for the process-model rows, since the
# schedule-direct rows predict one whole-case curve and can never have a hole.
# Filling the gap makes the comparison whole-case on both sides.
#
# Set PIPELINE_COMPLETE_CURVE_FILL_MISSING=false to restore the old skip.
COMPLETE_CURVE_FILL_MISSING = _os_seed.environ.get(
    'PIPELINE_COMPLETE_CURVE_FILL_MISSING', 'true').lower() == 'true'


def _build_gap_fill_curves(energy_pipelines, sensors):
    """
    One fallback curve per sensor: the pointwise median of that sensor's trained
    leaf `reference_curve`s, resampled to their median length. Same pooling idea
    as the 'baseline' row (everything the sensor has, no conditioning) but taken
    from the pipelines being evaluated, so no extra training pass and no access
    to the test split — and it keeps a SHAPE, unlike the flat 'baseline' level,
    because a gap in a simulated day is better filled with a representative
    profile than with a constant.

    Returns {sensor: np.ndarray}; a sensor with no usable reference curve is
    absent from the dict and its gaps stay unfilled.
    """
    out = {}
    for sensor in sensors:
        refs = []
        for _act_map in (energy_pipelines.get(sensor) or {}).values():
            for _ep in (_act_map or {}).values():
                _rc = (_ep or {}).get('reference_curve')
                if _rc is not None and len(_rc) >= 2:
                    refs.append(np.asarray(_rc, dtype=float))
        if not refs:
            continue
        L = max(2, int(np.median([len(r) for r in refs])))
        stack = np.array([np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(r)), r)
                          for r in refs])
        out[sensor] = _clip_physical(np.median(stack, axis=0))
    return out


def _predict_case_curves_all_sensors(sim_case_df, energy_pipelines, sensors,
                                     activity_exog_means=None,
                                     temporal_resolution_minutes=15.0,
                                     activity_col='activity', object_col='object',
                                     exog_lookup=None,
                                     gap_fill_curves=None, fill_stats=None):
    """
    Predict every sensor's (relative_time_minutes, value) curve for one
    simulated case (already filtered to that case_id) in a single pass over
    its activities — one predict_curve_for_instance call per (activity,
    sensor), same as before, but only one iteration over the case's rows
    total instead of one per sensor. That matters: iterating a DataFrame is
    the expensive part here (row-tuple construction), not the predict call
    itself, so doing it len(sensors) times per case was the dominant cost
    for processes with many cases (verified: a 657-case process took ~2.5
    min per (mode, approach) combo, a 15-case process ~2 seconds).

    Uses itertuples (fast) rather than iterrows (constructs a full Series
    per row — much slower at this scale).

    Returns {sensor: (times, values)}, only for sensors with at least one
    usable predicted curve for this case.

    `gap_fill_curves` (from _build_gap_fill_curves) covers activities with no
    trained leaf, so the assembled case curve spans the whole case instead of
    carrying a hole — see COMPLETE_CURVE_FILL_MISSING for why that matters.
    `fill_stats` is an optional dict the caller passes in to collect what was
    filled; a fill is a real gap in the model's coverage, so it is counted and
    reported rather than left to look like a successful prediction.
    """
    sim_case_df = sim_case_df.sort_values('timestamp_start')
    case_start = sim_case_df['timestamp_start'].iloc[0]
    times_acc  = {s: [] for s in sensors}
    values_acc = {s: [] for s in sensors}

    # Predecessor context from the simulated log itself — the same two
    # event-log features (activity name + wall-clock duration) the
    # ml_external training side gets from split_curves_with_prev_activity
    # (include_prev_energy=False). 'none'/0.0 mirror that function's
    # first-of-case sentinels; no meter readings are involved, so the
    # simulation is not peeking at any energy.
    prev_name, prev_dur_min = 'none', 0.0

    for row in sim_case_df.itertuples(index=False):
        ts_start = getattr(row, 'timestamp_start')
        ts_end   = getattr(row, 'timestamp_end')
        duration_minutes = (ts_end - ts_start).total_seconds() / 60.0
        activity = getattr(row, activity_col)
        obj      = getattr(row, object_col)
        object_attributes = dict(getattr(row, 'object_attributes', {}) or {})
        object_attributes['prev_act_name'] = prev_name
        object_attributes['prev_act_duration_min'] = prev_dur_min
        t_start = (ts_start - case_start).total_seconds() / 60.0
        t_end   = (ts_end   - case_start).total_seconds() / 60.0

        for sensor in sensors:
            curve = predict_curve_for_instance(
                activity, obj, duration_minutes,
                object_attributes, energy_pipelines, sensor,
                activity_exog_means, temporal_resolution_minutes,
                exog_lookup=exog_lookup, ts_start=ts_start, ts_end=ts_end,
            )
            if curve is None or len(curve) == 0:
                # No trained leaf for this (sensor, activity, object). Fill with
                # the sensor's fallback curve on this activity's own grid rather
                # than emitting nothing — see COMPLETE_CURVE_FILL_MISSING.
                _fb = (gap_fill_curves or {}).get(sensor)
                if _fb is None or not COMPLETE_CURVE_FILL_MISSING:
                    continue
                _n_ts = max(2, round(duration_minutes / max(temporal_resolution_minutes, 1e-9)))
                curve = np.interp(np.linspace(0, 1, _n_ts),
                                  np.linspace(0, 1, len(_fb)), _fb)
                if fill_stats is not None:
                    fill_stats['filled'] = fill_stats.get('filled', 0) + 1
                    fill_stats['minutes'] = fill_stats.get('minutes', 0.0) + float(duration_minutes)
                    fill_stats.setdefault('activities', {})
                    fill_stats['activities'][str(activity)] = (
                        fill_stats['activities'].get(str(activity), 0.0) + float(duration_minutes))
            n = len(curve)
            t = np.linspace(t_start, t_end, n) if n > 1 else np.array([t_start])
            times_acc[sensor].append(t)
            values_acc[sensor].append(np.asarray(curve, dtype=float))

        prev_name, prev_dur_min = str(activity), float(duration_minutes)

    result = {}
    for sensor in sensors:
        if times_acc[sensor]:
            result[sensor] = (np.concatenate(times_acc[sensor]), np.concatenate(values_acc[sensor]))
    return result


def compare_complete_case_curves(real_expanded_df, simulated_df, energy_pipelines, sensors,
                                 activity_exog_means=None,
                                 temporal_resolution_minutes=15.0,
                                 real_case_col='case_id_log', sim_case_col='case_id',
                                 save_curves=False, exog_lookup=None):
    """
    Per real test case (matched to the simulated log by case_id), per sensor:
    build the real case's complete profile and the simulated case's complete
    predicted profile, on a shared case-relative time axis.

    Returns a DataFrame with one row per (case_id, sensor):
    ['case_id', 'sensor', 'n_real_pts', 'n_sim_pts'].

    When save_curves=True, also returns a second, long-format DataFrame with
    one row per (case_id, sensor, series, timestep) — series in {'real',
    'predicted'} — columns ['case_id', 'sensor', 'series', 't_minutes',
    'value'], so the exact curves can be reloaded later for other metrics or
    plots without re-simulating anything.
    """
    # Match by case_id as strings, not raw values — the real expanded df and
    # a simulated log can carry the same case_id in different dtypes (e.g.
    # float vs str for numeric-looking IDs), which would silently zero out
    # every match under a raw-value set intersection despite full overlap.
    # The real expanded frame already carries the ef_* series, so the
    # timestamp-resolved external factors need no extra input from the caller.
    if exog_lookup is None:
        exog_lookup = build_exog_lookup(real_expanded_df)

    # idle_min rebuilt from this simulation's own schedule — see the helper.
    simulated_df = _inject_schedule_idle_min(simulated_df, sim_case_col)

    real_ids_str = real_expanded_df[real_case_col].astype(str)
    sim_ids_str  = simulated_df[sim_case_col].astype(str)
    real_valid   = real_expanded_df[real_case_col].notna()
    sim_valid    = simulated_df[sim_case_col].notna()

    real_cases = set(real_ids_str[real_valid].unique())
    sim_cases  = set(sim_ids_str[sim_valid].unique())
    shared_cases = sorted(real_cases & sim_cases)

    # Group once instead of re-filtering the full dataframe with a boolean
    # mask for every case — get_group is an O(1) amortised lookup after this,
    # vs. an O(n_rows) scan per case with the old `df[ids == cid]` approach.
    real_groups = real_expanded_df[real_valid].groupby(real_ids_str[real_valid])
    sim_groups  = simulated_df[sim_valid].groupby(sim_ids_str[sim_valid])

    # Per-sensor fallback curves for activities with no trained leaf, built once
    # for the whole comparison — see COMPLETE_CURVE_FILL_MISSING.
    _gap_fill = _build_gap_fill_curves(energy_pipelines, sensors) if COMPLETE_CURVE_FILL_MISSING else {}
    _fill_stats = {}

    rows = []
    curve_rows = [] if save_curves else None
    for cid in shared_cases:
        real_g = real_groups.get_group(cid)
        sim_g  = sim_groups.get_group(cid)
        if real_g.empty or sim_g.empty:
            continue

        # One pass over this case's rows per side, covering every sensor —
        # not one pass per sensor. See _predict_case_curves_all_sensors'
        # docstring for why this is the change that actually matters at scale.
        real_curves = _extract_real_case_curves_all_sensors(real_g, sensors)
        if not real_curves:
            continue
        sim_curves = _predict_case_curves_all_sensors(
            sim_g, energy_pipelines, list(real_curves.keys()),
            activity_exog_means, temporal_resolution_minutes,
            exog_lookup=exog_lookup,
            gap_fill_curves=_gap_fill, fill_stats=_fill_stats,
        )

        for sensor, (t_real, v_real) in real_curves.items():
            sim_pair = sim_curves.get(sensor)
            if sim_pair is None:
                continue
            t_sim, v_sim = sim_pair

            # Only curves with positive total mass are kept — this guard
            # filters which (case, sensor) curves reach the saved
            # predicted_curves parquet outputs downstream.
            w_real = np.clip(v_real, 0, None)
            w_sim  = np.clip(v_sim, 0, None)
            if w_real.sum() <= 0 or w_sim.sum() <= 0:
                continue

            rows.append({
                'case_id':           cid,
                'sensor':            sensor,
                'n_real_pts':        int(len(v_real)),
                'n_sim_pts':         int(len(v_sim)),
            })

            if save_curves:
                curve_rows.extend({
                    'case_id': cid, 'sensor': sensor, 'series': 'real',
                    't_minutes': float(t), 'value': float(v),
                } for t, v in zip(t_real, v_real))
                curve_rows.extend({
                    'case_id': cid, 'sensor': sensor, 'series': 'predicted',
                    't_minutes': float(t), 'value': float(v),
                } for t, v in zip(t_sim, v_sim))

    # Never silent: a filled interval is a gap in the model's coverage, and the
    # whole reason this exists is that the previous behaviour (drop the interval)
    # was invisible in the outputs and quietly biased every metric.
    if _fill_stats.get('filled'):
        _acts = sorted(_fill_stats.get('activities', {}).items(), key=lambda kv: -kv[1])
        print(f"  ℹ️ complete-curve: filled {_fill_stats['filled']:,} (activity, sensor) "
              f"intervals with no trained leaf, {_fill_stats['minutes']:,.0f} activity-minutes "
              f"total, using the per-sensor fallback curve. Top: "
              + ', '.join(f'{a} ({m:,.0f} min)' for a, m in _acts[:3]))
    _no_fb = [s for s in sensors if s not in _gap_fill]
    if COMPLETE_CURVE_FILL_MISSING and _no_fb:
        print(f"  ⚠️ complete-curve: no fallback curve for {len(_no_fb)} sensor(s) "
              f"(no trained leaf at all) — their gaps are still dropped: {_no_fb[:3]}")

    if save_curves:
        return pd.DataFrame(rows), pd.DataFrame(curve_rows)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# "Schedule Profile Evaluation" — whole-case reference generators.
#
# Everything above (compare_complete_case_curves) predicts a case's energy
# profile by first simulating its activities/durations via the discovered
# process model, then predicting each activity's curve. This section works at
# CASE granularity instead: one complete real curve plus schedule-level
# features per case (build_case_level_curves), a dedicated case-duration
# regressor (train_case_duration_pipeline), and two naive reference
# generators — a stochastic generator (per-canonical-position Normal fit,
# sampled), the "DES + stochastic distributions" style of prior energy-DES
# literature (e.g. Kouki et al. 2017), plus a bootstrap resampler of whole
# training curves.
#
# Since there's no simulated duration for a generated profile, each sampled
# profile is placed on the case's predicted duration when a duration
# pipeline is available, else on the median real case duration over the
# training set.
# ---------------------------------------------------------------------------

def build_case_level_curves(real_expanded_df, sensor, ef_cols=None,
                            case_col='case_id_log', time_col='datetime_energy',
                            start_col='timestamp_start_log'):
    """
    Build one complete real curve plus a schedule-only feature dict per case,
    for one sensor. Deliberately no per-activity information — this is the
    same granularity a production schedule offers before any process
    simulation: case/recipe attributes, case start time, external factors.

    Returns a list of dicts: {'case_id', 'values', 'duration_minutes', 'attributes'}.
    """
    ef_cols = ef_cols or []
    out = []
    for cid, g in real_expanded_df.groupby(case_col):
        curves = _extract_real_case_curves_all_sensors(g, [sensor], time_col=time_col, start_col=start_col)
        if sensor not in curves:
            continue
        _, v = curves[sensor]
        if len(v) < 2:
            continue

        raw_attrs = g['object_attributes_log'].iloc[0] if 'object_attributes_log' in g.columns and not g.empty else {}
        attrs = dict(raw_attrs) if isinstance(raw_attrs, dict) else {}
        ts = pd.to_datetime(g[start_col]).min()
        attrs['hour_of_day'] = float(ts.hour) if pd.notnull(ts) else 0.0
        attrs['day_of_week'] = float(ts.dayofweek) if pd.notnull(ts) else 0.0
        for col in ef_cols:
            if col in g.columns:
                vals = g[col].dropna()
                attrs[col] = float(vals.mean()) if len(vals) else float('nan')

        case_start = pd.to_datetime(g[start_col]).min()
        case_end   = pd.to_datetime(g[time_col]).max()
        duration_minutes = (case_end - case_start).total_seconds() / 60.0 if pd.notnull(case_start) and pd.notnull(case_end) else float(len(v))

        out.append({'case_id': cid, 'values': v, 'duration_minutes': max(duration_minutes, 1e-6),
                    'attributes': attrs})
    return out


def train_case_duration_pipeline(train_cases, val_size=0.2, random_state=42,
                                 model_class=None, verbose=0,
                                 allow_median_fallback=True):
    """
    Train a case-level TOTAL DURATION predictor: schedule-only attributes
    (see build_case_level_curves) -> scalar case duration (minutes). One row
    per case, with a single scalar target.

    Used to correct "Best, duration-corrected": the process-simulation
    timeline (discovered process model + per-activity duration draws) has
    no mechanism forcing its total elapsed time to be realistic -- local
    per-activity errors and resource/shift queueing artifacts compound
    freely, so a case's simulated total span can drift arbitrarily far from
    its real duration even when the individual predicted values are good.
    This pipeline gives an independent, dedicated estimate of what the total
    should be, so the simulated timeline can be rescaled to match it (see
    rescale_case_curve_to_duration) instead of trusting the raw simulated
    span.

    Returns a pipeline dict compatible with predict_case_duration, or None
    if there isn't enough data to train on.
    """
    if model_class is None:
        model_class = GradientBoostingRegressor
    if len(train_cases) < 4:
        if verbose:
            print(f"  [case-duration] skipped: only {len(train_cases)} cases (need >= 4)")
        return None

    all_keys, key_types = _infer_key_types(train_cases)

    rows = []
    for c in train_cases:
        row = {'case_id': c['case_id'], 'y': c['duration_minutes']}
        for key in all_keys:
            value = c['attributes'].get(key, None)
            if key_types[key] == 'numeric':
                try:
                    row[key] = float(value) if value is not None else np.nan
                except (ValueError, TypeError):
                    row[key] = np.nan
            else:
                row[key] = str(value) if value is not None else 'None'
        rows.append(row)
    df_reg = pd.DataFrame(rows)

    categorical_cols = [k for k in all_keys if key_types[k] == 'category']
    X_all = pd.DataFrame(index=df_reg.index)
    for key in all_keys:
        X_all = X_all.assign(**{key: df_reg[key].values})
    if categorical_cols:
        X_all = pd.get_dummies(X_all, columns=categorical_cols, drop_first=True)
    y_all = df_reg['y'].copy()

    unique_cases = df_reg['case_id'].unique()
    if len(unique_cases) < 4:
        train_inst, val_inst = unique_cases, unique_cases
    else:
        train_inst, val_inst = train_test_split(unique_cases, test_size=val_size, random_state=random_state)
    train_mask = df_reg['case_id'].isin(train_inst)
    val_mask   = df_reg['case_id'].isin(val_inst)

    X_train, X_val = X_all[train_mask].copy(), X_all[val_mask].copy()
    y_train, y_val = y_all[train_mask].copy(), y_all[val_mask].copy()
    feature_columns = X_all.columns.tolist()

    numeric_feature_cols = [k for k in all_keys if key_types[k] == 'numeric']
    numeric_feature_cols = [c for c in numeric_feature_cols if c in X_train.columns]

    feature_scaler = None
    if numeric_feature_cols:
        X_train[numeric_feature_cols] = X_train[numeric_feature_cols].astype('float64')
        X_val[numeric_feature_cols]   = X_val[numeric_feature_cols].astype('float64')
        feature_scaler = StandardScaler()
        X_train.loc[:, numeric_feature_cols] = feature_scaler.fit_transform(X_train[numeric_feature_cols])
        X_val.loc[:, numeric_feature_cols]   = feature_scaler.transform(X_val[numeric_feature_cols])

    X_train = X_train.fillna(0)
    X_val   = X_val.fillna(0)

    model = model_class(n_estimators=150, max_depth=5, learning_rate=0.1,
                        subsample=0.8, random_state=random_state)
    model.fit(X_train, y_train)
    val_mae = float(mean_absolute_error(y_val, model.predict(X_val))) if len(X_val) else float('nan')

    # Guard: only trust the learned model if it actually beats a trivial
    # "always predict the median training duration" baseline on the held-out
    # validation cases. With few training cases (common per sensor/process)
    # a GBR can easily be noisier than just guessing the median -- and since
    # this pipeline's whole job is to correct a wrong simulated duration,
    # feeding it a worse-than-median prediction would actively hurt the
    # duration-corrected reconstruction rather than help it.
    median_duration = float(np.median(y_train))
    baseline_val_mae = (float(mean_absolute_error(y_val, np.full(len(y_val), median_duration)))
                       if len(y_val) else float('nan'))
    use_median_fallback = not (pd.notna(val_mae) and pd.notna(baseline_val_mae) and val_mae < baseline_val_mae)
    # allow_median_fallback=False keeps the model even when it loses to the median.
    # Used for the schedule-direct span: this floor puts a bound on how wrong that
    # span can be, and the simulated Petri-net span has no equivalent bound, so
    # leaving it on hands one side of the comparison a guard the other never gets.
    # Everything else (notably "Best, duration-corrected", whose whole job is to
    # REPAIR a wrong span) keeps the floor, where the guard is the point.
    if not allow_median_fallback:
        use_median_fallback = False

    if verbose:
        _status = 'median fallback (model did not beat it)' if use_median_fallback else 'model'
        print(f"  [case-duration] trained on {len(train_cases)} cases, val_mae={val_mae:.2f} min "
              f"vs. median baseline={baseline_val_mae:.2f} min -> using {_status}")

    return {
        'model': model, 'all_keys': all_keys, 'key_types': key_types,
        'feature_columns': feature_columns, 'feature_scaler': feature_scaler,
        'numeric_feature_cols': numeric_feature_cols, 'val_mae': val_mae,
        'median_duration': median_duration, 'baseline_val_mae': baseline_val_mae,
        'use_median_fallback': use_median_fallback,
    }


def predict_case_duration(attributes, pipeline):
    """
    Predict a scalar total case duration (minutes) from schedule-only
    attributes -- falls back to the median training duration if the learned
    model didn't beat that trivial baseline on held-out validation cases
    (see use_median_fallback in train_case_duration_pipeline).
    """
    if pipeline.get('use_median_fallback'):
        return pipeline['median_duration']

    feature_columns      = pipeline['feature_columns']
    feature_scaler       = pipeline.get('feature_scaler')
    numeric_feature_cols = pipeline.get('numeric_feature_cols', [])
    all_keys             = pipeline['all_keys']
    key_types            = pipeline['key_types']

    row = {}
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

    categorical_cols = [k for k in all_keys if key_types[k] == 'category']
    if categorical_cols:
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
    X = X.fillna(0)

    pred = pipeline['model'].predict(X)
    return max(float(pred[0]), 1e-6)


def rescale_case_curve_to_duration(t_minutes, predicted_duration, raw_simulated_duration):
    """
    Rescale a simulated case's reconstructed curve time axis so its total
    span matches predicted_duration, given the raw simulated schedule's own
    (potentially very wrong) total span raw_simulated_duration. Preserves
    the relative sequencing/shape the process simulation contributes --
    which activity happens before/after which, roughly how big one is next
    to another -- only the absolute scale is corrected.
    """
    if raw_simulated_duration <= 1e-6:
        return t_minutes
    scale = predicted_duration / raw_simulated_duration
    return t_minutes * scale


def fit_stochastic_profile_generator(train_cases, fixed_length=None):
    """
    Population-level reference generator with no schedule conditioning at
    all: fit a Normal distribution per canonical position from the real
    training case curves (same fractional-progress resampling as the
    barycenter above). Matches the "DES + stochastic distributions" style of
    prior energy-DES literature (e.g. Kouki et al. 2017), as opposed to a
    learned model — the floor a schedule-aware model needs to beat.
    """
    if len(train_cases) < 2:
        return None
    if fixed_length is None:
        fixed_length = max(2, int(round(np.median([len(c['values']) for c in train_cases]))))
    resampled = np.array([
        np.interp(np.linspace(0, 1, fixed_length), np.linspace(0, 1, len(c['values'])), c['values'])
        for c in train_cases
    ])
    return {'mean': resampled.mean(axis=0), 'std': resampled.std(axis=0), 'fixed_length': fixed_length}


def sample_stochastic_profile(generator, case_duration_minutes, rng=None):
    """Draw one sample curve from the fitted per-position Normal distributions.

    The time axis is placed on ``case_duration_minutes`` — pass a per-case
    predicted duration (see predict_case_duration) for a schedule-aware span,
    or a single median duration for the classic constant-span behaviour.
    """
    rng = rng if rng is not None else np.random.default_rng(GLOBAL_RANDOM_SEED)
    mean, std = generator['mean'], generator['std']
    sample = rng.normal(mean, np.clip(std, 1e-6, None))
    sample = np.clip(sample, 0, None)
    t = np.linspace(0, case_duration_minutes, generator['fixed_length'])
    return t, sample


def fit_bootstrap_profile_generator(train_cases, fixed_length=None):
    """
    Non-parametric empirical-bootstrap reference generator: instead of a
    fitted per-position Normal (fit_stochastic_profile_generator), it keeps the
    real training case curves themselves (resampled to a common canonical
    length) and, at sample time, draws one of them at random. Because a whole
    real curve is returned, the within-case temporal SHAPE and autocorrelation
    are preserved exactly — precisely what the independent per-position Gaussian
    destroys — while still using no schedule conditioning. This is the classic
    resampling / block-bootstrap load-profile baseline; it is also robust with
    very few training cases (no distribution to estimate), unlike a Markov
    load model. Returns None if there aren't at least 2 cases to draw from.
    """
    if len(train_cases) < 2:
        return None
    if fixed_length is None:
        fixed_length = max(2, int(round(np.median([len(c['values']) for c in train_cases]))))
    resampled = np.array([
        np.interp(np.linspace(0, 1, fixed_length), np.linspace(0, 1, len(c['values'])), c['values'])
        for c in train_cases
    ])
    return {'curves': resampled, 'fixed_length': fixed_length}


def sample_bootstrap_profile(generator, case_duration_minutes, rng=None):
    """Draw one whole real training curve at random and place it on the given
    (per-case predicted or median) span."""
    rng = rng if rng is not None else np.random.default_rng(GLOBAL_RANDOM_SEED)
    curves = generator['curves']
    sample = np.clip(curves[rng.integers(len(curves))], 0, None)
    t = np.linspace(0, case_duration_minutes, generator['fixed_length'])
    return t, sample


def compare_schedule_and_stochastic_profiles(test_cases, stochastic_generator,
                                             median_case_duration_minutes, random_state=42,
                                             save_curves=False, bootstrap_generator=None,
                                             duration_pipeline=None):
    """
    For each real test case (from build_case_level_curves), place its real
    complete profile alongside one stochastic-generator sample (and, when
    given, one bootstrap sample) on a shared time axis.

    Returns a DataFrame with one row per case_id: ['case_id'].

    When save_curves=True, also returns a second, long-format DataFrame with
    one row per (case_id, series, timestep) — series in {'real', 'stochastic',
    'bootstrap'} — columns ['case_id', 'series', 't_minutes', 'value'], so the
    exact curves can be reloaded later for other metrics or plots. The caller
    adds a 'sensor' column since this function is called once per sensor.

    Span handling: when ``duration_pipeline`` is given, each predicted series
    (stochastic, bootstrap) is placed on that case's OWN predicted total
    duration (predict_case_duration on the case attributes), exactly like the
    "Best, duration-corrected" mode — so these baselines get the same
    schedule-aware span the best modes do, instead of a single median span for
    every case. Falls back to ``median_case_duration_minutes`` per case
    whenever no predictor is available or the prediction is non-positive.
    """
    rng = np.random.default_rng(random_state)
    rows = []
    curve_rows = [] if save_curves else None
    for c in test_cases:
        v_real = c['values']
        t_real = np.linspace(0, c['duration_minutes'], len(v_real))
        w_real = np.clip(v_real, 0, None)
        if w_real.sum() <= 0:
            continue

        # Per-case predicted span (schedule-aware), else the median fallback.
        case_span = median_case_duration_minutes
        if duration_pipeline is not None:
            try:
                _pred_span = float(predict_case_duration(c['attributes'], duration_pipeline))
                if _pred_span > 0:
                    case_span = _pred_span
            except Exception:
                pass

        row = {'case_id': c['case_id']}
        if save_curves:
            curve_rows.extend({'case_id': c['case_id'], 'series': 'real', 't_minutes': float(t), 'value': float(v)}
                              for t, v in zip(t_real, v_real))

        if stochastic_generator is not None:
            t_stoch, v_stoch = sample_stochastic_profile(stochastic_generator, case_span, rng=rng)
            # Positive-mass guard — filters which curves reach the saved
            # predicted_curves parquet downstream.
            w_stoch = np.clip(v_stoch, 0, None)
            if w_stoch.sum() > 0:
                if save_curves:
                    curve_rows.extend({'case_id': c['case_id'], 'series': 'stochastic', 't_minutes': float(t), 'value': float(v)}
                                      for t, v in zip(t_stoch, v_stoch))

        if bootstrap_generator is not None:
            t_boot, v_boot = sample_bootstrap_profile(bootstrap_generator, case_span, rng=rng)
            w_boot = np.clip(v_boot, 0, None)
            if w_boot.sum() > 0:
                if save_curves:
                    curve_rows.extend({'case_id': c['case_id'], 'series': 'bootstrap', 't_minutes': float(t), 'value': float(v)}
                                      for t, v in zip(t_boot, v_boot))

        rows.append(row)

    if save_curves:
        return pd.DataFrame(rows), pd.DataFrame(curve_rows)
    return pd.DataFrame(rows)


# %%

# ══════════════════════════════════════════════════════════════════════════════
# TRAINED-MODEL PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════
# The pipelines exist only in 02_modelling's memory and die with the run, so any
# follow-up question needing a NEW prediction (a different span policy, a new
# eval slice, a what-if schedule) costs a full retraining pass. These helpers
# persist the trained pipeline dicts to the run folder and rebuild them later.
#
# What is saved is the picklable payload: the per-leaf 'full_pipeline' dicts
# (sklearn models, arrays, scalars, torch modules) plus 'reference_curve'. The
# 'predict_fn' closures are NOT saved — they are lambdas 02_modelling wraps
# around full_pipeline, so they are rebuilt on load by rebuild_energy_predict_fns
# from the same approach -> module-level predictor mapping. The duration
# pipelines and stochastic generators are plain dicts whose predictors are
# already module-level functions, so they reload as-is.
#
# Set PIPELINE_SAVE_TRAINED_MODELS=false to skip saving (e.g. disk-tight runs).

SAVE_TRAINED_MODELS = _os_seed.environ.get(
    'PIPELINE_SAVE_TRAINED_MODELS', 'true').lower() == 'true'


def _sanitize_for_pickle(obj):
    """Recursive copy with every callable dropped (predict_fn lambdas and any
    other closure); everything else is kept as-is. Containers are rebuilt so the
    live in-memory dicts are never mutated."""
    if callable(obj) and not hasattr(obj, 'state_dict'):
        # torch nn.Modules are callable but picklable and wanted; plain
        # functions/lambdas are neither.
        return None
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            sv = _sanitize_for_pickle(v)
            if sv is not None or v is None:
                out[k] = sv
        return out
    if isinstance(obj, (list, tuple)):
        vals = [_sanitize_for_pickle(v) for v in obj]
        return type(obj)(vals) if not isinstance(obj, tuple) else tuple(vals)
    return obj


def save_trained_pipelines(pipelines, path, label=''):
    """
    Persist one pipelines structure (e.g. {sensor: {activity: {object: leaf}}}
    for an energy approach, or {sensor: {...}} for the schedule family) to
    `path` with joblib. Never raises: a failure (an unpicklable torch state, a
    corrupt leaf) prints a warning and returns False, because model saving must
    not be able to kill a multi-hour training run at the last step.
    """
    import joblib
    try:
        _os_seed.makedirs(_os_seed.path.dirname(path), exist_ok=True)
        joblib.dump(_sanitize_for_pickle(pipelines), path, compress=3)
        return True
    except Exception as exc:
        print(f"  ⚠️ save_trained_pipelines{f' [{label}]' if label else ''}: "
              f"{type(exc).__name__}: {exc} — models NOT saved to {path}")
        return False


def load_trained_pipelines(path):
    """joblib.load counterpart of save_trained_pipelines. For energy-approach
    files, follow with rebuild_energy_predict_fns to restore 'predict_fn'."""
    import joblib
    return joblib.load(path)


# approach -> callable(full_pipeline) -> predict_fn, mirroring exactly the
# lambdas 02_modelling builds at reassembly time (same signatures per family:
# exog-aware approaches take a 4th `exog` argument, the rest three args).
ENERGY_PREDICT_REBUILDERS = {
    'baseline':               lambda ep: lambda rv, act, attrs: predict_raw_curve_median(rv, act, attrs, pipeline=ep),
    'median_activity_sensor': lambda ep: lambda rv, act, attrs: predict_raw_curve_median(rv, act, attrs, pipeline=ep),
    'ml_dtw':                 lambda ep: lambda rv, act, attrs: predict_raw_curve(rv, act, attrs, pipeline=ep),
    'ml_only':                lambda ep: lambda rv, act, attrs: predict_raw_curve_ml_only(rv, act, attrs, pipeline=ep),
    'ml_external':            lambda ep: lambda rv, act, attrs, exog=None: predict_raw_curve_exog_prev_activity(rv, act, attrs, pipeline=ep, exog_values=exog or {}),
    'ml_external_wcounts':    lambda ep: lambda rv, act, attrs, exog=None: predict_raw_curve_exog_prev_activity(rv, act, attrs, pipeline=ep, exog_values=exog or {}),
    'ml_external_wmetric':    lambda ep: lambda rv, act, attrs, exog=None: predict_raw_curve_exog_prev_activity(rv, act, attrs, pipeline=ep, exog_values=exog or {}),
    'ml_step_dtw':            lambda ep: lambda rv, act, attrs, exog=None: predict_raw_curve_step_dtw(rv, act, attrs, pipeline=ep, exog_values=exog or {}),
    'ml_step_dtw_smooth':     lambda ep: lambda rv, act, attrs, exog=None: predict_raw_curve_step_dtw(rv, act, attrs, pipeline=ep, exog_values=exog or {}),
    'seq2seq':                lambda ep: lambda rv, act, attrs: predict_raw_curve_seq2seq(rv, act, attrs, pipeline=ep),
    'seq2seq_only':           lambda ep: lambda rv, act, attrs: predict_raw_curve_seq2seq_only(rv, act, attrs, pipeline=ep),
    'seq2seq_iom':            lambda ep: lambda rv, act, attrs: predict_raw_curve_seq2seq_iom(rv, act, attrs, pipeline=ep),
    'seq2seq_external':       lambda ep: lambda rv, act, attrs, exog=None: predict_raw_curve_seq2seq_external(rv, act, attrs, pipeline=ep, exog_values=exog or {}),
}


def rebuild_energy_predict_fns(pipelines_by_sensor, approach):
    """
    Reattach 'predict_fn' to every leaf of a loaded energy-approach structure
    ({sensor: {activity: {object: {'reference_curve', 'full_pipeline'}}}}),
    returning it ready for predict_curve_for_instance /
    compare_complete_case_curves. Mutates and returns the structure.

    Round trip:
        ep = load_trained_pipelines('<run>/trained_models/<proc>/ml_step_dtw_smooth.joblib')
        ep = rebuild_energy_predict_fns(ep, 'ml_step_dtw_smooth')
        compare_complete_case_curves(real_df, sim_df, ep, sensors, ...)
    """
    builder = ENERGY_PREDICT_REBUILDERS.get(approach)
    if builder is None:
        raise ValueError(f"No predict_fn rebuilder for approach {approach!r} — "
                         f"known: {sorted(ENERGY_PREDICT_REBUILDERS)}")
    for _act_map in (pipelines_by_sensor or {}).values():
        for _obj_map in (_act_map or {}).values():
            for _leaf in (_obj_map or {}).values():
                if isinstance(_leaf, dict) and _leaf.get('full_pipeline') is not None:
                    _leaf['predict_fn'] = builder(_leaf['full_pipeline'])
    return pipelines_by_sensor
