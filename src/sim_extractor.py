from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import warnings

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
# Process extractor
# ---------------------------------------------------------------------------

def extract_process(df):
    """
    Extract process statistics with PROBABILISTIC END transitions.
    End is treated as a transition probability, not a hard stop.

    Returns
    -------
    stats_df : pd.DataFrame
        One row per (activity, object, object_type, higher_level_activity).
        Columns include *dist_name* and *dist_params* in addition to the
        original duration / transition columns.
    raw_df : pd.DataFrame
        One row per activity *instance* with the raw duration, the resolved
        next activity (``'__END__'`` when none), and any flattened
        ``object_attributes`` keys (prefixed with ``attr_``).
    """

    stats = []
    raw_rows = []   # ← new: per-instance data for ML training

    # Group by the combination that defines a unique process configuration
    grouped = df.groupby(['object', 'object_type', 'higher_level_activity'])
    
    for (object_name, object_type, higher_level_activity), group in grouped:
        print(f"\nProcessing: {object_name} ({object_type}) - {higher_level_activity}")

        # Get all activities for this object configuration
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

        # For each activity, calculate statistics
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
                    row_here = case_acts.iloc[idx]

                    # resolved next activity for this instance
                    if idx + 1 < len(case_acts):
                        next_act = case_acts.iloc[idx + 1]['activity']
                        transition_counts[next_act] += 1
                    else:
                        next_act = '__END__'
                        end_counts += 1

                    # ── raw row for ML training ───────────────────────────
                    inst_duration = (
                        (row_here['timestamp_end'] - row_here['timestamp_start'])
                        .total_seconds() / 60
                    )
                    attr_raw = row_here.get('object_attributes', {}) or {}
                    attr_flat = {f'attr_{k}': v for k, v in attr_raw.items()}

                    raw_rows.append({
                        'case_id':               case_id,
                        'activity':              activity,
                        'object':                object_name,
                        'object_type':           object_type,
                        'higher_level_activity': higher_level_activity,
                        'duration':              inst_duration,
                        'next_activity':         next_act,
                        'timestamp_start':       row_here['timestamp_start'],
                        **attr_flat,
                    })

            # ── Transition probabilities ──────────────────────────────────
            transitions = {}
            if total_occurrences > 0:
                for next_act, count in transition_counts.items():
                    transitions[next_act] = count / total_occurrences
                if end_counts > 0:
                    transitions['__END__'] = end_counts / total_occurrences

            is_end = end_counts > 0

            print(f"      Events: {n_events}, Start: {is_start}, Can End: {is_end}")
            print(f"      Total occurrences: {total_occurrences}")
            print(f"      Transitions: {transitions}")

            stats.append({
                'activity':              activity,
                'object':                object_name,
                'object_type':           object_type,
                'higher_level_activity': higher_level_activity,
                'duration':              duration_median,
                'duration_std':          duration_std,
                'dist_name':             dist_name,
                'dist_params':           dist_params,
                'n_events':              n_events,
                'transition':            transitions,
                'is_start':              is_start,
                'is_end':                is_end,
            })

    stats_df = pd.DataFrame(stats)
    raw_df   = pd.DataFrame(raw_rows)

    return stats_df, raw_df
