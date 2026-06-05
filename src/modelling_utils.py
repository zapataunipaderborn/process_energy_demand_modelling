"""
modelling_utils.py
==================
Standalone utility functions extracted from modelling.py so they can be
imported into notebooks or other scripts without executing the full pipeline.

Quick start in a notebook::

    import sys; sys.path.insert(0, 'src')
    from modelling_utils import (
        load_process_datasets,
        _split_process_datasets,
        comprehensive_simulation_evaluation,
        plot_simulation_comparison,
        visualize_heuristic_nets,
        _plot_short_heatmap,
    )
"""
from __future__ import annotations

import os
import re
import tempfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pm4py
from collections import defaultdict
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import jensenshannon

from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

try:
    from pm4py.algo.evaluation.generalization import algorithm as pm4py_generalization
except Exception:
    pm4py_generalization = None

try:
    from pm4py.algo.evaluation.simplicity import algorithm as pm4py_simplicity
except Exception:
    pm4py_simplicity = None


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODES_TO_COMPARE = [
    'statistical',
    'petri_net_alpha',
    'petri_net_heuristic',
    'petri_net_inductive',
    'petri_net_combined',
    'petri_net_median_duration',
    'petri_net_ilp',
    'petri_net_energy_duration_aware',
    'petri_net_energy_transition_aware',
    'petri_net_energy_aware',
    'petri_net_energy_direct_duration_only',
    'petri_net_energy_direct_transition_only',
    'petri_net_energy_direct',
    'petri_net_energy_dist',
    'petri_net_direct_test',
    'petri_net_quantile_blend',
    'petri_net_blend_duration',
    'petri_net_test_2',
]

METRICS_LOWER_IS_BETTER = {
    'basic_metrics_event_count_error',
    'basic_metrics_case_count_error',
    'activity_metrics_js_divergence',
    'activity_metrics_frequency_mae',
    'duration_metrics_ks_statistic',
    'duration_metrics_mean_duration_error',
    'duration_metrics_median_duration_error',
    'duration_metrics_std_duration_error',
    'duration_metrics_activity_duration_error',
    'duration_metrics_dur_js_whole',
    'duration_metrics_dur_js_activ',
    'case_metrics_events_per_case_ks',
    'case_metrics_median_events_per_case_error',
    'overall_error',
}

METRICS_HIGHER_IS_BETTER = {
    'basic_metrics_event_count_ratio',
    'basic_metrics_case_count_ratio',
    'activity_metrics_activity_coverage_ratio',
    'duration_metrics_ks_pvalue',
    'case_metrics_events_per_case_pvalue',
    'conformance_metrics_fitness',
    'conformance_metrics_precision',
    'conformance_metrics_generalization',
    'conformance_metrics_simplicity',
    'control_flow_metrics_edge_precision',
    'control_flow_metrics_edge_recall',
    'control_flow_metrics_edge_f1_score',
    'control_flow_metrics_start_activities_jaccard',
    'control_flow_metrics_end_activities_jaccard',
}

CORE_METRIC_BASES = [
    'overall_error',
    'basic_metrics_event_count_ratio',
    'duration_metrics_mean_duration_error',
    'duration_metrics_activity_duration_error',
    'duration_metrics_dur_js_whole',
    'duration_metrics_dur_js_activ',
    'activity_metrics_js_divergence',
    'control_flow_metrics_edge_f1_score',
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_process_datasets(experiment: str = '5', base_path: str | Path | None = None) -> dict:
    """Load all process datasets for a given experiment from the gold data layer.

    Parameters
    ----------
    experiment : str
        Experiment identifier (e.g. '5').
    base_path : path-like, optional
        Root of the repository.  Defaults to the parent of ``src/``.

    Returns
    -------
    dict
        ``{process_name: {'event_log': df, 'expanded': df, 'production_plan': df}}``
    """
    if base_path is None:
        base_path = Path(__file__).resolve().parent.parent
    folder_gold = Path(base_path) / 'data' / 'gold' / f'experiment_{experiment}'

    process_datasets: dict = {}
    for process_dir in sorted(folder_gold.glob('process_*')):
        datasets_dir = process_dir / 'datasets'
        if not datasets_dir.exists():
            continue
        paths = {
            'event_log':       datasets_dir / 'df_event_log.parquet',
            'expanded':        datasets_dir / 'df_expanded.parquet',
            'production_plan': datasets_dir / 'df_production_plan.parquet',
        }
        if not all(p.exists() for p in paths.values()):
            continue
        process_datasets[process_dir.name] = {k: pd.read_parquet(v) for k, v in paths.items()}
        print(f"Loaded {process_dir.name}: "
              f"event_log={len(process_datasets[process_dir.name]['event_log'])} rows, "
              f"expanded={len(process_datasets[process_dir.name]['expanded'])} rows")
    return process_datasets


def _parse_resolution_minutes(res_str: str) -> float:
    if res_str == 'original':
        return 1.0
    m = re.match(r'^(\d+(?:\.\d+)?)(min|h)$', res_str.strip())
    if m:
        val = float(m.group(1))
        return val * 60.0 if m.group(2) == 'h' else val
    return 15.0


def _aggregate_expanded(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.copy()
    df['datetime_energy'] = pd.to_datetime(df['datetime_energy'])
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    non_numeric_cols = [c for c in df.columns if c != 'datetime_energy' and c not in numeric_cols]
    agg_dict = {c: 'mean' for c in numeric_cols}
    agg_dict.update({c: 'first' for c in non_numeric_cols})
    return df.resample(freq, on='datetime_energy').agg(agg_dict).reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Train / test split
# ─────────────────────────────────────────────────────────────────────────────

def _split_process_datasets(datasets: dict, train_ratio: float = 0.80) -> tuple[dict, dict]:
    """Temporally split all processes into train / test by case start time.

    Returns ``(train_datasets, test_datasets)``.
    """
    train_datasets: dict = {}
    test_datasets: dict = {}

    for proc_name, proc_data in datasets.items():
        event_log       = proc_data['event_log']
        production_plan = proc_data['production_plan']
        expanded        = proc_data.get('expanded')

        el = event_log.dropna(subset=['case_id']).copy()
        case_start = el.groupby('case_id')['timestamp_start'].min().sort_values()
        n_train = max(1, int(len(case_start) * train_ratio))

        train_cases = set(case_start.index[:n_train])
        test_cases  = set(case_start.index[n_train:])

        el_train = el[el['case_id'].isin(train_cases)].copy()
        el_test  = el[el['case_id'].isin(test_cases)].copy()

        pp_train = production_plan[production_plan['case_id'].isin(train_cases)].copy()
        pp_test  = production_plan[production_plan['case_id'].isin(test_cases)].copy()

        if len(pp_train) == 0 and len(pp_test) == 0 and len(production_plan) > 0:
            pp_case_start = (production_plan.groupby('case_id')['timestamp_start'].min().sort_values())
            n_pp_train = max(1, int(len(pp_case_start) * train_ratio))
            pp_train_ids = set(pp_case_start.index[:n_pp_train])
            pp_test_ids  = set(pp_case_start.index[n_pp_train:])
            pp_train = production_plan[production_plan['case_id'].isin(pp_train_ids)].copy()
            pp_test  = production_plan[production_plan['case_id'].isin(pp_test_ids)].copy()

        exp_train = exp_test = None
        if expanded is not None:
            if 'case_id_log' in expanded.columns:
                exp_train = expanded[expanded['case_id_log'].isin(train_cases)].copy()
                exp_test  = expanded[expanded['case_id_log'].isin(test_cases)].copy()
            else:
                cutoff = case_start.iloc[n_train - 1]
                if 'datetime_energy' in expanded.columns:
                    exp_train = expanded[expanded['datetime_energy'] <= cutoff].copy()
                    exp_test  = expanded[expanded['datetime_energy'] > cutoff].copy()
                else:
                    exp_train = expanded.copy()
                    exp_test  = expanded.iloc[0:0].copy()

        train_datasets[proc_name] = {'event_log': el_train, 'production_plan': pp_train, 'expanded': exp_train}
        test_datasets[proc_name]  = {'event_log': el_test,  'production_plan': pp_test,  'expanded': exp_test}

        print(f"  {proc_name}: {len(train_cases)} train / {len(test_cases)} test cases  |  "
              f"event_log {len(el_train)}/{len(el_test)}  |  "
              f"production_plan {len(pp_train)}/{len(pp_test)}"
              + (f"  |  expanded {len(exp_train)}/{len(exp_test)}" if exp_train is not None else ""))

    return train_datasets, test_datasets


# ─────────────────────────────────────────────────────────────────────────────
# Conformance helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mean_trace_fitness(log_df, net, im, fm) -> float:
    try:
        replay = token_replay.apply(log_df, net, im, fm,
                                    parameters={'consider_remaining_in_fitness': True})
        scores = []
        for item in replay:
            if 'trace_fitness' in item and item['trace_fitness'] is not None:
                scores.append(float(item['trace_fitness']))
            elif item.get('trace_is_fit') is True:
                scores.append(1.0)
            else:
                scores.append(0.0)
        return float(np.mean(scores)) if scores else np.nan
    except Exception:
        return np.nan


def _safe_precision(log_df, net, im, fm) -> float:
    try:
        return float(pm4py.precision_token_based_replay(log_df, net, im, fm))
    except Exception:
        return np.nan


def _safe_generalization(log_df, net, im, fm) -> float:
    if pm4py_generalization is None:
        return np.nan
    try:
        return float(pm4py_generalization.apply(log_df, net, im, fm))
    except Exception:
        return np.nan


def _safe_simplicity(net) -> float:
    if pm4py_simplicity is not None:
        try:
            return float(pm4py_simplicity.apply(net))
        except Exception:
            pass
    complexity = len(net.places) + len(net.transitions) + len(net.arcs)
    return float(1.0 / (1.0 + 0.005 * float(complexity)))


# ─────────────────────────────────────────────────────────────────────────────
# Per-case metrics (median across cases to avoid large-case dominance)
# ─────────────────────────────────────────────────────────────────────────────

def _dur_js(a, b, n_bins: int = 20) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    b = b[np.isfinite(b) & (b > 0)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    lo = max(np.percentile(np.concatenate([a, b]), 1), 1e-3)
    hi = np.percentile(np.concatenate([a, b]), 99)
    if lo >= hi:
        return 0.0
    bins = np.exp(np.linspace(np.log(lo), np.log(hi), n_bins + 1))
    p, _ = np.histogram(a, bins=bins)
    q, _ = np.histogram(b, bins=bins)
    p = p.astype(float) + 1e-9
    q = q.astype(float) + 1e-9
    p /= p.sum(); q /= q.sum()
    return float(jensenshannon(p, q))


def _per_case_median_metrics(simulated_df, real_df,
                             case_col='case_id', activity_col='activity',
                             start_col='timestamp_start', end_col='timestamp_end') -> dict:
    common = sorted(set(simulated_df[case_col].dropna().unique()) &
                    set(real_df[case_col].dropna().unique()))
    if not common:
        return {}

    rows = []
    for cid in common:
        rc = real_df[real_df[case_col] == cid]
        sc = simulated_df[simulated_df[case_col] == cid]
        if rc.empty:
            continue

        evt_ratio_err = abs(len(sc) / len(rc) - 1.0)

        real_durs = (rc[end_col] - rc[start_col]).dt.total_seconds() / 60.0
        sim_durs  = (sc[end_col] - sc[start_col]).dt.total_seconds() / 60.0
        real_mu = real_durs.mean()
        dur_err_whole = abs(sim_durs.mean() - real_mu) / real_mu if real_mu > 0 else np.nan

        r_act = (rc.assign(_d=(rc[end_col]-rc[start_col]).dt.total_seconds()/60.0)
                   .groupby(activity_col)['_d'].mean())
        s_act = (sc.assign(_d=(sc[end_col]-sc[start_col]).dt.total_seconds()/60.0)
                   .groupby(activity_col)['_d'].mean())
        common_acts = r_act.index.intersection(s_act.index)
        act_errs = [abs(s_act[a] - r_act[a]) / r_act[a]
                    for a in common_acts if r_act[a] > 0]
        dur_err_activ = float(np.mean(act_errs)) if act_errs else np.nan

        all_acts = sorted(set(rc[activity_col].dropna()) | set(sc[activity_col].dropna()))
        r_cnt = rc[activity_col].value_counts()
        s_cnt = sc[activity_col].value_counts()
        p = np.array([r_cnt.get(a, 0) for a in all_acts], dtype=float) + 1e-9
        q = np.array([s_cnt.get(a, 0) for a in all_acts], dtype=float) + 1e-9
        p /= p.sum(); q /= q.sum()
        m = 0.5 * (p + q)
        js_div = float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))

        def _case_edges(df):
            grp = df.sort_values(start_col)
            acts = grp[activity_col].tolist()
            return set(zip(acts, acts[1:]))

        r_edges = _case_edges(rc)
        s_edges = _case_edges(sc)
        if r_edges or s_edges:
            tp   = len(r_edges & s_edges)
            prec = tp / len(s_edges) if s_edges else 0.0
            rec  = tp / len(r_edges) if r_edges else 0.0
            ef1  = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        else:
            ef1 = np.nan

        rows.append({'evt_ratio_err': evt_ratio_err,
                     'dur_err_whole': dur_err_whole,
                     'dur_err_activ': dur_err_activ,
                     'js_div':        js_div,
                     'edge_f1':       ef1})

    if not rows:
        return {}
    df = pd.DataFrame(rows)
    return {
        'evt_ratio_err': float(df['evt_ratio_err'].median()),
        'dur_err_whole': float(df['dur_err_whole'].median()),
        'dur_err_activ': float(df['dur_err_activ'].median()),
        'js_div':        float(df['js_div'].median()),
        'edge_f1':       float(df['edge_f1'].median()),
        'n_cases':       len(rows),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive evaluation
# ─────────────────────────────────────────────────────────────────────────────

def comprehensive_simulation_evaluation(
        simulated_df, real_df, real_expanded_df=None,
        case_col='case_id', activity_col='activity',
        start_col='timestamp_start', end_col='timestamp_end',
        process_models=None, station_col='higher_level_activity',
        verbose: bool = True) -> dict:
    """Evaluate simulation quality.  Returns a nested results dict."""

    def _report(*args):
        if verbose:
            print(*args, flush=True)

    results = {}

    simulated_df = simulated_df.copy()
    real_df = real_df.copy()

    for col in [case_col, activity_col, start_col, end_col]:
        if col not in simulated_df.columns:
            simulated_df[col] = pd.Series(dtype='object')
        if col not in real_df.columns:
            real_df[col] = pd.Series(dtype='object')

    simulated_df[start_col] = pd.to_datetime(simulated_df[start_col], errors='coerce')
    simulated_df[end_col]   = pd.to_datetime(simulated_df[end_col],   errors='coerce')
    real_df[start_col]      = pd.to_datetime(real_df[start_col],      errors='coerce')
    real_df[end_col]        = pd.to_datetime(real_df[end_col],        errors='coerce')

    _report("=" * 80)
    _report("COMPREHENSIVE SIMULATION EVALUATION")
    _report("=" * 80)

    # ── 1. Basic metrics ──────────────────────────────────────────────────────
    _report("\n1. BASIC PROCESS METRICS")
    sim_events  = len(simulated_df)
    real_events = len(real_df)
    event_ratio = sim_events / real_events if real_events > 0 else 0
    sim_cases   = simulated_df[case_col].nunique() if case_col in simulated_df.columns else 0
    real_cases  = real_df[case_col].nunique()       if case_col in real_df.columns      else 0
    case_ratio  = sim_cases / real_cases if real_cases > 0 else 0
    _report(f"Events — Real: {real_events}, Sim: {sim_events}, Ratio: {event_ratio:.3f}")
    _report(f"Cases  — Real: {real_cases},  Sim: {sim_cases},  Ratio: {case_ratio:.3f}")
    results['basic_metrics'] = {
        'event_count_ratio': event_ratio,
        'case_count_ratio':  case_ratio,
        'event_count_error': abs(1 - event_ratio),
        'case_count_error':  abs(1 - case_ratio),
    }

    # ── 2. Activity frequency ─────────────────────────────────────────────────
    _report("\n2. ACTIVITY FREQUENCY ANALYSIS")
    sim_activity_freq  = simulated_df[activity_col].dropna().value_counts(normalize=True).sort_index()
    real_activity_freq = real_df[activity_col].dropna().value_counts(normalize=True).sort_index()
    all_activities = sorted(set(sim_activity_freq.index) | set(real_activity_freq.index))
    if all_activities:
        sim_fa  = pd.Series([sim_activity_freq.get(a, 0)  for a in all_activities], index=all_activities)
        real_fa = pd.Series([real_activity_freq.get(a, 0) for a in all_activities], index=all_activities)
        js_divergence = jensenshannon(sim_fa.values, real_fa.values)
        freq_mae      = np.mean(np.abs(sim_fa.values - real_fa.values))
    else:
        js_divergence = freq_mae = 1.0
    _report(f"JS Divergence (activities): {js_divergence:.4f}")
    _report(f"Freq MAE: {freq_mae:.4f}")
    results['activity_metrics'] = {
        'js_divergence':            js_divergence,
        'frequency_mae':            freq_mae,
        'activity_coverage_ratio':  (len(sim_activity_freq) / len(real_activity_freq)
                                     if len(real_activity_freq) > 0 else 0),
    }

    # ── 3. Duration analysis ──────────────────────────────────────────────────
    _report("\n3. DURATION ANALYSIS")
    sim_durs  = (simulated_df[end_col] - simulated_df[start_col]).dt.total_seconds() / 60
    real_durs = (real_df[end_col] - real_df[start_col]).dt.total_seconds() / 60
    sim_durs  = sim_durs.replace([np.inf, -np.inf], np.nan).dropna()
    real_durs = real_durs.replace([np.inf, -np.inf], np.nan).dropna()

    if len(sim_durs) > 0 and len(real_durs) > 0:
        ks_stat, ks_pval = stats.ks_2samp(sim_durs, real_durs)
        dur_stats = pd.DataFrame({
            'Real':      [real_durs.mean(), real_durs.median(), real_durs.std()],
            'Simulated': [sim_durs.mean(),  sim_durs.median(),  sim_durs.std()],
        }, index=['Mean', 'Median', 'Std'])
        dur_stats['Error'] = np.where(
            dur_stats['Real'] != 0,
            np.abs(dur_stats['Simulated'] - dur_stats['Real']) / np.abs(dur_stats['Real']),
            1.0)
    else:
        ks_stat, ks_pval = 1.0, 0.0
        dur_stats = pd.DataFrame({'Real': [np.nan]*3, 'Simulated': [np.nan]*3, 'Error': [1.0]*3},
                                 index=['Mean', 'Median', 'Std'])

    _real_act_dur = (real_df.assign(_d=(real_df[end_col]-real_df[start_col]).dt.total_seconds())
                    .groupby(activity_col)['_d'].mean())
    _sim_act_dur  = (simulated_df.assign(_d=(simulated_df[end_col]-simulated_df[start_col]).dt.total_seconds())
                    .groupby(activity_col)['_d'].mean())
    _common_acts  = _real_act_dur.index.intersection(_sim_act_dur.index)
    _act_dur_errs = [abs(_sim_act_dur[a] - _real_act_dur[a]) / _real_act_dur[a]
                     for a in _common_acts if _real_act_dur[a] != 0]
    activity_duration_error = float(np.mean(_act_dur_errs)) if _act_dur_errs else np.nan

    dur_js_whole = _dur_js(sim_durs.values, real_durs.values)
    _act_dur_js = []
    for _act in _common_acts:
        _r = (real_df[real_df[activity_col] == _act]
              .pipe(lambda d: (d[end_col] - d[start_col]).dt.total_seconds() / 60.0)
              .replace([np.inf, -np.inf], np.nan).dropna())
        _s = (simulated_df[simulated_df[activity_col] == _act]
              .pipe(lambda d: (d[end_col] - d[start_col]).dt.total_seconds() / 60.0)
              .replace([np.inf, -np.inf], np.nan).dropna())
        _js = _dur_js(_r.values, _s.values)
        if pd.notna(_js):
            _act_dur_js.append(_js)
    dur_js_activ = float(np.mean(_act_dur_js)) if _act_dur_js else np.nan

    _report(dur_stats.round(3))
    _report(f"KS={ks_stat:.4f}  p={ks_pval:.4f}  |  act-dur-err={activity_duration_error:.4f}  |  "
            f"JS-whole={dur_js_whole:.4f}  JS-activ={dur_js_activ:.4f}")
    results['duration_metrics'] = {
        'ks_statistic':             ks_stat,
        'ks_pvalue':                ks_pval,
        'mean_duration_error':      dur_stats.loc['Mean', 'Error'],
        'median_duration_error':    dur_stats.loc['Median', 'Error'],
        'std_duration_error':       dur_stats.loc['Std', 'Error'],
        'activity_duration_error':  activity_duration_error,
        'dur_js_whole':             dur_js_whole,
        'dur_js_activ':             dur_js_activ,
    }

    # ── 4. Case-level ─────────────────────────────────────────────────────────
    _report("\n4. CASE-LEVEL ANALYSIS")
    sim_epc  = simulated_df.groupby(case_col).size() if len(simulated_df) > 0 else pd.Series(dtype=float)
    real_epc = real_df.groupby(case_col).size()      if len(real_df) > 0       else pd.Series(dtype=float)
    if len(sim_epc) > 0 and len(real_epc) > 0:
        epc_ks, epc_pval = stats.ks_2samp(sim_epc, real_epc)
        cs = pd.DataFrame({
            'Real':      [real_epc.mean(), real_epc.median(), real_epc.std()],
            'Simulated': [sim_epc.mean(),  sim_epc.median(),  sim_epc.std()],
        }, index=['Mean', 'Median', 'Std'])
        cs['Error'] = np.where(cs['Real'] != 0,
                               np.abs(cs['Simulated'] - cs['Real']) / np.abs(cs['Real']), 1.0)
    else:
        epc_ks, epc_pval = 1.0, 0.0
        cs = pd.DataFrame({'Real': [np.nan]*3, 'Simulated': [np.nan]*3, 'Error': [1.0]*3},
                          index=['Mean', 'Median', 'Std'])
    _report(cs.round(3))
    results['case_metrics'] = {
        'events_per_case_ks':              epc_ks,
        'events_per_case_pvalue':          epc_pval,
        'mean_events_per_case_error':      cs.loc['Mean', 'Error'],
        'median_events_per_case_error':    cs.loc['Median', 'Error'],
    }

    # ── 5. Control-flow (DFG) ─────────────────────────────────────────────────
    _report("\n5. CONTROL-FLOW ANALYSIS (DFG)")
    sim_for_dfg  = simulated_df.dropna(subset=[case_col, activity_col, start_col])
    real_for_dfg = real_df.dropna(subset=[case_col, activity_col, start_col])
    if len(sim_for_dfg) > 0 and len(real_for_dfg) > 0:
        sim_log  = pm4py.format_dataframe(sim_for_dfg,  case_id=case_col, activity_key=activity_col, timestamp_key=start_col)
        real_log = pm4py.format_dataframe(real_for_dfg, case_id=case_col, activity_key=activity_col, timestamp_key=start_col)
        sim_dfg,  sim_start,  sim_end  = pm4py.discover_dfg(sim_log)
        real_dfg, real_start, real_end = pm4py.discover_dfg(real_log)
        sim_edges  = set(sim_dfg.keys())
        real_edges = set(real_dfg.keys())
    else:
        sim_edges = real_edges = set()
        sim_start = real_start = sim_end = real_end = {}

    ep = len(sim_edges & real_edges) / len(sim_edges) if sim_edges else 0
    er = len(sim_edges & real_edges) / len(real_edges) if real_edges else 0
    ef = 2*(ep*er)/(ep+er) if (ep+er) > 0 else 0
    start_j = (len(set(sim_start) & set(real_start)) / len(set(sim_start) | set(real_start))
               if (set(sim_start) | set(real_start)) else 0)
    end_j   = (len(set(sim_end) & set(real_end)) / len(set(sim_end) | set(real_end))
               if (set(sim_end) | set(real_end)) else 0)
    _report(f"Edges — Real: {len(real_edges)}, Sim: {len(sim_edges)}, Common: {len(sim_edges & real_edges)}")
    _report(f"EdgeF1={ef:.4f}  StartJaccard={start_j:.4f}  EndJaccard={end_j:.4f}")
    results['control_flow_metrics'] = {
        'edge_precision': ep, 'edge_recall': er, 'edge_f1_score': ef,
        'start_activities_jaccard': start_j, 'end_activities_jaccard': end_j,
    }

    # ── 6. Classic PM conformance ─────────────────────────────────────────────
    _report("\n6. CLASSIC PM CONFORMANCE")
    conformance_metrics = {'fitness': np.nan, 'precision': np.nan,
                           'generalization': np.nan, 'simplicity': np.nan}
    if process_models is not None:
        fit_list, prec_list, gen_list, sim_list = [], [], [], []
        seen: set = set()
        for key, pm_entry in process_models.items():
            _obj, obj_type, station = key
            if (obj_type, station) in seen:
                continue
            seen.add((obj_type, station))
            net_s, im_s, fm_s = pm_entry['net'], pm_entry['im'], pm_entry['fm']
            df_st = (real_df[real_df[station_col] == station].dropna(subset=[case_col, activity_col, start_col])
                     if station_col in real_df.columns
                     else real_df.dropna(subset=[case_col, activity_col, start_col]))
            if len(df_st) == 0:
                continue
            try:
                log_st = pm4py.format_dataframe(df_st, case_id=case_col,
                                                activity_key=activity_col, timestamp_key=start_col)
                f = _mean_trace_fitness(log_st, net_s, im_s, fm_s)
                p = _safe_precision(log_st, net_s, im_s, fm_s)
                g = _safe_generalization(log_st, net_s, im_s, fm_s)
                s = _safe_simplicity(net_s)
                if not np.isnan(f):  fit_list.append(f)
                if not np.isnan(p):  prec_list.append(p)
                if not np.isnan(g):  gen_list.append(g)
                if not np.isnan(s):  sim_list.append(s)
            except Exception as exc:
                _report(f"  Conformance skipped for '{station}': {exc}")
        if fit_list:  conformance_metrics['fitness']        = float(np.mean(fit_list))
        if prec_list: conformance_metrics['precision']      = float(np.mean(prec_list))
        if gen_list:  conformance_metrics['generalization'] = float(np.mean(gen_list))
        if sim_list:  conformance_metrics['simplicity']     = float(np.mean(sim_list))
    results['conformance_metrics'] = conformance_metrics

    # ── 8. Overall quality score ──────────────────────────────────────────────
    _report("\n7. OVERALL QUALITY SCORE")
    _pc = _per_case_median_metrics(simulated_df, real_df,
                                   case_col=case_col, activity_col=activity_col,
                                   start_col=start_col, end_col=end_col)
    _evt_ratio_global = results['basic_metrics']['event_count_ratio']
    _js_div_global    = results['activity_metrics'].get('js_divergence', np.nan)
    _edge_f1_global   = results['control_flow_metrics'].get('edge_f1_score', np.nan)
    _dur_whole_global = results['duration_metrics']['mean_duration_error']
    _dur_activ_global = results['duration_metrics'].get('activity_duration_error', np.nan)

    short_components = {
        'evt_ratio_err':   _pc.get('evt_ratio_err',
                                   abs(_evt_ratio_global - 1.0) if pd.notna(_evt_ratio_global) else np.nan),
        'dur_err_whole':   _pc.get('dur_err_whole', _dur_whole_global),
        'dur_err_activ':   _pc.get('dur_err_activ', _dur_activ_global),
        'js_div':          _pc.get('js_div', _js_div_global),
        'edge_err':        ((1.0 - _pc['edge_f1']) if 'edge_f1' in _pc
                            else ((1.0 - _edge_f1_global) if pd.notna(_edge_f1_global) else np.nan)),
        'dur_js_whole':    results['duration_metrics'].get('dur_js_whole', np.nan),
        'dur_js_activ':    results['duration_metrics'].get('dur_js_activ', np.nan),
    }
    err_values = [float(v) for v in short_components.values() if pd.notna(v)]
    overall_error = float(np.mean(err_values)) if err_values else np.nan
    quality = ("EXCELLENT" if pd.notna(overall_error) and overall_error <= 0.05 else
               "GOOD"      if pd.notna(overall_error) and overall_error <= 0.15 else
               "FAIR"      if pd.notna(overall_error) and overall_error <= 0.30 else "POOR")
    _report(f"Overall error: {overall_error:.4f}  ({quality})")
    if _pc:
        results['basic_metrics']['event_count_ratio']             = 1.0 + _pc['evt_ratio_err']
        results['basic_metrics']['event_count_error']             = _pc['evt_ratio_err']
        results['duration_metrics']['mean_duration_error']        = _pc['dur_err_whole']
        if pd.notna(_pc.get('dur_err_activ', np.nan)):
            results['duration_metrics']['activity_duration_error'] = _pc['dur_err_activ']
        if pd.notna(_pc.get('js_div', np.nan)):
            results['activity_metrics']['js_divergence']           = _pc['js_div']
        if 'edge_f1' in _pc:
            results['control_flow_metrics']['edge_f1_score']       = _pc['edge_f1']
    results['overall_error']      = overall_error
    results['quality_assessment'] = quality

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_simulation_comparison(simulated_df, real_df,
                               case_col='case_id', activity_col='activity',
                               start_col='timestamp_start', end_col='timestamp_end'):
    """Four-panel comparison: activity freq, duration dist, events/case, case duration."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Simulation vs Real Data Comparison', fontsize=16, fontweight='bold')

    sim_freq  = simulated_df[activity_col].value_counts()
    real_freq = real_df[activity_col].value_counts()
    all_acts  = list(set(sim_freq.index) | set(real_freq.index))
    x = np.arange(len(all_acts))
    axes[0, 0].bar(x - 0.2, [real_freq.get(a, 0) for a in all_acts], 0.4,
                   label='Real', alpha=0.7, color='skyblue')
    axes[0, 0].bar(x + 0.2, [sim_freq.get(a, 0)  for a in all_acts], 0.4,
                   label='Simulated', alpha=0.7, color='orange')
    axes[0, 0].set_title('Activity Frequencies'); axes[0, 0].legend()
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(all_acts, rotation=45, ha='right')

    sim_durs  = (simulated_df[end_col] - simulated_df[start_col]).dt.total_seconds() / 60
    real_durs = (real_df[end_col] - real_df[start_col]).dt.total_seconds() / 60
    axes[0, 1].hist(real_durs, alpha=0.7, label='Real', bins=30, density=True, color='skyblue')
    axes[0, 1].hist(sim_durs,  alpha=0.7, label='Simulated', bins=30, density=True, color='orange')
    axes[0, 1].set_title('Activity Duration Distributions'); axes[0, 1].legend()

    sim_epc  = simulated_df.groupby(case_col).size()
    real_epc = real_df.groupby(case_col).size()
    axes[1, 0].hist(real_epc, alpha=0.7, label='Real', bins=20, density=True, color='skyblue')
    axes[1, 0].hist(sim_epc,  alpha=0.7, label='Simulated', bins=20, density=True, color='orange')
    axes[1, 0].set_title('Events per Case'); axes[1, 0].legend()

    sim_case_dur  = simulated_df.groupby(case_col).apply(
        lambda x: (x[end_col].max() - x[start_col].min()).total_seconds() / 3600)
    real_case_dur = real_df.groupby(case_col).apply(
        lambda x: (x[end_col].max() - x[start_col].min()).total_seconds() / 3600)
    axes[1, 1].hist(real_case_dur, alpha=0.7, label='Real', bins=20, density=True, color='skyblue')
    axes[1, 1].hist(sim_case_dur,  alpha=0.7, label='Simulated', bins=20, density=True, color='orange')
    axes[1, 1].set_title('Case Duration Distribution (hours)'); axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def visualize_heuristic_nets(df_compare, simulated_log):
    """Plot real vs simulated heuristic nets side by side."""
    required_cols = {'case_id', 'activity', 'timestamp_start'}
    if simulated_log is None or simulated_log.empty:
        print("Skipping: simulated log is empty.")
        return
    if not required_cols.issubset(set(simulated_log.columns)):
        print("Skipping: simulated log is missing required columns.")
        return

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf1, \
         tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf2:
        p1, p2 = tf1.name, tf2.name

    try:
        el_real = pm4py.format_dataframe(df_compare,    case_id='case_id', activity_key='activity', timestamp_key='timestamp_start')
        el_sim  = pm4py.format_dataframe(simulated_log, case_id='case_id', activity_key='activity', timestamp_key='timestamp_start')
        pm4py.save_vis_heuristics_net(pm4py.discover_heuristics_net(el_real), p1, bgcolor='white', dpi=300)
        pm4py.save_vis_heuristics_net(pm4py.discover_heuristics_net(el_sim),  p2, bgcolor='white', dpi=300)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(mpimg.imread(p1)); ax1.set_title('Real', fontsize=16, fontweight='bold'); ax1.axis('off')
        ax2.imshow(mpimg.imread(p2)); ax2.set_title('Simulated', fontsize=16, fontweight='bold'); ax2.axis('off')
        plt.tight_layout(); plt.show()
    finally:
        for p in [p1, p2]:
            if os.path.exists(p):
                os.unlink(p)


def _normalise_metrics(df, cols, lower_set=None):
    """Min-max normalise; inverts lower-is-better metrics so 1 = best."""
    norm_df = df.copy()
    base_lower = set(lower_set or [])
    base_lower.update(METRICS_LOWER_IS_BETTER)
    for c in cols:
        vals = df[c].dropna()
        if len(vals) == 0:
            continue
        vmin, vmax = vals.min(), vals.max()
        rng = vmax - vmin if vmax != vmin else 1.0
        clean_c = c.replace('test_', '').replace('train_', '')
        is_lower = clean_c in base_lower or any(m in clean_c for m in ['MAE', 'RMSE', 'WAPE'])
        if 'R2' in clean_c:
            is_lower = False
        norm_df[c] = (vmax - df[c]) / rng if is_lower else (df[c] - vmin) / rng
    return norm_df


def _plot_short_heatmap(target_df, title, save_path=None, agg='mean', split='test'):
    """7-metric error heatmap (0 = best) + Overall column."""
    if target_df is None or target_df.empty or 'mode' not in target_df.columns:
        return

    _col_evt    = f'{split}_basic_metrics_event_count_ratio'
    _col_dur_w  = f'{split}_duration_metrics_mean_duration_error'
    _col_dur_a  = f'{split}_duration_metrics_activity_duration_error'
    _col_dur_jw = f'{split}_duration_metrics_dur_js_whole'
    _col_dur_ja = f'{split}_duration_metrics_dur_js_activ'
    _col_js     = f'{split}_activity_metrics_js_divergence'
    _col_f1     = f'{split}_control_flow_metrics_edge_f1_score'
    _col_ov     = f'{split}_overall_error'

    all_metric_cols = [_col_evt, _col_dur_w, _col_dur_a, _col_dur_jw,
                       _col_dur_ja, _col_js, _col_f1, _col_ov]
    agg_cols = [c for c in all_metric_cols if c in target_df.columns]
    if not agg_cols:
        return

    mode_avg = (target_df.groupby('mode')[agg_cols].median()
                if agg == 'median' else target_df.groupby('mode')[agg_cols].mean())
    hm = pd.DataFrame(index=mode_avg.index)
    if _col_evt  in mode_avg.columns: hm['EvtRatioErr']  = (mode_avg[_col_evt] - 1.0).abs()
    if _col_dur_w  in mode_avg.columns: hm['DurErr(whole)'] = mode_avg[_col_dur_w]
    if _col_dur_a  in mode_avg.columns: hm['DurErr(activ)'] = mode_avg[_col_dur_a]
    if _col_dur_jw in mode_avg.columns: hm['DurJS(whole)']  = mode_avg[_col_dur_jw]
    if _col_dur_ja in mode_avg.columns: hm['DurJS(activ)']  = mode_avg[_col_dur_ja]
    if _col_js   in mode_avg.columns: hm['JS div']        = mode_avg[_col_js]
    if _col_f1   in mode_avg.columns: hm['1-EdgeF1']      = 1.0 - mode_avg[_col_f1]
    if _col_ov   in mode_avg.columns: hm['Overall']       = mode_avg[_col_ov]
    elif hm.shape[1] > 0:
        hm['Overall'] = hm.mean(axis=1)
    hm = hm.sort_values('Overall', ascending=True) if 'Overall' in hm.columns else hm

    n_rows = max(2, len(hm))
    fig, ax = plt.subplots(figsize=(max(10, len(hm.columns) * 1.5), n_rows * 0.9 + 1.8))
    sns.heatmap(hm.round(3), annot=True, fmt='.3f', cmap='RdYlGn_r',
                vmin=0, vmax=1, linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Error (0=best)', 'shrink': 0.7}, ax=ax)
    ax.set_title(f'{title}\n0 = best', fontsize=11, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n{title}\n" + "-" * 60)
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(hm.round(4).to_string())
