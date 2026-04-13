"""Minimal PM4Py quality metrics for log-model evaluation.

Computes exactly these dimensions for a Petri net model against an event log:
- Fitness (token-based replay and alignments)
- Precision (token-based replay and alignments)
- Generalization
- Simplicity
"""

from __future__ import annotations

from typing import Any, Dict

from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator


def _to_float(value: Any) -> float | None:
    """Best-effort conversion to float, returning None when conversion fails."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_fitness_score(result: Any) -> float | None:
    """Extract comparable scalar from PM4Py replay fitness outputs."""
    if isinstance(result, dict):
        # Typical keys across PM4Py replay fitness variants.
        for key in ("log_fitness", "average_trace_fitness", "perc_fit_traces"):
            if key in result:
                return _to_float(result[key])
    return _to_float(result)


def evaluate_log_model_metrics(event_log, net, initial_marking, final_marking) -> Dict[str, float | None]:
    """Return PM4Py quality metrics for one log-model pair.

    Parameters
    ----------
    event_log:
        PM4Py EventLog (or compatible object).
    net:
        PM4Py Petri net.
    initial_marking:
        Initial marking for the Petri net.
    final_marking:
        Final marking for the Petri net.
    """
    metrics: Dict[str, float | None] = {
        "fitness_token_based_replay": None,
        "fitness_alignments": None,
        "precision_token_based_replay": None,
        "precision_alignments": None,
        "generalization": None,
        "simplicity": None,
    }

    # Fitness
    try:
        tbr_fit = fitness_evaluator.apply(
            event_log,
            net,
            initial_marking,
            final_marking,
            variant=fitness_evaluator.Variants.TOKEN_BASED,
        )
        metrics["fitness_token_based_replay"] = _extract_fitness_score(tbr_fit)
    except (TypeError, ValueError, KeyError, AttributeError):
        pass

    try:
        align_fit = fitness_evaluator.apply(
            event_log,
            net,
            initial_marking,
            final_marking,
            variant=fitness_evaluator.Variants.ALIGNMENT_BASED,
        )
        metrics["fitness_alignments"] = _extract_fitness_score(align_fit)
    except (TypeError, ValueError, KeyError, AttributeError):
        pass

    # Precision
    try:
        precision_tbr = precision_evaluator.apply(
            event_log,
            net,
            initial_marking,
            final_marking,
            variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN,
        )
        metrics["precision_token_based_replay"] = _to_float(precision_tbr)
    except (TypeError, ValueError, KeyError, AttributeError):
        pass

    try:
        precision_align = precision_evaluator.apply(
            event_log,
            net,
            initial_marking,
            final_marking,
            variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE,
        )
        metrics["precision_alignments"] = _to_float(precision_align)
    except (TypeError, ValueError, KeyError, AttributeError):
        pass

    # Generalization and simplicity
    try:
        metrics["generalization"] = _to_float(
            generalization_evaluator.apply(event_log, net, initial_marking, final_marking)
        )
    except (TypeError, ValueError, KeyError, AttributeError):
        pass

    try:
        metrics["simplicity"] = _to_float(simplicity_evaluator.apply(net))
    except (TypeError, ValueError, KeyError, AttributeError):
        pass

    return metrics
