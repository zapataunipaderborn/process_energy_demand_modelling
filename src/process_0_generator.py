import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def _get_shift_label(ts):
    hour = ts.hour
    if 6 <= hour < 14:
        return "1"
    if 14 <= hour < 22:
        return "2"
    return "3"


def _sample_minutes(mean_minutes, std_minutes, min_minutes=1.0):
    return float(max(min_minutes, np.random.normal(mean_minutes, std_minutes)))


def generate_process_0_dataset(
    num_cases=60,
    num_machines=3,
    rework_probability=0.20,
    failure_probability=0.10,
    random_seed=42,
    start_datetime=None,
):
    """Generate a synthetic process_0 dataset with both material and machine events.

    Returns a dict with keys: expanded, event_log, production_plan.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    if start_datetime is None:
        start_datetime = datetime(2025, 1, 6, 6, 0, 0)

    machines = [f"machine_{i+1}" for i in range(num_machines)]
    events = []

    for idx in range(num_cases):
        case_id = f"order_{idx+1:04d}"
        material_id = f"material_{idx+1:04d}"
        case_start = start_datetime + timedelta(minutes=15 * idx)

        # Material route with optional rework in v1.
        route = list(machines)
        if random.random() < rework_probability and len(route) >= 2:
            rework_machine = random.choice(route[1:])
            route.append(rework_machine)

        current_time = case_start

        for step_idx, machine in enumerate(route):
            shift = _get_shift_label(current_time)
            machine_hla = f"machine_state_shift_{shift}"
            material_hla = f"material_flow_shift_{shift}"

            setup_minutes = _sample_minutes(4.0, 1.0, min_minutes=0.5)
            run_minutes = _sample_minutes(18.0, 5.0, min_minutes=2.0)
            cleanup_minutes = _sample_minutes(3.0, 0.8, min_minutes=0.5)

            # Machine setup
            setup_start = current_time
            setup_end = setup_start + timedelta(minutes=setup_minutes)
            events.append({
                "case_id": case_id,
                "activity": "setup",
                "timestamp_start": setup_start,
                "timestamp_end": setup_end,
                "object": machine,
                "object_type": "production_machine",
                "higher_level_activity": machine_hla,
                "object_attributes": {
                    "material_id": material_id,
                    "machine_id": machine,
                    "step_index": step_idx,
                },
                "perspective": "machine_state",
            })

            # Material processing + machine running
            run_start = setup_end
            run_end = run_start + timedelta(minutes=run_minutes)

            events.append({
                "case_id": case_id,
                "activity": f"material_process_{machine}",
                "timestamp_start": run_start,
                "timestamp_end": run_end,
                "object": material_id,
                "object_type": "material_unit",
                "higher_level_activity": material_hla,
                "object_attributes": {
                    "material_id": material_id,
                    "machine_id": machine,
                    "step_index": step_idx,
                },
                "perspective": "material_flow",
            })

            events.append({
                "case_id": case_id,
                "activity": "running",
                "timestamp_start": run_start,
                "timestamp_end": run_end,
                "object": machine,
                "object_type": "production_machine",
                "higher_level_activity": machine_hla,
                "object_attributes": {
                    "material_id": material_id,
                    "machine_id": machine,
                    "step_index": step_idx,
                },
                "perspective": "machine_state",
            })

            current_time = run_end

            # Stochastic failure/rework in v1.
            if random.random() < failure_probability:
                fail_minutes = _sample_minutes(6.0, 2.0, min_minutes=1.0)
                fail_start = current_time
                fail_end = fail_start + timedelta(minutes=fail_minutes)

                events.append({
                    "case_id": case_id,
                    "activity": "fault",
                    "timestamp_start": fail_start,
                    "timestamp_end": fail_end,
                    "object": machine,
                    "object_type": "production_machine",
                    "higher_level_activity": machine_hla,
                    "object_attributes": {
                        "material_id": material_id,
                        "machine_id": machine,
                        "step_index": step_idx,
                    },
                    "perspective": "machine_state",
                })
                current_time = fail_end

                # Material rework event aligned to failure interval.
                events.append({
                    "case_id": case_id,
                    "activity": f"material_rework_{machine}",
                    "timestamp_start": fail_start,
                    "timestamp_end": fail_end,
                    "object": material_id,
                    "object_type": "material_unit",
                    "higher_level_activity": material_hla,
                    "object_attributes": {
                        "material_id": material_id,
                        "machine_id": machine,
                        "step_index": step_idx,
                    },
                    "perspective": "material_flow",
                })

            # Cleanup state
            clean_start = current_time
            clean_end = clean_start + timedelta(minutes=cleanup_minutes)
            events.append({
                "case_id": case_id,
                "activity": "cleanup",
                "timestamp_start": clean_start,
                "timestamp_end": clean_end,
                "object": machine,
                "object_type": "production_machine",
                "higher_level_activity": machine_hla,
                "object_attributes": {
                    "material_id": material_id,
                    "machine_id": machine,
                    "step_index": step_idx,
                },
                "perspective": "machine_state",
            })
            current_time = clean_end + timedelta(seconds=1)

    event_log = pd.DataFrame(events).sort_values("timestamp_start").reset_index(drop=True)

    # Production plan from case boundaries.
    bounds = (
        event_log.groupby("case_id", as_index=False)
        .agg(timestamp_start=("timestamp_start", "min"), timestamp_end=("timestamp_end", "max"))
    )
    plan_rows = []
    for _, row in bounds.iterrows():
        case_events = event_log[event_log["case_id"] == row["case_id"]]
        attrs = case_events.iloc[0]["object_attributes"]
        plan_rows.append({
            "case_id": row["case_id"],
            "activity": "production",
            "timestamp_start": row["timestamp_start"],
            "timestamp_end": row["timestamp_end"],
            "object_attributes": attrs,
        })
    production_plan = pd.DataFrame(plan_rows)

    # Expanded table with simple synthetic energy profile.
    expanded_rows = []
    for _, ev in event_log.iterrows():
        start = pd.to_datetime(ev["timestamp_start"])  # defensive cast
        end = pd.to_datetime(ev["timestamp_end"])
        minute_index = pd.date_range(start=start, end=end, freq="1min")
        if len(minute_index) == 0:
            minute_index = pd.DatetimeIndex([start])

        for ts in minute_index:
            if ev["perspective"] == "machine_state":
                base_power = {
                    "setup": 1.8,
                    "running": 6.5,
                    "fault": 2.4,
                    "cleanup": 1.2,
                }.get(ev["activity"], 1.0)
            else:
                base_power = 0.8 if "rework" in ev["activity"] else 1.4

            expanded_rows.append({
                "datetime_energy": ts,
                "power_kw_energy": float(max(0.1, np.random.normal(base_power, 0.2))),
                "case_id_log": ev["case_id"],
                "activity_log": ev["activity"],
                "timestamp_start_log": ev["timestamp_start"],
                "timestamp_end_log": ev["timestamp_end"],
                "object_log": ev["object"],
                "object_type_log": ev["object_type"],
                "higher_level_activity_log": ev["higher_level_activity"],
                "object_attributes_log": ev["object_attributes"],
                "perspective_log": ev["perspective"],
            })

    expanded = pd.DataFrame(expanded_rows).sort_values("datetime_energy").reset_index(drop=True)

    return {
        "expanded": expanded,
        "event_log": event_log,
        "production_plan": production_plan,
    }
