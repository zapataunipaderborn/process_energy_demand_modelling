import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

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
    mode : {'statistical', 'ml', 'ml_duration_only'}
        'statistical'      – sample durations/transitions from fitted
        distributions (or normal as fallback).
        'ml'               – use XGBoost models from *ml_models* when
        available; fall back to statistical automatically.
        'ml_duration_only' – use XGBoost only for the duration *median*;
        the standard deviation and transition probabilities still come
        from the statistical extraction (``activity_stats_df``).
    ml_models : SimModeller | None
        A trained ``SimModeller`` instance (required when mode='ml').
    random_seed : int
        NumPy / random seed for reproducibility.
    """

    def __init__(self, activity_stats_df, production_plan,
                 mode='statistical', ml_models=None, random_seed=42):
        self.activity_stats = activity_stats_df
        self.production_plan = production_plan
        self.mode = mode
        self.ml_models = ml_models
        random.seed(random_seed)
        np.random.seed(random_seed)

        if self.mode in ('ml', 'ml_duration_only') and self.ml_models is None:
            print(f"[ProcessSimulation] WARNING: mode='{self.mode}' but no ml_models "
                  "provided – falling back to statistical mode.")
            self.mode = 'statistical'

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
                               higher_level_activity, object_attributes=None):
        """
        Generate a sampled duration (minutes) for one activity instance.

        In 'ml' mode the XGBoost duration model is tried first; it falls back
        to the statistical path when no model exists for this key.
        In 'ml_duration_only' mode, the ML model provides only the median
        prediction while the noise (std) comes from the statistical config.
        In 'statistical' mode the best-fit distribution stored in
        activity_config is used directly.
        """
        activity              = str(activity).strip()
        object_name           = str(object_name).strip()
        object_type           = str(object_type).strip()
        higher_level_activity = (str(higher_level_activity).strip()
                                 if pd.notna(higher_level_activity) else None)
        object_attributes     = object_attributes or {}

        key = (activity, object_name, object_type, higher_level_activity)

        # ── Full ML path (duration + ML std) ──────────────────────────────
        if self.mode == 'ml' and self.ml_models is not None:
            ml_dur = self.ml_models.predict_duration(
                activity, object_name, object_type,
                higher_level_activity, object_attributes
            )
            if ml_dur is not None:
                return ml_dur
            # else: fall through to statistical

        # ── ML duration-only path (ML median + statistical std) ──────────
        if self.mode == 'ml_duration_only' and self.ml_models is not None:
            ml_median = self.ml_models.predict_duration_median(
                activity, object_name, object_type,
                higher_level_activity, object_attributes
            )
            if ml_median is not None:
                # Use the statistical std from the extracted data
                stat_std = 0.0
                if key in self.activity_config:
                    stat_std = self.activity_config[key].get('duration_std', 0.0)
                if stat_std > 0:
                    sampled = np.random.normal(ml_median, stat_std)
                else:
                    sampled = ml_median
                return max(0.1, float(sampled))
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

        activities    = list(transitions.keys())
        probabilities = list(transitions.values())
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
    
    def _simulate_process_for_case(self, case_id, object_attributes, start_time):
        """Simulate process for one case using probabilistic end transitions"""
        unique_objects = self.activity_stats[['object', 'object_type', 'higher_level_activity']].drop_duplicates()
        
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
            
            # Follow the process flow with probabilistic ending - NO ARTIFICIAL LIMITS
            activity_count = 0
            
            while current_activity:
                print(f"    Processing activity {activity_count + 1}: {current_activity}")
                
                # Get duration
                activity_duration = self._get_activity_duration(
                    current_activity, object_name, object_type,
                    higher_level_activity, object_attributes
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