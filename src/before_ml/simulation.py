import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class ProcessSimulation:
    def __init__(self, activity_stats_df, production_plan, random_seed=42):
        self.activity_stats = activity_stats_df
        self.production_plan = production_plan
        random.seed(random_seed)
        np.random.seed(random_seed)
        
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
                'duration': row['duration'],
                'duration_std': row['duration_std'],
                'transitions': row['transition'],
                'is_start': row['is_start'],
                'is_end': row['is_end'],
                'n_events': row['n_events']
            }
            
            print(f"  Config: {key}")
            print(f"    Transitions: {row['transition']}")
            print(f"    Start: {row['is_start']}")
    
    def _get_activity_duration(self, activity, object_name, object_type, higher_level_activity):
        """Generate duration for an activity based on statistics"""
        activity = str(activity).strip()
        object_name = str(object_name).strip()
        object_type = str(object_type).strip()
        higher_level_activity = str(higher_level_activity).strip() if pd.notna(higher_level_activity) else None
        
        key = (activity, object_name, object_type, higher_level_activity)
        
        if key in self.activity_config:
            config = self.activity_config[key]
            duration = config['duration']
            duration_std = config['duration_std']
            
            if duration_std > 0:
                sampled_duration = max(0.1, np.random.normal(duration, duration_std))
            else:
                sampled_duration = duration
                
            return sampled_duration
        else:
            print(f"WARNING: No config found for {key}")
            return 10.0
    
    def _get_next_activity(self, current_activity, object_name, object_type, higher_level_activity):
        """
        Determine next activity based on transition probabilities
        NOW HANDLES PROBABILISTIC END instead of hard stop
        """
        current_activity = str(current_activity).strip()
        object_name = str(object_name).strip()
        object_type = str(object_type).strip()
        higher_level_activity = str(higher_level_activity).strip() if pd.notna(higher_level_activity) else None
        
        key = (current_activity, object_name, object_type, higher_level_activity)
        
        print(f"    Looking for transitions from: {current_activity}")
        
        if key in self.activity_config:
            transitions = self.activity_config[key]['transitions']
            print(f"    Found transitions: {transitions}")
            
            if transitions and len(transitions) > 0:
                activities = list(transitions.keys())
                probabilities = list(transitions.values())
                
                # Normalize probabilities
                prob_sum = sum(probabilities)
                if prob_sum > 0:
                    probabilities = [p/prob_sum for p in probabilities]
                    
                    # Choose next transition (including possible __END__)
                    next_transition = np.random.choice(activities, p=probabilities)
                    
                    # Check if we selected the END transition
                    if next_transition == '__END__':
                        print(f"    Selected END transition - stopping process")
                        return None
                    else:
                        print(f"    Selected next activity: {next_transition}")
                        return str(next_transition).strip()
            else:
                print(f"    No transitions available - ending process")
        else:
            print(f"    Key not found in config: {key}")
        
        return None  # End of process
    
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
                    current_activity, object_name, object_type, higher_level_activity
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
                    current_activity, object_name, object_type, higher_level_activity
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