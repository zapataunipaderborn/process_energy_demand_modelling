from collections import defaultdict
import pandas as pd
import numpy as np

def extract_process(df):
    """
    Extract process statistics with PROBABILISTIC END transitions
    End is treated as a transition probability, not a hard stop
    """

    
    stats = []
    
    # Group by the combination that defines a unique process configuration
    grouped = df.groupby(['object', 'object_type', 'higher_level_activity'])
    
    for (object_name, object_type, higher_level_activity), group in grouped:
        print(f"\nProcessing: {object_name} ({object_type}) - {higher_level_activity}")
        
        # Get all activities for this object configuration
        activities = group['activity'].unique()
        print(f"  Found {len(activities)} unique activities: {activities}")
        
        # For each activity, calculate statistics
        for activity in activities:
            activity_data = group[group['activity'] == activity]
            
            # Basic statistics
            durations = (activity_data['timestamp_end'] - activity_data['timestamp_start']).dt.total_seconds() / 60
            duration_mean = durations.mean()
            duration_std = durations.std() if len(durations) > 1 else 0
            n_events = len(activity_data)
            
            # Start detection - appears first in cases
            is_start = False
            for case_id in activity_data['case_id'].unique():
                if pd.isna(case_id):
                    continue
                    
                case_activities = group[group['case_id'] == case_id].sort_values('timestamp_start')
                if len(case_activities) > 0:
                    first_activity = case_activities.iloc[0]['activity']
                    if first_activity == activity:
                        is_start = True
                        break
            
            # PROBABILISTIC Transition detection - INCLUDING END PROBABILITY
            transition_counts = defaultdict(int)
            end_counts = 0  # Count how many times this activity ends without transition
            total_occurrences = 0
            
            for case_id in activity_data['case_id'].unique():
                if pd.isna(case_id):
                    continue
                    
                case_activities = group[group['case_id'] == case_id].sort_values('timestamp_start')
                case_activities = case_activities.reset_index(drop=True)
                
                # Find all occurrences of current activity in this case
                current_indices = case_activities[case_activities['activity'] == activity].index
                
                for idx in current_indices:
                    total_occurrences += 1
                    
                    # Check if there's a next activity
                    if idx + 1 < len(case_activities):
                        next_activity = case_activities.iloc[idx + 1]['activity']
                        transition_counts[next_activity] += 1
                    else:
                        # This activity is the last one - count as "end"
                        end_counts += 1
            
            # Convert counts to probabilities INCLUDING end probability
            transitions = {}
            if total_occurrences > 0:
                # Regular transitions to other activities
                for next_act, count in transition_counts.items():
                    transitions[next_act] = count / total_occurrences
                
                # Add END probability (probability of process ending after this activity)
                if end_counts > 0:
                    transitions['__END__'] = end_counts / total_occurrences
            
            # is_end is now just informational (if this activity can end the process)
            is_end = end_counts > 0
            
            print(f"    {activity}:")
            print(f"      Events: {n_events}, Start: {is_start}, Can End: {is_end}")
            print(f"      Total occurrences: {total_occurrences}")
            print(f"      Transitions: {transitions}")
            
            stats.append({
                'activity': activity,
                'object': object_name,
                'object_type': object_type,
                'higher_level_activity': higher_level_activity,
                'duration': duration_mean,
                'duration_std': duration_std,
                'n_events': n_events,
                'transition': transitions,
                'is_start': is_start,
                'is_end': is_end,  # Keep for info, but don't use as hard stop
            })
    
    return pd.DataFrame(stats)
