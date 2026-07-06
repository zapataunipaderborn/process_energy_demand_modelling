import re

file_path = "src/generate_process_1.py"
with open(file_path, "r") as f:
    content = f.read()

target = """\
        # Batch clock: when the batch is ready to enter the next station
        batch_ready = start_date + timedelta(minutes=b * 240)  # 4-h stagger → ~1 year for 2190 batches

        for station in PROCESS_SEQUENCE:
            cfg         = PROCESS_CONFIG[station]
            vol_affected = cfg['volume_affected']
            n_res        = NUM_RESOURCES[station]
            res_times    = resource_free[station]

            # Assign to the resource that becomes free soonest
            best_slot = int(np.argmin([t.timestamp() for t in res_times]))
            slot_free = res_times[best_slot]
            station_start = max(batch_ready, slot_free)

            # Resource name
            if n_res == 1:
                resource_name = station.lower()
            else:
                resource_name = f'{station.lower()}_{best_slot + 1}'

            object_type = 'sterilization_line'

            current_time = station_start
            for event_def in cfg['events']:
                act_name = event_def['name']
                rec_cfg  = event_def['recipes'][recipe]
                dur      = sample_duration(
                    rec_cfg['duration'], rec_cfg['variability'],
                    volume_L, vol_affected, rng
                )
                ts_start = current_time
                ts_end   = current_time + timedelta(minutes=dur)

                events.append({
                    'case_id':               case_id,
                    'activity':              f'{resource_name}_{act_name}',
                    'timestamp_start':       ts_start,
                    'timestamp_end':         ts_end,
                    'higher_level_activity': 'sterilization_process',
                    'station':               station,
                    'object_type':           object_type,
                    'object':                resource_name,
                    'object_attributes':     obj_attrs,
                })
                current_time = ts_end + timedelta(seconds=1)

            # Update resource and batch clocks
            res_times[best_slot] = current_time
            batch_ready          = current_time
"""

replacement = """\
        # Batch clock: when the batch is ready to enter the next station
        batch_ready = start_date + timedelta(minutes=b * 240)  # 4-h stagger → ~1 year for 2190 batches

        stations_queue = list(PROCESS_SEQUENCE)
        
        # Dynamic Skipping: Cleaning recipes don't go to Packaging or Warehousing
        if recipe == 'cleaning':
            stations_queue = [s for s in stations_queue if s not in ('Packaging', 'Warehousing')]

        while stations_queue:
            station = stations_queue.pop(0)

            cfg         = PROCESS_CONFIG[station]
            vol_affected = cfg['volume_affected']
            n_res        = NUM_RESOURCES[station]
            res_times    = resource_free[station]

            # Assign to the resource that becomes free soonest
            best_slot = int(np.argmin([t.timestamp() for t in res_times]))
            slot_free = res_times[best_slot]
            station_start = max(batch_ready, slot_free)

            # Resource name
            if n_res == 1:
                resource_name = station.lower()
            else:
                resource_name = f'{station.lower()}_{best_slot + 1}'

            object_type = 'sterilization_line'

            current_time = station_start
            
            events_to_process = list(cfg['events'])
            idx = 0
            
            while idx < len(events_to_process):
                event_def = events_to_process[idx]
                act_name = event_def['name']
                rec_cfg  = event_def['recipes'][recipe]
                dur      = sample_duration(
                    rec_cfg['duration'], rec_cfg['variability'],
                    volume_L, vol_affected, rng
                )
                
                # Machine Breakdowns: 5% chance before working/heat
                if act_name in ('working', 'heat') and rng.random() < 0.05:
                    breakdown_dur = float(rng.uniform(60.0, 120.0))
                    events.append({
                        'case_id':               case_id,
                        'activity':              f'{resource_name}_maintenance',
                        'timestamp_start':       current_time,
                        'timestamp_end':         current_time + timedelta(minutes=breakdown_dur),
                        'higher_level_activity': 'sterilization_process',
                        'station':               station,
                        'object_type':           object_type,
                        'object':                resource_name,
                        'object_attributes':     obj_attrs,
                    })
                    current_time += timedelta(minutes=breakdown_dur) + timedelta(seconds=1)

                ts_start = current_time
                ts_end   = current_time + timedelta(minutes=dur)

                events.append({
                    'case_id':               case_id,
                    'activity':              f'{resource_name}_{act_name}',
                    'timestamp_start':       ts_start,
                    'timestamp_end':         ts_end,
                    'higher_level_activity': 'sterilization_process',
                    'station':               station,
                    'object_type':           object_type,
                    'object':                resource_name,
                    'object_attributes':     obj_attrs,
                })
                current_time = ts_end + timedelta(seconds=1)
                
                # Microscopic rework inside Autoclave
                if station == 'Autoclaving' and act_name == 'cool':
                    if rng.random() < 0.15:  # 15% chance to rework
                        rework_events = [e for e in cfg['events'] if e['name'] in ('heat', 'hold', 'cool')]
                        events_to_process = events_to_process[:idx+1] + rework_events + events_to_process[idx+1:]
                
                idx += 1

            # Update resource and batch clocks
            res_times[best_slot] = current_time
            batch_ready          = current_time
            
            # Macroscopic rework: Warehousing -> Packaging
            if station == 'Warehousing':
                if rng.random() < 0.10: # 10% chance to fail QA and go back to Packaging
                    stations_queue.append('Packaging')
                    stations_queue.append('Warehousing')
"""

if target in content:
    content = content.replace(target, replacement)
    with open(file_path, "w") as f:
        f.write(content)
    print("Replaced successfully!")
else:
    print("Target not found.")
