def driving_actions(edge_info: dict, vehicles_info: list[dict]) -> list[dict]:
    """Cooperative driving algorithm combining IDM for longitudinal control with traffic-aware lane changing."""
    
    # Initialize persistent state if not exists
    if not hasattr(driving_actions, 'state'):
        driving_actions.state = {
            'vehicle_first_seen': {},
            'target_lanes': {},
            'lane_change_cooldown': {},
            'swap_pairs': set()
        }
    
    state = driving_actions.state
    
    # Vehicle type characteristics
    VEHICLE_LENGTHS = {
        'passenger': 5.0,
        'bus': 12.0,
        'truck': 7.5,
        'emergency': 5.0
    }
    
    # IDM parameters
    IDM_PARAMS = {
        'desired_speed': edge_info['max_speed'],
        'time_headway': 1.5,
        'min_gap': 2.0,
        'max_accel': 2.0,
        'comfortable_decel': 3.0,
        'delta': 4
    }
    
    # Lane change parameters
    ENTRY_WAIT_STEPS = 3
    SAFE_GAP_BUFFER = 3.0
    COOLDOWN_STEPS = 5
    URGENCY_START_DISTANCE = 100.0
    PLATOON_BUFFER = 2.0
    
    actions = []
    current_step = len(state['vehicle_first_seen'])
    
    # Track vehicles and update state
    current_vehicle_ids = {v['veh_id'] for v in vehicles_info}
    for veh_id in list(state['vehicle_first_seen'].keys()):
        if veh_id not in current_vehicle_ids:
            state['vehicle_first_seen'].pop(veh_id, None)
            state['target_lanes'].pop(veh_id, None)
            state['lane_change_cooldown'].pop(veh_id, None)
    
    for veh in vehicles_info:
        if veh['veh_id'] not in state['vehicle_first_seen']:
            state['vehicle_first_seen'][veh['veh_id']] = current_step
    
    # Build lane structure
    lane_vehicles = {}
    for veh in vehicles_info:
        lane_id = veh['current_lane']
        if lane_id not in lane_vehicles:
            lane_vehicles[lane_id] = []
        lane_vehicles[lane_id].append(veh)
    
    # Sort vehicles by position in each lane
    for lane_id in lane_vehicles:
        lane_vehicles[lane_id].sort(key=lambda v: v['position'])
    
    def get_vehicle_length(veh_type):
        return VEHICLE_LENGTHS.get(veh_type, 5.0)
    
    def get_lane_index(lane_id):
        return int(lane_id.split('_')[-1])
    
    def find_leader_follower(veh, lane_id):
        if lane_id not in lane_vehicles:
            return None, None
        
        lane_vehs = lane_vehicles[lane_id]
        leader = None
        follower = None
        
        for other in lane_vehs:
            if other['veh_id'] == veh['veh_id']:
                continue
            if other['position'] > veh['position']:
                if leader is None or other['position'] < leader['position']:
                    leader = other
            elif other['position'] < veh['position']:
                if follower is None or other['position'] > follower['position']:
                    follower = other
        
        return leader, follower
    
    def calculate_idm_acceleration(veh, leader):
        v = veh['speed']
        v0 = IDM_PARAMS['desired_speed']
        T = IDM_PARAMS['time_headway']
        s0 = IDM_PARAMS['min_gap']
        a = IDM_PARAMS['max_accel']
        b = IDM_PARAMS['comfortable_decel']
        delta = IDM_PARAMS['delta']
        
        if leader is None:
            return a * (1 - (v / v0) ** delta)
        
        s = leader['position'] - veh['position'] - get_vehicle_length(leader['veh_type'])
        dv = v - leader['speed']
        
        s_star = s0 + max(0, v * T + (v * dv) / (2 * (a * b) ** 0.5))
        
        accel = a * (1 - (v / v0) ** delta - (s_star / max(s, 0.1)) ** 2)
        return accel
    
    def get_best_target_lane(veh):
        lanes = veh['potential_target_lanes']
        
        # Filter valid continuation lanes
        valid_lanes = [l for l in lanes if l[4]]  # allowsContinuation
        if not valid_lanes:
            valid_lanes = lanes
        
        if not valid_lanes:
            return get_lane_index(veh['current_lane'])
        
        # Sort by occupation (less congested first)
        valid_lanes.sort(key=lambda l: l[2])
        
        return get_lane_index(valid_lanes[0][0])
    
    def calculate_urgency(veh, target_lane_idx):
        current_idx = get_lane_index(veh['current_lane'])
        if current_idx == target_lane_idx:
            return 0.0
        
        remaining_distance = edge_info['length'] - veh['position']
        
        if remaining_distance > URGENCY_START_DISTANCE:
            return 0.0
        
        # Gradual increase from 0 to 1
        urgency = 1.0 - (remaining_distance / URGENCY_START_DISTANCE)
        return max(0.0, min(1.0, urgency))
    
    def is_gap_safe(gap, veh_length, speed_diff):
        min_safe_gap = veh_length + SAFE_GAP_BUFFER
        
        # If follower is slower or equal, relax gap requirement
        if speed_diff <= 0:
            min_safe_gap = veh_length + 1.0
        
        return gap >= min_safe_gap
    
    def detect_swap_pairs(vehicles_info):
        swap_pairs = []
        
        for i, veh1 in enumerate(vehicles_info):
            target1 = state['target_lanes'].get(veh1['veh_id'])
            if target1 is None:
                continue
            
            current1 = get_lane_index(veh1['current_lane'])
            
            for veh2 in vehicles_info[i+1:]:
                target2 = state['target_lanes'].get(veh2['veh_id'])
                if target2 is None:
                    continue
                
                current2 = get_lane_index(veh2['current_lane'])
                
                # Check if they want to swap
                if current1 == target2 and current2 == target1:
                    # Check if they're close enough to interfere
                    if abs(veh1['position'] - veh2['position']) < 50.0:
                        swap_pairs.append((veh1, veh2))
        
        return swap_pairs
    
    # Determine target lanes for all vehicles
    for veh in vehicles_info:
        if veh['veh_id'] not in state['target_lanes']:
            state['target_lanes'][veh['veh_id']] = get_best_target_lane(veh)
    
    # Detect swap deadlocks
    swap_pairs = detect_swap_pairs(vehicles_info)
    swap_vehicles = set()
    for v1, v2 in swap_pairs:
        swap_vehicles.add(v1['veh_id'])
        swap_vehicles.add(v2['veh_id'])
    
    # Process each vehicle
    for veh in vehicles_info:
        veh_id = veh['veh_id']
        current_lane_idx = get_lane_index(veh['current_lane'])
        target_lane_idx = state['target_lanes'][veh_id]
        
        # Check cooldown
        cooldown = state['lane_change_cooldown'].get(veh_id, 0)
        if cooldown > 0:
            state['lane_change_cooldown'][veh_id] = cooldown - 1
        
        # Conservative entry behavior
        steps_since_entry = current_step - state['vehicle_first_seen'][veh_id]
        if steps_since_entry < ENTRY_WAIT_STEPS:
            leader, _ = find_leader_follower(veh, veh['current_lane'])
            accel = calculate_idm_acceleration(veh, leader)
            actions.append({
                'veh_id': veh_id,
                'acceleration': accel,
                'lane_changing': 0
            })
            continue
        
        # Calculate base acceleration
        leader, follower = find_leader_follower(veh, veh['current_lane'])
        base_accel = calculate_idm_acceleration(veh, leader)
        
        # Determine lane change direction
        lane_change = 0
        
        if current_lane_idx != target_lane_idx and state['lane_change_cooldown'].get(veh_id, 0) == 0:
            direction = 1 if target_lane_idx > current_lane_idx else -1
            target_lane_id = f"{edge_info['edge_id']}_{current_lane_idx + direction}"
            
            # Calculate urgency
            urgency = calculate_urgency(veh, target_lane_idx)
            
            # Check if safe to change
            if target_lane_id in lane_vehicles:
                target_leader, target_follower = find_leader_follower(veh, target_lane_id)
                
                front_gap = float('inf')
                rear_gap = float('inf')
                
                if target_leader:
                    front_gap = target_leader['position'] - veh['position'] - get_vehicle_length(target_leader['veh_type'])
                    front_speed_diff = veh['speed'] - target_leader['speed']
                else:
                    front_gap = edge_info['length'] - veh['position']
                    front_speed_diff = 0
                
                if target_follower:
                    rear_gap = veh['position'] - target_follower['position'] - get_vehicle_length(veh['veh_type'])
                    rear_speed_diff = target_follower['speed'] - veh['speed']
                else:
                    rear_gap = veh['position']
                    rear_speed_diff = 0
                
                front_safe = is_gap_safe(front_gap, get_vehicle_length(veh['veh_type']), front_speed_diff)
                rear_safe = is_gap_safe(rear_gap, get_vehicle_length(target_follower['veh_type']) if target_follower else 5.0, rear_speed_diff)
                
                # Handle swap situation
                if veh_id in swap_vehicles:
                    # Find swap partner
                    swap_partner = None
                    for v1, v2 in swap_pairs:
                        if v1['veh_id'] == veh_id:
                            swap_partner = v2
                        elif v2['veh_id'] == veh_id:
                            swap_partner = v1
                    
                    if swap_partner:
                        offset = abs(veh['position'] - swap_partner['position'])
                        required_offset = get_vehicle_length(veh['veh_type']) + get_vehicle_length(swap_partner['veh_type']) + PLATOON_BUFFER * 2
                        
                        if offset < required_offset:
                            # Create offset through speed adjustment
                            if veh['speed'] > swap_partner['speed']:
                                base_accel = min(base_accel + 0.5, IDM_PARAMS['max_accel'])
                            else:
                                base_accel = max(base_accel - 0.5, -IDM_PARAMS['comfortable_decel'])
                            
                            # Don't change lane yet
                            front_safe = False
                            rear_safe = False
                
                if front_safe and rear_safe:
                    if urgency > 0.3:
                        lane_change = direction
                        state['lane_change_cooldown'][veh_id] = COOLDOWN_STEPS
                else:
                    # Active gap creation
                    if urgency > 0.5:
                        if not front_safe and target_leader and front_gap < 20.0:
                            base_accel = max(base_accel - 0.8, -IDM_PARAMS['comfortable_decel'])
                        elif not rear_safe and target_follower and rear_gap < 20.0:
                            base_accel = min(base_accel + 0.8, IDM_PARAMS['max_accel'])
        
        actions.append({
            'veh_id': veh_id,
            'acceleration': base_accel,
            'lane_changing': lane_change
        })
    
    return actions