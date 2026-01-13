import traci

def driving_actions(edge_info, vehicles_info):
    """
    Decide the next actions for all vehicles for the current time step.
    
    Args:
        edge_info (dict): target edge info; dict keys are:
        - 'edge_id' (str)
        - 'lane_count' (int)
        - 'length' (float, m)
        - 'max_speed' (float, m/s)

        vehicles_info (list[dict]): info of vehicles on the edge; one dict for each vehicle; dict keys are:
        - 'veh_id' (str)
        - 'veh_type' (str)
        - 'position' (float) - x-coordinate position
        - 'speed' (float, m/s)
        - 'current_lane' (str) - e.g., "E0_0"
        - 'wants_left'/'wants_right' (bool)
        - 'potential_target_lanes' (list[tuple]) - (laneID, length, occupation, bestLaneOffset, allowsContinuation, nextLanes)

    Returns:
        actions (list[dict]): actions to take; one dict for each vehicle; dict keys are:
        - 'veh_id'
        - 'acceleration' (float, m/s^2); positive for acceleration, negative for deceleration
        - 'lane_changing' (int: +1=left, 0=keep, -1=right)
    """
    # =====PARAMETERS & VARIABLES=====
    # -----CONSTANTS-----
    # Safety parameters
    SAFE_TTC = 3.0                      # Safe time-to-collision threshold (seconds)
    MIN_GAP = 1.6                       # Minimum gap for car following (meters); Note: gap does not include vehicle length
    LANE_CHANGE_GAP_BASE = 10.0         # Base gap required for lane changes (meters); Note: gap does not include vehicle length
    OFFSET_BUFFER = 1.6                 # Safety buffer added to vehicle length for lane-changing (meters); Note: buffer does not include vehicle length
    
    # Entry behavior parameters
    ENTRY_DISTANCE = 15.0               # Distance from start considered as entry zone (meters)
    ENTRY_TIME = 3.0                    # Time after entering considered as entry period (seconds)
    
    # Lane change control parameters
    LANE_CHANGE_COOLDOWN = 2.5          # Minimum time between lane changes (seconds)
    TARGET_LANE_PATIENCE = 60.0         # Distance to stay in target lane before changing again (meters)
    
    # Urgency calculation parameters
    URGENCY_START_DIST = 400.0          # Distance from end where urgency starts increasing (meters)
    CRITICAL_DIST = 80.0                # Distance from end where urgency reaches maximum (meters)
    
    # Acceleration parameters
    MIN_ACCEL = 0.3                     # Minimum acceleration to maintain (m/s²)
    MAX_ACCEL_BOOST = 1.2               # Multiplier for maximum acceleration in free flow
    
    # Cooperation parameters
    COOPERATION_RANGE = 80.0           # Range for cooperative behavior (meters)
    COOPERATION_WINDOW = 6.0            # Time window for cooperative intent signaling (seconds)
    SPEED_MATCH_FACTOR = 0.6            # Factor for speed matching adjustments
    
    # Gap creation parameters
    GAP_SEARCH_ACCEL = 3.2              # Acceleration when searching for gaps (m/s²)
    GAP_SEARCH_DECEL = 1.8              # Deceleration when searching for gaps (m/s²)
    
    # Swap coordination parameters
    SWAP_DETECT_RANGE = 50.0            # Range to detect swap conflicts (meters)
    SWAP_COORDINATION_ACCEL = 2.5       # Acceleration for swap coordination (m/s²)
    SWAP_COORDINATION_DECEL = 2.0       # Deceleration for swap coordination (m/s²)
    
    # Platoon formation parameters
    PLATOON_DETECT_RANGE = 18.0         # Range to detect platoon opportunities (meters)
    PLATOON_ACCEL = 3.0                 # Acceleration adjustment for platoon coordination (m/s²)
    PLATOON_DECEL = 1.5                 # Deceleration adjustment for platoon coordination (m/s²)
    
    # -----VEHICLE TYPE PARAMETERS-----
    
    # Vehicle-specific lengths (meters)
    veh_length = {
        "passenger": 5.0,
        "bus": 12.0,
        "truck": 7.1,
        "emergency": 6.5
    }
    
    # Vehicle-specific acceleration capabilities (m/s²)
    max_accel = {
        "passenger": 2.6,
        "bus": 1.2,
        "truck": 1.3,
        "emergency": 2.6
    }
    
    # Vehicle-specific deceleration capabilities (m/s²)
    max_decel = {
        "passenger": 4.5,
        "bus": 4.0,
        "truck": 4.0,
        "emergency": 4.5
    }
    
    # -----MEMORY INITIALIZATION-----
    
    if not hasattr(driving_actions, 'last_lane_change'):
        driving_actions.last_lane_change = {}
    
    if not hasattr(driving_actions, 'entry_time'):
        driving_actions.entry_time = {}
    
    if not hasattr(driving_actions, 'target_lane_reached'):
        driving_actions.target_lane_reached = {}
    
    if not hasattr(driving_actions, 'lane_change_intent'):
        driving_actions.lane_change_intent = {}
    
    if not hasattr(driving_actions, 'swap_partners'):
        driving_actions.swap_partners = {}
    
    if not hasattr(driving_actions, 'platoon_pairs'):
        driving_actions.platoon_pairs = {}
    
    current_time = traci.simulation.getTime()
    
    #-----ORGANIZE VEHICLES BY LANE-----
    
    lanes = {}
    for i in range(edge_info['lane_count']):
        lanes[f"{edge_info['edge_id']}_{i}"] = []
    
    for veh in vehicles_info:
        lanes[veh['current_lane']].append(veh)
    
    for lane in lanes:
        lanes[lane].sort(key=lambda x: x['position'], reverse=True)  
        # Note: vehicles are sorted by position, largest first
        # this makes it easy to iterate vehicle from upstream to downstream
    
    # =====HELPER FUNCTIONS=====
    
    def compute_accel(veh_speed, veh_type, leader_speed, gap, max_speed):
        """Compute optimal acceleration based on IDM"""
        if leader_speed is None or gap is None:
            return min(max_accel[veh_type] * MAX_ACCEL_BOOST, max_speed - veh_speed)
        
        if gap < MIN_GAP:
            return -max_decel[veh_type]
        
        desired_gap = max(MIN_GAP, 1.5 * veh_speed + 2.0)  # desired gap calculation; Note: gap does not include vehicle length
        relative_speed = veh_speed - leader_speed
        
        if relative_speed > 0:
            ttc = gap / relative_speed
            if ttc < SAFE_TTC:
                decel = -min(max_decel[veh_type], relative_speed / SAFE_TTC * 1.2)
                return max(-max_decel[veh_type], decel)
        
        if gap < desired_gap:
            return max(-max_decel[veh_type] * 0.4, -0.6 * (desired_gap - gap) / SAFE_TTC)
        
        return min(max_accel[veh_type] * MAX_ACCEL_BOOST, max_speed - veh_speed)
    
    def get_target_lane(veh):
        """Determine target lane based on route"""
        lane_index = int(veh['current_lane'].split('_')[-1])
        
        for lane_info in veh['potential_target_lanes']:
            if lane_info[3] == 0 and lane_info[4]:
                return int(lane_info[0].split('_')[-1])
        
        return lane_index
    
    def get_current_lane_length(veh):
        """Determine target lane based on route"""
        ans = edge_info['length']  # default to full length if not found
        
        for lane_info in veh['potential_target_lanes']:
            if lane_info[0] == veh['current_lane']:
                return float(lane_info[1])
        return ans
    
    def is_lane_change_safe(veh, target_lane_id, lanes, urgency, min_gap_override=None):
        """Check if lane change is safe with relaxed requirements for safe speeds"""
        position = veh['position']
        speed = veh['speed']
        veh_type = veh['veh_type']
        my_length = veh_length[veh_type]
        
        if min_gap_override is not None:
            gap_threshold = min_gap_override
        else:
            gap_threshold = LANE_CHANGE_GAP_BASE * (1.0 - 0.35 * urgency)
        
        ttc_threshold = SAFE_TTC * (1.0 - 0.25 * urgency)
        
        target_vehs = lanes.get(target_lane_id, [])
        
        # Check leader in target lane
        vehicles_ahead = [v for v in target_vehs if v['position'] > position]
        vehicles_ahead = sorted(vehicles_ahead, key=lambda x: x['position'])
        leader = vehicles_ahead[0] if vehicles_ahead else None
        if leader:
            leader_length = veh_length[leader['veh_type']]
            gap = leader['position'] - position - leader_length
            rel_speed = speed - leader['speed']
            
            # If we're slower or equal speed, just need one vehicle length
            if rel_speed <= 0 and gap >= OFFSET_BUFFER:
                pass  # Safe
            else:
                if gap < gap_threshold:
                    return False
                if rel_speed > 0:
                    ttc = gap / rel_speed
                    if ttc < ttc_threshold:
                        return False
        
        # Check follower in target lane
        follower = next((v for v in target_vehs if v['position'] < position), None)
        if follower:
            gap = position - follower['position'] - my_length
            rel_speed = follower['speed'] - speed
            
            # If follower is slower or equal speed, just need one vehicle length
            if rel_speed <= 0 and gap >= OFFSET_BUFFER:
                pass  # Safe
            else:
                if gap < gap_threshold * 0.75:
                    return False
                if rel_speed > 0:
                    ttc = gap / rel_speed
                    if ttc < ttc_threshold:
                        return False
        
        return True
    
    def detect_swap_conflict(veh1, veh2, target1, target2):
        """Detect if two vehicles want to swap lanes"""
        lane1 = int(veh1['current_lane'].split('_')[-1])
        lane2 = int(veh2['current_lane'].split('_')[-1])
        
        # Target lanes may be more than one lane away; but vehicle can only change one lane at a time
        # so we normalize to adjacent lanes
        if target1 > lane1:
            target1 = lane1 + 1
        elif target1 < lane1:
            target1 = lane1 - 1
        if target2 > lane2:
            target2 = lane2 + 1
        elif target2 < lane2:
            target2 = lane2 - 1
            
        if abs(lane1 - lane2) == 1:
            if target1 == lane2 and target2 == lane1:
                pos_diff = abs(veh1['position'] - veh2['position'])
                if pos_diff < SWAP_DETECT_RANGE:
                    return True
        
        return False
    
    def detect_platoon_opportunity(veh1, veh2, target1, target2):
        """
        Detect if two vehicles can form a platoon (FLEXIBLE VERSION)
        
        This version supports bidirectional platoon formation:
        - If follower is behind: lane vehicle accelerates, joining vehicle decelerates
        - If follower is ahead: lane vehicle decelerates, joining vehicle accelerates
        
        Returns:
            (lane_veh_id, joining_veh_id, position_relation) tuple if platoon possible
            position_relation: 'lane_ahead' if lane_vehicle is ahead, 'lane_behind' if behind
            None if no platoon opportunity
        """
        lane1 = int(veh1['current_lane'].split('_')[-1])
        lane2 = int(veh2['current_lane'].split('_')[-1])
            
        if current_time == 215 and veh1['veh_id'] in ['veh_1_185_0', 'veh_4_192_0'] and veh2['veh_id'] in ['veh_1_185_0', 'veh_4_192_0']:  # for debugging
            print('', end='')  # put a breakpoint here if needed
        
        # Target lanes may be more than one lane away; but vehicle can only change one lane at a time
        # so we normalize to adjacent lanes
        if target1 > lane1:
            target1 = lane1 + 1
        elif target1 < lane1:
            target1 = lane1 - 1
        if target2 > lane2:
            target2 = lane2 + 1
        elif target2 < lane2:
            target2 = lane2 - 1
            
        # Must have same target lane
        if target1 != target2:
            return None
        
        # Must be in adjacent lanes
        if abs(lane1 - lane2) != 1:
            return None
        
        # Check if close enough
        pos_diff = abs(veh1['position'] - veh2['position'])
        if pos_diff > PLATOON_DETECT_RANGE:
            return None
        
        threshold = 0.8  # threshold to decide that vehicles are very close (meters); then we need their types to decide who should lead
        # Determine who is in target lane and who needs to join
        if lane1 == target1 and lane2 != target2:
            # veh1 is in target lane, veh2 needs to join
            lane_veh_id = veh1['veh_id']
            joining_veh_id = veh2['veh_id']
            if veh1['position'] - veh2['position'] >= threshold:
                position_relation = 'lane_ahead'
            elif (0 <= veh1['position'] - veh2['position'] < threshold) and not (veh1['veh_type'] in ['bus', 'truck'] and veh2['veh_type'] in ['passenger', 'emergency']):
                position_relation = 'lane_ahead'
            elif -threshold <= veh1['position'] - veh2['position'] < 0 and (veh1['veh_type'] in ['passenger', 'emergency']and veh2['veh_type'] in ['bus', 'truck']):
                position_relation = 'lane_ahead'
            else:
                position_relation = 'lane_behind'
            return (lane_veh_id, joining_veh_id, position_relation)
            
        elif lane2 == target2 and lane1 != target1:
            # veh2 is in target lane, veh1 needs to join
            lane_veh_id = veh2['veh_id']
            joining_veh_id = veh1['veh_id']
            if veh2['position'] - veh1['position'] >= threshold:
                position_relation = 'lane_ahead'
            elif (0 <= veh2['position'] - veh1['position'] < threshold) and not (veh2['veh_type'] in ['bus', 'truck'] and veh1['veh_type'] in ['passenger', 'emergency']):
                position_relation = 'lane_ahead'
            elif -threshold <= veh2['position'] - veh1['position'] < 0 and (veh2['veh_type'] in ['passenger', 'emergency'] and veh1['veh_type'] in ['bus', 'truck']):
                position_relation = 'lane_ahead'
            else:
                position_relation = 'lane_behind'
            return (lane_veh_id, joining_veh_id, position_relation)
        
        return None
    
    def update_platoon(platoon_pairs, lane_veh_id, joining_veh_id, position_relation):
        """ 
        Avoid duplicate platoon entries
        Check if this pair is already recorded in either order
        if so, decide whether current platoon pair is better
        """
        joining_veh = vehicle_states[joining_veh_id]['veh']
        joining_veh_position = joining_veh['position']
        
        lane_veh = vehicle_states[lane_veh_id]['veh']
        lane_veh_position = lane_veh['position']
        
        if current_time == 215 and joining_veh['veh_id'] in ['veh_1_185_0'] and lane_veh['veh_id'] in ['veh_4_192_0', 'veh_1_186_0']:  # for debugging
            print('', end='')  # put a breakpoint here if needed
        
        # absolute difference between the new platoon pair proposal
        abs_dist = abs(lane_veh_position - joining_veh_position)
        
        # The following logic ensures that a joining vehicle only forms a platoon with the closest lane vehicle
        # we need to check `driving_actions.platoon_pairs`
        for i, pair in enumerate(platoon_pairs):
            if joining_veh_id == pair[1]:
                last_lane_veh_id = pair[0]
                last_lane_veh = vehicle_states[last_lane_veh_id]['veh']
                last_lane_veh_position = last_lane_veh['position']
                
                if abs_dist < abs(last_lane_veh_position - joining_veh_position):
                    # if new platoon pair is closer, replace the old one
                    # remove old pair from platoon_pairs
                    platoon_pairs.pop(i)
                    platoon_pairs.append((lane_veh_id, joining_veh_id, position_relation,
                                        vehicle_states[lane_veh_id], vehicle_states[joining_veh_id]))
                    return platoon_pairs
                else:
                    return platoon_pairs # keep the old one
        # if no conflict, just add the new pair
        platoon_pairs.append((lane_veh_id, joining_veh_id, position_relation,
                             vehicle_states[lane_veh_id], vehicle_states[joining_veh_id]))
        return platoon_pairs
                
    
    def calculate_min_offset(leader_veh_type):
        """
        Calculate minimum offset required for safe lane change
        Uses the length of the leader vehicle plus a safety buffer.
            
        Returns:
            Minimum offset distance in meters
        """
        # Use the larger vehicle length plus safety buffer
        return veh_length[leader_veh_type] + OFFSET_BUFFER
    
    # =====MEMORY CLEANUP=====
    
    actions = []
    current_veh_ids = set(veh['veh_id'] for veh in vehicles_info)
    
    driving_actions.last_lane_change = {k: v for k, v in driving_actions.last_lane_change.items() if k in current_veh_ids}
    driving_actions.entry_time = {k: v for k, v in driving_actions.entry_time.items() if k in current_veh_ids}
    driving_actions.target_lane_reached = {k: v for k, v in driving_actions.target_lane_reached.items() if k in current_veh_ids}
    driving_actions.lane_change_intent = {k: v for k, v in driving_actions.lane_change_intent.items() if k in current_veh_ids and current_time - v['time'] < COOPERATION_WINDOW}
    driving_actions.swap_partners = {k: v for k, v in driving_actions.swap_partners.items() if k in current_veh_ids and v in current_veh_ids}
    driving_actions.platoon_pairs = {k: v for k, v in driving_actions.platoon_pairs.items() if k in current_veh_ids and v in current_veh_ids}
    
    # =====FIRST PASS: ANALYZE SITUATION=====
    
    vehicle_states = {}
    swap_pairs = []
    platoon_pairs = []
    
    for veh in vehicles_info:
        veh_id = veh['veh_id']
        
        if traci.simulation.getTime() == 66 and veh_id in ['veh_5_52_0']:  # for debugging
            print('', end='')  # put a breakpoint here if needed
            
        position = veh['position']
        lane_index = int(veh['current_lane'].split('_')[-1])
        
        if veh_id not in driving_actions.entry_time:
            driving_actions.entry_time[veh_id] = current_time
        
        entry_age = current_time - driving_actions.entry_time[veh_id]
        is_entry = position < ENTRY_DISTANCE or entry_age < ENTRY_TIME
        
        target_lane_index = get_target_lane(veh)
        dist_to_end = get_current_lane_length(veh) - position
        
        if dist_to_end > URGENCY_START_DIST:
            urgency = 0.0
        elif dist_to_end > CRITICAL_DIST:
            urgency = (URGENCY_START_DIST - dist_to_end) / (URGENCY_START_DIST - CRITICAL_DIST)
        else:
            urgency = 1.0
        
        vehicle_states[veh_id] = {
            'veh': veh,
            'is_entry': is_entry,
            'target_lane_index': target_lane_index,
            'urgency': urgency,
            'needs_change': lane_index != target_lane_index and not is_entry,
            'swap_role': None,
            'swap_partner': None,
            'platoon_role': None,
            'platoon_partner': None,
            'platoon_relation': None
        }
    
    # -----DETECT SWAP CONFLICTS-----
    
    veh_list = list(vehicle_states.keys())
    for i in range(len(veh_list)):
        for j in range(i + 1, len(veh_list)):
            veh1_id = veh_list[i]
            veh2_id = veh_list[j]
            
            if traci.simulation.getTime() == 417 and (veh1_id in ['veh_3_404_0', 'veh_1_397_0'] and veh2_id in ['veh_3_404_0', 'veh_1_397_0']):  # for debugging
                print('', end='')  # put a breakpoint here if needed
            
            state1 = vehicle_states[veh1_id]
            state2 = vehicle_states[veh2_id]
            
            if state1['needs_change'] and state2['needs_change']:
                if detect_swap_conflict(state1['veh'], state2['veh'], 
                                       state1['target_lane_index'], state2['target_lane_index']):
                    swap_pairs.append((veh1_id, veh2_id, state1, state2))
    
    # -----DETECT PLATOON OPPORTUNITIES-----

    for i in range(len(veh_list)):
        for j in range(i + 1, len(veh_list)):
            veh1_id = veh_list[i]
            veh2_id = veh_list[j]
            
            if traci.simulation.getTime() == 20 and veh1_id in ['veh_6_9_0', 'veh_4_5_0'] and veh2_id in ['veh_6_9_0', 'veh_4_5_0']:  # for debugging
                print('', end='')  # put a breakpoint here if needed
                
            state1 = vehicle_states[veh1_id]
            state2 = vehicle_states[veh2_id]
            
            # Skip if already in swap coordination
            if state1.get('swap_partner') or state2.get('swap_partner'):
                continue
            
            # Check for platoon opportunity
            platoon_result = detect_platoon_opportunity(
                state1['veh'], state2['veh'],
                state1['target_lane_index'], state2['target_lane_index']
            )
            
            if platoon_result:
                lane_veh_id, joining_veh_id, position_relation = platoon_result
                platoon_pairs = update_platoon(platoon_pairs, lane_veh_id, joining_veh_id, position_relation)
                # Avoid duplicate platoon entries
                # Check if this pair is already recorded in either order
                # if so, decide whether current platoon pair is better
    
    
    # -----CHECK SWAP & PLATOON DUPLICATION-----
    # If a vehicle is in swap_pairs, namely this vehicle want to swap lane with another vehicle,
    # at the same time, this vehicle is in platoon_pairs, namely this vehicle want to change lane to join another vehicle to form a platoon,
    # this creates a conflict, we need to resolve it by removing one action
    # this is done by compare the position of the vehicle to swap and the position of the "lane vehicle" in the platoon pair
    # the closer one gets to keep its action, the other one is removed
    
    # Build maps for easier lookup
    swap_map = {}  # {veh_id: (partner_id, state1, state2)}
    for veh1_id, veh2_id, state1, state2 in swap_pairs:
        swap_map[veh1_id] = (veh2_id, state1, state2)
        swap_map[veh2_id] = (veh1_id, state2, state1)
    
    platoon_map = {}  # {veh_id: (partner_id, relation, my_state, partner_state)}
    for lane_veh_id, joining_veh_id, position_relation, lane_state, joining_state in platoon_pairs:
        #platoon_map[lane_veh_id] = (joining_veh_id, position_relation, lane_state, joining_state)
        platoon_map[joining_veh_id] = (lane_veh_id, position_relation, joining_state, lane_state)
    
    # Find vehicles that are in both swap and platoon pairs
    conflicting_vehicles = set(swap_map.keys()) & set(platoon_map.keys())
    
    vehicles_to_remove_from_swap = set()
    vehicles_to_remove_from_platoon = set()
    
    for veh_id in conflicting_vehicles:
        # Get swap partner info
        swap_partner_id, my_swap_state, partner_swap_state = swap_map[veh_id]
        swap_partner_veh = my_swap_state['veh'] if my_swap_state['veh']['veh_id'] == veh_id else partner_swap_state['veh']
        actual_swap_partner_veh = partner_swap_state['veh'] if my_swap_state['veh']['veh_id'] == veh_id else my_swap_state['veh']
        
        # Get platoon partner info
        platoon_partner_id, platoon_relation, my_platoon_state, partner_platoon_state = platoon_map[veh_id]
        platoon_partner_veh = my_platoon_state['veh'] if my_platoon_state['veh']['veh_id'] == veh_id else partner_platoon_state['veh']
        actual_platoon_partner_veh = partner_platoon_state['veh'] if my_platoon_state['veh']['veh_id'] == veh_id else my_platoon_state['veh']
        
        # Get current vehicle info
        current_veh = my_swap_state['veh'] if my_swap_state['veh']['veh_id'] == veh_id else partner_swap_state['veh']
        
        '''
        current_position = current_veh['position']
        # Calculate distances to both partners
        swap_distance = abs(current_position - actual_swap_partner_veh['position'])
        platoon_distance = abs(current_position - actual_platoon_partner_veh['position'])
        
        # Keep the action with the closer partner
        if swap_distance < platoon_distance:
            # Keep swap, remove platoon
            vehicles_to_remove_from_platoon.add(veh_id)
            if platoon_partner_id in conflicting_vehicles:
                vehicles_to_remove_from_platoon.add(platoon_partner_id)
        else:
            # Keep platoon, remove swap
            vehicles_to_remove_from_swap.add(veh_id)
            if swap_partner_id in conflicting_vehicles:
                vehicles_to_remove_from_swap.add(swap_partner_id)
        '''
        # Keep platoon, remove swap
        vehicles_to_remove_from_swap.add(veh_id)
        if swap_partner_id in conflicting_vehicles:
            vehicles_to_remove_from_swap.add(swap_partner_id)
        
    # Remove conflicts from swap_pairs
    if vehicles_to_remove_from_swap:
        swap_pairs_filtered = []
        for veh1_id, veh2_id, state1, state2 in swap_pairs:
            if veh1_id not in vehicles_to_remove_from_swap and veh2_id not in vehicles_to_remove_from_swap:
                swap_pairs_filtered.append((veh1_id, veh2_id, state1, state2))
        swap_pairs = swap_pairs_filtered
    
    # Remove conflicts from platoon_pairs
    if vehicles_to_remove_from_platoon:
        platoon_pairs_filtered = []
        for lane_veh_id, joining_veh_id, position_relation, lane_state, joining_state in platoon_pairs:
            if lane_veh_id not in vehicles_to_remove_from_platoon and joining_veh_id not in vehicles_to_remove_from_platoon:
                platoon_pairs_filtered.append((lane_veh_id, joining_veh_id, position_relation, lane_state, joining_state))
        platoon_pairs = platoon_pairs_filtered
        
    
    # -----ASSIGN SWAP RESOLUTION ROLES-----
    for veh1_id, veh2_id, state1, state2 in swap_pairs:
        veh1 = state1['veh']
        veh2 = state2['veh']
        #speed_diff = veh1['speed'] - veh2['speed']  # not used
        pos_diff = veh1['position'] - veh2['position']
        
        # Decide roles based on positions; the one ahead accelerates, the one behind decelerates
        if pos_diff >= 0:
            # veh1 is ahead and faster or equal speed
            state1['swap_role'] = 'accelerate'
            state2['swap_role'] = 'decelerate'
        else:
            # veh2 is ahead and faster or equal speed
            state2['swap_role'] = 'accelerate'
            state1['swap_role'] = 'decelerate'
        
        state1['swap_partner'] = veh2_id
        state2['swap_partner'] = veh1_id
        driving_actions.swap_partners[veh1_id] = veh2_id
        driving_actions.swap_partners[veh2_id] = veh1_id
    
    # -----ASSIGN PLATOON ROLES-----
    
    for lane_veh_id, joining_veh_id, position_relation, lane_state, joining_state in platoon_pairs:
        # Only form platoon if joining vehicle needs to change lanes and has some urgency
        if joining_state['needs_change'] and joining_state['urgency'] > 0.1:
            lane_state['platoon_role'] = 'lane_vehicle'
            lane_state['platoon_partner'] = joining_veh_id
            lane_state['platoon_relation'] = position_relation
            
            joining_state['platoon_role'] = 'joining_vehicle'
            joining_state['platoon_partner'] = lane_veh_id
            joining_state['platoon_relation'] = position_relation
            
            driving_actions.platoon_pairs[lane_veh_id] = joining_veh_id
            driving_actions.platoon_pairs[joining_veh_id] = lane_veh_id
    
    # =====SECOND PASS: GENERATE ACTIONS=====
    
    # it's interesting to ask how to iterate over vehicles
    # in reality driving decision making is continuous process and simultaneously happening for all vehicles
    # in simulation we have to iterate one by one somehow
    # naturally we want to iterate from upstream to downstream
    # we can 1) iterate over all lanes at the same time, or 2) iterate lane by lane
    # implementation 1 (not used)
    #vehicles_info_sorted = sorted(vehicles_info, key=lambda x: x['position'], reverse=True)
    #for veh in vehicles_info_sorted:

    #  implementation 2
    
    #for veh in vehicles_info:
    for lane in lanes:
        for veh in lanes[lane]:
            veh_id = veh['veh_id']
            
            # -----DEBUGGING BREAKPOINT-----
            # start from here check decision reasoning
            if traci.simulation.getTime() == 226 and veh_id in ['veh_4_213_0']:  # for debugging
                print('', end='')  # put a breakpoint here if needed
            # --------------------------------
                
            veh_type = veh['veh_type']
            position = veh['position']
            speed = veh['speed']
            current_lane = veh['current_lane']
            lane_index = int(current_lane.split('_')[-1])
            my_length = veh_length[veh_type]
            
            state = vehicle_states[veh_id]
            is_entry = state['is_entry']
            target_lane_index = state['target_lane_index']
            urgency = state['urgency']
            swap_role = state['swap_role']
            swap_partner = state['swap_partner']
            platoon_role = state['platoon_role']
            platoon_partner = state['platoon_partner']
            platoon_relation = state['platoon_relation']
            dist_to_end = get_current_lane_length(veh) - position  # distance to the end of the lane
            
            # Find leader in current lane
            current_lane_vehs = lanes[current_lane]
            vehicles_ahead = [v for v in current_lane_vehs if v['position'] > position]
            vehicles_ahead = sorted(vehicles_ahead, key=lambda x: x['position'])
            leader = vehicles_ahead[0] if vehicles_ahead else None
            leader_speed = leader['speed'] if leader else None
            if leader:
                leader_length = veh_length[leader['veh_type']]
                leader_gap = leader['position'] - position - leader_length
            else:
                leader_gap = None
            
            # Base acceleration
            accel = compute_accel(speed, veh_type, leader_speed, leader_gap, edge_info['max_speed'])
            
            # Track target lane reached
            if lane_index == target_lane_index:
                if veh_id not in driving_actions.target_lane_reached:
                    driving_actions.target_lane_reached[veh_id] = {'time': current_time, 'position': position}
            else:
                if veh_id in driving_actions.target_lane_reached:
                    del driving_actions.target_lane_reached[veh_id]
            
            # Lane change decision
            lane_changing = 0
            last_change = driving_actions.last_lane_change.get(veh_id, 0)
            time_since_change = current_time - last_change
            
            # `in_target_recently` variable checks if a vehicle has recently reached its target lane and should therefore stay there for a while
            in_target_recently = veh_id in driving_actions.target_lane_reached and \
                                position - driving_actions.target_lane_reached[veh_id]['position'] < TARGET_LANE_PATIENCE
            
            # -----HANDLE PLATOON COORDINATION (WITH DYNAMIC OFFSET)-----
            if platoon_role is not None and platoon_partner is not None and platoon_relation is not None:
                partner_state = vehicle_states.get(platoon_partner)
                if partner_state:
                    partner_veh = partner_state['veh']
                    partner_veh_type = partner_veh['veh_type']
                    pos_offset = position - partner_veh['position']
                    
                    
                    if platoon_role == 'lane_vehicle':
                        # Vehicle already in target lane
                        if platoon_relation == 'lane_ahead':
                            # Lane vehicle is ahead: ACCELERATE to pull further ahead
                            accel = max_accel[veh_type]
                        else:  # 'lane_behind'
                            # Lane vehicle is behind: DECELERATE to create gap in front
                            accel = max( - PLATOON_DECEL, -max_decel[veh_type])
                    
                    elif platoon_role == 'joining_vehicle':
                        # Vehicle that needs to join the target lane
                        direction = 1 if target_lane_index > lane_index else -1
                        target_lane_id = f"{edge_info['edge_id']}_{lane_index + direction}"
                        
                        if platoon_relation == 'lane_ahead':
                            # Calculate required offset based on vehicle lengths
                            required_offset = calculate_min_offset(partner_veh_type)
                        
                            # Lane vehicle is ahead: DECELERATE to fall behind
                            accel = max(- PLATOON_DECEL, -max_decel[veh_type])
                            
                            # Change lane when offset is sufficient (negative = we're behind)
                            if pos_offset < -required_offset and time_since_change >= LANE_CHANGE_COOLDOWN:
                                if 0 <= lane_index + direction < edge_info['lane_count']:
                                    if is_lane_change_safe(veh, target_lane_id, lanes, urgency):
                                        lane_changing = direction
                                        # Clear platoon after merge
                                        if veh_id in driving_actions.platoon_pairs:
                                            partner = driving_actions.platoon_pairs[veh_id]
                                            if partner in driving_actions.platoon_pairs:
                                                del driving_actions.platoon_pairs[partner]
                                            del driving_actions.platoon_pairs[veh_id]
                        
                        else:  # 'lane_behind'
                            # Calculate required offset based on vehicle lengths
                            required_offset = calculate_min_offset(veh_type)
                            
                            # Lane vehicle is behind: ACCELERATE to get ahead
                            accel = max_accel[veh_type]
                            
                            # Change lane when offset is sufficient (positive = we're ahead)
                            if pos_offset > required_offset and time_since_change >= LANE_CHANGE_COOLDOWN:
                                if 0 <= lane_index + direction < edge_info['lane_count']:
                                    if is_lane_change_safe(veh, target_lane_id, lanes, urgency):
                                        lane_changing = direction
                                        # Clear platoon after merge
                                        if veh_id in driving_actions.platoon_pairs:
                                            partner = driving_actions.platoon_pairs[veh_id]
                                            if partner in driving_actions.platoon_pairs:
                                                del driving_actions.platoon_pairs[partner]
                                            del driving_actions.platoon_pairs[veh_id]
            
            # -----HANDLE SWAP COORDINATION-----
            elif swap_role is not None and swap_partner is not None:
                partner_state = vehicle_states.get(swap_partner)
                if partner_state:
                    partner_veh = partner_state['veh']
                    partner_veh_type = partner_veh['veh_type']
                    partner_veh_length = veh_length[partner_veh_type]
                    partner_speed = partner_veh['speed']
                    pos_offset = position - partner_veh['position']
                    
                    direction = 1 if target_lane_index > lane_index else -1
                    target_lane_id = f"{edge_info['edge_id']}_{lane_index + direction}"
                    
                    if swap_role == 'accelerate':
                        accel = min(SWAP_COORDINATION_ACCEL, max_accel[veh_type])
                        
                        # Calculate required offset based on vehicle lengths
                        required_offset = calculate_min_offset(veh_type)
                                
                        if pos_offset > required_offset and time_since_change >= LANE_CHANGE_COOLDOWN:
                            if 0 <= lane_index + direction < edge_info['lane_count']:
                                my_safe = is_lane_change_safe(veh, target_lane_id, lanes, urgency, required_offset)
                                
                                partner_direction = 1 if partner_state['target_lane_index'] > int(partner_veh['current_lane'].split('_')[-1]) else -1
                                partner_target_lane = f"{edge_info['edge_id']}_{int(partner_veh['current_lane'].split('_')[-1]) + partner_direction}"
                                
                                if my_safe:
                                    lane_changing = direction
                    
                    elif swap_role == 'decelerate':
                        accel = max(- SWAP_COORDINATION_DECEL, -max_decel[veh_type])
                        # Calculate required offset based on vehicle lengths
                        required_offset = calculate_min_offset(partner_veh_type)
                        
                        if pos_offset < -required_offset and speed <= partner_speed and time_since_change >= LANE_CHANGE_COOLDOWN:
                            if 0 <= lane_index + direction < edge_info['lane_count']:
                                my_safe = is_lane_change_safe(veh, target_lane_id, lanes, urgency, required_offset)
                                
                                partner_direction = 1 if partner_state['target_lane_index'] > int(partner_veh['current_lane'].split('_')[-1]) else -1
                                partner_target_lane = f"{edge_info['edge_id']}_{int(partner_veh['current_lane'].split('_')[-1]) + partner_direction}"
                                
                                if my_safe:
                                    lane_changing = direction
                                            
            # -----NORMAL LANE CHANGE LOGIC-----
            elif not in_target_recently and lane_index != target_lane_index and time_since_change >= LANE_CHANGE_COOLDOWN:
                if not is_entry and urgency > 0.05:
                    direction = 1 if target_lane_index > lane_index else -1
                    target_lane_id = f"{edge_info['edge_id']}_{lane_index + direction}"
                    
                    if 0 <= lane_index + direction < edge_info['lane_count']:
                        if is_lane_change_safe(veh, target_lane_id, lanes, urgency):
                            lane_changing = direction
                            driving_actions.lane_change_intent[veh_id] = {
                                'time': current_time,
                                'direction': direction,
                                'target': target_lane_id,
                                'urgency': urgency
                            }
                        else:
                            # Gap creation logic
                            target_vehs = lanes.get(target_lane_id, [])
                            vehicles_ahead = [v for v in target_vehs if v['position'] > position]
                            vehicles_ahead = sorted(vehicles_ahead, key=lambda x: x['position'])
                            target_leader = vehicles_ahead[0] if vehicles_ahead else None
                            target_follower = next((v for v in target_vehs if v['position'] < position), None)
                            
                            if target_leader and target_follower:
                                leader_length = veh_length[target_leader['veh_type']]
                                leader_gap = target_leader['position'] - position - leader_length
                                follower_gap = position - target_follower['position'] - my_length
                                
                                if leader_gap < follower_gap:
                                    if urgency > 0.2:
                                        accel = max( - GAP_SEARCH_DECEL, -max_decel[veh_type] * 0.5)
                                else:
                                    if urgency > 0.2:
                                        accel = min( GAP_SEARCH_ACCEL, max_accel[veh_type])
                            elif target_follower:
                                if urgency > 0.2:
                                    accel = min( GAP_SEARCH_ACCEL, max_accel[veh_type])
                            elif target_leader:
                                if urgency > 0.2:
                                    accel = max( - GAP_SEARCH_DECEL, -max_decel[veh_type] * 0.5)
                            
                            # Speed matching
                            if target_vehs and urgency > 0.3:
                                avg_speed = sum(v['speed'] for v in target_vehs) / len(target_vehs)
                                speed_diff = avg_speed - speed
                                if abs(speed_diff) > 2.0:
                                    accel += speed_diff * SPEED_MATCH_FACTOR
            
            # -----COOPERATIVE BEHAVIOR-----
            # this is the cooperative behavior to help others merge in, even when no platoon or swap is formed
            for other_id, intent in driving_actions.lane_change_intent.items():
                if other_id != veh_id and other_id != swap_partner and other_id != platoon_partner:
                    other_state = vehicle_states.get(other_id)
                    if not other_state:
                        continue
                    
                    other_veh = other_state['veh']
                    other_urgency = intent.get('urgency', 0)
                    
                    if abs(other_veh['position'] - position) < COOPERATION_RANGE:
                        if intent['target'] == current_lane:
                            # other vehicle is behind us and want to merge in
                            if position > other_veh['position'] and position - other_veh['position'] < 60:  
                                if other_urgency > 0.4:
                                    # increase acceleration to create gap
                                    accel = min(accel + 1.2 * other_urgency, max_accel[veh_type])  
                            # Other vehicle is ahead of us and want to merge in
                            elif position < other_veh['position'] and other_veh['position'] - position < 60:  
                                if other_urgency > 0.4:
                                    # decrease acceleration to create gap
                                    accel = max(accel - 1.0 * other_urgency, -max_decel[veh_type] * 0.4)  
                        
                        other_lane_idx = int(other_veh['current_lane'].split('_')[-1])
                        if abs(other_lane_idx - lane_index) == 1:
                            if abs(position - other_veh['position']) < 30 and other_urgency > 0.5:
                                if position > other_veh['position']:
                                    accel = min(accel + 0.8, max_accel[veh_type] * 0.7)
                                else:
                                    accel = max(accel - 0.6, -max_decel[veh_type] * 0.3)
        
                
            # -----FINAL ACCELERATION CONSTRAINTS-----
            accel = max(-max_decel[veh_type], min(max_accel[veh_type], accel))
            if accel > 0:
                accel = max(MIN_ACCEL, accel)
            
            # -----UPDATE MEMORY-----
            if lane_changing != 0:
                driving_actions.last_lane_change[veh_id] = current_time
                
                # Clear swap partner after lane change
                if veh_id in driving_actions.swap_partners:
                    partner = driving_actions.swap_partners[veh_id]
                    if partner in driving_actions.swap_partners:
                        del driving_actions.swap_partners[partner]
                    del driving_actions.swap_partners[veh_id]
            
            # -----APPEND FINAL ACTION-----
            actions.append({
                'veh_id': veh_id,
                'acceleration': accel,
                'lane_changing': lane_changing
            })
    

    # =====CAR-FOLLOWING SAFETY CHECK=====
    # Car-following may not be safe; say, if the leader brakes hard or just not accelerating fast enough
    # (Note: compute_accel function does not consider future leader deceleration)
    # For each lane, start from the furthest vehicle (largest position), check if the following vehicle can maintain safe headway at next time step
    # for headway, we use MIN_GAP and SAFE_TTC
    
    # Process each lane separately
    for lane_id, lane_vehicles in lanes.items():
        if not lane_vehicles:
            continue
        
        # Vehicles are already sorted by position (descending - front to back)
        # Start from the front and check each follower
        for i in range(len(lane_vehicles) - 1):
            leader_veh = lane_vehicles[i]
            follower_veh = lane_vehicles[i + 1]
            
            leader_id = leader_veh['veh_id']
            follower_id = follower_veh['veh_id']
            
            # Get actions for both vehicles
            leader_action = next((a for a in actions if a['veh_id'] == leader_id), None)
            follower_action = next((a for a in actions if a['veh_id'] == follower_id), None)
            
            if not leader_action and not follower_action:
                continue
            
            '''it's interesting to see that no skipping makes it safer
            # Skip if either vehicle is changing lanes (they won't be in the same lane next step)
            if leader_action['lane_changing'] != 0 or follower_action['lane_changing'] != 0:
                continue
            '''
            
            # Get vehicle properties
            leader_type = leader_veh['veh_type']
            follower_type = follower_veh['veh_type']
            leader_length = veh_length[leader_type]
            follower_length = veh_length[follower_type]
            
            # Current state
            leader_pos = leader_veh['position']
            follower_pos = follower_veh['position']
            leader_speed = leader_veh['speed']
            follower_speed = follower_veh['speed']
            
            # Current gap (without leader length)
            current_gap = leader_pos - follower_pos - leader_length
            
            # Get accelerations from actions
            leader_accel = leader_action['acceleration']
            follower_accel = follower_action['acceleration']
            
            # time step is 1 second
            # Predict next positions and speeds (kinematic equations)
            leader_pos_next = leader_pos + leader_speed + 0.5 * leader_accel
            follower_pos_next = follower_pos + follower_speed + 0.5 * follower_accel
            
            # Gap at next time step
            gap_next = leader_pos_next - follower_pos_next - leader_length
            
            if gap_next < MIN_GAP:
                gap_to_add = MIN_GAP - gap_next
                accel_adjustment = gap_to_add * 2
                # Calculate new acceleration (ensure it doesn't exceed max deceleration)
                new_accel = follower_accel - accel_adjustment
                if new_accel >= -max_decel[follower_type]:
                    new_accel = max(new_accel, -max_decel[follower_type])
                    # Update follower's acceleration
                    follower_action['acceleration'] = new_accel
                else:
                    follower_action['acceleration'] = -max_decel[follower_type]
                    leader_accel_needed = (- new_accel) - max_decel[follower_type] - 0.5  # in this case, we allow smaller buffer; namely MIN_GAP decrease by 1 meter
                    leader_action['acceleration'] = leader_accel + leader_accel_needed
                
    
    # =====MERGING SAFETY CHECK=====
    # Check before returning actions
    # When a vehicle is driving straight on a lane, it may not be aware of future merging intentions
    # In that case, it may collide with the merging vehicle at next time step
    # To prevent this, we check if there is any vehicle in adjacent lanes that may merge into our lane soon
    # If so, we adjust our acceleration to maintain a safe distance, namely we're being polite drivers
    
    # Build a map of lane changes from actions
    lane_change_map = {}  # {veh_id: (direction, current_lane, target_lane)}
    for action in actions:
        if action['lane_changing'] != 0:
            veh_id = action['veh_id']
            veh = next(v for v in vehicles_info if v['veh_id'] == veh_id)
            current_lane = veh['current_lane']
            lane_idx = int(current_lane.split('_')[-1])
            target_lane_idx = lane_idx + action['lane_changing']
            target_lane = f"{edge_info['edge_id']}_{target_lane_idx}"
            lane_change_map[veh_id] = (action['lane_changing'], current_lane, target_lane)
    
    # Check each vehicle not changing lanes for potential collision with merging vehicles
    for i, action in enumerate(actions):
        if action['lane_changing'] == 0:  # Vehicle is staying in current lane
            veh_id = action['veh_id']
            veh = next(v for v in vehicles_info if v['veh_id'] == veh_id)
            veh_type = veh['veh_type']
            position = veh['position']
            speed = veh['speed']
            current_lane = veh['current_lane']
            my_length = veh_length[veh_type]
            
            # Check adjacent lanes for vehicles merging into our lane
            lane_idx = int(current_lane.split('_')[-1])
            
            for other_id, (direction, other_current_lane, other_target_lane) in lane_change_map.items():
                if other_target_lane == current_lane:  # Other vehicle is merging into our lane
                    other_veh = next(v for v in vehicles_info if v['veh_id'] == other_id)
                    other_position = other_veh['position']
                    other_speed = other_veh['speed']
                    other_type = other_veh['veh_type']
                    other_length = veh_length[other_type]
                    
                    # Calculate potential conflict zone
                    position_diff = abs(position - other_position)
                    
                    # Check if vehicles are close enough to be a concern
                    if position_diff < 30.0:  # Within 30m range
                        # Determine relative positions after merge
                        if other_position > position:
                            # Other vehicle will be ahead after merging
                            gap = other_position - position - other_length
                            rel_speed = speed - other_speed
                            
                            # Check if we're approaching them
                            if rel_speed > 0 and gap < 20.0:
                                # Calculate required deceleration to maintain safe distance
                                ttc = gap / rel_speed if rel_speed > 0 else float('inf')
                                if ttc < SAFE_TTC:
                                    # Reduce our acceleration to be polite
                                    politeness_decel = min(1.5, rel_speed / SAFE_TTC)
                                    action['acceleration'] = max(
                                        action['acceleration'] - politeness_decel,
                                        -max_decel[veh_type] * 0.5
                                    )
                        
                        else:
                            # Other vehicle will be behind after merging
                            gap = position - other_position - my_length
                            rel_speed = other_speed - speed
                            
                            # Check if they're approaching us from behind
                            if rel_speed > 0 and gap < 20.0:
                                # Speed up slightly to create space
                                ttc = gap / rel_speed if rel_speed > 0 else float('inf')
                                if ttc < SAFE_TTC:
                                    politeness_accel = min(1.0, rel_speed / SAFE_TTC * 0.5)
                                    action['acceleration'] = min(
                                        action['acceleration'] + politeness_accel,
                                        max_accel[veh_type] * 0.8
                                    )

    # =====SIMULTANEOUS MERGING SAFETY CHECK=====
    # Check before returning actions
    # Check if multiple vehicles are trying to merge into the same position
    # If collision is imminent, cancel the merge for the vehicle with lower urgency
    
    # Group vehicles by their target lanes
    merging_by_target = {}  # {target_lane: [(veh_id, position, urgency, action_index)]}
    
    for i, action in enumerate(actions):
        if action['lane_changing'] != 0:
            veh_id = action['veh_id']
            veh = next(v for v in vehicles_info if v['veh_id'] == veh_id)
            position = veh['position']
            urgency = vehicle_states[veh_id]['urgency']
            
            current_lane = veh['current_lane']
            lane_idx = int(current_lane.split('_')[-1])
            target_lane_idx = lane_idx + action['lane_changing']
            target_lane = f"{edge_info['edge_id']}_{target_lane_idx}"
            
            if target_lane not in merging_by_target:
                merging_by_target[target_lane] = []
            
            merging_by_target[target_lane].append((veh_id, position, urgency, i))
    
    # Check each target lane for simultaneous merges
    for target_lane, merging_vehicles in merging_by_target.items():
        if len(merging_vehicles) > 1:
            # Sort by position to check for potential collisions
            merging_vehicles.sort(key=lambda x: x[1])  # Sort by position
            
            # Check each pair of adjacent merging vehicles
            for j in range(len(merging_vehicles) - 1):
                veh1_id, pos1, urgency1, idx1 = merging_vehicles[j]
                veh2_id, pos2, urgency2, idx2 = merging_vehicles[j + 1]
                
                veh1 = next(v for v in vehicles_info if v['veh_id'] == veh1_id)
                veh2 = next(v for v in vehicles_info if v['veh_id'] == veh2_id)
                
                veh1_length = veh_length[veh1['veh_type']]
                veh2_length = veh_length[veh2['veh_type']]
                
                # Calculate gap after both vehicles merge
                gap = pos2 - pos1 - veh2_length
                
                # Calculate relative speed
                rel_speed = veh1['speed'] - veh2['speed']
                
                # Check if collision is imminent
                collision_imminent = False
                
                if gap < OFFSET_BUFFER * 2:  # Gap is too small
                    collision_imminent = True
                elif rel_speed > 0 and gap > 0:  # veh1 is approaching veh2
                    ttc = gap / rel_speed
                    if ttc < SAFE_TTC * 0.5:  # Very short TTC
                        collision_imminent = True
                
                if collision_imminent:
                    # Cancel the merge for the vehicle with lower urgency
                    if urgency1 < urgency2:
                        # Cancel veh1's lane change
                        actions[idx1]['lane_changing'] = 0
                        # Apply gentle deceleration to create more space
                        veh1_type = veh1['veh_type']
                        actions[idx1]['acceleration'] = max(
                            actions[idx1]['acceleration'] - 1.0,
                            -max_decel[veh1_type] * 0.4
                        )
                    elif urgency2 < urgency1:
                        # Cancel veh2's lane change
                        actions[idx2]['lane_changing'] = 0
                        # Apply gentle deceleration to create more space
                        veh2_type = veh2['veh_type']
                        actions[idx2]['acceleration'] = max(
                            actions[idx2]['acceleration'] - 1.0,
                            -max_decel[veh2_type] * 0.4
                        )
                    else:
                        # Equal urgency - cancel the one behind (veh1)
                        actions[idx1]['lane_changing'] = 0
                        veh1_type = veh1['veh_type']
                        actions[idx1]['acceleration'] = max(
                            actions[idx1]['acceleration'] - 1.0,
                            -max_decel[veh1_type] * 0.4
                        )
        
    # =====RETURN ACTIONS=====
    return actions             
                                