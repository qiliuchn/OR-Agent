import traci

def driving_actions(edge_info, vehicles_info):
    """
    Ideas:
        - **Safety First**: Enforce continuous gap checks with a larger buffer.
        - **Necessary Lane Changes**: Coordinate step-by-step with abort capability.
        - **Early Lane Change Priority**: Adjust target lane speeds aggressively.
        - **Gap Waiting**: Require persistent safe gaps with real-time validation.
        - **Fail-Safe Progression**: Abort unsafe changes to maintain flow.
        - **Improved Safety Checks**: Reassess TTC and gaps at each step.
        - **Efficiency Optimization**: Preserve speed with adaptive adjustments.
        - **Gridlock Prevention**: Avoid conflicts via dynamic abort logic.
        - **Collision Reduction**: Eliminate lane change collisions with stricter rules.
        - **Stuck Prevention**: Ensure progression with fallback to original lane.

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

    # Constants
    SAFE_TTC = 10.0
    MIN_GAP_BASE = 22.0
    LANE_CHANGE_COOLDOWN = 5.0
    COOP_ACCEL = 2.0 * 2.5  # Increased to 5.0
    COOP_DECEL = 0.4 * 2.5  # Increased to 1.0
    DECEL_FOR_CHANGE = 0.9
    VEH_LENGTH = 5.0
    OCC_MARGIN = 20.0
    COOP_RANGE = 300.0
    PLATOON_RANGE = 50.0
    URGENCY_DIST = 450.0
    MIN_SPEED_FOR_DECEL = 1.0
    MIN_ACCEL = 1.5
    B_SAFE = 4.0
    POLITENESS = 0.85
    CHANGING_THRESHOLD = 0.35
    STOP_LINE_BUFFER = 20.0
    WAIT_DECEL = -0.6
    MIN_LANE_CHANGE_GAP = 28.0
    MAX_OCCUPANCY_THRESHOLD = 0.7
    ENTRY_BUFFER = 50.0
    ENTRY_DELAY = 5.0
    LEFT_TURN_BOOST_DIST = 50.0
    LANE_CHANGE_BUFFER_BASE = 15.0  # Increased base buffer
    COOP_INTENT_WINDOW = 3.0
    SAFE_GAP_PERSISTENCE = 1.0

    # Memory
    if not hasattr(driving_actions, 'prev_actions'):
        driving_actions.prev_actions = {}
    if 'last_move_time' not in driving_actions.__dict__:
        driving_actions.last_move_time = {}
    if 'entry_time' not in driving_actions.__dict__:
        driving_actions.entry_time = {}
    if 'coop_intent' not in driving_actions.__dict__:
        driving_actions.coop_intent = {}

    current_time = traci.simulation.getTime()

    # Organize vehicles by lane and compute occupancy
    lanes = {f"{edge_info['edge_id']}_{i}": [] for i in range(edge_info['lane_count'])}
    lane_lengths = {}
    lane_occupancy = {f"{edge_info['edge_id']}_{i}": 0.0 for i in range(edge_info['lane_count'])}
    for veh in vehicles_info:
        current_lane = veh['current_lane']
        lanes[current_lane].append(veh)
        for lane_info in veh['potential_target_lanes']:
            if lane_info[0] == current_lane:
                lane_lengths[current_lane] = lane_info[1]
                lane_occupancy[current_lane] = lane_info[2]
                break

    # Sort by position descending
    for lane in lanes:
        lanes[lane].sort(key=lambda x: x['position'], reverse=True)

    def compute_accel(veh_speed, veh_type, leader_speed=None, gap=None, safe_ttc=SAFE_TTC, max_speed=edge_info['max_speed'], near_stop=False, is_entry=False, is_left_turn=False, platoon_leader_accel=None):
        if leader_speed is None or gap is None:
            base_accel = max(MIN_ACCEL, min(max_acceleration_dict[veh_type], max_speed - veh_speed))
            effective_gap = float('inf') if gap is None else gap
            return base_accel * 1.1 if is_entry and effective_gap > MIN_GAP_BASE else base_accel
        dynamic_min_gap = max(MIN_GAP_BASE, 1.5 * leader_speed + 10.0)
        if gap < dynamic_min_gap / 2:
            return -max_deceleration_dict[veh_type]
        relative = veh_speed - leader_speed
        ttc = gap / relative if relative > 0 else float('inf')
        adjusted_ttc = safe_ttc / 4 if near_stop and is_left_turn else safe_ttc / 6 if near_stop else safe_ttc * 0.9 if is_entry else safe_ttc
        if ttc < adjusted_ttc or gap < dynamic_min_gap:
            decel = max(-max_deceleration_dict[veh_type], (leader_speed - veh_speed + (gap - veh_speed * adjusted_ttc)) / adjusted_ttc)
            return max(MIN_ACCEL, decel) if decel < 0 else decel
        if platoon_leader_accel is not None:
            return max(MIN_ACCEL, min(max_acceleration_dict[veh_type], platoon_leader_accel * 0.98))
        base_accel = max(MIN_ACCEL, min(max_acceleration_dict[veh_type], max_speed - veh_speed))
        return base_accel * 1.1 if is_entry and gap > dynamic_min_gap else base_accel

    max_acceleration_dict = {
        "passenger": 2.7,
        "bus": 1.3,
        "truck": 1.4,
        "emergency": 2.7,
    }
    max_deceleration_dict = {
        "passenger": 4.5,
        "bus": 4.0,
        "truck": 4.0,
        "emergency": 4.5,
    }

    # First pass: compute intended actions and detect lane requirements
    intended_actions = {}
    gridlock_clusters = {}
    stop_line = edge_info['length']
    for veh in vehicles_info:
        veh_id = veh['veh_id']
        veh_type = veh['veh_type']
        position = veh['position']
        speed = veh['speed']
        current_lane = veh['current_lane']
        lane_index = int(current_lane.split('_')[-1])
        dist_to_end = lane_lengths.get(current_lane, edge_info['length']) - position
        urgency = max(0, 1 - (dist_to_end / URGENCY_DIST)) if dist_to_end < URGENCY_DIST else 0
        dist_to_stop = stop_line - position
        is_left_turn = lane_index == 2 and veh.get('wants_left', False)
        is_entry = position < ENTRY_BUFFER and (veh_id not in driving_actions.entry_time or current_time - driving_actions.entry_time.get(veh_id, 0) < ENTRY_DELAY)

        # Determine required lane with delayed intent
        required_lane = lane_index
        initial_required = lane_index
        for lane_info in veh['potential_target_lanes']:
            lane_id, _, occupation, offset, allows, _ = lane_info
            target_index = int(lane_id.split('_')[-1])
            if offset == 0 and allows:
                initial_required = target_index
                break
        if veh.get('wants_left', False) and not is_entry:
            required_lane = 2
        elif veh.get('wants_right', False) and not is_entry:
            required_lane = 0
        else:
            required_lane = 0 if lane_index == 1 or (position / edge_info['length']) % 2 < 1 else 1
        if lane_index == initial_required and dist_to_stop > STOP_LINE_BUFFER / 2:
            required_lane = lane_index

        # Adjust urgency, delay for entry
        if lane_occupancy[current_lane] > MAX_OCCUPANCY_THRESHOLD and lane_index != required_lane and not is_entry:
            urgency = max(urgency, 0.4)
        if is_left_turn and dist_to_stop < LEFT_TURN_BOOST_DIST and not is_entry:
            urgency = max(urgency, 0.6)

        effective_safe_ttc = max(3.0, SAFE_TTC * (1 - 0.5 * urgency))

        leader = None
        leader_dist = float('inf')
        for other_veh in lanes[current_lane]:
            if other_veh['position'] > position and other_veh['position'] - position < leader_dist:
                leader_dist = other_veh['position'] - position
                leader = other_veh

        leader_speed = leader['speed'] if leader else None
        leader_gap = leader['position'] - position - VEH_LENGTH if leader else None
        dynamic_min_gap = max(MIN_GAP_BASE, 1.5 * (leader_speed if leader else speed) + 10.0)
        if dist_to_stop <= STOP_LINE_BUFFER:
            dynamic_min_gap = max(12.0, dynamic_min_gap * 0.9)

        # Platoon coordination
        platoon_leader_accel = None
        if leader and leader_dist < PLATOON_RANGE and abs(speed - leader['speed']) < 1.0:
            lead_action = intended_actions.get(leader['veh_id'], {}).get('acceleration', 0.0)
            platoon_leader_accel = lead_action

        accel = compute_accel(speed, veh_type, leader_speed, leader_gap, effective_safe_ttc, near_stop=(dist_to_stop <= STOP_LINE_BUFFER), is_entry=is_entry, is_left_turn=is_left_turn, platoon_leader_accel=platoon_leader_accel)

        lane_changing = 0
        necessary = False
        direction = 0
        if lane_index != required_lane and dist_to_stop > STOP_LINE_BUFFER and not is_entry:
            direction = 1 if required_lane > lane_index else -1
            necessary = True

            target_lane_index = lane_index + direction
            if 0 <= target_lane_index < edge_info['lane_count']:
                target_lane_id = f"{edge_info['edge_id']}_{target_lane_index}"
                for lane_info in veh['potential_target_lanes']:
                    if lane_info[0] == target_lane_id:
                        if lane_info[4] or target_lane_index == required_lane:
                            target_lane_vehs = lanes.get(target_lane_id, [])
                            new_leader = None
                            new_leader_dist = float('inf')
                            for other in target_lane_vehs:
                                if other['position'] > position and other['position'] - position < new_leader_dist:
                                    new_leader_dist = other['position'] - position
                                    new_leader = other

                            new_follower = None
                            new_follower_dist = float('inf')
                            for other in target_lane_vehs:
                                if other['position'] < position and position - other['position'] < new_follower_dist:
                                    new_follower_dist = position - other['position']
                                    new_follower = other

                            # Preemptive cooperative lane change coordination
                            safe_leader = True
                            if new_leader:
                                speed_diff_l = abs(speed - new_leader['speed'])
                                lane_change_buffer = LANE_CHANGE_BUFFER_BASE + 0.15 * speed_diff_l
                                gap_l = new_leader['position'] - position - VEH_LENGTH
                                rel_s_l = speed - new_leader['speed']
                                ttc_l = gap_l / rel_s_l if rel_s_l > 0 else float('inf')
                                if gap_l < MIN_LANE_CHANGE_GAP + lane_change_buffer or ttc_l < SAFE_TTC:
                                    safe_leader = False
                            safe_follower = True
                            if new_follower:
                                speed_diff_f = abs(speed - new_follower['speed'])
                                lane_change_buffer = LANE_CHANGE_BUFFER_BASE + 0.15 * speed_diff_f
                                gap_f = position - new_follower['position'] - VEH_LENGTH
                                rel_s_f = new_follower['speed'] - speed
                                ttc_f = gap_f / rel_s_f if rel_s_f > 0 else float('inf')
                                if gap_f < MIN_LANE_CHANGE_GAP + lane_change_buffer or ttc_f < SAFE_TTC:
                                    safe_follower = False

                            ttc_safe = safe_leader and safe_follower

                            if ttc_safe:
                                mobil_safe = True
                                if new_follower:
                                    tilde_a_n = compute_accel(new_follower['speed'], new_follower['veh_type'], speed, position - new_follower['position'] - VEH_LENGTH, effective_safe_ttc)
                                    if tilde_a_n < -B_SAFE:
                                        mobil_safe = False

                                if mobil_safe:
                                    a_c = accel
                                    new_leader_speed = new_leader['speed'] if new_leader else None
                                    new_leader_gap = new_leader['position'] - position - VEH_LENGTH if new_leader else None
                                    tilde_a_c = compute_accel(speed, veh_type, new_leader_speed, new_leader_gap, effective_safe_ttc)

                                    delta_a_self = tilde_a_c - a_c

                                    delta_a_new = 0.0
                                    if new_follower:
                                        new_foll_leader = None
                                        new_foll_leader_dist = float('inf')
                                        for other in target_lane_vehs:
                                            if other['position'] > new_follower['position'] and other['position'] - new_follower['position'] < new_foll_leader_dist:
                                                new_foll_leader_dist = other['position'] - new_follower['position']
                                                new_foll_leader = other
                                        a_n_leader_speed = new_foll_leader['speed'] if new_foll_leader else None
                                        a_n_gap = new_foll_leader['position'] - new_follower['position'] - VEH_LENGTH if new_foll_leader else None
                                        a_n = compute_accel(new_follower['speed'], new_follower['veh_type'], a_n_leader_speed, a_n_gap, effective_safe_ttc)

                                        tilde_a_n = compute_accel(new_follower['speed'], new_follower['veh_type'], speed, position - new_follower['position'] - VEH_LENGTH, effective_safe_ttc)

                                        delta_a_new = tilde_a_n - a_n

                                    delta_a_old = 0.0
                                    if leader:
                                        a_o_gap = position - leader['position'] - VEH_LENGTH
                                        a_o = compute_accel(leader['speed'], leader['veh_type'], speed, a_o_gap, effective_safe_ttc)

                                        tilde_o_leader_speed = new_leader['speed'] if new_leader else None
                                        tilde_o_gap = new_leader['position'] - leader['position'] - VEH_LENGTH if new_leader else None
                                        tilde_a_o = compute_accel(leader['speed'], leader['veh_type'], tilde_o_leader_speed, tilde_o_gap, effective_safe_ttc)

                                        delta_a_old = tilde_a_o - a_o

                                    incentive = delta_a_self + POLITENESS * (delta_a_new + delta_a_old)
                                    effective_threshold = CHANGING_THRESHOLD - urgency * 0.2

                                    if incentive > effective_threshold:
                                        # Signal intent with preemptive coordination
                                        driving_actions.coop_intent[veh_id] = {'time': current_time, 'direction': direction, 'position': position, 'lane': current_lane, 'target_lane': target_lane_id, 'safe_since': None}

        pos_key = round(position, 1)
        if speed < 0.01:
            if pos_key not in gridlock_clusters:
                gridlock_clusters[pos_key] = []
            gridlock_clusters[pos_key].append((veh_id, lane_index, direction, necessary))

        # Track entry time
        if position < ENTRY_BUFFER and veh_id not in driving_actions.entry_time:
            driving_actions.entry_time[veh_id] = current_time

        intended_actions[veh_id] = {'acceleration': accel, 'lane_changing': lane_changing, 'urgency': urgency, 'direction': direction, 'necessary': necessary, 'required_lane': required_lane, 'last_change_time': driving_actions.prev_actions.get(veh_id, {}).get('last_change_time', 0), 'is_left_turn': is_left_turn, 'is_entry': is_entry}

    # Second pass: apply lane enforcement, cooperation, and platoon coordination
    actions = []
    processed_clusters = set()
    for veh in vehicles_info:
        veh_id = veh['veh_id']
        veh_type = veh['veh_type']
        position = veh['position']
        speed = veh['speed']
        current_lane = veh['current_lane']
        lane_index = int(current_lane.split('_')[-1])
        dist_to_stop = edge_info['length'] - position

        action = intended_actions[veh_id].copy()
        accel = action['acceleration']
        lane_changing = action['lane_changing']
        urgency = action['urgency']
        direction = action['direction']
        necessary = action['necessary']
        required_lane = action['required_lane']
        last_change_time = action['last_change_time']
        is_left_turn = action['is_left_turn']
        is_entry = action['is_entry']

        # Cooperative lane change adjustment
        if lane_changing != 0 and veh_id in driving_actions.coop_intent:
            intent = driving_actions.coop_intent[veh_id]
            if current_time - intent['time'] < COOP_INTENT_WINDOW:
                target_lane_id = intent['target_lane']
                target_lane_vehs = lanes.get(target_lane_id, [])
                new_leader = next((v for v in target_lane_vehs if v['position'] > position), None)
                new_follower = next((v for v in target_lane_vehs if v['position'] < position), None)

                # Continuous target lane coordination
                if new_leader:
                    speed_diff_l = abs(speed - new_leader['speed'])
                    lane_change_buffer = LANE_CHANGE_BUFFER_BASE + 0.15 * speed_diff_l
                    leader_gap = new_leader['position'] - position - VEH_LENGTH
                    lead_action = intended_actions.get(new_leader['veh_id'], {})
                    rel_s_l = speed - new_leader['speed']
                    ttc_l = leader_gap / rel_s_l if rel_s_l > 0 else float('inf')
                    if leader_gap < MIN_LANE_CHANGE_GAP + lane_change_buffer or ttc_l < SAFE_TTC or not intent.get('safe_since'):
                        accel_new = max(lead_action.get('acceleration', 0) - COOP_ACCEL, MIN_ACCEL) if lead_action and lead_action.get('acceleration', 0) > 0 else lead_action.get('acceleration', 0)
                        intended_actions[new_leader['veh_id']] = {'acceleration': accel_new, 'lane_changing': 0, 'urgency': lead_action.get('urgency', 0), 'direction': 0, 'necessary': False, 'required_lane': lane_index + direction, 'last_change_time': lead_action.get('last_change_time', 0), 'is_left_turn': False, 'is_entry': False}
                if new_follower:
                    speed_diff_f = abs(speed - new_follower['speed'])
                    lane_change_buffer = LANE_CHANGE_BUFFER_BASE + 0.15 * speed_diff_f
                    follower_gap = position - new_follower['position'] - VEH_LENGTH
                    foll_action = intended_actions.get(new_follower['veh_id'], {})
                    rel_s_f = new_follower['speed'] - speed
                    ttc_f = follower_gap / rel_s_f if rel_s_f > 0 else float('inf')
                    if follower_gap < MIN_LANE_CHANGE_GAP + lane_change_buffer or ttc_f < SAFE_TTC or not intent.get('safe_since'):
                        accel_new = min(foll_action.get('acceleration', 0) + COOP_DECEL, -MIN_ACCEL) if foll_action and foll_action.get('acceleration', 0) < 0 else foll_action.get('acceleration', 0)
                        intended_actions[new_follower['veh_id']] = {'acceleration': accel_new, 'lane_changing': 0, 'urgency': foll_action.get('urgency', 0), 'direction': 0, 'necessary': False, 'required_lane': lane_index + direction, 'last_change_time': foll_action.get('last_change_time', 0), 'is_left_turn': False, 'is_entry': False}

                # Confirm persistent safe gap with abort
                leader_gap = (new_leader['position'] - position - VEH_LENGTH) if new_leader else float('inf')
                follower_gap = (position - new_follower['position'] - VEH_LENGTH) if new_follower else float('inf')
                rel_s_l = speed - (new_leader['speed'] if new_leader else speed)
                rel_s_f = (new_follower['speed'] if new_follower else speed) - speed
                ttc_l = leader_gap / rel_s_l if rel_s_l > 0 else float('inf')
                ttc_f = follower_gap / rel_s_f if rel_s_f > 0 else float('inf')
                speed_diff_l = abs(speed - (new_leader['speed'] if new_leader else speed))
                speed_diff_f = abs(speed - (new_follower['speed'] if new_follower else speed))
                lane_change_buffer_l = LANE_CHANGE_BUFFER_BASE + 0.15 * speed_diff_l
                lane_change_buffer_f = LANE_CHANGE_BUFFER_BASE + 0.15 * speed_diff_f

                safe_now = (not new_leader or (leader_gap >= MIN_LANE_CHANGE_GAP + lane_change_buffer_l and ttc_l >= SAFE_TTC)) and \
                           (not new_follower or (follower_gap >= MIN_LANE_CHANGE_GAP + lane_change_buffer_f and ttc_f >= SAFE_TTC))
                if safe_now:
                    if intent.get('safe_since') is None:
                        intent['safe_since'] = current_time
                    elif current_time - intent['safe_since'] >= SAFE_GAP_PERSISTENCE:
                        if current_time - last_change_time >= LANE_CHANGE_COOLDOWN:
                            lane_changing = direction
                            accel = max(MIN_ACCEL, accel + COOP_ACCEL * urgency)
                        else:
                            lane_changing = 0
                else:
                    intent['safe_since'] = None
                    lane_changing = 0

        # Platoon coordination with enhanced safety
        if (is_left_turn and lane_index == 2) or not is_entry:
            leaders = [v for v in lanes[current_lane] if v['position'] > position and v['position'] - position < COOP_RANGE]
            if leaders:
                lead_veh = max(leaders, key=lambda x: x['position'])
                lead_action = intended_actions.get(lead_veh['veh_id'], {})
                lead_gap = lead_veh['position'] - position - VEH_LENGTH
                dynamic_min_gap = max(MIN_GAP_BASE, 1.5 * lead_veh['speed'] + 10.0)
                if lead_gap < dynamic_min_gap:
                    accel = min(-0.15, accel)
                elif lead_action and lead_action['acceleration'] > MIN_ACCEL:
                    accel = max(accel, lead_action['acceleration'] * 0.97)

        if dist_to_stop > STOP_LINE_BUFFER and lane_index != required_lane and necessary and not is_entry:
            if current_time - last_change_time >= LANE_CHANGE_COOLDOWN:
                lane_changing = direction
                accel = max(MIN_ACCEL, accel + COOP_ACCEL * urgency)
            else:
                lane_changing = 0
        elif dist_to_stop <= STOP_LINE_BUFFER and lane_index != required_lane:
            lane_changing = direction
            accel = min(WAIT_DECEL, accel)
        else:
            lane_changing = 0

        pos_key = round(position, 1)
        in_gridlock = pos_key in gridlock_clusters and any(v[0] == veh_id for v in gridlock_clusters[pos_key])
        if in_gridlock and pos_key not in processed_clusters and dist_to_stop > STOP_LINE_BUFFER:
            cluster = gridlock_clusters[pos_key]
            deadlock_vehs = [(vid, lidx, dir, nec) for vid, lidx, dir, nec in cluster if any(v['veh_id'] == vid and v['speed'] < 0.01 for v in vehicles_info)]
            if deadlock_vehs and len(deadlock_vehs) > 1:
                priority_veh = max(deadlock_vehs, key=lambda x: (x[1], x[3]), default=None)
                if priority_veh and priority_veh[0] == veh_id:
                    target_lane_id = f"{edge_info['edge_id']}_{lane_index + priority_veh[2]}"
                    target_lane_vehs = lanes.get(target_lane_id, [])
                    new_leader = next((v for v in target_lane_vehs if v['position'] > position), None)
                    gap_l = (new_leader['position'] - position - VEH_LENGTH) if new_leader else float('inf')
                    if gap_l >= MIN_LANE_CHANGE_GAP and current_time - last_change_time >= LANE_CHANGE_COOLDOWN:
                        lane_changing = priority_veh[2]
                        accel = max(MIN_ACCEL, accel + 1.5)
                else:
                    lane_changing = 0
                    accel = min(WAIT_DECEL, accel - COOP_DECEL)
                processed_clusters.add(pos_key)
                driving_actions.last_move_time[veh_id] = current_time

        for other_veh in vehicles_info:
            if other_veh['veh_id'] != veh_id and abs(other_veh['position'] - position) < COOP_RANGE:
                other_urgency = intended_actions[other_veh['veh_id']]['urgency']
                if other_veh['current_lane'] == current_lane:
                    if other_veh['position'] > position:
                        accel -= COOP_DECEL * other_urgency
                    else:
                        accel += COOP_ACCEL * other_urgency * (1.5 if is_entry else 1.0)
                elif in_gridlock and any(t[0] == other_veh['veh_id'] for t in deadlock_vehs):
                    target_index = lane_index + lane_changing if lane_changing else lane_index
                    other_lane_index = int(other_veh['current_lane'].split('_')[-1])
                    if abs(target_index - other_lane_index) == 1:
                        if other_veh['position'] > position:
                            accel -= COOP_DECEL * other_urgency * 2
                        else:
                            accel += COOP_ACCEL * other_urgency * 2

        accel = max(MIN_ACCEL, max(-max_deceleration_dict[veh_type], min(max_acceleration_dict[veh_type], accel)))
        actions.append({'veh_id': veh_id, 'acceleration': accel, 'lane_changing': lane_changing, 'last_change_time': current_time if lane_changing != 0 else last_change_time})

    # Clean up expired intents
    driving_actions.coop_intent = {k: v for k, v in driving_actions.coop_intent.items() if current_time - v['time'] < COOP_INTENT_WINDOW}
    current_veh_ids = set(veh['veh_id'] for veh in vehicles_info)
    driving_actions.prev_actions = {k: v for k, v in [(a['veh_id'], {'last_change_time': a['last_change_time']}) for a in actions] if k in current_veh_ids}
    driving_actions.last_move_time = {k: v for k, v in driving_actions.last_move_time.items() if k in current_veh_ids}
    driving_actions.entry_time = {k: v for k, v in driving_actions.entry_time.items() if k in current_veh_ids}

    return actions