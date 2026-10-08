""" 
The main idea is: create safe merge slots proactively, reserve them, and then validate the entire traffic configuration before applying commands.
1. Collision-aware car following
For every lane, vehicles are sorted from front to back. Each follower receives a safe next speed based on:
- Its current gap to the leader
- The leader's speed
- Vehicle-specific braking capability
- A reaction-time allowance
- The configured minimum following gap
Conceptually:
    next speed = min(
        speed limit,
        acceleration-limited speed,
        braking-safe speed
    )
The braking-safe speed estimates how fast the follower can travel while retaining enough distance to brake safely if the leader slows down.

2. Route-aware lane requirements
For each vehicle, the controller finds the lane required by its route using potential_target_lanes.
Vehicles that are already in the correct lane stay there. Vehicles in the wrong lane become lane-change candidates.
Candidates are prioritized by:
- Urgency: how close they are to the end of their current lane
- Position: downstream vehicles receive priority when urgency is equal
This prevents vehicles close to an exit from being perpetually blocked by less urgent changes.

3. Entry-zone protection
The controller does not initiate lane changes:
- During the first two seconds after a vehicle enters
- Within the first 30 metres of the road
Newly arriving vehicles may not yet be present in vehicles_info when another vehicle makes its decision. The entry buffer prevents changing into a lane containing a vehicle that becomes visible only on the next step.

4. Front-and-rear merge safety
Before approving a lane change, the controller finds the closest leader and follower in the target lane.
A change is allowed only when both sides satisfy:
- Minimum physical gap
- Minimum time-to-collision
- Lane-change cooldown
- Entry-zone restrictions
The base merge gap is 6 metres, with a small urgency adjustment. Urgency can slightly relax spacing, but it cannot bypass TTC checks.

5. Merge reservations
Approved lane changes reserve a longitudinal region in their target lane.
Another vehicle cannot simultaneously merge into approximately the same position. This prevents two vehicles from different source lanes independently deciding that the same target-lane gap is available.

6. Cooperative gap creation
When a lane change is unsafe, the vehicle does not simply stop and wait.
Instead, the controller coordinates the three relevant vehicles:
- The target-lane leader accelerates to enlarge the front gap.
- The target-lane follower yields by slowing down.
- The requesting vehicle accelerates or decelerates toward the emerging slot.
This was important for maintaining speed under high demand. Merely increasing the required merge gap reduced collisions but caused gridlock.

7. Post-coordination synchronization
Several vehicles can request gap creation during the same step. A later request might modify the speed of a vehicle involved in an already-approved merge.
Therefore, after processing all requests, the controller revisits every approved merge and synchronizes the speeds again:
- The merging vehicle does not approach its new leader too quickly.
- Its new follower stays slightly slower during the lateral transition.
This step eliminated the remaining side collision where a merger slowed down after approval while its new follower accelerated.

8. Projected dual-lane safety pass
During the evaluator's five-second lane-change command, a changing vehicle can interact with traffic in both its original lane and its target lane.
The controller therefore builds a projected configuration where every lane-changing vehicle temporarily occupies both lanes. It then performs several front-to-back safety passes, propagating necessary braking through each queue.
This final pass ensures that cooperative adjustments do not accidentally create a rear-end collision elsewhere.

Instead of handling platoons, swaps, normal changes, cooperation, and final safety as several interacting special cases, the new logic uses one consistent pipeline:
safe longitudinal speeds
-> prioritized merge requests
-> gap and TTC validation
-> merge reservations
-> cooperative gap creation
-> merge-speed synchronization
-> projected whole-road safety validation
"""
import traci

def driving_actions(edge_info, vehicles_info) -> list[dict]:
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
    # The centralized controller below computes every speed and lane-change command
    VEH_LENGTH = {
        "passenger": 5.0,
        "bus": 12.0,
        "truck": 7.1,
        "emergency": 6.5,
    }
    MAX_ACCEL = {
        "passenger": 2.6,
        "bus": 1.2,
        "truck": 1.3,
        "emergency": 2.6,
    }
    MAX_DECEL = {
        "passenger": 4.5,
        "bus": 4.0,
        "truck": 4.0,
        "emergency": 4.5,
    }

    FOLLOW_GAP = 2.0
    REACTION_TIME = 0.50
    MERGE_BASE_GAP = 6.0
    MERGE_TTC = 3.5
    MERGE_COOLDOWN = 3.0
    ENTRY_DELAY = 2.0
    ENTRY_MERGE_DISTANCE = 30.0
    RESERVATION_DISTANCE = 16.0
    COOP_BRAKE = 2.0

    current_time = traci.simulation.getTime()
    if not hasattr(driving_actions, 'v2_entry_time'):
        driving_actions.v2_entry_time = {}
    if not hasattr(driving_actions, 'v2_last_lane_change'):
        driving_actions.v2_last_lane_change = {}

    active_ids = {veh['veh_id'] for veh in vehicles_info}
    driving_actions.v2_entry_time = {
        veh_id: entry_time
        for veh_id, entry_time in driving_actions.v2_entry_time.items()
        if veh_id in active_ids
    }
    driving_actions.v2_last_lane_change = {
        veh_id: change_time
        for veh_id, change_time in driving_actions.v2_last_lane_change.items()
        if veh_id in active_ids
    }

    lanes = {
        f"{edge_info['edge_id']}_{lane_idx}": []
        for lane_idx in range(edge_info['lane_count'])
    }
    vehicles_by_id = {}
    for veh in vehicles_info:
        vehicles_by_id[veh['veh_id']] = veh
        lanes[veh['current_lane']].append(veh)
        driving_actions.v2_entry_time.setdefault(veh['veh_id'], current_time)
    for lane_vehs in lanes.values():
        lane_vehs.sort(key=lambda item: item['position'], reverse=True)

    def target_lane_index(veh):
        current_idx = int(veh['current_lane'].split('_')[-1])
        for lane_info in veh['potential_target_lanes']:
            if lane_info[3] == 0 and lane_info[4]:
                return int(lane_info[0].split('_')[-1])
        return current_idx

    def current_lane_length(veh):
        for lane_info in veh['potential_target_lanes']:
            if lane_info[0] == veh['current_lane']:
                return float(lane_info[1])
        return float(edge_info['length'])

    def neighbors(target_lane, position):
        lane_vehs = lanes.get(target_lane, [])
        leaders = [other for other in lane_vehs if other['position'] > position]
        followers = [other for other in lane_vehs if other['position'] < position]
        leader = min(leaders, key=lambda item: item['position'], default=None)
        follower = max(followers, key=lambda item: item['position'], default=None)
        return leader, follower

    # First compute collision-aware longitudinal speeds in current lanes.
    next_speeds = {}
    for lane_vehs in lanes.values():
        for index, veh in enumerate(lane_vehs):
            veh_id = veh['veh_id']
            veh_type = veh['veh_type']
            speed = veh['speed']
            desired_speed = min(
                edge_info['max_speed'],
                speed + MAX_ACCEL[veh_type],
            )
            if index > 0:
                leader = lane_vehs[index - 1]
                gap = (
                    leader['position']
                    - veh['position']
                    - VEH_LENGTH[leader['veh_type']]
                )
                braking_term = (
                    (MAX_DECEL[veh_type] * REACTION_TIME) ** 2
                    + leader['speed'] ** 2
                    + 2 * MAX_DECEL[veh_type] * max(0.0, gap - FOLLOW_GAP)
                )
                safe_speed = max(
                    0.0,
                    -MAX_DECEL[veh_type] * REACTION_TIME
                    + braking_term ** 0.5,
                )
                desired_speed = min(desired_speed, safe_speed)
            next_speeds[veh_id] = max(
                0.0,
                speed - MAX_DECEL[veh_type],
                desired_speed,
            )

    lane_changes = {veh_id: 0 for veh_id in active_ids}
    reservations = {
        lane_id: [] for lane_id in lanes
    }

    # Process urgent vehicles first, then vehicles furthest downstream.
    candidates = []
    for veh in vehicles_info:
        current_idx = int(veh['current_lane'].split('_')[-1])
        target_idx = target_lane_index(veh)
        if current_idx == target_idx:
            continue
        dist_to_end = current_lane_length(veh) - veh['position']
        urgency = max(0.0, min(1.0, 1.0 - dist_to_end / 180.0))
        candidates.append((urgency, veh['position'], veh, target_idx))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    for urgency, _, veh, target_idx in candidates:
        veh_id = veh['veh_id']
        veh_type = veh['veh_type']
        current_idx = int(veh['current_lane'].split('_')[-1])
        if current_time - driving_actions.v2_entry_time[veh_id] < ENTRY_DELAY:
            continue
        if veh['position'] < ENTRY_MERGE_DISTANCE:
            continue
        if current_time - driving_actions.v2_last_lane_change.get(veh_id, -1000.0) < MERGE_COOLDOWN:
            continue

        direction = 1 if target_idx > current_idx else -1
        adjacent_idx = current_idx + direction
        if not 0 <= adjacent_idx < edge_info['lane_count']:
            continue
        target_lane = f"{edge_info['edge_id']}_{adjacent_idx}"
        if any(
            abs(veh['position'] - reserved_position) < RESERVATION_DISTANCE
            for reserved_position in reservations[target_lane]
        ):
            continue

        leader, follower = neighbors(target_lane, veh['position'])
        leader_gap = float('inf')
        leader_ttc = float('inf')
        if leader is not None:
            leader_gap = (
                leader['position']
                - veh['position']
                - VEH_LENGTH[leader['veh_type']]
            )
            closing_speed = veh['speed'] - leader['speed']
            if closing_speed > 0:
                leader_ttc = leader_gap / closing_speed

        follower_gap = float('inf')
        follower_ttc = float('inf')
        if follower is not None:
            follower_gap = (
                veh['position']
                - follower['position']
                - VEH_LENGTH[veh_type]
            )
            closing_speed = follower['speed'] - veh['speed']
            if closing_speed > 0:
                follower_ttc = follower_gap / closing_speed

        merge_gap = MERGE_BASE_GAP - 0.75 * urgency
        safe_to_merge = (
            leader_gap >= merge_gap
            and follower_gap >= merge_gap
            and leader_ttc >= MERGE_TTC
            and follower_ttc >= MERGE_TTC
        )

        if safe_to_merge:
            lane_changes[veh_id] = direction
            reservations[target_lane].append(veh['position'])
            driving_actions.v2_last_lane_change[veh_id] = current_time

            if leader is not None and leader_gap < merge_gap + 5.0:
                next_speeds[veh_id] = min(
                    next_speeds[veh_id],
                    next_speeds[leader['veh_id']],
                )
            if follower is not None and follower_gap < merge_gap + 5.0:
                next_speeds[follower['veh_id']] = min(
                    next_speeds[follower['veh_id']],
                    max(0.0, next_speeds[veh_id] - 0.5),
                )
        else:
            # Create a slot cooperatively rather than stopping the requester.
            if leader is not None and (
                leader_gap < merge_gap or leader_ttc < MERGE_TTC
            ):
                next_speeds[veh_id] = min(
                    next_speeds[veh_id],
                    max(0.0, veh['speed'] - 1.5),
                )
                next_speeds[leader['veh_id']] = min(
                    edge_info['max_speed'],
                    leader['speed'] + MAX_ACCEL[leader['veh_type']],
                )
            if follower is not None and (
                follower_gap < merge_gap or follower_ttc < MERGE_TTC
            ):
                next_speeds[follower['veh_id']] = min(
                    next_speeds[follower['veh_id']],
                    max(0.0, follower['speed'] - COOP_BRAKE),
                )
                if leader_gap >= merge_gap:
                    next_speeds[veh_id] = min(
                        edge_info['max_speed'],
                        veh['speed'] + MAX_ACCEL[veh_type],
                    )

    # Re-synchronize accepted merges after all cooperative adjustments.
    # This prevents later gap-creation requests from making a merger brake
    # while its new follower accelerates into the lateral transition.
    for veh_id, direction in lane_changes.items():
        if direction == 0:
            continue
        veh = vehicles_by_id[veh_id]
        current_idx = int(veh['current_lane'].split('_')[-1])
        target_lane = f"{edge_info['edge_id']}_{current_idx + direction}"
        leader, follower = neighbors(target_lane, veh['position'])
        if leader is not None:
            leader_gap = (
                leader['position']
                - veh['position']
                - VEH_LENGTH[leader['veh_type']]
            )
            if leader_gap < MERGE_BASE_GAP + 6.0:
                next_speeds[veh_id] = min(
                    next_speeds[veh_id],
                    max(0.0, next_speeds[leader['veh_id']] - 0.5),
                )
        if follower is not None:
            follower_gap = (
                veh['position']
                - follower['position']
                - VEH_LENGTH[veh['veh_type']]
            )
            if follower_gap < MERGE_BASE_GAP + 6.0:
                next_speeds[follower['veh_id']] = min(
                    next_speeds[follower['veh_id']],
                    max(0.0, next_speeds[veh_id] - 0.5),
                )

    # Enforce projected one-step gaps while lane changers occupy both source
    # and target lanes. Iterate to propagate braking through each queue.
    projected_lanes = {lane_id: list(lane_vehs) for lane_id, lane_vehs in lanes.items()}
    for veh_id, direction in lane_changes.items():
        if direction == 0:
            continue
        veh = vehicles_by_id[veh_id]
        current_idx = int(veh['current_lane'].split('_')[-1])
        target_lane = f"{edge_info['edge_id']}_{current_idx + direction}"
        projected_lanes[target_lane].append(veh)

    for _ in range(3):
        for lane_vehs in projected_lanes.values():
            ordered = sorted(
                lane_vehs,
                key=lambda item: item['position'],
                reverse=True,
            )
            for index in range(1, len(ordered)):
                leader = ordered[index - 1]
                follower = ordered[index]
                leader_id = leader['veh_id']
                follower_id = follower['veh_id']
                gap = (
                    leader['position']
                    - follower['position']
                    - VEH_LENGTH[leader['veh_type']]
                )
                max_follower_speed = (
                    next_speeds[leader_id] + gap - FOLLOW_GAP
                )
                next_speeds[follower_id] = max(
                    0.0,
                    follower['speed'] - MAX_DECEL[follower['veh_type']],
                    min(next_speeds[follower_id], max_follower_speed),
                )

    actions = []
    for veh in vehicles_info:
        veh_id = veh['veh_id']
        acceleration = next_speeds[veh_id] - veh['speed']
        acceleration = max(
            -MAX_DECEL[veh['veh_type']],
            min(MAX_ACCEL[veh['veh_type']], acceleration),
        )
        actions.append({
            'veh_id': veh_id,
            'acceleration': acceleration,
            'lane_changing': lane_changes[veh_id],
        })
    return actions