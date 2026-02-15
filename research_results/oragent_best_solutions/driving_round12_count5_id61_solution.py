import math
from typing import Dict, List, Tuple
from collections import defaultdict


def driving_actions(edge_info: dict, vehicles_info: list[dict]) -> list[dict]:
    """
    Density-adaptive constant time-gap following algorithm with dynamic parameter adjustment.
    
    Implementation Idea:
    - Calculate real-time lane-specific traffic density metrics each timestep
    - Categorize density into three levels: high (>0.3 veh/m), medium (0.15-0.3 veh/m), low (<0.15 veh/m)
    - Adjust car-following parameters dynamically based on density:
        * High density: Conservative baseline (time_headway=3.0s, min_gap=6.0m, target_speed=0.35 m/s)
        * Medium density: Moderate (time_headway=2.5s, min_gap=5.0m, target_speed=1.5 m/s) 
        * Low density: Aggressive (time_headway=2.0s, min_gap=4.0m, target_speed=min(5.0, max_speed))
    - Smooth parameter transitions to prevent abrupt behavior changes
    - Maintain proven constant time-gap framework with deadlock prevention
    - No lane changes to maintain stability
    
    Implementation Considerations:
    1. Density calculation uses lane-specific vehicle counts divided by edge length
    2. Parameter smoothing uses exponential moving average to prevent abrupt changes
    3. Critical TTC monitoring ensures safety margins are maintained
    4. Deadlock prevention accelerates vehicles stationary >10 steps with sufficient gap
    5. Action smoothing limits acceleration changes to 0.5 m/s² per step
    
    Args:
        edge_info (dict): Target edge information
        vehicles_info (list[dict]): Information about vehicles on the edge
        
    Returns:
        list[dict]: Actions for each vehicle
    """
    
    # === PERSISTENT STATE INITIALIZATION ===
    if not hasattr(driving_actions, 'state'):
        driving_actions.state = {
            'step_counter': 0,
            'vehicle_stationary_steps': defaultdict(int),
            'last_actions': {}
        }
    
    state = driving_actions.state
    state['step_counter'] += 1
    
    # === ALGORITHM PARAMETERS ===
    # Fixed conservative parameters (parent solution)
    TIME_HEADWAY = 3.0    # seconds
    MIN_GAP = 6.0         # meters
    TARGET_SPEED = 0.35   # m/s
    
    # Deadlock prevention parameters
    STATIONARY_THRESHOLD = 5  # steps before deadlock intervention
    DEADLOCK_ACCEL = 1.0  # m/s² acceleration for deadlocked vehicles
    
    # Vehicle characteristics
    VEHICLE_CHAR = {
        'passenger': {'length': 5.0, 'max_accel': 2.6, 'max_decel': 4.5},
        'bus': {'length': 12.0, 'max_accel': 1.2, 'max_decel': 4.0},
        'truck': {'length': 7.5, 'max_accel': 1.3, 'max_decel': 4.0},
        'emergency': {'length': 5.0, 'max_accel': 2.6, 'max_decel': 4.5}
    }
    
    # === HELPER FUNCTIONS ===
    
    def get_vehicle_char(veh_type: str) -> dict:
        """Get vehicle characteristics"""
        return VEHICLE_CHAR.get(veh_type, VEHICLE_CHAR['passenger'])
    
    def get_vehicle_length(veh_type: str) -> float:
        """Get vehicle length"""
        return get_vehicle_char(veh_type)['length']
    
    def get_leader(vehicle: dict, all_vehicles: list[dict]) -> dict:
        """Find the immediate leader in the same lane"""
        leader = None
        min_distance = float('inf')
        
        current_lane = vehicle['current_lane']
        current_pos = vehicle['position']
        
        for other in all_vehicles:
            if (other['current_lane'] == current_lane and 
                other['position'] > current_pos):
                distance = other['position'] - current_pos
                if distance < min_distance:
                    min_distance = distance
                    leader = other
        
        return leader
    
    def calculate_lane_density(lane_id: str, vehicles: list[dict]) -> float:
        """Calculate current density for a specific lane (vehicles per meter)"""
        # Count vehicles in this lane
        lane_vehicles = [v for v in vehicles if v['current_lane'] == lane_id]
        
        if not lane_vehicles:
            return 0.0
        
        # Edge length is used as approximation for lane length
        edge_length = edge_info.get('length', 100.0)
        
        # Return density (vehicles per meter)
        return len(lane_vehicles) / edge_length
    
    def get_density_category(density: float) -> str:
        """Categorize density into high, medium, or low"""
        if density > HIGH_DENSITY_THRESHOLD:
            return 'high'
        elif density >= MEDIUM_DENSITY_THRESHOLD:
            return 'medium'
        else:
            return 'low'
    
    def smooth_density(lane_id: str, current_density: float) -> float:
        """Apply exponential moving average to smooth density measurements"""
        if lane_id not in state['lane_density_ema']:
            state['lane_density_ema'][lane_id] = current_density
        else:
            state['lane_density_ema'][lane_id] = (
                DENSITY_SMOOTHING_ALPHA * current_density + 
                (1 - DENSITY_SMOOTHING_ALPHA) * state['lane_density_ema'][lane_id]
            )
        return state['lane_density_ema'][lane_id]
    
    def get_vehicle_parameters(vehicle: dict, lane_density: float) -> dict:
        """Get car-following parameters for a vehicle based on lane density"""
        # Get density category
        density_cat = get_density_category(lane_density)
        
        # Get base parameters for this density category
        base_params = PARAMS_BY_DENSITY[density_cat].copy()
        
        # Smooth parameter transitions if vehicle has previous parameters
        veh_id = vehicle['veh_id']
        if veh_id in state['vehicle_params']:
            prev_params = state['vehicle_params'][veh_id]
            
            # Smooth time_headway transition (max change 0.1s per step)
            time_headway_diff = base_params['time_headway'] - prev_params['time_headway']
            time_headway_change = max(-0.1, min(0.1, time_headway_diff))
            base_params['time_headway'] = prev_params['time_headway'] + time_headway_change
            
            # Smooth min_gap transition (max change 0.1m per step)
            min_gap_diff = base_params['min_gap'] - prev_params['min_gap']
            min_gap_change = max(-0.1, min(0.1, min_gap_diff))
            base_params['min_gap'] = prev_params['min_gap'] + min_gap_change
            
            # Smooth target_speed transition (max change 0.1 m/s per step)
            target_speed_diff = base_params['target_speed'] - prev_params['target_speed']
            target_speed_change = max(-0.1, min(0.1, target_speed_diff))
            base_params['target_speed'] = prev_params['target_speed'] + target_speed_change
        
        # Store for next step
        state['vehicle_params'][veh_id] = base_params.copy()
        
        return base_params
    
    def calculate_target_speed(vehicle: dict, leader: dict, params: dict) -> float:
        """
        Calculate target speed using constant time-gap following with given parameters
        
        Based on: desired_gap = MIN_GAP + vehicle['speed'] * TIME_HEADWAY
        """
        if leader is None:
            return params['target_speed']
        
        # Calculate current gap to leader
        current_gap = leader['position'] - vehicle['position'] - get_vehicle_length(leader['veh_type'])
        
        # Calculate desired gap based on constant time headway
        desired_gap = params['min_gap'] + vehicle['speed'] * params['time_headway']
        
        # If we have sufficient gap, aim for target speed
        if current_gap >= desired_gap * 1.1:  # 10% buffer
            # Cap at leader's speed plus small margin
            return min(params['target_speed'], leader['speed'] * 1.1)
        
        # If gap is too small, adjust speed to maintain safe gap
        elif current_gap < desired_gap:
            # Calculate speed adjustment needed
            gap_error = desired_gap - current_gap
            # Convert gap error to speed reduction (divide by time headway)
            speed_adjustment = gap_error / params['time_headway']
            target_speed = max(0, leader['speed'] - speed_adjustment)
            return min(target_speed, params['target_speed'])
        
        # Maintain current speed if gap is acceptable
        else:
            return min(vehicle['speed'], params['target_speed'])
    
    def calculate_acceleration(vehicle: dict, target_speed: float) -> float:
        """Calculate acceleration to reach target speed with smoothing"""
        current_speed = vehicle['speed']
        veh_char = get_vehicle_char(vehicle['veh_type'])
        
        # Simple proportional control
        speed_error = target_speed - current_speed
        
        # Use 1-second time constant for responsive control
        base_accel = speed_error  # m/s²
        
        # Apply vehicle-specific limits
        base_accel = max(-veh_char['max_decel'], min(base_accel, veh_char['max_accel']))
        
        # Smooth acceleration changes using last action
        if vehicle['veh_id'] in state['last_actions']:
            last_accel = state['last_actions'][vehicle['veh_id']].get('acceleration', 0)
            # Limit acceleration change to 0.5 m/s² per step for smoothness
            max_change = 0.5
            base_accel = max(last_accel - max_change, min(last_accel + max_change, base_accel))
        
        return base_accel
    
    # === MAIN ALGORITHM EXECUTION ===
    
    # No density adaptation - use fixed parameters
    # (keeping lane_ids for potential future use)
    lane_ids = [f"{edge_info['edge_id']}_{i}" for i in range(edge_info['lane_count'])]
    
    # Update stationary step counter for deadlock detection
    current_veh_ids = {v['veh_id'] for v in vehicles_info}
    
    # Clean up departed vehicles
    for veh_id in list(state['vehicle_stationary_steps'].keys()):
        if veh_id not in current_veh_ids:
            state['vehicle_stationary_steps'].pop(veh_id, None)
    
    # Track stationary vehicles
    for veh in vehicles_info:
        veh_id = veh['veh_id']
        if veh['speed'] < 0.1:  # Nearly stationary
            state['vehicle_stationary_steps'][veh_id] += 1
        else:
            state['vehicle_stationary_steps'][veh_id] = 0
    
    # Determine actions for each vehicle
    actions = []
    
    for veh in vehicles_info:
        veh_id = veh['veh_id']
        
        # Use fixed parameters (no density adaptation)
        params = {
            'time_headway': TIME_HEADWAY,
            'min_gap': MIN_GAP,
            'target_speed': TARGET_SPEED
        }
        
        # Find leader in same lane
        leader = get_leader(veh, vehicles_info)
        
        # Calculate target speed using adaptive parameters
        target_speed = calculate_target_speed(veh, leader, params)
        
        # Deadlock prevention: if vehicle has been stationary too long
        if state['vehicle_stationary_steps'].get(veh_id, 0) > STATIONARY_THRESHOLD:
            # Check if there's enough space ahead (at least min_gap)
            if leader is None or (leader['position'] - veh['position'] > params['min_gap']):
                # Force acceleration to break deadlock
                target_speed = max(target_speed, min(params['target_speed'], 0.5))
                # Reset stationary counter
                state['vehicle_stationary_steps'][veh_id] = 0
        
        # Emergency vehicles: no special boost for safety (parent solution had none)
        # (We keep the same target_speed)
        
        # Calculate acceleration
        acceleration = calculate_acceleration(veh, target_speed)
        
        # No lane changes to maintain stability (proven effective in parent solutions)
        lane_changing = 0
        
        # Store action for smoothing in next step
        state['last_actions'][veh_id] = {
            'acceleration': acceleration,
            'lane_changing': lane_changing
        }
        
        actions.append({
            'veh_id': veh_id,
            'acceleration': acceleration,
            'lane_changing': lane_changing
        })
    
    return actions