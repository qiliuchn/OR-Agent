"""
# Evaluation script for driving problem
The function `driving_actions` implements the driving logic; it will be invoked for each simulation step. 
Eval script will be invoked by evaluator.run() method.
Pass argument to this script as command line arguments.
Eval results are collected by stdout string.
"""
import os
import sys
import traceback
import inspect
import json
import argparse
from datetime import datetime
import traci 
import xml.etree.ElementTree as ET
from typing import Dict, Mapping
import seed_solution2 as solution_module  # Note: solution module script is generated and saved on the fly


# =====Load function to evolve=====
problem = "driving"
driving_actions = getattr(solution_module, "driving_actions")  # Get function to evolve


# =====Helper functions=====
# Type aliases
MetricsPerTest = Mapping[str, Dict]  # map from test case (str) identifier to eval metric (a dict)
ScoresPerTest = Mapping[str, float]  # map from test case identifier to score; this defines ScoresPerTest as a mapping from any test case (str) to a float (the score for that test).
Signature = tuple[int, ...] # Defines Signature as a tuple of ints of any length. It is used to represent a canonical "signature" for a solution.

# Function to convert test metrics to a signature. Different versions are provided below. 
# Choose according to your needs. The choice maybe problem dependent!
def test_metrics_to_scores(metrics_per_test: MetricsPerTest) -> ScoresPerTest:
    ''' 
    Converts a mapping of test metrics to a mapping of test scores for driving problem. 
    
    Args:
        metrics_per_test (Dict): a dict that map test name (str) to metrics (Dict);
            Test metric Example:
            {
                'critical_ttc_count': 28, 
                'collisions': 0, 
                'emergencyStops': 0, 
                'emergencyBraking': 4, 
                'teleports': 0, 
                'avg_fuel_consumption': 8.32, 
                'avg_speed': 12.51, 
                'speed_variance': 16.22
            }
    
    Returns:
        scores_per_test (Dict): a dict that map test name (str) to score (float).

    Notes on `metrics_per_test`:
        Higher scores are better. For driving solution fitness, we prioritizes:
        1. Safety (no collisions, minimal emergency events)
        2. Efficiency (good speed)
        3. Smoothness (low speed variance)
        There aspects are modeled by safety_score, speed_score and smoothness_score respectively.
        safety_score/speed_score/smoothness_score are all in range [0, 100].
        The final score is weighted average of these three metrics.
    '''
    scores_per_test = {}
    
    for test_id, metrics in metrics_per_test.items():
        # Extract metrics with defaults
        critical_ttc = metrics.get('critical_ttc_count', 0)
        collisions = metrics.get('collisions', 0)
        emergency_stops = metrics.get('emergencyStops', 0)
        emergency_braking = metrics.get('emergencyBraking', 0)
        teleports = metrics.get('teleports', 0)
        fuel_consumption = metrics.get('avg_fuel_consumption', 0)
        avg_speed = metrics.get('avg_speed', 0)
        speed_variance = metrics.get('speed_variance', 0)
        
        # Safety Score - Most Important
        # Severe penalties for critical safety events
        safety_score = 100.0
        safety_score -= collisions * 50          # -50 per collision
        safety_score -= emergency_stops * 5     # -5 per emergency stop
        safety_score -= teleports * 30          # -30 per teleport (simulation failure)
        safety_score -= emergency_braking * 2   # -2 per emergency brake
        safety_score -= critical_ttc * 0.5      # -0.5 per critical time-to-collision
        #safety_score = max(0, safety_score)     # Ensure non-negative
        
        # Speed Efficiency Score (0-100)
        # Optimal speed range: 30-50 km/h for urban driving
        # You may need to adjust these ranges based on your simulation
        target_speed = 13.89  # m/s
        speed_deviation = abs(avg_speed - target_speed)
        if speed_deviation <= 1.5:  # close to target speed
            speed_score = 100
        elif speed_deviation <= 3.5:  # still feel good
            speed_score = 100 - (speed_deviation - 1.5) * 5
        else:
            speed_score = max(0, 90 - (speed_deviation - 3.5) * 8.662)  # reduced to zero at zero speed
        
        # Smoothness Score
        # Lower speed variance indicates smoother driving
        if speed_variance <= 5.0:
            smoothness_score = 100
        elif speed_variance <= 10.0:
            smoothness_score = 100 - (speed_variance - 5.0) * 4
        else:
            smoothness_score = 80 - (speed_variance - 10.0) * 8  # reduced to zero at 20.0
            smoothness_score = max(0, 80 - (speed_variance - 10.0) * 8)  # reduced to zero at 20.0
        
        # Weighted Combined Score
        # Safety is most important, followed by efficiency and smoothness
        weights = {
            'safety': 0.5,      # Most critical
            'speed': 0.3,       # Speed efficiency
            'smoothness': 0.2  # Driving smoothness
        }
        
        combined_score = (
            weights['safety'] * safety_score +
            weights['speed'] * speed_score +
            weights['smoothness'] * smoothness_score
        )
        
        scores_per_test[test_id] = combined_score
    
    return scores_per_test
    

def reduce_score(scores_per_test: ScoresPerTest) -> float:
    """
    Reduces per-test scores into a single score, which is used as the overall score for a solution for driving problem.
    
    Args:
        scores_per_test (Dict): a dict that map test name (str) to score (float).
        
    Returns:
        score (float): a single float score.


    Notes on `score`:
    We need find a way to reduce test results (may contain multiple values or even string) of multiple test cases into a single score,
    so that we can use it as the overall score (the "fitness") for a solution in the evolutionary process.
    
    Cf. Funsearch, paper, p3:
    "The scores across different inputs are then combined into an overall score of the
    program using an aggregation function, such as the mean."
    FunSearch implementation use the score of the last test case instead of the mean.
    Here we use the mean instead; users can need customize the reduce function according to their needs.
    """
    #return scores_per_test[list(scores_per_test.keys())[-1]]  # from FunSearch implementation
    score = sum(scores_per_test.values()) / len(scores_per_test)  # here use mean instead
    return score 
    
    
def average_test_metrics(metrics_per_test: MetricsPerTest) -> dict[str, float]:
    """ 
    Aggregates metrics per test into a single metrics dict.
    
    Args:
        metrics_per_test (MetricsPerTest): A dictionary where keys are test identifiers and values are dictionaries of metrics.
        
    Returns:
        averaged_metrics (dict[str, float]): A dictionary where keys are metric name and values are averaged metrics.
    """
    if not metrics_per_test:
        return {}
    
    # Collect all unique metric names
    all_metric_names = set()
    for test_metrics in metrics_per_test.values():
        all_metric_names.update(test_metrics.keys())
    
    # Calculate averages for each metric
    averaged_metrics = {}
    for metric_name in all_metric_names:
        values = []
        for test_metrics in metrics_per_test.values():
            if metric_name in test_metrics:
                # Convert to float to handle numeric values
                try:
                    values.append(float(test_metrics[metric_name]))
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    continue
        
        if values:  # Only calculate average if we have valid numeric values
            averaged_metrics[metric_name] = sum(values) / len(values)
    
    return averaged_metrics
    

def get_feature(metrics_per_test: MetricsPerTest) -> Signature:
    """ 
    Get the feature vector for a algorithm.
    This signature is used for clustering the algorithms. Each island will have several clusters of algorithms.
    
    **Behavioral Signature ** (Recommended) is used.
    MAP-Elites will be used. So, signature is the `feature` vector.
    
    Args:
        test_metrics (MetricsPerTest): A mapping of test metrics to a signature.
    
    Returns:
        Signature: (safety_level, speed_efficiency, fuel_efficiency, traffic_smoothness) on 0-3 scales.
        - Safety level: How safe the algorithm is
        - Speed Efficiency: How fast the avg speed is
        - Fuel Efficiency: Fuel usage
        - Traffic smoothness: Predictability
        Level 0: terrible; 1: bad; 2: good; 3: perfect
        This creates interpretable clusters like "Safe-Conservative-Efficient" vs "Risky-Aggressive-Fast".
        Total number of combinations decides the potential number of clusters in an island. Here 4^3 = 128.
    """
    # Aggregate metrics across all tests
    avg_test_metrics = average_test_metrics(metrics_per_test)
    
    # Safety Level: Higher = Safer
    if avg_test_metrics['collisions'] >= 1.0 or avg_test_metrics['teleports'] >= 1.0:
        safety_level = 0
    elif avg_test_metrics['emergencyStops'] >= 3.0:
        safety_level = 1
    elif avg_test_metrics['emergencyBraking'] >=  5.0 or avg_test_metrics['critical_ttc_count'] >= 50.0:
        safety_level = 2
    else:
        safety_level = 3
    
    # Speed efficiency (not too slow, not too fast is best)
    if 40 / 3.6 <= avg_test_metrics['avg_speed'] <= 50 / 3.6:
        speed_efficiency = 3
    elif 30 / 3.6 <= avg_test_metrics['avg_speed'] <= 40 / 3.6 or \
        50 / 3.6 <= avg_test_metrics['avg_speed'] <= 55 / 3.6:
        speed_efficiency = 2
    elif 15 / 3.6 <= avg_test_metrics['avg_speed'] <= 30 / 3.6:
        speed_efficiency = 1
    else:
        speed_efficiency = 0
    
    # Traffic smoothness (0-4): Higher = More smooth
    if avg_test_metrics['speed_variance'] <= 5:
        traffic_smoothness = 3
    elif avg_test_metrics['speed_variance'] <= 10:
        traffic_smoothness  = 2
    elif avg_test_metrics['speed_variance'] <= 20:
        traffic_smoothness  = 1
    else:
        traffic_smoothness  = 0
    
    return (safety_level, speed_efficiency, traffic_smoothness)



# =====Evaluation Function=====
def evaluate(root_dir, file_output_prefix=''):
    """
    Evaluate the driving_actions function in the SUMO environment; 
    Each test is done on a single (directional) road segment with id `edge_id`;
    For all test cases, this road segment to analyze starts from coordinate (0, 0), and stretch horizontally to (length, 0).
    
    Args:
        root_dir: root directory of the project
    
    Returns:
        metrics_per_test: a dictionary containing metrics for each test case. Example: 
        {
            'case_1': {
                'critical_ttc_count': 28, 
                'collisions': 0, 
                'emergencyStops': 0, 
                'emergencyBraking': 4, 
                'teleports': 0, 
                'avg_speed': 12.51, 
            }
        }
    """
    current_module = sys.modules[__name__]
    if hasattr(current_module, 'Callbacks'):
        callbacks = getattr(current_module, 'Callbacks')()  # callbacks class definition will be concatenated to eval script at runtime
    else:
        callbacks = None
            
    test_names = ["case_0", "case_1"]  # list of test names; options: case_0 (low demand), case_1 (high demand)
    print("\nTest Cases List:\n\"case_0\": low traffic demand\n\"Case_1\": high traffic demand")
    metrics_per_test = {}  # initialize the return
    
    for test_name in test_names:
        print(f"\nTest case \"{test_name}\" starts...")
        # -----Set up SUMO environment for specified test case-----
        # Note the current directory will be changed by hydra!
        sumo_binary = "sumo" # "sumo-gui" for GUI, or "sumo" for command-line mode
        driving_dataset_dir = root_dir + "/problems/driving/dataset/"  # dataset for the driving problem
        sumo_config = driving_dataset_dir  + f"{test_name}.sumocfg"  # sumo config file
        with open(driving_dataset_dir  + f"{test_name}.json", "r") as f:
            test_case_config = json.load(f)
        # simulation parameters
        step_length = 1.0  # simulation step length (seconds); one step = 1 second in SUMO by default
        max_acceleration = {
            "passenger": 2.6,
            "bus": 1.2,
            "truck": 1.3,
            "emergency": 2.6,
        }  # maximum acceleration (m/s^2)
        max_deceleration = {
            "passenger": 4.5,
            "bus": 4.0,
            "truck": 4.0,
            "emergency": 4.5,
        }  # maximum deceleration (m/s^2)
        max_speed = test_case_config['max_speed']  # maximum speed (m/s); default: 13.89 m/s
        horizon = test_case_config['horizon']  # simulation steps; default: 600 seconds
        
        # start sumo simulation
        if traci.isLoaded():
            traci.close()

        # command for SUMO traci API
        sumo_cmd = [
            sumo_binary,
            "-c", sumo_config,
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--step-length", str(step_length),
            # override <output> section from .sumocfg
            "--statistic-output", file_output_prefix + "stats.xml",  # Note: this is path relative to the working directory
            #"--emission-output",  out_prefix + "emissions.xml",
            "--collision-output", file_output_prefix + "collisions.xml",
            "--device.ssm.probability", "1.0",
            "--device.ssm.file",  os.getcwd() + '/' + file_output_prefix + "ssm.xml" if file_output_prefix else os.getcwd() + '/ssm.xml',  # Note: The SSM output file is being created relative to the SUMO config file's directory if relative path is specified. Use absolute path for ssm.xml.
            "--device.ssm.measures", "TTC",
            "--device.ssm.thresholds", "1.5",
        ]
        traci.start(sumo_cmd, numRetries=100)
        
        
        # -----Create `edge_info` var - holds the information of the road segment to analyze-----
        # Fixed for the test
        # edge_info is a dictionary with keys:
        # - 'edge_id'  # target edge id, e.g. "E_0"
        # - 'lane_count'  # number of lanes
        # - 'length'  # length of road segment to analyze (m)
        # - 'max_speed'  # maximum speed (m/s)
        edge_id = test_case_config['edge_id']
        edge_info = {
            "edge_id": edge_id,
            "lane_count": traci.edge.getLaneNumber(edge_id),   # int
            "length": test_case_config['length'],      # float (meters); Note lane length can be obtained by traci.lane.getLength(lane_id); lane length is not the same as edge length
            "max_speed": test_case_config['max_speed']    # float (m/s); default: 13.89 m/s
        }


        # -----Data for evaluation initialization-----
        running_mean_speed = 0.0
        running_mean_speed_squared = 0.0
        total_speed_samples = 0

        try:
            # -----Run the simulation-----
            while traci.simulation.getTime() < horizon:
                step = traci.simulation.getTime()
                if traci.simulation.getMinExpectedNumber() <= 0:
                    print("No more vehicles in simulation")
                    break
                
                # ---Create `vehicles_info` var - holds information about all vehicles on the road segment to analyze---
                # Updated for each simulation time step
                # vehicle_info is a list of dictionaries; each dictionary represents a vehicle on the road segment to analysis
                # vehicle_info[i] is a dictionary with keys:
                # - 'veh_id'  # e.g. 'veh_5_3_0'
                # - 'veh_type'  # e.g. 'passenger', 'bus', 'truck', 'emergency'
                # - 'position'  # position of the vehicle on the road segment (m)
                # - 'speed'  # speed of the vehicle (m/s)
                # - 'current_lane'  # current lane's id, e.g., "E5_0"; note in SUMO, lane index starts from 0 and counts from rightmost to leftmost
                # - 'wants_left'  # whether the vehicle wants to change lane to the left for the current time step
                # - 'wants_right'  # whether the vehicle wants to change lane to the right for the current time step
                # - 'potential_target_lanes'  # info of potential target lanes for the vehicle
                vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_info['edge_id'])
                vehicles_info = []
                for veh_id in vehicle_ids:
                    # Get various vehicle information
                    position = traci.vehicle.getLanePosition(veh_id)  # meters from start of the current lane
                    speed = traci.vehicle.getSpeed(veh_id)
                    current_lane = traci.vehicle.getLaneID(veh_id) # Lane IDs are formatted as <edge_id>_<lane_index> (e.g., E5_0, E5_1, …); '0' is for the rightmost
                    veh_type = traci.vehicle.getTypeID(veh_id)
                    
                    # Get lane change information
                    lane_changing_left_model, lane_changing_left_traci = traci.vehicle.getLaneChangeState(veh_id, +1)
                    lane_changing_right_model, lane_changing_right_traci = traci.vehicle.getLaneChangeState(veh_id, -1)
                    # the current lane-changing intention of the vehicle
                    wants_left  = bool(lane_changing_left_model  & traci.constants.LCA_LEFT)
                    wants_right = bool(lane_changing_right_model & traci.constants.LCA_RIGHT)
                    # Note: Current lane-changing intention (`wants_left`, `wants_right`) is not guaranteed to be well-judged; you may want to check `potential_target_lanes` (see below) for better decision at each time step.
                    
                    # Get `potential_target_lanes` var - info of potential target lanes for the vehicle 
                    potential_target_lanes = traci.vehicle.getBestLanes(veh_id)
                    # traci.vehicle.getBestLanes() returns a list of tuples, where each tuple contains information about a lane that the vehicle can potentially use from its current position.
                    # The "best lane" in SUMO refers to the lane that is optimal for a vehicle to achieve its routing goals efficiently at free flow condition; 
                    # 'bestLaneOffset' tells you how many lanes away the optimal lane is; usually there is exactly one “best lane” (bestLaneOffset = 0).
                    # Note: traci.vehicle.getBestLanes recommends the "best lane" only under free flow condition; this choice does NOT consider the real-time traffic situation! alternatives lanes may actually be better choices! 
                    #
                    # potential_target_lanes format:
                    # (laneID, length, occupation, bestLaneOffset, allowsContinuation, nextLanes)
                    # - 'laneID': ID of that lane (on the current edge)
                    # - 'length': The length that can be driven without lane change (measured from the start of that lane)
                    # - 'occupation': Forecast “brutto vehicle lengths” on the future lanes (a congestion proxy)
                    # - 'bestLaneOffset': Offset from the "best" lane (e.g., -1 = one lane right of the "best lane", +1 = one lane left of the "best lane", 0 = is the "best lane")
                    # - 'allowsContinuation': Whether this lane allows continuation of the route (boolean)
                    # - 'nextLanes': The list of lanes on the next edge that the vehicle will reach if it stays on this lane
                    # 
                    # Example: for a vehicle traveling on lane E0_1 heading straight through next intersection, traci.vehicle.getBestLanes(veh_id) returns:
                    # [('E0_0', 172.8, 0.0, 1, True, ('E0_0', 'E4_0')),
                    #('E0_1', 209.20000000000002, 22.5, 0, True, ('E0_1', 'E4_1')),
                    #('E0_2', 172.8, 0.0, -1, False, ('E0_2',))]
                    # 
                    # How to find alternative lanes?
                    # The 'allowsContinuation' field being True means that the lane continuing the route (for at least one more edge)
                    # 
                    # How to find traffic situation?
                    # The 'occupation' field provides the congestion level for each lane;
                    # For the last example, lane 'E0_0' has 'occupation' level 0.0, whereas 'E0_1' has 'occupation' level 22.5; this means 'E0_1' is much more congested than lane 'E0_0';
                    # Hence although lane 'E0_1' is the 'best lane' recommended by traci, 'E0_0' may be a better choice considering the traffic situation.
                    
                    vehicles_info.append({
                        "veh_id": veh_id,
                        "veh_type": veh_type,
                        "position": position,
                        "speed": speed,
                        "current_lane": current_lane,
                        "wants_left": wants_left,
                        "wants_right": wants_right,
                        "potential_target_lanes": potential_target_lanes,
                    })
                veh_id_to_state = {veh_info["veh_id"]: veh_info for veh_info in vehicles_info}  # vehicles_info converted to a dict
                
                
                # ---Collect data for evaluation (during simulation run)---
                # Aggregated performance indices are collected at the end of the simulation run
                # Here during each step, we only collect and accumulate speed info
                # Collect speed related data
                current_speeds = [veh_info["speed"] for veh_info in vehicles_info]
                n_vehicles = len(current_speeds)
                if n_vehicles > 0:
                    # Use current_speeds in calculations
                    step_mean_speed = sum(current_speeds) / n_vehicles
                    step_mean_speed_squared = sum(speed * speed for speed in current_speeds) / n_vehicles
                    # Update running means incrementally
                    old_n = total_speed_samples
                    total_speed_samples += n_vehicles
                    # Incremental weighted average
                    running_mean_speed = (running_mean_speed * old_n + step_mean_speed * n_vehicles) / total_speed_samples
                    running_mean_speed_squared = (running_mean_speed_squared * old_n + step_mean_speed_squared * n_vehicles) / total_speed_samples
                
                
                # ---Invoke driving_actions (the function to evolve) to get the next actions of all vehicles---
                actions = driving_actions(edge_info, vehicles_info)
                # actions is a list of dictionaries
                # actions[i] is a dictionary with keys:
                # - 'veh_id'
                # - 'acceleration'  # acceleration of the vehicle (float); positive means acceleration, negative means deceleration, '0' means speed no change
                # - 'lane_changing'  # whether the vehicle wants to change lanes; "0" for no change, "1" for changing to the lane on the left, "-1" for changing to the lane on the right
                
                
                # ---Apply the actions to the simulation---
                for action in actions:
                    veh_id = action["veh_id"]
                    if veh_id not in vehicle_ids:
                        continue  # Don't apply actions to vehicles that are not on the road segment to analysis
                    veh_type = veh_id_to_state[veh_id]["veh_type"]
                    current_speed = veh_id_to_state[veh_id]["speed"]
                    current_lane = veh_id_to_state[veh_id]["current_lane"]
                    if "acceleration" in action:
                        accel = action["acceleration"]
                        # impose acceleration limit and speed limit
                        if accel > 0:
                            accel = min(accel, max_acceleration[veh_type])
                        elif accel < 0:
                            accel = max(accel, -max_deceleration[veh_type])
                        next_speed = max(0.0, min(current_speed + accel * step_length, max_speed))
                        # set speed
                        traci.vehicle.setSpeed(veh_id, next_speed)
                    if "lane_changing" in action:
                        # set lane changing
                        duration = 5.0
                        traci.vehicle.setLaneChangeMode(veh_id, 0b000000000000)  # All checks disabled
                        traci.vehicle.changeLaneRelative(veh_id, action["lane_changing"], duration)
                
                # ---Invoke callbacks---
                event_name = "on_step_end"
                if hasattr(callbacks, event_name):
                    getattr(callbacks, event_name)(step=step, edge_info=edge_info, vehicles_info=vehicles_info)
                
                # ---Advance the simulation by one step---
                traci.simulationStep()
                
        finally:
            # Always close cleanly so SUMO doesn't see an abrupt client disconnect
            try:
                traci.close()
            except Exception:
                pass
            sys.stdout.flush()
        
        
        # -----Collect data for evaluation (after the simulation run)-----
        # We collect the aggregated performance indices at the end of the simulation run
        # Performance metrics collected:
        # - 'collisions': total number of collisions
        # - 'critical_ttc_count': critical TTC Events (≤1.5s)
        # - 'emergency_stops': number of emergency_stops
        # - 'emergency_braking': number of emergency_braking
        # - 'avg_speed': average Speed (m/s)
        # - 'speed_variance': Speed Variance
        # - 'teleports': number of teleport vehicles
        
        # Load aggregated performance metrics from SUMO output files
        performance_metrics = load_performance_metrics_from_sumo_files(file_output_prefix)
        # Add mean speed and speed variance metrics that are collected during simulation steps
        performance_metrics['avg_speed'] = round(running_mean_speed, 2)
        performance_metrics['speed_variance'] = round(running_mean_speed_squared - (running_mean_speed * running_mean_speed), 2)
        
        metrics_per_test[test_name] = performance_metrics
    
    return metrics_per_test
        
        
def load_performance_metrics_from_sumo_files(out_prefix=''):
    """
    Load SUMO output files and extract performance metrics.
    
    Returns:
            dict: Performance metrics loaded from files
    """
    metrics = {}
    
    # Critical TTC Count (TTC ≤ 1.5s) - from ssm.xml
    CRITICAL_TTC_THRESHOLD = 1.5
    try:
        ssm_tree = ET.parse(f"{out_prefix}ssm.xml")
        ssm_root = ssm_tree.getroot()
        critical_ttc_count = 0

        for conflict in ssm_root.findall('conflict'):
            for min_ttc_elem in conflict.findall('minTTC'):
                ttc_value_str = min_ttc_elem.get('value')
                if ttc_value_str:
                    ttc_value = float(ttc_value_str)
                    if ttc_value <= CRITICAL_TTC_THRESHOLD:
                        critical_ttc_count += 1
            
        metrics['critical_ttc_count'] = critical_ttc_count
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Warning: Could not read {out_prefix}ssm.xml: {e}")
    
    # Number of collisions, emergency stops, emergency braking - from stats.xml
    try:
        stats_tree = ET.parse(f"{out_prefix}stats.xml")
        stats_root = stats_tree.getroot()
        safety = stats_root.find("safety")
        if safety is not None:
            metrics["collisions"] = int(float(safety.get("collisions", "0")))
            metrics["emergencyStops"] = int(float(safety.get("emergencyStops", "0")))
            metrics["emergencyBraking"] = int(float(safety.get("emergencyBraking", "0")))

        tele = stats_root.find("teleports")
        if tele is not None:
            metrics["teleports"] = int(float(tele.get("total", "0")))

    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Warning: Could not read {out_prefix}stats.xml: {e}")

    return metrics



# =====Main Function=====
if __name__ == "__main__":
    print("Evaluation script running ...")
    # -----Parse command line arguments (same for all problems)-----
    parser = argparse.ArgumentParser(description='Evaluation script.')
    parser.add_argument(
        '--root_dir',
        type=str,
        default=os.getcwd(),
        help='Project root directory for loading data (default: current working directory)'
    )
    parser.add_argument(
        '--file_output_prefix',
        type=str,
        default='',
        help='Output file prefix for saving evaluation results. '
             'Absolute path recommended. Files saved as {prefix}filename '
             '(default: empty string, saves to current directory)')
    parser.add_argument(
        '--mode',
        type=str,
        default='val',
        choices=['train', 'val'],
        help='Execution mode: train or val (default: val)'
    )
    parser.add_argument(
        '--problem_size',
        type=int,
        default=None,  # Customize this to your needs
        help='Problem size parameter'
    )
    # Parse arguments
    args = parser.parse_args()
    root_dir = args.root_dir
    file_output_prefix = args.file_output_prefix
    mode = args.mode  # Not used in this problem
    problem_size = args.problem_size  # Not used in this problem
    # Print parsed arguments for verification
    print(f"root_dir: {root_dir}")
    print(f"file_output_prefix: {file_output_prefix}")
    #print(f"mode: {mode}")
    #print(f"problem_size: {problem_size}")
    
    try:        
        # -----Run the evaluation-----
        metrics_per_test = evaluate(root_dir, file_output_prefix)
        scores_per_test = test_metrics_to_scores(metrics_per_test)
        print("\nMetrics for all tests:")
        for test_name in metrics_per_test:
            print(f"{test_name}: {str(metrics_per_test[test_name])}")
        print("\nScores for all tests:")
        for test_name in scores_per_test:
            print(f"{test_name}: {str(scores_per_test[test_name])}")
        # Compute aggregate performance indices
        metrics = average_test_metrics(metrics_per_test)
        features = get_feature(metrics_per_test)
        score = reduce_score(scores_per_test)
        print()
        
        # -----Print results to stdout (same for all problems)-----
        print('__SANDBOX_RESULT__')

        print('__METRICS_START__')
        print(repr(metrics))
        print('__METRICS_END__')
        
        print('__FEATURES_START__')
        print(repr(features))
        print('__FEATURES_END__')
        
        print('__SCORE_START__')
        print(repr(score))
        print('__SCORE_END__')
        
        print('__SANDBOX_SUCCESS__')
        
    except Exception as e:
        print('__SANDBOX_ERROR__:')
        print(f'Error type: {type(e).__name__}')
        print(f'Error message: {str(e)}')
        print('Full traceback:')
        traceback.print_exc()