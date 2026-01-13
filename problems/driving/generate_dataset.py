# Generating demand for single intersection
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import numpy as np
import random
import os
from pathlib import Path

# Get just the directory containing the file
current_dir = Path(__file__).parent

# Set simulation parameters
NUM_ROUTES = 7
SIMULATION_STEPS = 600 - 30  # Total number of simulation steps (e.g., 1 hour=3600 steps);
# no generating demand for the last 30 steps, so that all vehicles can go to the destination during the horizon


# Vehicle type configuration
VEHICLE_TYPES = {
    'passenger': {'vClass': 'passenger', 'ratio': 0.7},
    'truck': {'vClass': 'truck', 'ratio': 0.2},
    'bus': {'vClass': 'bus', 'ratio': 0.1},
    'emergency': {'vClass': 'emergency', 'ratio': 0.0}  # Set to 0 if not needed, or adjust as desired
}


def get_random_vehicle_type():
    """Select a random vehicle type based on configured ratios"""
    types = list(VEHICLE_TYPES.keys())
    weights = [VEHICLE_TYPES[vtype]['ratio'] for vtype in types]
    
    # Filter out types with 0 ratio
    filtered_types = [vtype for vtype, weight in zip(types, weights) if weight > 0]
    filtered_weights = [weight for weight in weights if weight > 0]
    
    return random.choices(filtered_types, weights=filtered_weights)[0]


# Function to generate stochastic demand based on the binomial distribution
def generate_vehicle_flows(b=2, p=0.03):
    # Binomial distribution parameters
    # Note: Adjust these based on desired traffic flow characteristics
    binomial_b = [b] * NUM_ROUTES  # Maximum number of arriving vehicles per second (2 to 5)
    binomial_p = [p] * NUM_ROUTES  # Probability for binomial distribution 

    flows = []
    
    for step in range(SIMULATION_STEPS):
        for route_id in range(NUM_ROUTES):
            route_id_str = f"r_{route_id}"
        
            # Generate vehicle arrivals for each second
            # Number of vehicles arriving in this second
            arrivals = np.random.binomial(binomial_b[route_id], binomial_p[route_id])
            
            # Create vehicles with random intervals within the current second
            for i in range(arrivals):
                departure_time = step + random.uniform(0, 1)  # Departure time within the current second
                vehicle_id = f"veh_{route_id}_{step}_{i}"
                vehicle_type = get_random_vehicle_type()
                
                flows.append({
                    "vehicle_id": vehicle_id,
                    "departure_time": departure_time,
                    "route_id": route_id_str,
                    "vehicle_type": vehicle_type
                })
                
    # Sort flows by departure time before returning
    flows.sort(key=lambda x: x["departure_time"])
    return flows


# Function to write flows to a SUMO-compatible XML route file with pretty formatting
def write_to_route_file(output_path, flows):
    root = ET.Element("routes")
    
    # Add vehicle type definitions
    for vtype_id, vtype_config in VEHICLE_TYPES.items():
        if vtype_config['ratio'] > 0:  # Only add types that are actually used
            ET.SubElement(root, "vType", id=vtype_id, vClass=vtype_config['vClass'])
    
    # Define routes for each route ID (uncomment if needed)
    # for route_id in range(NUM_ROUTES):
    #     ET.SubElement(root, "route", id=f"r_{route_id}", edges=f"edge_{route_id}_start edge_{route_id}_end")
    
    # Create vehicles
    for flow in flows:
        ET.SubElement(
            root, "vehicle",
            id=flow["vehicle_id"],
            depart=str(flow["departure_time"]),
            route=flow["route_id"],
            type=flow["vehicle_type"]
        )
    
    # Convert ElementTree to a string
    rough_string = ET.tostring(root, 'utf-8')
    # Use minidom to pretty print the XML
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")
    
    # Save pretty-printed XML to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)
    
    print(f"Generated route file saved at: {output_path}")
    
    # Print statistics about generated vehicle types
    type_counts = {}
    for flow in flows:
        vtype = flow["vehicle_type"]
        type_counts[vtype] = type_counts.get(vtype, 0) + 1
    
    total_vehicles = len(flows)
    print(f"\nGenerated {total_vehicles} vehicles:")
    for vtype, count in type_counts.items():
        percentage = (count / total_vehicles) * 100 if total_vehicles > 0 else 0
        print(f"  {vtype}: {count} vehicles ({percentage:.1f}%)")


if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    random.seed(42)
    np.random.seed(42)
    
    # Case settings
    case_names = ["case_0", "case_1"]
    b_params = [2, 3]
    p_params = [0.03, 0.03]
    
    # Generate data
    for name, b, p in zip(case_names, b_params, p_params):
        # Output path for the generated route file
        output_path = f"{current_dir}/dataset/{name}_demand.rou.xml"
        
        # Generate demand flows using binomial distribution
        flows = generate_vehicle_flows(b, p)
        
        # Write to SUMO route file with pretty formatting
        write_to_route_file(output_path, flows)