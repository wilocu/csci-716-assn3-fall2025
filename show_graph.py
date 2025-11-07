"""
Standalone script to visualize trapezoidal map without entering query mode.
"""
import utils
from main import TrapezoidalMap
from visualize import visualize_trapezoidal_map

# Read and build map
num_segments, bbox, segments_data = utils.read_file('./input/benjamin.txt')
trap_map = TrapezoidalMap.from_segments(bbox, segments_data, seed=None, randomize=False)

print(f"Constructed trapezoidal map with {len(trap_map.trapezoids)} trapezoids")
print("Displaying visualization...")

# Show visualization
visualize_trapezoidal_map(trap_map, "Trapezoidal Map - benjamin.txt")
