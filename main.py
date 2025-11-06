import argparse
import os, sys
import utils
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List


########################################
#   Trapezoidal Map & Point Location   #
########################################


class TrapezoidalMap:
    def __init__(self, bbox: Tuple[float, float, float, float]):
        x_min, _, x_max, _ = bbox
        self.bbox = bbox
        # init bbox to be the first trapezoid
        t0 = utils.Trapezoid(top=None, bottom=None, leftx=x_min, rightx=x_max)
        self.root: utils.Node = utils.Leaf(t0)
        self.trapezoids: List[utils.Trapezoid] = [t0]
    
    @classmethod
    def from_segments(cls, bbox: Tuple, segments: List[Tuple], seed: Optional[float]) -> "TrapezoidalMap":
        # (1) get segments from raw data
        labeled_segments: List[utils.Segments] = utils.construct_segments(segments)
        # (2) randomize the segments (in-place)
        rng = np.random.default_rng(seed)
        rng.shuffle(labeled_segments)
        # (3) initialize trapezoidal map
        trap_map = cls(bbox)
        # (4) iteratively add segments to the trapezoidal map
        for seg in labeled_segments:
            trap_map._insert_segment(seg)
        return trap_map
        
    def _insert_segment(self, s: utils.Segment):
        raise NotImplementedError()


#####################
#   Orchestration   #
#####################

def parse_input_file():
    parser = argparse.ArgumentParser(description="CSCI-716 Assn3 Trapezoidal Maps & Planar Point Location")
    parser.add_argument("file", type=str, help="Input file containing the planar subdivision data")
    parser.add_argument("-o", "--output", type=str, default="./out/out.txt", help="Output file to save planar subdivision matrix")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed for segment insertion order")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize the segments and trapezoidal map using matplotlib")

    args = parser.parse_args()

    if args.seed is None:
        # if None, use a time-based seed
        args.seed = int(time.time_ns() % (2**32 - 1))

    return args

def main():
    args = parse_input_file()
    input_path = args.file
    output_path = args.output
    seed = args.seed
    visualize = args.visualize

    print("="*60)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Random seed: {seed}")
    print("="*60)

    # (1) Read input segments from file:
    try:
        num_segments, bbox, segments_data = utils.read_file(input_path)
        print(f"\nRead {num_segments} segments from {input_path}")
        min_x, min_y, max_x, max_y = bbox
        print(f"Bounding box: ({min_x}, {min_y}) to ({max_x}, {max_y})")
        print()
    except Exception as e:
        print(f"Error reading input file:", e, file=sys.stderr)
        sys.exit(1)
    
    # visualization of segments in bounding box using matplotlib
    # if visualize:
    #     utils.visualize_input_segments(segments, bbox)


    # (2) Build trapezoidal map (uses randomized-incremental algorithm)
    trap_map = TrapezoidalMap.from_segments(bbox, segments_data, seed=seed)
    



if __name__ == "__main__":
    main()
