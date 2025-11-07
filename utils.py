#
# file:   utils.py
# desc:   Provides generic utilities for reading/writing files of points.
#         Note: this implementation is taken from Benjamin Piro's assignment 2 submission.
#         It has been adapted here for use in assignment 3.
# author: Benjamin Piro (brp8396@rit.edu)
# date:   8 October 2025
#

import os, sys
from typing import Optional, List, Tuple, Dict
import matplotlib.pyplot as plt

#
# Read/write points from/to file utilities
#

def read_file(file_path: str) -> tuple[int, tuple, list[tuple]]:
    """
    @desc: Reads an input `.txt` file containing line segment endpoints.
        This function may behave unpredictably if the proper format is not used.
        Expected format:
            Line 1: Number of line segments (integer)
            Line 2: Bounding box as "x_min y_min x_max y_max" (lower-left and upper-right corners)
            Lines 3+: Line segments as "x1 y1 x2 y2"

    @type file_path: str
    @param file_path: The full path of the file being read.

    @rtype: tuple[int, tuple, list[tuple]]
    @returns: A tuple containing:
        - num_points: The number of line segments (integer)
        - bounding_box: (x_min, y_min, x_max, y_max) representing the bounding box (lower-left, upper-right)
        - points: A list of (x1, y1, x2, y2) point tuples, representing line segments
    """
    file_basename: str = os.path.basename(file_path)
    if len(file_basename) < 4 or file_path[-4:] != ".txt":
        raise ValueError("Invalid file type, expected `.txt`.")
    try:
        # read the file
        points: list[tuple[float]] = []
        with open(file_path, 'r') as f:
            # first line should be an integer
            num_points: int = int(f.readline().strip())

            # second line should be the bounding box: x_min y_min x_max y_max (lower-left, upper-right)
            bbox_vec: list[float] = [float(c) for c in f.readline().strip().split()]
            if len(bbox_vec) != 4:
                raise ValueError(
                    "Invalid bounding box format. Expected 4 values " \
                    f"(x_min y_min x_max y_max), but got {len(bbox_vec)} instead."
                )
            bounding_box: tuple[float, float, float, float] = tuple(bbox_vec)

            # read line segments
            for i in range(num_points):
                vec: list[float] = [float(c) for c in f.readline().strip().split()]
                val_check: int = len(vec)
                if val_check != 4:
                    raise ValueError(
                        "Invalid format. Expected 4 values, " \
                        f"but got {val_check} at {file_basename}:{i+3} instead."
                    )
                # order points by x coordinate
                x1, y1, x2, y2 = vec
                if x2 < x1 or (x1 == x2 and y2 < y1):
                    vec = [x2, y2, x1, y1]
                points.append(tuple(vec))
            np_check = len(points)
            if np_check != num_points:
                raise ValueError(
                    "Read an unexpected number of points. " \
                    f"Expected {num_points}, but got {np_check} instead."
                )
        return num_points, bounding_box, points
    except Exception as e:
        print(f"Failed to parse file at {file_path}", file=sys.stderr)
        raise e


def write_file(intersections, path: str) -> None:
    """
    Write intersections to a `.txt` file
    """
    raise NotImplementedError("write_file function is not yet implemented.")


#################
#   Utilities   #
#################

EPS = 1e-12     # used for handling rounding errors

# === Geometries ===

class Point:
    def __init__(self, x, y, label=""):
        self.x = x
        self.y = y
        self.label = label

    def __repr__(self):
        return f"{self.label}({self.x}, {self.y})"
    
    def __eq__(self, other):
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9
    

class Segment:
    def __init__(self, p1, p2, label=""):
        # Ensure left point has smaller x (or smaller y if vertical)
        if p1.x < p2.x or (p1.x == p2.x and p1.y < p2.y):
            self.left = p1
            self.right = p2
        else:
            self.left = p2
            self.right = p1
        self.label = label
    
    @property
    def xmin(self):
        return self.left.x
    
    @property
    def xmax(self):
        return self.right.x
    
    def y_at(self, x: float) -> float:
        diff = self.xmax - self.xmin
        t = 0.0 if abs(diff) < EPS else (x - self.xmin) / (diff)
        return self.left.y + t * (self.right.y - self.left.y)

    def __repr__(self):
        return f"{self.label}[{self.left} -> {self.right}]"


class Trapezoid:
    def __init__(self, top: Optional[Segment], bottom: Optional[Segment], leftx: float, rightx: float):
        self.label = None
        # Trapezoid bounds
        self.top = top
        self.bottom = bottom
        self.leftx = leftx
        self.rightx = rightx
        # Neighbor pointers - now lists to support multiple neighbors
        self.left_neighbors: List[Trapezoid] = []
        self.right_neighbors: List[Trapezoid] = []
        # Back-pointer to DAG leaf
        self.leaf: Optional[Leaf] = None
    
    def contains_x(self, x: float) -> bool:
        return self.leftx - EPS <= x <= self.rightx + EPS
    
    def y_top(self, x: float, y_top_boundary: float) -> float:
        return self.top.y_at(x) if self.top is not None else y_top_boundary
    
    def y_bottom(self, x: float, y_bottom_boundary: float) -> float:
        return self.bottom.y_at(x) if self.bottom is not None else y_bottom_boundary


# === DAG Structures ===

class Node:
    # Exists purely as a parent class to XNode and YNode
    pass


class XNode(Node):
    def __init__(self, x: float, left: Node, right: Node):
        self.x = x
        self.left = left
        self.right = right


class YNode(Node):
    def __init__(self, seg: Segment, above: Node, below: Node):
        self.seg = seg
        self.above = above
        self.below = below


class Leaf(Node):
    def __init__(self, trap: Trapezoid):
        self.trap = trap
        trap.leaf = self    # set trapezoid back-pointer



def construct_segments(segments_data):
    # deduplicate points so identical (x, y) reuse the same point object
    point_map: Dict[Tuple[float, float], Point] = {}

    def get_point(x: float, y: float, label: str) -> Point:
        key = (x, y)
        p = point_map.get(key)
        if p is None:
            p = Point(x, y, label)
            point_map[key] = p
        return p
    
    labeled_segments: List[Segment] = []
    for i, (x1, y1, x2, y2) in enumerate(segments_data, start=1):
        # Assign P{i} to the left endpoint and Q{i} to the right endpoint by x
        if (x1 < x2) or (abs(x1 - x2) < EPS and y1 <= y2):
            p_left = get_point(x1, y1, f"P{i}")
            p_right = get_point(x2, y2, f"Q{i}")
        else:
            p_left = get_point(x2, y2, f"P{i}")
            p_right = get_point(x1, y1, f"Q{i}")
        s = Segment(p_left, p_right, label=f"S{i}")
        labeled_segments.append(s)
    
    return labeled_segments


def visualize_input_segments(segments, bbox):
    min_x, min_y, max_x, max_y = bbox
    _, ax = plt.subplots()
    ax.set_title("Input Line Segments")
    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(min_y - 1, max_y + 1)
    ax.set_aspect('equal', adjustable='box')
    # Draw bounding box
    bbox_rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                              linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(bbox_rect)
    # Draw segments
    for seg in segments:
        lx, ly, rx, ry = seg
        ax.plot([lx, rx], [ly, ry], marker='o')
    plt.grid(True)
    plt.show()