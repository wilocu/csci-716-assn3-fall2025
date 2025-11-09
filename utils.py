"""
Utility classes and functions for trapezoidal map construction.
"""
import sys
from typing import Optional, List, Tuple, Dict, Set

EPS = 1e-9  # Epsilon for floating point comparisons


class Point:
    """Represents a 2D point with a label."""

    def __init__(self, x: float, y: float, label: str = ""):
        self.x = x
        self.y = y
        self.label = label

    def __repr__(self):
        return f"{self.label}({self.x}, {self.y})"

    def __eq__(self, other):
        return abs(self.x - other.x) < EPS and abs(self.y - other.y) < EPS


class Segment:
    """Represents a line segment with left and right endpoints."""

    def __init__(self, p1: Point, p2: Point, label: str = ""):
        # Ensure left point has smaller x-coordinate
        if p1.x < p2.x or (abs(p1.x - p2.x) < EPS and p1.y < p2.y):
            self.left = p1
            self.right = p2
        else:
            self.left = p2
            self.right = p1
        self.label = label

    @property
    def xmin(self) -> float:
        return self.left.x

    @property
    def xmax(self) -> float:
        return self.right.x

    def y_at(self, x: float) -> float:
        """Return y-coordinate of segment at given x."""
        if abs(self.xmax - self.xmin) < EPS:
            # Vertical segment
            return self.left.y
        t = (x - self.xmin) / (self.xmax - self.xmin)
        return self.left.y + t * (self.right.y - self.left.y)

    def __repr__(self):
        return f"{self.label}[{self.left} -> {self.right}]"


class Node:
    """Base class for DAG nodes."""
    pass


class XNode(Node):
    """DAG node that splits on x-coordinate (vertical line through a point)."""

    def __init__(self, point: Point, left: Optional[Node] = None, right: Optional[Node] = None):
        self.point = point
        self.left = left   # x < point.x
        self.right = right  # x >= point.x

    @property
    def a(self) -> Node:
        return self.left

    @a.setter
    def a(self, new_a: Node):
        self.left = new_a

    @property
    def b(self) -> Node:
        return self.right

    @b.setter
    def b(self, new_b: Node):
        self.right = new_b

    @property
    def x(self) -> float:
        return self.point.x


class YNode(Node):
    """DAG node that splits on segment (horizontal comparison)."""

    def __init__(self, seg: Segment, above: Optional[Node] = None, below: Optional[Node] = None):
        self.seg = seg
        self.above = above  # y > seg.y_at(x)
        self.below = below  # y <= seg.y_at(x)

    @property
    def a(self) -> Node:
        return self.above

    @a.setter
    def a(self, new_a: Node):
        self.above = new_a

    @property
    def b(self) -> Node:
        return self.below

    @b.setter
    def b(self, new_b: Node):
        self.below = new_b


class Leaf(Node):
    """DAG leaf node containing a trapezoid."""

    def __init__(self, trap: 'Trapezoid', parent: Optional[Node] = None):
        self.parent = parent
        self.trap = trap
        trap.leaves.add(self)  # Back-pointer



class Trapezoid:
    """Represents a trapezoid in the map."""

    def __init__(self, top: Optional[Segment], bottom: Optional[Segment],
                 leftx: float, rightx: float,
                 left_point: Optional[Point] = None,
                 right_point: Optional[Point] = None):
        self.top = top          # Top boundary segment (None = bbox top)
        self.bottom = bottom    # Bottom boundary segment (None = bbox bottom)
        self.leftx = leftx      # Left x-coordinate
        self.rightx = rightx    # Right x-coordinate
        self.left_point = left_point   # Point at left boundary
        self.right_point = right_point # Point at right boundary
        self.label = None       # Will be assigned later
        self.leaves: Set[Leaf] = set()        # Back-pointer to DAG leaves that reference the trapezoid
        self.u_left: Optional[Trapezoid] = None
        self.u_right: Optional[Trapezoid] = None
        self.d_left: Optional[Trapezoid] = None
        self.d_right: Optional[Trapezoid] = None

    @property
    def right_neighbors(self) -> Set["Trapezoid"]:
        return {t for t in (self.u_right, self.d_right) if t is not None}

    @property
    def left_neighbors(self) -> Set["Trapezoid"]:
        return {t for t in (self.u_left, self.d_left) if t is not None}

    def y_top(self, x: float, bbox_y_max: float) -> float:
        """Return y-coordinate of top boundary at given x."""
        return self.top.y_at(x) if self.top else bbox_y_max

    def y_bottom(self, x: float, bbox_y_min: float) -> float:
        """Return y-coordinate of bottom boundary at given x."""
        return self.bottom.y_at(x) if self.bottom else bbox_y_min

    def __repr__(self):
        top_label = self.top.label if self.top else "bbox_top"
        bot_label = self.bottom.label if self.bottom else "bbox_bot"
        return f"{self.label}[{self.leftx:.1f}, {self.rightx:.1f}] top={top_label}, bot={bot_label}"


def read_file(file_path: str) -> Tuple[int, Tuple[float, float, float, float], List[Tuple[float, float, float, float]]]:
    """
    Read input file containing line segments.

    Expected format:
        Line 1: Number of segments (integer)
        Line 2: Bounding box as "x_min y_min x_max y_max"
        Lines 3+: Segments as "x1 y1 x2 y2"

    Returns:
        (num_segments, bbox, segments_data)
    """
    try:
        with open(file_path, 'r') as f:
            num_segments = int(f.readline().strip())

            bbox_values = [float(x) for x in f.readline().strip().split()]
            if len(bbox_values) != 4:
                raise ValueError(f"Expected 4 bbox values, got {len(bbox_values)}")
            bbox = tuple(bbox_values)

            segments = []
            for i in range(num_segments):
                values = [float(x) for x in f.readline().strip().split()]
                if len(values) != 4:
                    raise ValueError(f"Expected 4 values for segment {i+1}, got {len(values)}")
                segments.append(tuple(values))

            return num_segments, bbox, segments

    except Exception as e:
        print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        raise


def construct_segments(segments_data: List[Tuple[float, float, float, float]]) -> List[Segment]:
    """
    Construct labeled Segment objects from raw data.

    Points are labeled P1, P2, ... (left endpoints) and Q1, Q2, ... (right endpoints).
    Segments are labeled S1, S2, ...
    Identical coordinates reuse the same Point object.
    """
    point_map: Dict[Tuple[float, float], Point] = {}
    p_counter = 1
    q_counter = 1

    def get_or_create_point(x: float, y: float, is_left: bool) -> Point:
        """Get existing point or create new one with consecutive label."""
        nonlocal p_counter, q_counter
        key = (x, y)

        if key in point_map:
            return point_map[key]

        # Create new point
        label = f"P{p_counter}" if is_left else f"Q{q_counter}"
        if is_left:
            p_counter += 1
        else:
            q_counter += 1

        point = Point(x, y, label)
        point_map[key] = point
        return point

    labeled_segments = []
    for i, (x1, y1, x2, y2) in enumerate(segments_data, start=1):
        # Determine which is left and which is right
        if x1 < x2 or (abs(x1 - x2) < EPS and y1 <= y2):
            p_left = get_or_create_point(x1, y1, is_left=True)
            p_right = get_or_create_point(x2, y2, is_left=False)
        else:
            p_left = get_or_create_point(x2, y2, is_left=True)
            p_right = get_or_create_point(x1, y1, is_left=False)

        seg = Segment(p_left, p_right, label=f"S{i}")
        labeled_segments.append(seg)

    return labeled_segments
