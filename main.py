"""
Trapezoidal map construction using randomized incremental algorithm.
"""
import argparse
import sys
import numpy as np
from typing import List, Tuple, Optional
import utils
from utils import Point, Segment, Trapezoid, Node, XNode, YNode, Leaf


class TrapezoidalMap:
    """Trapezoidal map using randomized incremental construction."""

    def __init__(self, bbox: Tuple[float, float, float, float]):
        """Initialize map with bounding box."""
        self.bbox = bbox  # (x_min, y_min, x_max, y_max)

        # Create initial trapezoid spanning entire bounding box
        t0 = Trapezoid(
            top=None,  # bbox top
            bottom=None,  # bbox bottom
            leftx=bbox[0],
            rightx=bbox[2],
            left_point=None,
            right_point=None
        )
        t0.label = "T0"

        # Initialize DAG with single leaf
        self.root = Leaf(t0)

        # Track all trapezoids
        self.trapezoids = [t0]

    @classmethod
    def from_segments(cls, bbox: Tuple, segments: List[Tuple],
                     seed: Optional[int], randomize: bool = False):
        """Build map from segments."""
        labeled_segments = utils.construct_segments(segments)

        if randomize:
            rng = np.random.default_rng(seed)
            rng.shuffle(labeled_segments)

        trap_map = cls(bbox)

        print("Segments:")
        for seg in labeled_segments:
            print(f"   {seg}")
            trap_map.insert_segment(seg)
        print()

        # Label trapezoids
        trap_map.trapezoids.sort(key=lambda t: (t.leftx, t.rightx))
        for i, t in enumerate(trap_map.trapezoids, 1):
            t.label = f"T{i}"

        return trap_map

    def insert_segment(self, seg: Segment):
        """Insert segment into map."""
        # 1. Find all trapezoids crossed by segment
        crossed = self._find_crossed(seg)

        # 2. Create new trapezoids
        new_traps = self._create_trapezoids(crossed, seg)

        # 3. Build sub-DAG for new trapezoids
        sub_dag = self._build_dag(crossed, new_traps, seg)

        # 4. Replace crossed trapezoids in DAG
        self._replace_dag(crossed, sub_dag)

        # 5. Update trapezoid list
        self.trapezoids = [t for t in self.trapezoids if t not in crossed]
        self.trapezoids.extend(new_traps)

    def _find_crossed(self, seg: Segment) -> List[Trapezoid]:
        """Find all trapezoids crossed by segment."""
        # Start at trapezoid containing left endpoint
        t = self.locate_point(seg.left)
        crossed = [t]

        # Walk rightward until we reach segment's right endpoint
        while seg.right.x > t.rightx + utils.EPS:
            # Find next trapezoid to the right
            next_t = None
            for candidate in self.trapezoids:
                # Check if candidate's left boundary matches current's right boundary
                if abs(candidate.leftx - t.rightx) < utils.EPS:
                    # Check if segment passes through this trapezoid at x = t.rightx
                    y_seg = seg.y_at(t.rightx)
                    y_top = candidate.y_top(t.rightx, self.bbox[3])
                    y_bot = candidate.y_bottom(t.rightx, self.bbox[1])

                    if y_bot - utils.EPS <= y_seg <= y_top + utils.EPS:
                        next_t = candidate
                        break

            if next_t is None:
                # Shouldn't happen if map is consistent
                break

            t = next_t
            crossed.append(t)

        return crossed

    def _create_trapezoids(self, crossed: List[Trapezoid], seg: Segment) -> List[Trapezoid]:
        """Create new trapezoids from crossed ones."""
        new_traps = []

        first, last = crossed[0], crossed[-1]
        seg_left_x = max(seg.xmin, first.leftx)
        seg_right_x = min(seg.xmax, last.rightx)

        # Left remainder trapezoid (if segment doesn't start at left boundary)
        if seg_left_x > first.leftx + utils.EPS:
            t = Trapezoid(first.top, first.bottom, first.leftx, seg_left_x,
                         first.left_point, seg.left)
            new_traps.append(t)

        # Right remainder trapezoid (if segment doesn't end at right boundary)
        if last.rightx > seg_right_x + utils.EPS:
            t = Trapezoid(last.top, last.bottom, seg_right_x, last.rightx,
                         seg.right, last.right_point)
            new_traps.append(t)

        # Trapezoids above segment (merge consecutive with same top boundary)
        i = 0
        while i < len(crossed):
            boundary = crossed[i].top
            # Find all consecutive trapezoids with same top
            j = i
            while j < len(crossed) and crossed[j].top == boundary:
                j += 1

            # Determine x-boundaries
            left_x = seg_left_x if i == 0 else crossed[i-1].rightx
            right_x = seg_right_x if j == len(crossed) else crossed[j-1].rightx
            left_pt = seg.left if i == 0 else crossed[i-1].right_point
            right_pt = seg.right if j == len(crossed) else crossed[j-1].right_point

            t = Trapezoid(boundary, seg, left_x, right_x, left_pt, right_pt)
            new_traps.append(t)
            i = j

        # Trapezoids below segment (merge consecutive with same bottom boundary)
        i = 0
        while i < len(crossed):
            boundary = crossed[i].bottom
            # Find all consecutive trapezoids with same bottom
            j = i
            while j < len(crossed) and crossed[j].bottom == boundary:
                j += 1

            # Determine x-boundaries
            left_x = seg_left_x if i == 0 else crossed[i-1].rightx
            right_x = seg_right_x if j == len(crossed) else crossed[j-1].rightx
            left_pt = seg.left if i == 0 else crossed[i-1].right_point
            right_pt = seg.right if j == len(crossed) else crossed[j-1].right_point

            t = Trapezoid(seg, boundary, left_x, right_x, left_pt, right_pt)
            new_traps.append(t)
            i = j

        return new_traps

    def _build_dag(self, crossed: List[Trapezoid], new_traps: List[Trapezoid],
                   seg: Segment) -> Node:
        """Build sub-DAG for new trapezoids."""
        first, last = crossed[0], crossed[-1]
        seg_left_x = max(seg.xmin, first.leftx)
        seg_right_x = min(seg.xmax, last.rightx)

        # Classify trapezoids
        left_rem = None
        right_rem = None
        above = []
        below = []

        for t in new_traps:
            # Check for left remainder
            if (t.top == first.top and t.bottom == first.bottom and
                abs(t.rightx - seg_left_x) < utils.EPS):
                left_rem = t
            # Check for right remainder
            elif (t.top == last.top and t.bottom == last.bottom and
                  abs(t.leftx - seg_right_x) < utils.EPS):
                right_rem = t
            # Trapezoid above segment
            elif t.bottom == seg:
                above.append(t)
            # Trapezoid below segment
            elif t.top == seg:
                below.append(t)

        # Build core DAG: YNode with first above/below trapezoids
        if not above or not below:
            raise RuntimeError(f"Missing trapezoids for segment {seg.label}")

        dag = YNode(seg, Leaf(above[0]), Leaf(below[0]))

        # Wrap with XNodes at endpoints
        if left_rem:
            dag = XNode(seg.left, Leaf(left_rem), dag)
        if right_rem:
            dag = XNode(seg.right, dag, Leaf(right_rem))

        return dag

    def _replace_dag(self, crossed: List[Trapezoid], sub_dag: Node):
        """Replace crossed trapezoid leaves with sub-DAG."""
        leaves_to_replace = {t.leaf for t in crossed if t.leaf}

        def replace(node: Node) -> Node:
            if node in leaves_to_replace:
                return sub_dag
            if isinstance(node, XNode):
                node.left = replace(node.left)
                node.right = replace(node.right)
            elif isinstance(node, YNode):
                node.above = replace(node.above)
                node.below = replace(node.below)
            return node

        self.root = replace(self.root)

    def locate_point(self, p: Point) -> Trapezoid:
        """Find trapezoid containing point by traversing DAG."""
        node = self.root

        while not isinstance(node, Leaf):
            if isinstance(node, XNode):
                # Go left if point.x < node.x, otherwise right
                node = node.left if p.x < node.x else node.right
            elif isinstance(node, YNode):
                # Clamp x to segment range to handle endpoints
                x = max(node.seg.xmin, min(node.seg.xmax, p.x))
                y_seg = node.seg.y_at(x)
                # Go above if point.y > segment.y, otherwise below
                node = node.above if p.y > y_seg else node.below

        return node.trap

    def query_point(self, x: float, y: float) -> Tuple[str, List[Node]]:
        """Query point and return path through DAG."""
        p = Point(x, y)
        node = self.root
        path = []

        # Traverse DAG
        while not isinstance(node, Leaf):
            path.append(node)
            if isinstance(node, XNode):
                node = node.left if p.x < node.x else node.right
            elif isinstance(node, YNode):
                x_clamped = max(node.seg.xmin, min(node.seg.xmax, p.x))
                y_seg = node.seg.y_at(x_clamped)
                node = node.above if p.y > y_seg else node.below

        path.append(node)

        # Build path string
        path_str = []
        for n in path:
            if isinstance(n, XNode):
                path_str.append(n.point.label)
            elif isinstance(n, YNode):
                path_str.append(n.seg.label)
            elif isinstance(n, Leaf):
                path_str.append(n.trap.label if n.trap.label else "T?")

        return " ".join(path_str), path

    def generate_adjacency_matrix(self) -> Tuple[List[str], List[List[int]]]:
        """Generate adjacency matrix from DAG."""
        all_nodes = []
        visited = set()

        def collect(node: Node):
            if id(node) in visited:
                return
            visited.add(id(node))
            all_nodes.append(node)

            if isinstance(node, XNode):
                collect(node.left)
                collect(node.right)
            elif isinstance(node, YNode):
                collect(node.above)
                collect(node.below)

        collect(self.root)

        # Assign labels
        labels = []
        for node in all_nodes:
            if isinstance(node, XNode):
                labels.append(node.point.label)
            elif isinstance(node, YNode):
                labels.append(node.seg.label)
            elif isinstance(node, Leaf):
                labels.append(node.trap.label if node.trap.label else "T?")

        # Build adjacency matrix
        n = len(all_nodes)
        matrix = [[0] * n for _ in range(n)]
        node_to_idx = {id(node): i for i, node in enumerate(all_nodes)}

        for i, node in enumerate(all_nodes):
            children = []
            if isinstance(node, XNode):
                children = [node.left, node.right]
            elif isinstance(node, YNode):
                children = [node.above, node.below]

            for child in children:
                j = node_to_idx.get(id(child))
                if j is not None:
                    matrix[i][j] = 1  # matrix[parent][child] = 1

        return labels, matrix


def write_adjacency_matrix(output_path: str, labels: List[str], matrix: List[List[int]]):
    """Write adjacency matrix to file."""
    import os
    n = len(labels)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        # Header row
        f.write(f"{'':>6}")
        for label in labels:
            f.write(f"{label:>4}")
        f.write("  Sum\n")

        # Matrix rows
        for i, label in enumerate(labels):
            f.write(f"{label:>6}")
            for j in range(n):
                f.write(f"{matrix[i][j]:>4}")
            row_sum = sum(matrix[i])
            f.write(f"{row_sum:>4}\n")

        # Column sums
        f.write(f"{'Sum':>6}")
        for j in range(n):
            col_sum = sum(matrix[i][j] for i in range(n))
            f.write(f"{col_sum:>4}")
        total = sum(sum(row) for row in matrix)
        f.write(f"{total:>4}\n")


def main():
    parser = argparse.ArgumentParser(description="Trapezoidal Map Construction")
    parser.add_argument("input_file", help="Input file with line segments")
    parser.add_argument("-o", "--output", default="./out/out.txt", help="Output file")
    parser.add_argument("-s", "--seed", type=int, help="Random seed")
    parser.add_argument("-r", "--randomize", action="store_true", help="Randomize insertion order")
    parser.add_argument("-v", "--visualize", action="store_true", help="Show visualization")
    args = parser.parse_args()

    if args.seed and not args.randomize:
        print("A seed parameter (-s <int> flag) was set without enabling randomization (-r flag). "
              "It's value will be ignored")

    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    print()

    try:
        # Read input
        num_segments, bbox, segments_data = utils.read_file(args.input_file)
        print(f"Read {num_segments} segments from {args.input_file}")
        print(f"Bounding box: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")
        print()

        # Build map
        trap_map = TrapezoidalMap.from_segments(bbox, segments_data, args.seed, args.randomize)
        print(f"Constructed trapezoidal map with {len(trap_map.trapezoids)} trapezoids")

        # Generate and write adjacency matrix
        labels, matrix = trap_map.generate_adjacency_matrix()
        write_adjacency_matrix(args.output, labels, matrix)
        print(f"Adjacency matrix written to {args.output}")
        print(f"DAG contains {len(labels)} nodes")

        # Visualization
        if args.visualize:
            from visualize import visualize_trapezoidal_map
            visualize_trapezoidal_map(trap_map, f"Trapezoidal Map - {args.input_file}")

        # Interactive query mode
        print()
        print("=" * 60)
        print("Point Location Query Mode")
        print("Enter a point (x y) to find its location in the trapezoidal map")
        print("Enter 'quit' or 'exit' to stop")
        print("=" * 60)
        print()

        while True:
            try:
                line = input("Enter point (x y): ").strip()
                if line.lower() in ['quit', 'exit', '']:
                    print("Exiting query mode.")
                    break

                parts = line.split()
                if len(parts) != 2:
                    print("Invalid input. Please enter two numbers (x y)")
                    continue

                x, y = float(parts[0]), float(parts[1])
                path_str, path_nodes = trap_map.query_point(x, y)

                print(f"\nQuery: ({x}, {y})")
                print(f"Path: {path_str}")
                print(f"Located in: {path_nodes[-1].trap.label}")
                print()

            except KeyboardInterrupt:
                print("\nExiting query mode.")
                break
            except Exception as e:
                print(f"Error: {e}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
