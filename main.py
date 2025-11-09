"""
Trapezoidal map construction using randomized incremental algorithm.
"""
import argparse
import sys
import numpy as np
from typing import List, Tuple, Optional
import utils
import visualize
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

    @property
    def bbox_xmin(self) -> float:
        return self.bbox[0]

    @property
    def bbox_ymin(self) -> float:
        return self.bbox[1]

    @property
    def bbox_xmax(self) -> float:
        return self.bbox[2]

    @property
    def bbox_ymax(self) -> float:
        return self.bbox[3]

    def _label_trapezoids(self):
        self.trapezoids.sort(key=lambda x: (x.leftx, x.rightx))
        for i, t in enumerate(self.trapezoids, 1):
            t.label = f"T{i}"

    @classmethod
    def from_segments(cls, bbox: Tuple, segments: List[Tuple],
                     seed: Optional[int], randomize: bool = False):
        """Build map from segments."""
        labeled_segments = utils.construct_segments(segments)

        if randomize:
            rng = np.random.default_rng(seed)
            rng.shuffle(labeled_segments)

        trap_map = cls(bbox)

        visualize.visualize_trapezoidal_map(trap_map, "Trapezoidal map")

        print("Segments:")
        for i, seg in enumerate(labeled_segments):
            print(f"   {seg}")
            trap_map.insert_segment(seg)
            trap_map._label_trapezoids()

            for t in trap_map.trapezoids:
                print(t)
                if t.left_neighbors:
                    print("   left_neighbors:")
                    for tl in (t.u_left, t.d_left):
                        print(f"      {tl}")
                if t.right_neighbors:
                    print("   right_neighbors:")
                    for tr in (t.u_right, t.d_right):
                        print(f"      {tr}")
            print("\n\n\n")

            visualize.visualize_trapezoidal_map(trap_map, "Trapezoidal map")
        print()

        # Label trapezoids
        trap_map._label_trapezoids()

        return trap_map

    def insert_segment(self, seg: Segment):
        """Insert segment into map."""
        # 1. Find all trapezoids crossed by segment
        crossed = self._find_crossed(seg)
        print("Crossed Trapezoids:")
        for t in crossed:
            print(f"    {t}")
        print()

        # 2. Create new trapezoids
        left_rem_trap, right_rem_trap, above_traps, below_traps = self._create_trapezoids(crossed, seg)

        # 3. Connect neighbors
        self._connect_neighbors(crossed, left_rem_trap, right_rem_trap, above_traps, below_traps, seg)

        # 3. Build sub-DAG for new trapezoids
        sub_dags = self._build_dag(crossed, left_rem_trap, right_rem_trap,
                                  above_traps, below_traps, seg)

        # 4. Replace crossed trapezoids in DAG
        self._replace_dag(crossed, sub_dags)

        # 5. Update trapezoid list
        self.trapezoids = [t for t in self.trapezoids if t not in crossed]
        new_traps = [left_rem_trap] + above_traps + below_traps + [right_rem_trap]
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
            for candidate in t.right_neighbors:
            # for candidate in t.right_neighbors:
                if abs(candidate.leftx - t.rightx) < utils.EPS:
                    # Check if segment passes through this trapezoid at x = t.rightx
                    y_seg = seg.y_at(t.rightx)
                    y_top = candidate.y_top(t.rightx, self.bbox_ymax)
                    y_bot = candidate.y_bottom(t.rightx, self.bbox_ymin)

                    if y_bot - utils.EPS <= y_seg <= y_top + utils.EPS:
                        next_t = candidate
                        break
            if next_t is None:
                # Shouldn't happen if map is consistent
                break

            t = next_t
            crossed.append(t)

        return crossed

    def _create_trapezoids(
        self,
        crossed: List[Trapezoid],
        seg: Segment
    ) -> Tuple[Trapezoid, Trapezoid, List[Trapezoid], List[Trapezoid]]:
        """Create new trapezoids from crossed ones."""
        above_traps = []
        below_traps = []

        first, last = crossed[0], crossed[-1]
        seg_left_x = max(seg.xmin, first.leftx)
        seg_right_x = min(seg.xmax, last.rightx)

        # create the left-remainder trapezoid (will always exist in the general position)
        left_rem_trap = Trapezoid(first.top, first.bottom, first.leftx, seg_left_x,
                     first.left_point, seg.left)

        # create the right-remainder trapezoid (will always exist in the general position)
        right_rem_trap = Trapezoid(last.top, last.bottom, seg_right_x, last.rightx,
                     seg.right, last.right_point)

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
            above_traps.append(t)
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
            below_traps.append(t)
            i = j

        return left_rem_trap, right_rem_trap, above_traps, below_traps

    def _connect_neighbors(
        self,
        crossed: List[Trapezoid],
        left_rem: Trapezoid,
        right_rem: Trapezoid,
        above: List[Trapezoid],
        below: List[Trapezoid],
        seg: Segment
    ) -> None:
        first, last = crossed[0], crossed[-1]
        # connect first's left neighbors to left_rem
        if first.u_left:
            if first.u_left.u_right == first:
                first.u_left.u_right = left_rem
            if first.u_left.d_right == first:
                first.u_left.d_right = left_rem
            left_rem.u_left = first.u_left
        if first.d_left:
            if first.d_left.d_right == first:
                first.d_left.d_right = left_rem
            if first.d_left.u_right == first:
                first.d_left.u_right = left_rem
            left_rem.d_left = first.d_left
        # connect last's right neighbors to left_rem
        if last.u_right:
            if last.u_right.u_left == last:
                last.u_right.u_left = right_rem
            if last.u_right.d_left == last:
                last.u_right.d_left = right_rem
            right_rem.u_right = first.u_right
        if last.d_right:
            if last.d_right.d_left == last:
                last.d_right.d_left = right_rem
            if last.d_right.u_left == last:
                last.d_right.u_left = right_rem
            right_rem.d_right = last.d_right

        # connect first above/below trapezoids to left remainder trapezoid
        above[0].u_left = left_rem
        above[0].d_left = left_rem
        left_rem.u_right = above[0]
        below[0].u_left = left_rem
        below[0].d_left = left_rem
        left_rem.d_right = below[0]
        # connect last above/below trapezoids to right remainder trapezoid
        above[-1].u_right = right_rem
        above[-1].d_right = right_rem
        right_rem.u_left = above[-1]
        below[-1].u_right = right_rem
        below[-1].d_right = right_rem
        right_rem.d_left = below[-1]

        # connect above neighboring trapezoids
        for i in range(1, len(above)):
            t1, t2 = above[i-1], above[i]
            # t1 and t2 must be sharing the same bottom segment
            t1.d_right = t2
            t2.d_left = t1
            # handle pairing top neighbors
            if t1.top == t2.top:
                # if they share the same top segment, then they are top neighbors
                t1.u_right = t2
                t2.u_left = t1
            else:
                # otherwise, we need to inherit from crossed trapezoid neighbors
                pass

        # connect below neighboring trapezoids
        t1_crossed_origin_j = 0
        t2_crossed_origin_j = 0
        for i in range(1, len(above)):
            t1, t2 = above[i-1], above[i]
            # t1 and t2 must be sharing the bottom segment
            t1.d_right = t2
            t2.d_left = t1
            # find the trapezoids that t1 and t2 originate from
            while crossed[t1_crossed_origin_j].top != t1.top:
                t1_crossed_origin_j += 1
            while crossed[t2_crossed_origin_j].top != t2.top:
                t2_crossed_origin_j += 1
            # handle pairing top neighbors
            if t1_crossed_origin_j == t2_crossed_origin_j:
                # if t1 and t2 are derived from the same trapezoid, then they share a top segment
                t1.u_right = t2
                t2.u_left = t1
            else:
                # if they are not top neighbors, then we need to figure out which origin to inherit a neighbor from
                x = t1.rightx
                # get the right upper y-bound for t1
                t1_y_top = t1.y_top(x, self.bbox_ymax)
                # and the left upper y-bound for t2
                t2_y_top = t2.y_top(x, self.bbox_ymax)
                if abs(t1_y_top - t2_y_top) < utils.EPS:
                    # if t1 and t2 have the same y_top, then they are neighbors of each other
                    t1.u_right = t2
                    t2.u_left = t1
                else:
                    if t1_y_top > t2_y_top:
                        # t1 has a higher y_top, so t1 inherits from its origin upper-right neighbor
                        t1.u_right = crossed[t1_crossed_origin_j].u_right
                        t2.u_left = t2
                        # handle patching t1.u_right
                        t1.u_right.d_left = t1
                        if t1.u_right.top == t1.top:
                            t1.u_right.u_left = t1
                    else:
                        # otherwise, t2 has a higher y_top, so t2 inherits from its origin upper-right neighbor
                        t2.u_left = crossed[t2_crossed_origin_j].u_left
                        t1.u_right = t2
                        # handle patching t2.u_left
                        t2.u_left.d_right = t2
                        if t2.u_left.top == t2.top:
                            t2.u_left.u_right = t2

        # connect below neighboring trapezoids
        t1_crossed_origin_j = 0
        t2_crossed_origin_j = 0
        for i in range(1, len(below)):
            t1, t2 = below[i-1], below[i]
            # t1 and t2 must be sharing the same top segment
            t1.u_right = t2
            t2.u_left = t1
            # find the trapezoids that t1 and t2 originate from
            while crossed[t1_crossed_origin_j].bottom != t1.bottom:
                t1_crossed_origin_j += 1
            while crossed[t2_crossed_origin_j].bottom != t2.bottom:
                t2_crossed_origin_j += 1
            # handle pairing bottom neighbors
            if t1_crossed_origin_j == t2_crossed_origin_j:
                # if t1 and t2 are derived from the same trapezoid, then they share a bottom segment
                t1.d_right = t2
                t2.d_left = t1
            else:
                # if they are not bottom neighbors, then we need to figure out which origin to inherit a neighbor from
                x = t1.rightx
                # get the right lower y-bound for t1
                t1_y_bot = t1.y_bottom(x, self.bbox_ymin)
                # and the left lower y-bound for t2
                t2_y_bot = t2.y_bottom(x, self.bbox_ymin)
                if abs(t1_y_bot - t2_y_bot) < utils.EPS:
                    # if t1 and t2 have the same y_bot, then they are neighbors of each other
                    t1.d_right = t2
                    t2.d_left = t1
                else:
                    if t1_y_bot < t2_y_bot:
                        # t1 has a lower y_bot, so t1 inherits from its origin lower-right neighbor
                        t1.d_right = crossed[t1_crossed_origin_j].d_right
                        t2.d_left = t1
                        # handle patching t1.d_right
                        t1.d_right.u_left = t1
                        if t1.d_right.bottom == t1.bottom:
                            t1.d_right.d_left = t1

                    else:
                        # otherwise, t2 has a lower y_bot, so t2 inherits from its origin lower-left neighbor
                        t2.d_left = crossed[t2_crossed_origin_j].d_left
                        t1.d_right = t2
                        # handle patching t2.d_left
                        t2.d_left.u_right = t2
                        if t2.d_left.bottom == t2.bottom:
                            t2.d_left.d_right = t2

    def _build_dag(
        self,
        crossed: List[Trapezoid],
        left_rem: Optional[Trapezoid],
        right_rem: Optional[Trapezoid],
        above: List[Trapezoid],
        below: List[Trapezoid],
        seg: Segment
    ) -> List[Node]:
        """Build sub-DAG for new trapezoids."""
        crossed_dags: List[Node] = []
        first, last = crossed[0], crossed[-1]

        # handle construction by the 3 cases:
        if first == last:
            dag = self._case_1_dag(left_rem, right_rem, above[0], below[0], seg)
            crossed_dags.append(dag)
        else:
            # build sub-dag for case 2a
            first_dag = self._case_2_dag(left_rem, above[0], below[0], seg, is_left=True)
            crossed_dags.append(first_dag)
            # build middle sub-dags for case 3
            middle_dags = self._case_3_dags(crossed[1:-1], above, below, seg)
            crossed_dags.extend(middle_dags)
            # build sub-dag for case 2b
            last_dag = self._case_2_dag(right_rem, above[-1], below[-1], seg, is_left=False)
            crossed_dags.append(last_dag)

        return crossed_dags

    def _case_1_dag(
        self,
        left_rem: Trapezoid,
        right_rem: Trapezoid,
        above: Trapezoid,
        below: Trapezoid,
        seg: Segment
    ) -> Node:
        # case 1: both segments are in the same trapezoid
        #            a(both)
        #               P
        #             /   \
        #            T    Q
        #               /   \
        #              S    T
        #            /   \
        #           T    T

        # build new leaves for trapezoids (there are 4)
        left_rem_leaf = Leaf(left_rem)
        right_rem_leaf = Leaf(right_rem)
        above_leaf = Leaf(above)
        below_leaf = Leaf(below)
        # build XNode split for the left endpoint (root of the sub-DAG)
        P = XNode(point=seg.left)
        # left-rem trapezoid leaf
        P.left = left_rem_leaf
        left_rem_leaf.parent = P
        # build XNode split for the right endpoint (right child of P)
        Q = XNode(point=seg.right)
        P.right = Q
        # right-rem trapezoid leaf
        Q.right = right_rem_leaf
        right_rem_leaf.parent = Q
        # build YNode split for the segment (left child of Q)
        S = YNode(seg=seg)
        Q.left = S
        # connect above and below trapezoids
        S.above = above_leaf
        above_leaf.parent = S
        S.below = below_leaf
        below_leaf.parent = S
        return P

    def _case_2_dag(
        self,
        rem: Trapezoid,
        above: Trapezoid,
        below: Trapezoid,
        seg: Segment,
        is_left: bool
    ) -> Node:
        # case 2: only one endpoint is in a trapezoid
        #  a(left + left rem)  b(right + right rem)
        #           P                     Q
        #         /   \                 /   \
        #        T    S                S     T
        #           /   \            /   \
        #          T    T           T    T

        # build trapezoid leaves
        rem_leaf = Leaf(rem)
        above_leaf = Leaf(above)
        below_leaf = Leaf(below)
        # build root XNode
        root = XNode(point=(seg.left if is_left else seg.right))
        # build S node
        S = YNode(seg=seg)
        S.above = above_leaf
        above_leaf.parent = S
        S.below = below_leaf
        below_leaf.parent = S
        if is_left:
            root.left = rem_leaf
            rem_leaf.parent = root
            root.right = S
        else:
            root.right = rem_leaf
            rem_leaf.parent = root
            root.left = S
        return root

    def _case_3_dags(
        self,
        crossed_traps: List[Trapezoid],
        above_traps: List[Trapezoid],
        below_traps: List[Trapezoid],
        seg: Segment
    ) -> List[Node]:
        """Build sub-DAG for new trapezoids, some of which may be built from split and merged crossed trapezoids."""
        # case 3: no endpoint is in a trapezoid
        #       S
        #     /   \
        #    T    T

        case_3_dags = []
        i_above = 0
        i_below = 0
        for crossed in crossed_traps:
            # determine if the current above/below trapezoids are merged from the current crossed trapezoid
            while above_traps[i_above].top != crossed.top:
                i_above += 1
            while below_traps[i_below].bottom != crossed.bottom:
                i_below += 1
            # create above/below trapezoid leaves
            above_leaf = Leaf(above_traps[i_above])
            below_leaf = Leaf(below_traps[i_below])
            # create root YNode
            S = YNode(seg=seg)
            S.above = above_leaf
            above_leaf.parent = S
            S.below = below_leaf
            below_leaf.parent = S
            case_3_dags.append(S)
        return case_3_dags

    def _replace_dag(self, crossed_traps: List[Trapezoid], sub_dags: List[Node]):
        """Replace crossed trapezoid leaves with sub-DAG."""
        for i in range(len(crossed_traps)):
            crossed = crossed_traps[i]
            sub_dag = sub_dags[i]
            # collect parent nodes of the crossed trapezoids
            parents = {leaf.parent for leaf in crossed.leaves}
            for parent in parents:
                if parent is None:
                    # this should ever happen once: when the parent node is a leaf (first segment insertion)
                    self.root = sub_dag
                    break
                # replace all leaves of parents where `crossed` is being pointed to
                if isinstance(parent.a, Leaf) and parent.a.trap == crossed:
                    parent.a = sub_dag
                elif isinstance(parent.b, Leaf) and parent.b.trap == crossed:
                    parent.b = sub_dag


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
