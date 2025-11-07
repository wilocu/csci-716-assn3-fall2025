import argparse
import os, sys
import utils
from utils import Point, Segment, Trapezoid, Node, XNode, YNode, Leaf
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from typing import Optional, Tuple, List


########################################
#   Trapezoidal Map & Point Location   #
########################################

class TrapezoidalMap:
    def __init__(self, bbox: Tuple[float, float, float, float]):
        x_min, _, x_max, _ = bbox
        self.bbox = bbox
        t0 = Trapezoid(top=None, bottom=None, leftx=x_min, rightx=x_max)
        self.root: Node = Leaf(t0)
        self.trapezoids: set[Trapezoid] = {t0}
    
    @property
    def bbox_x_min(self):
        return self.bbox[0]
    
    @property
    def bbox_y_min(self):
        return self.bbox[1]
    
    @property
    def bbox_x_max(self):
        return self.bbox[2]
    
    @property
    def bbox_y_max(self):
        return self.bbox[3]
    
    @classmethod
    def from_segments(cls, bbox: Tuple, segments: List[Tuple], seed: Optional[float], randomize: Optional[bool] = False) -> "TrapezoidalMap":
        """Build a trapezoidal map using randomized incremental construction."""
        labeled_segments: List[Segment] = utils.construct_segments(segments)
        if randomize:
            rng = np.random.default_rng(seed)
            rng.shuffle(labeled_segments)

        trap_map = cls(bbox)
        for seg in labeled_segments:
            print(seg)
            trap_map._insert_segment(seg)
        
        # Post-processing: label all trapezoids
        counter = 0
        for t in trap_map.trapezoids:
            if not t.label:
                t.label = f"T{counter+1}"
                counter += 1
        return trap_map

    def locate_point(self, p: Point) -> utils.Trapezoid:
        """Locate which trapezoid contains point p. Expected O(log n) time."""
        node = self.root
        while not isinstance(node, Leaf):
            if isinstance(node, XNode):
                node = node.left if p.x < node.x else node.right
            elif isinstance(node, YNode):
                xq = max(node.seg.xmin, min(p.x, node.seg.xmax))
                y_on_seg = node.seg.y_at(xq)
                node = node.above if p.y > y_on_seg else node.below
        return node.trap
        
    def _insert_segment(self, seg: Segment):
        """Insert segment into trapezoidal map. Expected O(log n) time."""
        # Locate starting trapezoid containing segment's left endpoint
        t = self.locate_point(seg.left)

        # Fallback: handle case where DAG returns obsolete trapezoid
        while t not in self.trapezoids:
            for candidate in self.trapezoids:
                if candidate.leftx - utils.EPS <= seg.left.x <= candidate.rightx + utils.EPS:
                    y_top = candidate.y_top(seg.left.x, self.bbox_y_max)
                    y_bot = candidate.y_bottom(seg.left.x, self.bbox_y_min)
                    if y_bot - utils.EPS <= seg.left.y <= y_top + utils.EPS:
                        t = candidate
                        break
            if t not in self.trapezoids:
                raise RuntimeError(f"Could not find valid trapezoid for start point {seg.left}")

        # Traverse right through trapezoids intersected by segment
        crossed_traps = []
        while True:
            crossed_traps.append(t)
            if seg.right.x <= t.rightx + utils.EPS:
                break

            xr, y_seg = t.rightx, seg.y_at(t.rightx)
            next_trap = None
            for neighbor in t.right_neighbors:
                if neighbor not in self.trapezoids:
                    continue
                y_top = neighbor.y_top(xr, self.bbox_y_max)
                y_bot = neighbor.y_bottom(xr, self.bbox_y_min)
                if y_bot - utils.EPS <= y_seg <= y_top + utils.EPS:
                    next_trap = neighbor
                    break

            if next_trap is None:
                raise RuntimeError(f"Could not find next trapezoid at x={xr}, y={y_seg}")
            t = next_trap

        # Split crossed trapezoids and update DAG
        sub = self._split_crossed(crossed_traps, seg)
        self._splice(crossed_traps, sub)
        self.trapezoids.difference_update(crossed_traps)

        # Clean up obsolete neighbor pointers
        for trap in self.trapezoids:
            trap.left_neighbors = [n for n in trap.left_neighbors if n in self.trapezoids]
            trap.right_neighbors = [n for n in trap.right_neighbors if n in self.trapezoids]
    
    def _split_crossed(self, crossed_traps: List[Trapezoid], seg: Segment) -> Node:
        """Split crossed trapezoids, creating new trapezoids above/below segment."""
        traps_above, traps_below = [], []
        left_remainder, right_remainder = None, None

        t_first, t_last = crossed_traps[0], crossed_traps[-1]
        seg_left_x = max(seg.xmin, t_first.leftx)
        seg_right_x = min(seg.xmax, t_last.rightx)

        if seg_left_x - t_first.leftx > utils.EPS:
            left_remainder = Trapezoid(t_first.top, t_first.bottom, t_first.leftx, seg_left_x)
            left_remainder.left_neighbors = t_first.left_neighbors.copy()
            self.trapezoids.add(left_remainder)

        if t_last.rightx - seg_right_x > utils.EPS:
            right_remainder = Trapezoid(t_last.top, t_last.bottom, seg_right_x, t_last.rightx)
            right_remainder.right_neighbors = t_last.right_neighbors.copy()
            self.trapezoids.add(right_remainder)

        # Merge consecutive crossed trapezoids with same top/bottom into single trapezoids
        i = 0
        while i < len(crossed_traps):
            same_top = crossed_traps[i].top
            j = i
            while j < len(crossed_traps) and crossed_traps[j].top == same_top:
                j += 1

            left_x = seg_left_x if i == 0 else crossed_traps[i-1].rightx
            right_x = seg_right_x if j == len(crossed_traps) else crossed_traps[j-1].rightx

            t_upper = Trapezoid(same_top, seg, left_x, right_x)
            self.trapezoids.add(t_upper)
            traps_above.append(t_upper)
            i = j

        i = 0
        while i < len(crossed_traps):
            same_bottom = crossed_traps[i].bottom
            j = i
            while j < len(crossed_traps) and crossed_traps[j].bottom == same_bottom:
                j += 1

            left_x = seg_left_x if i == 0 else crossed_traps[i-1].rightx
            right_x = seg_right_x if j == len(crossed_traps) else crossed_traps[j-1].rightx

            t_lower = Trapezoid(seg, same_bottom, left_x, right_x)
            self.trapezoids.add(t_lower)
            traps_below.append(t_lower)
            i = j

        # Neighbor pointer management - O(k × d) instead of O(k²)
        all_left_neighbors = set()
        all_right_neighbors = set()
        for trap in crossed_traps:
            all_left_neighbors.update(trap.left_neighbors)
            all_right_neighbors.update(trap.right_neighbors)

        # Group new trapezoids by boundary x-coordinates for O(1) lookup
        new_traps_by_leftx = {}
        new_traps_by_rightx = {}

        for trap in [left_remainder] + traps_above + traps_below + ([right_remainder] if right_remainder else []):
            if trap is None:
                continue
            if trap.leftx not in new_traps_by_leftx:
                new_traps_by_leftx[trap.leftx] = []
            new_traps_by_leftx[trap.leftx].append(trap)
            if trap.rightx not in new_traps_by_rightx:
                new_traps_by_rightx[trap.rightx] = []
            new_traps_by_rightx[trap.rightx].append(trap)

        # Update left neighbors with geometrically adjacent new trapezoids
        for neighbor in all_left_neighbors:
            if neighbor not in self.trapezoids:
                continue
            neighbor.right_neighbors = [t for t in neighbor.right_neighbors if t not in crossed_traps]
            candidates = new_traps_by_leftx.get(neighbor.rightx, [])
            for new_trap in candidates:
                if self._trapezoids_are_vertically_adjacent(neighbor, new_trap, neighbor.rightx):
                    neighbor.right_neighbors.append(new_trap)
                    new_trap.left_neighbors.append(neighbor)

        # Update right neighbors with geometrically adjacent new trapezoids
        for neighbor in all_right_neighbors:
            if neighbor not in self.trapezoids:
                continue
            neighbor.left_neighbors = [t for t in neighbor.left_neighbors if t not in crossed_traps]
            candidates = new_traps_by_rightx.get(neighbor.leftx, [])
            for new_trap in candidates:
                if self._trapezoids_are_vertically_adjacent(new_trap, neighbor, neighbor.leftx):
                    neighbor.left_neighbors.append(new_trap)
                    new_trap.right_neighbors.append(neighbor)

        # Stitch horizontal neighbors within the new trapezoids
        for i in range(len(traps_above) - 1):
            if traps_above[i+1] not in traps_above[i].right_neighbors:
                traps_above[i].right_neighbors.append(traps_above[i+1])
                traps_above[i+1].left_neighbors.append(traps_above[i])

        for i in range(len(traps_below) - 1):
            if traps_below[i+1] not in traps_below[i].right_neighbors:
                traps_below[i].right_neighbors.append(traps_below[i+1])
                traps_below[i+1].left_neighbors.append(traps_below[i])

        # Connect remainders to above/below trapezoids
        if left_remainder and traps_above:
            if traps_above[0] not in left_remainder.right_neighbors:
                left_remainder.right_neighbors.append(traps_above[0])
                traps_above[0].left_neighbors.append(left_remainder)
        if left_remainder and traps_below:
            if traps_below[0] not in left_remainder.right_neighbors:
                left_remainder.right_neighbors.append(traps_below[0])
                traps_below[0].left_neighbors.append(left_remainder)

        if right_remainder and traps_above:
            if traps_above[-1] not in right_remainder.left_neighbors:
                right_remainder.left_neighbors.append(traps_above[-1])
                traps_above[-1].right_neighbors.append(right_remainder)
        if right_remainder and traps_below:
            if traps_below[-1] not in right_remainder.left_neighbors:
                right_remainder.left_neighbors.append(traps_below[-1])
                traps_below[-1].right_neighbors.append(right_remainder)

        # Build the local sub-DAG
        def chain_by_bounds(traps: List[Trapezoid]) -> Node:
            if not traps:
                return Leaf(self._dummy_empty_trap())
            if len(traps) == 1:
                return Leaf(traps[0])
            traps_sorted = sorted(traps, key=lambda t: (t.leftx, t.rightx))
            mid = len(traps_sorted) // 2
            left_chain = chain_by_bounds(traps_sorted[:mid])
            right_chain = chain_by_bounds(traps_sorted[mid:])
            split_x = traps_sorted[mid].leftx
            return XNode(split_x, left_chain, right_chain)

        above_chain = chain_by_bounds(traps_above)
        below_chain = chain_by_bounds(traps_below)
        y_node = YNode(seg, above_chain, below_chain)

        # Wrap with XNodes at the ends if remainders were created
        if left_remainder is not None:
            y_node = XNode(traps_above[0].leftx, Leaf(left_remainder), y_node)
        if right_remainder is not None:
            y_node = XNode(traps_above[-1].rightx, y_node, Leaf(right_remainder))
        return y_node

    def _trapezoids_are_vertically_adjacent(self, left_trap: Trapezoid, right_trap: Trapezoid, x_boundary: float) -> bool:
        """Check if two trapezoids are vertically adjacent at x_boundary."""
        left_y_top = left_trap.y_top(x_boundary, self.bbox_y_max)
        left_y_bot = left_trap.y_bottom(x_boundary, self.bbox_y_min)
        right_y_top = right_trap.y_top(x_boundary, self.bbox_y_max)
        right_y_bot = right_trap.y_bottom(x_boundary, self.bbox_y_min)

        overlap_bot = max(left_y_bot, right_y_bot)
        overlap_top = min(left_y_top, right_y_top)
        return overlap_bot <= overlap_top + utils.EPS
    
    def _dummy_empty_trap(self) -> Trapezoid:
        t = Trapezoid(None, None, self.bbox_x_min, self.bbox_x_min)
        self.trapezoids.add(t)
        return t
    
    def _splice(self, crossed_traps: List[Trapezoid], sub_dag: Node):
        """
        Replace crossed trapezoids' leaves in DAG with sub_dag.
        Expected O(klogn) where k = # crossed trapezoids, E[k] = O(1).
        """
        if not crossed_traps:
            return

        leaves_to_replace = set()
        for trap in crossed_traps:
            if trap.leaf is not None:
                leaves_to_replace.add(trap.leaf)

        if not leaves_to_replace:
            return

        def replace_in(node: Node) -> Node:
            if node in leaves_to_replace:
                return sub_dag
            if isinstance(node, XNode):
                left_result = replace_in(node.left)
                right_result = replace_in(node.right)
                if left_result is right_result and left_result is sub_dag:
                    return sub_dag
                node.left = left_result
                node.right = right_result
                return node
            if isinstance(node, YNode):
                above_result = replace_in(node.above)
                below_result = replace_in(node.below)
                if above_result is below_result and above_result is sub_dag:
                    return sub_dag
                node.above = above_result
                node.below = below_result
                return node
            return node

        self.root = replace_in(self.root)

    def visualize(self, title: str = "Trapezoidal Map"):
        """Visualize the trapezoidal map with matplotlib."""
        trapezoids_list = list(self.trapezoids)

        _, ax = plt.subplots(figsize=(12, 8))
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')

        padding = 1.0
        ax.set_xlim(self.bbox_x_min - padding, self.bbox_x_max + padding)
        ax.set_ylim(self.bbox_y_min - padding, self.bbox_y_max + padding)

        bbox_rect = Rectangle(
            (self.bbox_x_min, self.bbox_y_min),
            self.bbox_x_max - self.bbox_x_min,
            self.bbox_y_max - self.bbox_y_min,
            linewidth=2, edgecolor='black', facecolor='none', linestyle='--'
        )
        ax.add_patch(bbox_rect)

        segments_set = set()
        for trap in trapezoids_list:
            if trap.top is not None:
                segments_set.add(trap.top)
            if trap.bottom is not None:
                segments_set.add(trap.bottom)

        colors = plt.cm.Set3(np.linspace(0, 1, len(trapezoids_list)))
        for i, trap in enumerate(trapezoids_list):
            xl, xr = trap.leftx, trap.rightx
            y_top_left = trap.y_top(xl, self.bbox_y_max)
            y_top_right = trap.y_top(xr, self.bbox_y_max)
            y_bot_left = trap.y_bottom(xl, self.bbox_y_min)
            y_bot_right = trap.y_bottom(xr, self.bbox_y_min)

            if trap.top is not None:
                if xl < trap.top.xmin:
                    y_top_left = trap.top.y_at(trap.top.xmin)
                if xr > trap.top.xmax:
                    y_top_right = trap.top.y_at(trap.top.xmax)

            if trap.bottom is not None:
                if xl < trap.bottom.xmin:
                    y_bot_left = trap.bottom.y_at(trap.bottom.xmin)
                if xr > trap.bottom.xmax:
                    y_bot_right = trap.bottom.y_at(trap.bottom.xmax)

            vertices = []
            if trap.bottom is not None and trap.bottom.xmin <= xr and trap.bottom.xmax >= xl:
                num_samples = max(2, int((xr - xl) * 10))
                for j in range(num_samples):
                    t = j / (num_samples - 1)
                    x = xl + t * (xr - xl)
                    x_clamped = max(trap.bottom.xmin, min(trap.bottom.xmax, x))
                    y = trap.bottom.y_at(x_clamped)
                    vertices.append([x, y])
            else:
                vertices.append([xl, y_bot_left])
                vertices.append([xr, y_bot_right])

            if trap.top is not None and trap.top.xmin <= xr and trap.top.xmax >= xl:
                num_samples = max(2, int((xr - xl) * 10))
                for j in range(num_samples):
                    t = 1 - j / (num_samples - 1)
                    x = xl + t * (xr - xl)
                    x_clamped = max(trap.top.xmin, min(trap.top.xmax, x))
                    y = trap.top.y_at(x_clamped)
                    vertices.append([x, y])
            else:
                vertices.append([xr, y_top_right])
                vertices.append([xl, y_top_left])

            if len(vertices) >= 3:
                poly = Polygon(vertices, alpha=0.3, facecolor=colors[i],
                             edgecolor='gray', linewidth=0.5)
                ax.add_patch(poly)

            ax.plot([xl, xl], [y_bot_left, y_top_left], 'k-', linewidth=1, alpha=0.5)
            ax.plot([xr, xr], [y_bot_right, y_top_right], 'k-', linewidth=1, alpha=0.5)

        for seg in segments_set:
            ax.plot([seg.left.x, seg.right.x], [seg.left.y, seg.right.y],
                   'b-', linewidth=2, marker='o', markersize=4, label=seg.label)
            ax.text(seg.left.x, seg.left.y, f'  {seg.left.label}',
                   fontsize=8, ha='left', va='bottom')
            ax.text(seg.right.x, seg.right.y, f'  {seg.right.label}',
                   fontsize=8, ha='left', va='bottom')

        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.text(0.02, 0.98, f'Trapezoids: {len(trapezoids_list)}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()


def build_adjmat(trap_map: TrapezoidalMap):
    pass


#####################
#   Orchestration   #
#####################

def parse_input_file():
    parser = argparse.ArgumentParser(description="CSCI-716 Assn3 Trapezoidal Maps & Planar Point Location")
    parser.add_argument("file", type=str, help="Input file containing the planar subdivision data.")
    parser.add_argument("-o", "--output", type=str, default="./out/out.txt", help="Output file to save planar subdivision matrix.")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed for segment insertion order.")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize the segments and trapezoidal map using matplotlib.")
    parser.add_argument("-r", "--randomize", action="store_true", help="Enables use of randomization for incremental construction.")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = int(time.time_ns() % (2**32 - 1))
    else:
        print("A seed parameter (-s <int> flag) was set without enabling randomization (-r flag). It's value will be ignored")
    return args

def main():
    args = parse_input_file()
    input_path = args.file
    output_path = args.output
    seed = args.seed
    visualize = args.visualize
    randomize = args.randomize

    print("="*60)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Random seed: {seed}")
    print("="*60)

    try:
        num_segments, bbox, segments_data = utils.read_file(input_path)
        print(f"\nRead {num_segments} segments from {input_path}")
        min_x, min_y, max_x, max_y = bbox
        print(f"Bounding box: ({min_x}, {min_y}) to ({max_x}, {max_y})")
        print()
    except Exception as e:
        print(f"Error reading input file:", e, file=sys.stderr)
        sys.exit(1)

    trap_map = TrapezoidalMap.from_segments(bbox, segments_data, seed=seed, randomize=randomize)
    print(f"Constructed trapezoidal map with {len(trap_map.trapezoids)} trapezoids")

    if visualize:
        trap_map.visualize(title=f"Trapezoidal Map ({num_segments} segments)")


if __name__ == "__main__":
    main()
