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
        # init bbox to be the first trapezoid
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
    def from_segments(cls, bbox: Tuple, segments: List[Tuple], seed: Optional[float]) -> "TrapezoidalMap":
        # (1) get segments from raw data
        labeled_segments: List[Segment] = utils.construct_segments(segments)
        # (2) randomize the segments (in-place)
        rng = np.random.default_rng(seed)
        rng.shuffle(labeled_segments)
        # (3) initialize trapezoidal map
        trap_map = cls(bbox)
        # (4) iteratively add segments to the trapezoidal map
        for seg in labeled_segments:
            trap_map._insert_segment(seg)
        return trap_map
    
    def locate_point(self, p: Point) -> utils.Trapezoid:
        node = self.root
        while not isinstance(node, Leaf):
            if isinstance(node, XNode):
                # get left/right node
                node = node.left if p.x < node.x else node.right
            elif isinstance(node, YNode):
                # get above/below segment node
                xq = p.x
                if xq < node.seg.xmin:
                    xq = node.seg.xmin
                elif xq > node.seg.xmax:
                    xq = node.seg.xmax
                y_on_seg = node.seg.y_at(xq)
                node = node.above if p.y > y_on_seg else node.below
        return node.trap
        
    def _insert_segment(self, seg: Segment):
        # (1) Locate trapezoid containing left endpoint
        t = self.locate_point(seg.left)
        # If locate_point returns an obsolete trapezoid, find a valid one at the same point
        while t not in self.trapezoids:
            # This shouldn't happen often, but if it does, search for a valid trapezoid
            # by checking the trapezoids set
            for candidate in self.trapezoids:
                if (candidate.leftx - utils.EPS <= seg.left.x <= candidate.rightx + utils.EPS):
                    y_top = candidate.y_top(seg.left.x, self.bbox_y_max)
                    y_bot = candidate.y_bottom(seg.left.x, self.bbox_y_min)
                    if y_bot - utils.EPS <= seg.left.y <= y_top + utils.EPS:
                        t = candidate
                        break
            if t not in self.trapezoids:
                raise RuntimeError(f"Could not find valid trapezoid for start point {seg.left}")

        crossed_traps: List[Trapezoid] = []
        # (2) walk through trapezoids intersected by s from left to right
        while True:
            crossed_traps.append(t)
            if seg.right.x <= t.rightx + utils.EPS:
                break
            # Find the next trapezoid by checking right neighbors
            # The segment intersects the right boundary at x = t.rightx
            xr = t.rightx
            y_seg = seg.y_at(xr)

            # Find which right neighbor the segment enters
            # Skip obsolete neighbors not in the trapezoids set
            next_trap = None
            for neighbor in t.right_neighbors:
                # Skip obsolete trapezoids
                if neighbor not in self.trapezoids:
                    continue

                # Check if the segment at x=xr falls within this neighbor's vertical span
                y_top = neighbor.y_top(xr, self.bbox_y_max)
                y_bot = neighbor.y_bottom(xr, self.bbox_y_min)

                # The segment enters this neighbor if y_seg is between its top and bottom
                if y_bot - utils.EPS <= y_seg <= y_top + utils.EPS:
                    next_trap = neighbor
                    break

            if next_trap is None:
                # Fallback: shouldn't happen in valid trapezoidal map
                raise RuntimeError(f"Could not find next trapezoid at x={xr}, y={y_seg}")

            t = next_trap
        # (3) split crossed trapezoids with the segment and build a local sub-DAG
        sub = self._split_crossed(crossed_traps, seg)
        # (4) splice the sub-DAG into the global DAG, replacing the first crossed leaf
        self._splice(crossed_traps, sub)
        # (5) remove all crossed trapezoids since they've been replaced
        self.trapezoids.difference_update(crossed_traps)
        # (6) Final cleanup: remove any remaining obsolete neighbors from all trapezoids
        for trap in self.trapezoids:
            trap.left_neighbors = [n for n in trap.left_neighbors if n in self.trapezoids]
            trap.right_neighbors = [n for n in trap.right_neighbors if n in self.trapezoids]
    
    def _split_crossed(self, crossed_traps: List[Trapezoid], seg: Segment) -> Node:
        """
        Split crossed trapezoids with a new segment, creating trapezoids above and below.
        This implementation ensures proper neighbor pointer management for all insertion orders.
        """
        traps_above: List[Trapezoid] = []
        traps_below: List[Trapezoid] = []
        left_remainder: Optional[Trapezoid] = None
        right_remainder: Optional[Trapezoid] = None

        t_first, t_last = crossed_traps[0], crossed_traps[-1]

        # Determine the x-range that the segment covers
        seg_left_x = max(seg.xmin, t_first.leftx)
        seg_right_x = min(seg.xmax, t_last.rightx)

        # Create left remainder if needed
        if seg_left_x - t_first.leftx > utils.EPS:
            left_remainder = Trapezoid(t_first.top, t_first.bottom, t_first.leftx, seg_left_x)
            left_remainder.left_neighbors = t_first.left_neighbors.copy()
            self.trapezoids.add(left_remainder)

        # Create right remainder if needed
        if t_last.rightx - seg_right_x > utils.EPS:
            right_remainder = Trapezoid(t_last.top, t_last.bottom, seg_right_x, t_last.rightx)
            right_remainder.right_neighbors = t_last.right_neighbors.copy()
            self.trapezoids.add(right_remainder)

        # Merge consecutive crossed trapezoids with the same top/bottom into single trapezoids
        # For trapezoids above the segment
        i = 0
        while i < len(crossed_traps):
            same_top = crossed_traps[i].top
            j = i
            while j < len(crossed_traps) and crossed_traps[j].top == same_top:
                j += 1

            # Determine x-coordinates
            if i == 0:
                left_x = seg_left_x
            else:
                left_x = crossed_traps[i-1].rightx

            if j == len(crossed_traps):
                right_x = seg_right_x
            else:
                right_x = crossed_traps[j-1].rightx

            t_upper = Trapezoid(same_top, seg, left_x, right_x)
            self.trapezoids.add(t_upper)
            traps_above.append(t_upper)
            i = j

        # For trapezoids below the segment
        i = 0
        while i < len(crossed_traps):
            same_bottom = crossed_traps[i].bottom
            j = i
            while j < len(crossed_traps) and crossed_traps[j].bottom == same_bottom:
                j += 1

            if i == 0:
                left_x = seg_left_x
            else:
                left_x = crossed_traps[i-1].rightx

            if j == len(crossed_traps):
                right_x = seg_right_x
            else:
                right_x = crossed_traps[j-1].rightx

            t_lower = Trapezoid(seg, same_bottom, left_x, right_x)
            self.trapezoids.add(t_lower)
            traps_below.append(t_lower)
            i = j

        # ===================================================================
        # Optimized neighbor pointer management - O(k × d) instead of O(k²)
        # ===================================================================

        # Collect ALL neighbors from ALL crossed trapezoids
        all_left_neighbors = set()
        all_right_neighbors = set()
        for trap in crossed_traps:
            all_left_neighbors.update(trap.left_neighbors)
            all_right_neighbors.update(trap.right_neighbors)

        # Group new trapezoids by their boundary x-coordinates for O(1) lookup
        # This optimization reduces complexity from O(neighbors × new_traps) to O(neighbors + new_traps)
        new_traps_by_leftx = {}  # Maps x-coordinate to list of trapezoids with that left boundary
        new_traps_by_rightx = {}  # Maps x-coordinate to list of trapezoids with that right boundary

        for trap in [left_remainder] + traps_above + traps_below + ([right_remainder] if right_remainder else []):
            if trap is None:
                continue
            # Index by left boundary
            if trap.leftx not in new_traps_by_leftx:
                new_traps_by_leftx[trap.leftx] = []
            new_traps_by_leftx[trap.leftx].append(trap)
            # Index by right boundary
            if trap.rightx not in new_traps_by_rightx:
                new_traps_by_rightx[trap.rightx] = []
            new_traps_by_rightx[trap.rightx].append(trap)

        # Update left neighbors: replace crossed trapezoids with appropriate new ones
        for neighbor in all_left_neighbors:
            if neighbor not in self.trapezoids:
                continue

            # Remove all crossed trapezoids from right neighbors
            neighbor.right_neighbors = [t for t in neighbor.right_neighbors if t not in crossed_traps]

            # Add new trapezoids that are geometrically adjacent
            # Only check trapezoids whose left boundary matches this neighbor's right boundary
            neighbor_rightx = neighbor.rightx
            candidates = new_traps_by_leftx.get(neighbor_rightx, [])
            for new_trap in candidates:
                if self._trapezoids_are_vertically_adjacent(neighbor, new_trap, neighbor_rightx):
                    neighbor.right_neighbors.append(new_trap)
                    new_trap.left_neighbors.append(neighbor)

        # Update right neighbors: replace crossed trapezoids with appropriate new ones
        for neighbor in all_right_neighbors:
            if neighbor not in self.trapezoids:
                continue

            # Remove all crossed trapezoids from left neighbors
            neighbor.left_neighbors = [t for t in neighbor.left_neighbors if t not in crossed_traps]

            # Add new trapezoids that are geometrically adjacent
            # Only check trapezoids whose right boundary matches this neighbor's left boundary
            neighbor_leftx = neighbor.leftx
            candidates = new_traps_by_rightx.get(neighbor_leftx, [])
            for new_trap in candidates:
                if self._trapezoids_are_vertically_adjacent(new_trap, neighbor, neighbor_leftx):
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
        """
        Check if two trapezoids are vertically adjacent at a given x boundary.
        They are adjacent if their vertical ranges overlap at the boundary.
        """
        # Get y-ranges at the boundary
        left_y_top = left_trap.y_top(x_boundary, self.bbox_y_max)
        left_y_bot = left_trap.y_bottom(x_boundary, self.bbox_y_min)

        right_y_top = right_trap.y_top(x_boundary, self.bbox_y_max)
        right_y_bot = right_trap.y_bottom(x_boundary, self.bbox_y_min)

        # Check if ranges overlap (with epsilon tolerance)
        overlap_bot = max(left_y_bot, right_y_bot)
        overlap_top = min(left_y_top, right_y_top)

        return overlap_bot <= overlap_top + utils.EPS
    
    def _dummy_empty_trap(self) -> Trapezoid:
        t = Trapezoid(None, None, self.bbox_x_min, self.bbox_x_min)
        self.trapezoids.add(t)
        return t
    
    def _splice(self, crossed_traps: List[Trapezoid], sub_dag: Node):
        """
        Replace crossed trapezoids' leaves in the DAG with the new sub-DAG.

        Expected complexity: O(k × log n) where k = # crossed trapezoids
        - Uses backward pointers to identify leaves in O(k)
        - For each leaf, traverses from root to update parent in O(log n) expected

        Total expected over n insertions: O(n log n) since E[k] = O(1)
        """
        if not crossed_traps:
            return

        # Collect all leaves that need to be replaced using backward pointers - O(k)
        leaves_to_replace = set()
        for trap in crossed_traps:
            if trap.leaf is not None:
                leaves_to_replace.add(trap.leaf)

        if not leaves_to_replace:
            return

        # Replace all crossed leaves in a single traversal
        # Expected depth O(log n), but we traverse once, checking k leaves
        def replace_in(node: Node) -> Node:
            if node in leaves_to_replace:
                return sub_dag
            if isinstance(node, XNode):
                left_result = replace_in(node.left)
                right_result = replace_in(node.right)
                # Optimization: if both children become the same sub_dag, collapse
                if left_result is right_result and left_result is sub_dag:
                    return sub_dag
                node.left = left_result
                node.right = right_result
                return node
            if isinstance(node, YNode):
                above_result = replace_in(node.above)
                below_result = replace_in(node.below)
                # Optimization: if both children become the same sub_dag, collapse
                if above_result is below_result and above_result is sub_dag:
                    return sub_dag
                node.above = above_result
                node.below = below_result
                return node
            return node

        self.root = replace_in(self.root)

    def visualize(self, title: str = "Trapezoidal Map"):
        """
        Visualize the trapezoidal map using matplotlib.
        Shows all trapezoids with their boundaries and the line segments.
        """
        # Convert set to list for visualization
        trapezoids_list = list(self.trapezoids)

        _, ax = plt.subplots(figsize=(12, 8))
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')

        # Set up the plot bounds with some padding
        padding = 1.0
        ax.set_xlim(self.bbox_x_min - padding, self.bbox_x_max + padding)
        ax.set_ylim(self.bbox_y_min - padding, self.bbox_y_max + padding)

        # Draw bounding box
        bbox_rect = Rectangle(
            (self.bbox_x_min, self.bbox_y_min),
            self.bbox_x_max - self.bbox_x_min,
            self.bbox_y_max - self.bbox_y_min,
            linewidth=2, edgecolor='black', facecolor='none', linestyle='--'
        )
        ax.add_patch(bbox_rect)

        # Collect all unique segments from trapezoids
        segments_set = set()
        for trap in trapezoids_list:
            if trap.top is not None:
                segments_set.add(trap.top)
            if trap.bottom is not None:
                segments_set.add(trap.bottom)

        # Draw each trapezoid
        colors = plt.cm.Set3(np.linspace(0, 1, len(trapezoids_list)))
        for i, trap in enumerate(trapezoids_list):
            # Get the four corners of the trapezoid
            xl, xr = trap.leftx, trap.rightx

            # Top boundary
            y_top_left = trap.y_top(xl, self.bbox_y_max)
            y_top_right = trap.y_top(xr, self.bbox_y_max)

            # Bottom boundary
            y_bot_left = trap.y_bottom(xl, self.bbox_y_min)
            y_bot_right = trap.y_bottom(xr, self.bbox_y_min)

            # Handle segments that might not span the full x-range
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

            # Create polygon for the trapezoid (counter-clockwise)
            # Handle potential degenerate cases where segments extend beyond vertical bounds
            vertices = []

            # Bottom-left to bottom-right
            if trap.bottom is not None and trap.bottom.xmin <= xr and trap.bottom.xmax >= xl:
                # Sample points along the bottom segment
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

            # Top-right to top-left
            if trap.top is not None and trap.top.xmin <= xr and trap.top.xmax >= xl:
                # Sample points along the top segment (reverse order)
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

            # Draw the trapezoid
            if len(vertices) >= 3:
                poly = Polygon(vertices, alpha=0.3, facecolor=colors[i],
                             edgecolor='gray', linewidth=0.5)
                ax.add_patch(poly)

            # Draw vertical boundaries of the trapezoid
            ax.plot([xl, xl], [y_bot_left, y_top_left], 'k-', linewidth=1, alpha=0.5)
            ax.plot([xr, xr], [y_bot_right, y_top_right], 'k-', linewidth=1, alpha=0.5)

        # Draw all segments on top
        for seg in segments_set:
            ax.plot([seg.left.x, seg.right.x], [seg.left.y, seg.right.y],
                   'b-', linewidth=2, marker='o', markersize=4, label=seg.label)

            # Label the segment endpoints
            ax.text(seg.left.x, seg.left.y, f'  {seg.left.label}',
                   fontsize=8, ha='left', va='bottom')
            ax.text(seg.right.x, seg.right.y, f'  {seg.right.label}',
                   fontsize=8, ha='left', va='bottom')

        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Add text showing number of trapezoids
        ax.text(0.02, 0.98, f'Trapezoids: {len(trapezoids_list)}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()


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
    
    # visualization of raw segments in bounding box using matplotlib
    # if visualize:
    #     utils.visualize_input_segments(segments_data, bbox)


    # (2) Build trapezoidal map (uses randomized-incremental algorithm)
    trap_map = TrapezoidalMap.from_segments(bbox, segments_data, seed=seed)
    print(f"Constructed trapezoidal map with {len(trap_map.trapezoids)} trapezoids")

    # (3) Visualize the trapezoidal map
    if visualize:
        trap_map.visualize(title=f"Trapezoidal Map ({num_segments} segments)")


if __name__ == "__main__":
    main()
