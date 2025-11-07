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
    
    def _find_next_trapezoid(self, current: Trapezoid, seg: Segment) -> Optional[Trapezoid]:
        """
        Find the next trapezoid that the segment enters when exiting current trapezoid.
        Returns None if no valid neighbor is found.
        """
        xr = current.rightx
        y_seg = seg.y_at(xr)

        # Check right neighbors to find which one the segment enters
        for neighbor in current.right_neighbors:
            # Check if segment at x=xr falls within neighbor's vertical span
            y_top = neighbor.y_top(xr, self.bbox_y_max)
            y_bot = neighbor.y_bottom(xr, self.bbox_y_min)

            if y_bot - utils.EPS <= y_seg <= y_top + utils.EPS:
                return neighbor

        return None

    def locate_point(self, p: Point, track_path: bool = False) -> utils.Trapezoid:
        node = self.root
        path = [] if track_path else None

        while not isinstance(node, Leaf):
            if isinstance(node, XNode):
                if track_path:
                    path.append(('X', node.x))
                # get left/right node
                node = node.left if p.x < node.x else node.right
            elif isinstance(node, YNode):
                if track_path:
                    path.append(('Y', node.seg))
                # get above/below segment node
                xq = p.x
                if xq < node.seg.xmin:
                    xq = node.seg.xmin
                elif xq > node.seg.xmax:
                    xq = node.seg.xmax
                y_on_seg = node.seg.y_at(xq)
                node = node.above if p.y > y_on_seg else node.below

        if track_path:
            path.append(('T', node.trap))
            return node.trap, path
        return node.trap
        
    def _insert_segment(self, seg: Segment):
        """Insert segment into trapezoidal map. Expected O(log n) time."""
        # Locate starting trapezoid containing segment's left endpoint
        t = self.locate_point(seg.left)

        crossed_traps: List[Trapezoid] = []
        # (2) Walk through trapezoids intersected by segment from left to right
        while True:
            crossed_traps.append(t)
            # Stop if segment's right endpoint is within current trapezoid
            if seg.right.x <= t.rightx + utils.EPS:
                break

            # Find next trapezoid by checking where segment crosses right boundary
            t = self._find_next_trapezoid(t, seg)
            if t is None:
                raise RuntimeError(f"Could not find next trapezoid for segment {seg.label}")
        # (3) split crossed trapezoids with the segment and build a local sub-DAG
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

        # Update neighbor pointers using helper methods
        new_traps = [left_remainder] + traps_above + traps_below + ([right_remainder] if right_remainder else [])
        self._connect_neighbors(crossed_traps, new_traps)
        self._connect_horizontal_neighbors(traps_above)
        self._connect_horizontal_neighbors(traps_below)
        self._connect_remainder_to_splits(left_remainder, traps_above, traps_below, is_left=True)
        self._connect_remainder_to_splits(right_remainder, traps_above, traps_below, is_left=False)

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

    def _connect_neighbors(self, old_traps: List[Trapezoid], new_traps: List[Trapezoid]):
        """
        Update neighbor pointers when replacing old trapezoids with new ones.
        Efficiently handles O(k × d) neighbor updates instead of O(k²).
        """
        # Collect all neighbors from crossed trapezoids
        all_left_neighbors = set()
        all_right_neighbors = set()
        for trap in old_traps:
            all_left_neighbors.update(trap.left_neighbors)
            all_right_neighbors.update(trap.right_neighbors)

        # Index new trapezoids by boundary coordinates for O(1) lookup
        new_by_leftx = {}
        new_by_rightx = {}
        for trap in new_traps:
            if trap is None:
                continue
            new_by_leftx.setdefault(trap.leftx, []).append(trap)
            new_by_rightx.setdefault(trap.rightx, []).append(trap)

        # Update left neighbors of old trapezoids
        for neighbor in all_left_neighbors:
            if neighbor not in self.trapezoids:
                continue
            neighbor.right_neighbors = [t for t in neighbor.right_neighbors if t not in old_traps]

            # Add geometrically adjacent new trapezoids
            for new_trap in new_by_leftx.get(neighbor.rightx, []):
                if self._trapezoids_are_vertically_adjacent(neighbor, new_trap, neighbor.rightx):
                    neighbor.right_neighbors.append(new_trap)
                    new_trap.left_neighbors.append(neighbor)

        # Update right neighbors of old trapezoids
        for neighbor in all_right_neighbors:
            if neighbor not in self.trapezoids:
                continue
            neighbor.left_neighbors = [t for t in neighbor.left_neighbors if t not in old_traps]

            # Add geometrically adjacent new trapezoids
            for new_trap in new_by_rightx.get(neighbor.leftx, []):
                if self._trapezoids_are_vertically_adjacent(new_trap, neighbor, neighbor.leftx):
                    neighbor.left_neighbors.append(new_trap)
                    new_trap.right_neighbors.append(neighbor)

    def _connect_horizontal_neighbors(self, traps: List[Trapezoid]):
        """Link consecutive trapezoids horizontally."""
        for i in range(len(traps) - 1):
            if traps[i+1] not in traps[i].right_neighbors:
                traps[i].right_neighbors.append(traps[i+1])
                traps[i+1].left_neighbors.append(traps[i])

    def _connect_remainder_to_splits(self, remainder: Optional[Trapezoid],
                                     above_traps: List[Trapezoid],
                                     below_traps: List[Trapezoid],
                                     is_left: bool):
        """Connect left or right remainder to above/below trapezoids."""
        if remainder is None or not above_traps or not below_traps:
            return

        # Left remainder connects to first above/below, right remainder to last
        target_above = above_traps[0] if is_left else above_traps[-1]
        target_below = below_traps[0] if is_left else below_traps[-1]

        if is_left:
            # Left remainder's right neighbors
            if target_above not in remainder.right_neighbors:
                remainder.right_neighbors.append(target_above)
                target_above.left_neighbors.append(remainder)
            if target_below not in remainder.right_neighbors:
                remainder.right_neighbors.append(target_below)
                target_below.left_neighbors.append(remainder)
        else:
            # Right remainder's left neighbors
            if target_above not in remainder.left_neighbors:
                remainder.left_neighbors.append(target_above)
                target_above.right_neighbors.append(remainder)
            if target_below not in remainder.left_neighbors:
                remainder.left_neighbors.append(target_below)
                target_below.right_neighbors.append(remainder)
    
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

    def query_point(self, x: float, y: float) -> Tuple[str, List[Node]]:
        """
        Locate a point and return the traversal path through the DAG as a string.
        Returns path in format: P1 Q1 S1 P3 S3 P4 S4 T7
        """
        p = Point(x, y)

        # Traverse and track actual node objects
        node = self.root
        path_nodes = []

        while not isinstance(node, Leaf):
            path_nodes.append(node)
            if isinstance(node, XNode):
                node = node.left if p.x < node.x else node.right
            elif isinstance(node, YNode):
                xq = p.x
                if xq < node.seg.xmin:
                    xq = node.seg.xmin
                elif xq > node.seg.xmax:
                    xq = node.seg.xmax
                y_on_seg = node.seg.y_at(xq)
                node = node.above if p.y > y_on_seg else node.below

        # Add final leaf node
        path_nodes.append(node)

        # Generate node-to-label mapping
        all_nodes = []
        node_to_index = {}

        def collect_nodes(n: Node):
            if n in node_to_index:
                return
            node_to_index[n] = len(all_nodes)
            all_nodes.append(n)
            if isinstance(n, XNode):
                collect_nodes(n.left)
                collect_nodes(n.right)
            elif isinstance(n, YNode):
                collect_nodes(n.above)
                collect_nodes(n.below)

        collect_nodes(self.root)

        # Separate and label nodes properly
        x_nodes = [n for n in all_nodes if isinstance(n, XNode)]
        y_nodes = [n for n in all_nodes if isinstance(n, YNode)]
        leaf_nodes = [n for n in all_nodes if isinstance(n, Leaf)]

        # Build node-to-label mapping
        node_to_label = {}

        # Sort and label X-nodes as P/Q
        x_nodes_sorted = sorted(x_nodes, key=lambda n: n.x)
        p_counter = 1
        q_counter = 1
        for xnode in x_nodes_sorted:
            # Check if it's a left endpoint
            is_left = any(abs(yseg.seg.left.x - xnode.x) < utils.EPS for yseg in y_nodes)
            if is_left:
                node_to_label[xnode] = f"P{p_counter}"
                p_counter += 1
            else:
                node_to_label[xnode] = f"Q{q_counter}"
                q_counter += 1

        # Label Y-nodes as S1, S2, etc.
        y_nodes_sorted = sorted(y_nodes, key=lambda n: int(n.seg.label[1:]))
        for i, ynode in enumerate(y_nodes_sorted, start=1):
            node_to_label[ynode] = f"S{i}"

        # Label Leaf nodes as T1, T2, etc.
        for i, leaf in enumerate(leaf_nodes, start=1):
            node_to_label[leaf] = f"T{i}"

        # Build path string
        path_labels = [node_to_label.get(n, '?') for n in path_nodes]
        return ' '.join(path_labels), path_nodes

    def generate_adjacency_matrix(self) -> Tuple[List[str], List[List[int]]]:
        """
        Generate the adjacency matrix for the DAG structure.
        Returns (node_labels, matrix) where matrix[child][parent] = 1.
        """
        # Collect all nodes by traversing the DAG
        all_nodes = []
        node_to_index = {}

        def collect_nodes(node: Node):
            if node in node_to_index:
                return
            node_to_index[node] = len(all_nodes)
            all_nodes.append(node)

            if isinstance(node, XNode):
                collect_nodes(node.left)
                collect_nodes(node.right)
            elif isinstance(node, YNode):
                collect_nodes(node.above)
                collect_nodes(node.below)

        collect_nodes(self.root)

        # Separate nodes by type and assign labels
        x_nodes = []  # Point nodes (P and Q)
        y_nodes = []  # Segment nodes (S)
        leaf_nodes = []  # Trapezoid nodes (T)

        for node in all_nodes:
            if isinstance(node, XNode):
                x_nodes.append(node)
            elif isinstance(node, YNode):
                y_nodes.append(node)
            elif isinstance(node, Leaf):
                leaf_nodes.append(node)

        # Create mapping from segments to their labels
        segment_to_label = {}
        for node in y_nodes:
            if node.seg not in segment_to_label:
                segment_to_label[node.seg] = f"S{node.seg.label[1:]}"  # Extract number from label

        # Build ordered node labels: P1...Pk, Q1...Qk, S1...Sn, T1...Tm
        node_labels = []
        ordered_nodes = []

        # Process X-nodes (points)
        point_nodes_with_coords = [(node, node.x) for node in x_nodes]
        point_nodes_with_coords.sort(key=lambda x: x[1])

        p_counter = 1
        q_counter = 1
        for node, x_coord in point_nodes_with_coords:
            # Determine if this is a P or Q node by checking segments
            # This is a heuristic - check if it's a left or right endpoint
            is_left_endpoint = False
            for seg_node in y_nodes:
                if abs(seg_node.seg.left.x - x_coord) < utils.EPS:
                    is_left_endpoint = True
                    break

            if is_left_endpoint:
                label = f"P{p_counter}"
                p_counter += 1
            else:
                label = f"Q{q_counter}"
                q_counter += 1

            node_labels.append(label)
            ordered_nodes.append(node)

        # Process Y-nodes (segments)
        segment_nodes_sorted = sorted(y_nodes, key=lambda n: int(n.seg.label[1:]))
        for node in segment_nodes_sorted:
            node_labels.append(segment_to_label[node.seg])
            ordered_nodes.append(node)

        # Process Leaf nodes (trapezoids)
        for i, node in enumerate(leaf_nodes, start=1):
            node_labels.append(f"T{i}")
            ordered_nodes.append(node)

        # Create index mapping for ordered nodes
        ordered_index = {node: i for i, node in enumerate(ordered_nodes)}

        # Build adjacency matrix: matrix[child][parent] = 1
        n = len(ordered_nodes)
        matrix = [[0] * n for _ in range(n)]

        for parent in ordered_nodes:
            parent_idx = ordered_index[parent]
            children = []

            if isinstance(parent, XNode):
                children = [parent.left, parent.right]
            elif isinstance(parent, YNode):
                children = [parent.above, parent.below]

            for child in children:
                if child in ordered_index:
                    child_idx = ordered_index[child]
                    matrix[child_idx][parent_idx] = 1

        return node_labels, matrix

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

def write_adjacency_matrix(output_path: str, node_labels: List[str], matrix: List[List[int]]):
    """
    Write the adjacency matrix to a file with row and column sums.
    Format:
    - First row: column headers (node labels) + "Sum"
    - Each data row: node label + matrix values + row sum
    - Last row: "Sum" + column sums + total sum
    """
    n = len(node_labels)

    # Calculate row and column sums
    row_sums = [sum(row) for row in matrix]
    col_sums = [sum(matrix[i][j] for i in range(n)) for j in range(n)]
    total_sum = sum(row_sums)

    # Create output directory if it doesn't exist
    import os
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path contains a directory component
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        # Write header row
        f.write("     ")  # Empty cell for row labels
        for label in node_labels:
            f.write(f"{label:>4}")
        f.write("  Sum\n")

        # Write data rows
        for i, label in enumerate(node_labels):
            f.write(f"{label:>4} ")
            for j in range(n):
                f.write(f"{matrix[i][j]:>4}")
            f.write(f"{row_sums[i]:>5}\n")

        # Write sum row
        f.write("Sum  ")
        for col_sum in col_sums:
            f.write(f"{col_sum:>4}")
        f.write(f"{total_sum:>5}\n")

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

    # (3) Generate and write adjacency matrix
    node_labels, matrix = trap_map.generate_adjacency_matrix()
    write_adjacency_matrix(output_path, node_labels, matrix)
    print(f"Adjacency matrix written to {output_path}")
    print(f"DAG contains {len(node_labels)} nodes")

    # (4) Visualize the trapezoidal map
    if visualize:
        trap_map.visualize(title=f"Trapezoidal Map ({num_segments} segments)")

    # (5) Interactive point location query mode
    print("\n" + "="*60)
    print("Point Location Query Mode")
    print("Enter a point (x y) to find its location in the trapezoidal map")
    print("Enter 'quit' or 'exit' to stop")
    print("="*60)

    while True:
        try:
            user_input = input("\nEnter point (x y): ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting query mode.")
                break

            parts = user_input.split()
            if len(parts) != 2:
                print("Error: Please enter exactly two numbers separated by a space (x y)")
                continue

            x, y = float(parts[0]), float(parts[1])

            # Check if point is within bounding box
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                print(f"Warning: Point ({x}, {y}) is outside bounding box!")
                print(f"Bounding box: ({min_x}, {min_y}) to ({max_x}, {max_y})")

            # Perform query
            path_string, _ = trap_map.query_point(x, y)
            print(f"Path: {path_string}")

        except ValueError as e:
            print(f"Error: Invalid input. Please enter two numbers. ({e})")
        except EOFError:
            print("\nExiting query mode.")
            break
        except KeyboardInterrupt:
            print("\nExiting query mode.")
            break


if __name__ == "__main__":
    main()
