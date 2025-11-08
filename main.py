import argparse
import os, sys
import utils
from utils import Point, Segment, Trapezoid, Node, XNode, YNode, Leaf
from visualize import visualize_trapezoidal_map
import time
import numpy as np
from typing import Optional, Tuple, List


########################################
#   Trapezoidal Map & Point Location   #
########################################

class TrapezoidalMap:
    def __init__(self, bbox: Tuple[float, float, float, float]):
        x_min, _, x_max, _ = bbox
        self.bbox = bbox
        t0 = Trapezoid(top=None, bottom=None, leftx=x_min, rightx=x_max,
                       left_point=None, right_point=None)
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
        
        print("Segments:")
        for seg in labeled_segments:
            print(f"   {seg}")
            trap_map._insert_segment(seg)
        print()
        
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
        """Insert segment into trapezoidal map. Expected O(logn) time."""
        # (1) Locate starting trapezoid containing segment's left endpoint
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

        # (3) Split crossed trapezoids and build sub-DAG
        # This creates new trapezoids and adds them to self.trapezoids
        sub_dag = self._split_crossed(crossed_traps, seg)

        # (4) Splice: replace ALL crossed trapezoid leaves with the same sub-DAG
        for trap in crossed_traps:
            self._splice_trapezoid(trap, sub_dag)

        # (5) Remove old crossed trapezoids from the set
        self.trapezoids.difference_update(crossed_traps)

        # (6) Clean up obsolete neighbor pointers
        for trap in self.trapezoids:
            trap.left_neighbors = [n for n in trap.left_neighbors if n in self.trapezoids]
            trap.right_neighbors = [n for n in trap.right_neighbors if n in self.trapezoids]

    def _update_external_neighbors(self, old_traps: List[Trapezoid], new_traps: List[Trapezoid]):
        """
        Update neighbor pointers of external trapezoids (not in old_traps)
        to point to new trapezoids instead of old trapezoids.
        """
        old_trap_set = set(old_traps)

        # Build spatial index for new trapezoids by their left and right x-coordinates
        new_by_leftx = {}
        new_by_rightx = {}
        for trap in new_traps:
            if trap.leftx not in new_by_leftx:
                new_by_leftx[trap.leftx] = []
            new_by_leftx[trap.leftx].append(trap)

            if trap.rightx not in new_by_rightx:
                new_by_rightx[trap.rightx] = []
            new_by_rightx[trap.rightx].append(trap)

        # Update all trapezoids that have old trapezoids as neighbors
        for trap in self.trapezoids:
            if trap in old_trap_set:
                continue  # Skip trapezoids that are being deleted

            # Update left neighbors
            new_left_neighbors = []
            for neighbor in trap.left_neighbors:
                if neighbor in old_trap_set:
                    # This neighbor is being deleted, find replacement
                    # Look for new trapezoids at the same x-boundary
                    replacements = new_by_rightx.get(trap.leftx, [])
                    for new_trap in replacements:
                        # Check if vertically adjacent
                        if self._trapezoids_are_vertically_adjacent(new_trap, trap, trap.leftx):
                            new_left_neighbors.append(new_trap)
                else:
                    new_left_neighbors.append(neighbor)
            trap.left_neighbors = new_left_neighbors

            # Update right neighbors
            new_right_neighbors = []
            for neighbor in trap.right_neighbors:
                if neighbor in old_trap_set:
                    # This neighbor is being deleted, find replacement
                    # Look for new trapezoids at the same x-boundary
                    replacements = new_by_leftx.get(trap.rightx, [])
                    for new_trap in replacements:
                        # Check if vertically adjacent
                        if self._trapezoids_are_vertically_adjacent(trap, new_trap, trap.rightx):
                            new_right_neighbors.append(new_trap)
                else:
                    new_right_neighbors.append(neighbor)
            trap.right_neighbors = new_right_neighbors

    def _determine_case(self, trap: Trapezoid, seg: Segment) -> int:
        """
        Determine which of the 3 cases applies for inserting a segment into a trapezoid.

        Case 1: Exactly one endpoint is strictly within the trapezoid (not on left/right boundary)
        Case 2: Both endpoints are strictly within the trapezoid
        Case 3: Neither endpoint is strictly within the trapezoid (segment passes through)

        Returns: 1, 2, or 3
        """
        EPS = utils.EPS

        # Check if left endpoint is strictly inside trapezoid (not on boundaries)
        left_in = (trap.leftx + EPS < seg.left.x < trap.rightx - EPS)

        # Check if right endpoint is strictly inside trapezoid (not on boundaries)
        right_in = (trap.leftx + EPS < seg.right.x < trap.rightx - EPS)

        if left_in and right_in:
            return 2  # Both endpoints inside
        elif left_in or right_in:
            return 1  # Exactly one endpoint inside
        else:
            return 3  # Neither endpoint inside (passes through)

    def _build_case3_subdag(self, trap: Trapezoid, seg: Segment) -> Tuple[Node, List[Trapezoid]]:
        """
        Case 3: Neither endpoint in trapezoid (segment passes through).
        Split into 2 trapezoids: above and below.
        Returns: (YNode root, [above_trap, below_trap])
        """
        # Create trapezoid above the segment
        above_trap = Trapezoid(
            top=trap.top,
            bottom=seg,
            leftx=trap.leftx,
            rightx=trap.rightx,
            left_point=trap.left_point,
            right_point=trap.right_point
        )

        # Create trapezoid below the segment
        below_trap = Trapezoid(
            top=seg,
            bottom=trap.bottom,
            leftx=trap.leftx,
            rightx=trap.rightx,
            left_point=trap.left_point,
            right_point=trap.right_point
        )

        # Inherit neighbor pointers from original trapezoid
        above_trap.left_neighbors = trap.left_neighbors.copy()
        above_trap.right_neighbors = trap.right_neighbors.copy()
        below_trap.left_neighbors = trap.left_neighbors.copy()
        below_trap.right_neighbors = trap.right_neighbors.copy()

        # Build sub-DAG: YNode with leaves for above and below
        above_leaf = Leaf(above_trap)
        below_leaf = Leaf(below_trap)
        y_node = YNode(seg, above_leaf, below_leaf)

        return y_node, [above_trap, below_trap]

    def _build_case1_subdag(self, trap: Trapezoid, seg: Segment) -> Tuple[Node, List[Trapezoid]]:
        """
        Case 1: One endpoint in trapezoid (other on or outside boundary).
        Split into 3 trapezoids: remainder (left OR right), above, below.
        Returns: (XNode root, [remainder_trap, above_trap, below_trap])
        """
        EPS = utils.EPS

        # Determine which endpoint is inside
        left_in = (trap.leftx + EPS < seg.left.x < trap.rightx - EPS)
        right_in = (trap.leftx + EPS < seg.right.x < trap.rightx - EPS)

        if left_in:
            # Left endpoint is inside, create left remainder
            endpoint = seg.left
            is_left_remainder = True

            # Remainder trapezoid (from trap.leftx to endpoint.x)
            remainder = Trapezoid(
                top=trap.top,
                bottom=trap.bottom,
                leftx=trap.leftx,
                rightx=endpoint.x,
                left_point=trap.left_point,
                right_point=endpoint
            )

            # Above trapezoid (from endpoint.x to trap.rightx)
            above_trap = Trapezoid(
                top=trap.top,
                bottom=seg,
                leftx=endpoint.x,
                rightx=trap.rightx,
                left_point=endpoint,
                right_point=trap.right_point
            )

            # Below trapezoid (from endpoint.x to trap.rightx)
            below_trap = Trapezoid(
                top=seg,
                bottom=trap.bottom,
                leftx=endpoint.x,
                rightx=trap.rightx,
                left_point=endpoint,
                right_point=trap.right_point
            )

            # Set up neighbor pointers
            remainder.left_neighbors = trap.left_neighbors.copy()
            remainder.right_neighbors = [above_trap, below_trap]
            above_trap.left_neighbors = [remainder]
            above_trap.right_neighbors = trap.right_neighbors.copy()
            below_trap.left_neighbors = [remainder]
            below_trap.right_neighbors = trap.right_neighbors.copy()
        else:
            # Right endpoint is inside, create right remainder
            endpoint = seg.right
            is_left_remainder = False

            # Remainder trapezoid (from endpoint.x to trap.rightx)
            remainder = Trapezoid(
                top=trap.top,
                bottom=trap.bottom,
                leftx=endpoint.x,
                rightx=trap.rightx,
                left_point=endpoint,
                right_point=trap.right_point
            )

            # Above trapezoid (from trap.leftx to endpoint.x)
            above_trap = Trapezoid(
                top=trap.top,
                bottom=seg,
                leftx=trap.leftx,
                rightx=endpoint.x,
                left_point=trap.left_point,
                right_point=endpoint
            )

            # Below trapezoid (from trap.leftx to endpoint.x)
            below_trap = Trapezoid(
                top=seg,
                bottom=trap.bottom,
                leftx=trap.leftx,
                rightx=endpoint.x,
                left_point=trap.left_point,
                right_point=endpoint
            )

            # Set up neighbor pointers
            remainder.left_neighbors = [above_trap, below_trap]
            remainder.right_neighbors = trap.right_neighbors.copy()
            above_trap.left_neighbors = trap.left_neighbors.copy()
            above_trap.right_neighbors = [remainder]
            below_trap.left_neighbors = trap.left_neighbors.copy()
            below_trap.right_neighbors = [remainder]

        # Build sub-DAG
        above_leaf = Leaf(above_trap)
        below_leaf = Leaf(below_trap)
        y_node = YNode(seg, above_leaf, below_leaf)
        remainder_leaf = Leaf(remainder)

        if is_left_remainder:
            # XNode: left = remainder, right = YNode
            x_node = XNode(endpoint, remainder_leaf, y_node)
        else:
            # XNode: left = YNode, right = remainder
            x_node = XNode(endpoint, y_node, remainder_leaf)

        return x_node, [remainder, above_trap, below_trap]

    def _build_case2_subdag(self, trap: Trapezoid, seg: Segment) -> Tuple[Node, List[Trapezoid]]:
        """
        Case 2: Both endpoints in same trapezoid.
        Split into 4 trapezoids: left_remainder, right_remainder, above, below.
        Returns: (XNode root, [left_rem, right_rem, above_trap, below_trap])
        """
        # Left remainder (from trap.leftx to seg.left.x)
        left_remainder = Trapezoid(
            top=trap.top,
            bottom=trap.bottom,
            leftx=trap.leftx,
            rightx=seg.left.x,
            left_point=trap.left_point,
            right_point=seg.left
        )

        # Right remainder (from seg.right.x to trap.rightx)
        right_remainder = Trapezoid(
            top=trap.top,
            bottom=trap.bottom,
            leftx=seg.right.x,
            rightx=trap.rightx,
            left_point=seg.right,
            right_point=trap.right_point
        )

        # Above trapezoid (from seg.left.x to seg.right.x)
        above_trap = Trapezoid(
            top=trap.top,
            bottom=seg,
            leftx=seg.left.x,
            rightx=seg.right.x,
            left_point=seg.left,
            right_point=seg.right
        )

        # Below trapezoid (from seg.left.x to seg.right.x)
        below_trap = Trapezoid(
            top=seg,
            bottom=trap.bottom,
            leftx=seg.left.x,
            rightx=seg.right.x,
            left_point=seg.left,
            right_point=seg.right
        )

        # Set up neighbor pointers
        left_remainder.left_neighbors = trap.left_neighbors.copy()
        left_remainder.right_neighbors = [above_trap, below_trap]
        right_remainder.left_neighbors = [above_trap, below_trap]
        right_remainder.right_neighbors = trap.right_neighbors.copy()
        above_trap.left_neighbors = [left_remainder]
        above_trap.right_neighbors = [right_remainder]
        below_trap.left_neighbors = [left_remainder]
        below_trap.right_neighbors = [right_remainder]

        # Build sub-DAG:
        # XNode(left_pt) -> left_rem, XNode(right_pt) -> YNode(seg) -> {above, below}, right_rem
        above_leaf = Leaf(above_trap)
        below_leaf = Leaf(below_trap)
        y_node = YNode(seg, above_leaf, below_leaf)

        right_remainder_leaf = Leaf(right_remainder)
        right_x_node = XNode(seg.right, y_node, right_remainder_leaf)

        left_remainder_leaf = Leaf(left_remainder)
        left_x_node = XNode(seg.left, left_remainder_leaf, right_x_node)

        return left_x_node, [left_remainder, right_remainder, above_trap, below_trap]

    def _split_crossed(self, crossed_traps: List[Trapezoid], seg: Segment) -> Node:
        """
        Split crossed trapezoids and build a single sub-DAG for the segment insertion.
        The 3 cases are:
        - Case 1: One endpoint inside the first/last crossed trapezoid
        - Case 2: Both endpoints in the same trapezoid (only 1 trapezoid crossed)
        - Case 3: Segment passes through (endpoints on boundaries or multiple trapezoids)
        Returns: Root node of the sub-DAG
        """
        t_first, t_last = crossed_traps[0], crossed_traps[-1]
        EPS = utils.EPS

        # Determine the effective left/right x-coordinates for the segment within crossed region
        seg_left_x = max(seg.left.x, t_first.leftx)
        seg_right_x = min(seg.right.x, t_last.rightx)

        # Check if segment endpoints are strictly inside the first/last trapezoids
        left_endpoint_inside = (t_first.leftx + EPS < seg.left.x < t_first.rightx - EPS)
        right_endpoint_inside = (t_last.leftx + EPS < seg.right.x < t_last.rightx - EPS)

        # Determine which case
        only_one_trapezoid = (len(crossed_traps) == 1)

        if only_one_trapezoid and left_endpoint_inside and right_endpoint_inside:
            # CASE 2: Both endpoints strictly inside one trapezoid
            sub_dag, new_traps = self._build_case2_subdag(t_first, seg)
            # Add new trapezoids to the set and update neighbors
            for trap in new_traps:
                self.trapezoids.add(trap)
            # Connect neighbors
            self._connect_neighbors(crossed_traps, new_traps)
            self._connect_horizontal_neighbors([new_traps[2]])  # above trapezoid
            self._connect_horizontal_neighbors([new_traps[3]])  # below trapezoid
            self._connect_remainder_to_splits(new_traps[0], [new_traps[2]], [new_traps[3]], is_left=True)  # left_remainder
            self._connect_remainder_to_splits(new_traps[1], [new_traps[2]], [new_traps[3]], is_left=False)  # right_remainder
            return sub_dag

        # Build remainders if endpoints are strictly inside boundaries
        left_remainder = None
        right_remainder = None

        if left_endpoint_inside:
            # CASE 1: Left endpoint inside, create left remainder
            left_remainder = Trapezoid(
                top=t_first.top,
                bottom=t_first.bottom,
                leftx=t_first.leftx,
                rightx=seg.left.x,
                left_point=t_first.left_point,
                right_point=seg.left
            )
            left_remainder.left_neighbors = t_first.left_neighbors.copy()
            self.trapezoids.add(left_remainder)

        if right_endpoint_inside:
            # CASE 1: Right endpoint inside, create right remainder
            right_remainder = Trapezoid(
                top=t_last.top,
                bottom=t_last.bottom,
                leftx=seg.right.x,
                rightx=t_last.rightx,
                left_point=seg.right,
                right_point=t_last.right_point
            )
            right_remainder.right_neighbors = t_last.right_neighbors.copy()
            self.trapezoids.add(right_remainder)

        # Merge consecutive crossed trapezoids with same top/bottom
        traps_above, traps_below = [], []

        # Merge above trapezoids
        i = 0
        while i < len(crossed_traps):
            same_top = crossed_traps[i].top
            j = i
            while j < len(crossed_traps) and crossed_traps[j].top == same_top:
                j += 1

            # Determine boundaries for merged trapezoid
            left_x = seg_left_x if i == 0 else crossed_traps[i].leftx
            right_x = seg_right_x if j == len(crossed_traps) else crossed_traps[j-1].rightx
            left_pt = seg.left if i == 0 and left_endpoint_inside else crossed_traps[i].left_point
            right_pt = seg.right if j == len(crossed_traps) and right_endpoint_inside else crossed_traps[j-1].right_point

            t_upper = Trapezoid(same_top, seg, left_x, right_x, left_pt, right_pt)
            self.trapezoids.add(t_upper)
            traps_above.append(t_upper)
            i = j

        # Merge below trapezoids
        i = 0
        while i < len(crossed_traps):
            same_bottom = crossed_traps[i].bottom
            j = i
            while j < len(crossed_traps) and crossed_traps[j].bottom == same_bottom:
                j += 1

            # Determine boundaries for merged trapezoid
            left_x = seg_left_x if i == 0 else crossed_traps[i].leftx
            right_x = seg_right_x if j == len(crossed_traps) else crossed_traps[j-1].rightx
            left_pt = seg.left if i == 0 and left_endpoint_inside else crossed_traps[i].left_point
            right_pt = seg.right if j == len(crossed_traps) and right_endpoint_inside else crossed_traps[j-1].right_point

            t_lower = Trapezoid(seg, same_bottom, left_x, right_x, left_pt, right_pt)
            self.trapezoids.add(t_lower)
            traps_below.append(t_lower)
            i = j

        # Update neighbor pointers
        new_traps = ([left_remainder] if left_remainder else []) + traps_above + traps_below + ([right_remainder] if right_remainder else [])
        self._connect_neighbors(crossed_traps, new_traps)
        self._connect_horizontal_neighbors(traps_above)
        self._connect_horizontal_neighbors(traps_below)
        self._connect_remainder_to_splits(left_remainder, traps_above, traps_below, is_left=True)
        self._connect_remainder_to_splits(right_remainder, traps_above, traps_below, is_left=False)

        # Build the sub-DAG correctly based on the case
        # Create YNode for the segment with above/below as single leaves
        above_node = Leaf(traps_above[0]) if len(traps_above) == 1 else self._build_chain(traps_above)
        below_node = Leaf(traps_below[0]) if len(traps_below) == 1 else self._build_chain(traps_below)
        y_node = YNode(seg, above_node, below_node)

        # Wrap with XNodes for endpoints if needed
        if left_remainder:
            y_node = XNode(seg.left, Leaf(left_remainder), y_node)
        if right_remainder:
            y_node = XNode(seg.right, y_node, Leaf(right_remainder))

        return y_node

    def _build_chain(self, traps: List[Trapezoid]) -> Node:
        """Build a left-to-right chain of trapezoids with XNodes at boundaries."""
        if len(traps) == 1:
            return Leaf(traps[0])

        # Sort left to right
        traps_sorted = sorted(traps, key=lambda t: t.leftx)

        # Build chain from right to left
        result = Leaf(traps_sorted[-1])
        for i in range(len(traps_sorted) - 2, -1, -1):
            trap = traps_sorted[i]
            split_point = trap.right_point
            result = XNode(split_point, Leaf(trap), result)

        return result

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
        Update neighbor pointers when replacing old trapezoids with new ones (O(kd) updates).
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
        t = Trapezoid(None, None, self.bbox_x_min, self.bbox_x_min, None, None)
        self.trapezoids.add(t)
        return t
    
    def _splice_trapezoid(self, old_trap: Trapezoid, new_subdag: Node):
        """
        Replace a single trapezoid's leaf with its sub-DAG using parent pointers.
        O(1) operation.
        """
        leaf = old_trap.leaf
        if leaf is None:
            return

        parent = leaf.parent

        if parent is None:
            # This leaf is the root of the entire DAG
            self.root = new_subdag
            new_subdag.parent = None
        elif isinstance(parent, XNode):
            # Replace in parent's left or right child
            if parent.left == leaf:
                parent.left = new_subdag  # Property setter handles parent pointer
            else:
                parent.right = new_subdag
        elif isinstance(parent, YNode):
            # Replace in parent's above or below child
            if parent.above == leaf:
                parent.above = new_subdag
            else:
                parent.below = new_subdag

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

        # Build path string using existing labels from Point/Segment/Trapezoid objects
        path_labels = []
        for n in path_nodes:
            if isinstance(n, XNode):
                # Use the label from the Point object
                path_labels.append(n.point.label)
            elif isinstance(n, YNode):
                # Use the label from the Segment object
                path_labels.append(n.seg.label)
            elif isinstance(n, Leaf):
                # Use the label from the Trapezoid object
                path_labels.append(n.trap.label if n.trap.label else "T?")

        return ' '.join(path_labels), path_nodes

    def generate_adjacency_matrix(self) -> Tuple[List[str], List[List[int]]]:
        """
        Generate the adjacency matrix for the DAG structure.
        Nodes referencing the same Point/Segment/Trapezoid are merged into a single entry.
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

        # Group nodes by their underlying Point/Segment/Trapezoid objects
        point_to_xnodes = {}    # Point object id → list of XNodes
        seg_to_ynodes = {}      # Segment object id → list of YNodes
        trap_to_leaves = {}     # Trapezoid object id → list of Leaves

        for node in all_nodes:
            if isinstance(node, XNode):
                point_id = id(node.point)
                point_to_xnodes.setdefault(point_id, []).append(node)
            elif isinstance(node, YNode):
                seg_id = id(node.seg)
                seg_to_ynodes.setdefault(seg_id, []).append(node)
            elif isinstance(node, Leaf):
                trap_id = id(node.trap)
                trap_to_leaves.setdefault(trap_id, []).append(node)

        # Create unique representative nodes (one per unique object)
        # Store both the representative object and the label
        unique_points = {}      # point_id → (Point object, list of XNodes)
        unique_segments = {}    # seg_id → (Segment object, list of YNodes)
        unique_traps = {}       # trap_id → (Trapezoid object, list of Leaves)

        for point_id, xnodes in point_to_xnodes.items():
            unique_points[point_id] = (xnodes[0].point, xnodes)

        for seg_id, ynodes in seg_to_ynodes.items():
            unique_segments[seg_id] = (ynodes[0].seg, ynodes)

        for trap_id, leaves in trap_to_leaves.items():
            unique_traps[trap_id] = (leaves[0].trap, leaves)

        # Build ordered list of unique objects with labels
        # Order: P1, P2, ..., Q1, Q2, ..., S1, S2, ..., T1, T2, ...
        node_labels = []
        ordered_objects = []  # List of (type, object_id, label)

        # Separate P and Q nodes
        p_nodes = []  # Left endpoints (P1, P2, ...)
        q_nodes = []  # Right endpoints (Q1, Q2, ...)

        for point_id, (point, xnodes) in unique_points.items():
            if point.label.startswith('P'):
                p_nodes.append((point_id, point))
            else:  # Q nodes
                q_nodes.append((point_id, point))

        # Sort P nodes by label number
        p_nodes_sorted = sorted(p_nodes, key=lambda item: int(item[1].label[1:]))
        for point_id, point in p_nodes_sorted:
            node_labels.append(point.label)
            ordered_objects.append(('point', point_id, point.label))

        # Sort Q nodes by label number
        q_nodes_sorted = sorted(q_nodes, key=lambda item: int(item[1].label[1:]))
        for point_id, point in q_nodes_sorted:
            node_labels.append(point.label)
            ordered_objects.append(('point', point_id, point.label))

        # Sort Segments by label number
        segments_sorted = sorted(unique_segments.items(),
                                key=lambda item: int(item[1][0].label[1:]))
        for seg_id, (seg, _) in segments_sorted:
            node_labels.append(seg.label)
            ordered_objects.append(('segment', seg_id, seg.label))

        # Sort Trapezoids by label number
        traps_sorted = sorted(unique_traps.items(),
                             key=lambda item: int(item[1][0].label[1:]) if item[1][0].label and item[1][0].label[0] == 'T' else float('inf'))
        for trap_id, (trap, _) in traps_sorted:
            label = trap.label if trap.label else "T?"
            node_labels.append(label)
            ordered_objects.append(('trapezoid', trap_id, label))

        # Create mapping from object_id to matrix index
        object_to_index = {}
        for idx, (obj_type, obj_id, _) in enumerate(ordered_objects):
            object_to_index[(obj_type, obj_id)] = idx

        # Build adjacency matrix by merging relationships from all nodes with same object
        n = len(ordered_objects)
        matrix = [[0] * n for _ in range(n)]

        # Helper to get object_id and type for a node
        def get_object_key(node: Node) -> Tuple[str, int]:
            if isinstance(node, XNode):
                return ('point', id(node.point))
            elif isinstance(node, YNode):
                return ('segment', id(node.seg))
            elif isinstance(node, Leaf):
                return ('trapezoid', id(node.trap))
            return None

        # Collect all parent-child relationships
        for obj_type, obj_id, _ in ordered_objects:
            # Get all nodes with this object
            if obj_type == 'point':
                nodes = unique_points[obj_id][1]
            elif obj_type == 'segment':
                nodes = unique_segments[obj_id][1]
            else:  # trapezoid
                nodes = unique_traps[obj_id][1]

            parent_idx = object_to_index[(obj_type, obj_id)]

            # Collect all children from all nodes with this object
            for node in nodes:
                children = []
                if isinstance(node, XNode):
                    children = [node.left, node.right]
                elif isinstance(node, YNode):
                    children = [node.above, node.below]

                for child in children:
                    child_key = get_object_key(child)
                    if child_key and child_key in object_to_index:
                        child_idx = object_to_index[child_key]
                        matrix[child_idx][parent_idx] = 1

        return node_labels, matrix


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
        visualize_trapezoidal_map(trap_map, title=f"Trapezoidal Map ({num_segments} segments)")

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
