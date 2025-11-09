"""
Generate diagrams for the assignment report:
1. Trapezoidal map visualization
2. DAG structure visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import utils
from main import TrapezoidalMap


class DiagramGenerator:
    """Generate publication-quality diagrams for the report."""

    def __init__(self, input_file='./input/benjamin.txt'):
        """Initialize with input file."""
        self.input_file = input_file

        # Read and build the trapezoidal map
        num_segments, bbox, segments_data = utils.read_file(input_file)
        self.trap_map = TrapezoidalMap.from_segments(bbox, segments_data, seed=None, randomize=False)
        self.bbox = bbox

        # Construct segments properly
        self.segments = utils.construct_segments(segments_data)

    def generate_trapezoidal_map_diagram(self, output_file='trapezoidal_map.png'):
        """Generate the trapezoidal map diagram."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        x_min, y_min, x_max, y_max = self.bbox

        # Draw bounding box
        bbox_rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='black', facecolor='none', linestyle='--'
        )
        ax.add_patch(bbox_rect)

        # Color palette for trapezoids
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.trap_map.trapezoids)))

        # Draw trapezoids
        for i, trap in enumerate(self.trap_map.trapezoids):
            # Get trapezoid boundaries
            left_x = trap.leftx
            right_x = trap.rightx

            # Get y-coordinates at left and right
            if trap.top is None:
                top_left_y = top_right_y = y_max
            else:
                top_left_y = trap.top.y_at(left_x)
                top_right_y = trap.top.y_at(right_x)

            if trap.bottom is None:
                bot_left_y = bot_right_y = y_min
            else:
                bot_left_y = trap.bottom.y_at(left_x)
                bot_right_y = trap.bottom.y_at(right_x)

            # Create polygon vertices (counterclockwise)
            vertices = [
                (left_x, bot_left_y),
                (right_x, bot_right_y),
                (right_x, top_right_y),
                (left_x, top_left_y)
            ]

            polygon = patches.Polygon(
                vertices, closed=True,
                facecolor=colors[i], edgecolor='gray',
                linewidth=1, alpha=0.6
            )
            ax.add_patch(polygon)

            # Add trapezoid label at center
            center_x = (left_x + right_x) / 2
            center_y = (top_left_y + top_right_y + bot_left_y + bot_right_y) / 4
            ax.text(center_x, center_y, trap.label,
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Draw segments on top
        for seg in self.segments:
            ax.plot([seg.left.x, seg.right.x], [seg.left.y, seg.right.y],
                   'b-', linewidth=2.5, zorder=10, label=seg.label if seg == self.segments[0] else "")

            # Draw endpoints
            ax.plot(seg.left.x, seg.left.y, 'ro', markersize=8, zorder=11)
            ax.plot(seg.right.x, seg.right.y, 'go', markersize=8, zorder=11)

            # Label endpoints
            ax.text(seg.left.x, seg.left.y + 3, seg.left.label,
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
            ax.text(seg.right.x, seg.right.y + 3, seg.right.label,
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.8))

            # Label segment at midpoint
            mid_x = (seg.left.x + seg.right.x) / 2
            mid_y = (seg.left.y + seg.right.y) / 2
            ax.text(mid_x, mid_y - 4, seg.label,
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.9))

        # Set axis properties
        ax.set_xlim(x_min - 5, x_max + 5)
        ax.set_ylim(y_min - 5, y_max + 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y', fontsize=12, fontweight='bold')
        ax.set_title('Trapezoidal Map - benjamin.txt (12 trapezoids)',
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Trapezoidal map diagram saved to {output_file}")
        plt.close()

    def generate_dag_diagram(self, output_file='dag_structure.png'):
        """Generate the DAG structure diagram."""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.axis('off')

        # Build node hierarchy for layout
        levels = self._build_dag_levels()

        # Position nodes
        node_positions = self._layout_dag_nodes(levels)

        # Draw edges first (so they appear behind nodes)
        self._draw_dag_edges(ax, node_positions)

        # Draw nodes
        self._draw_dag_nodes(ax, node_positions, levels)

        # Add legend
        self._add_dag_legend(ax)

        ax.set_xlim(-1, 17)
        ax.set_ylim(-1, len(levels) + 1)
        ax.set_title('DAG Structure for Point Location (23 nodes)',
                    fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"DAG structure diagram saved to {output_file}")
        plt.close()

    def _build_dag_levels(self):
        """Build level-by-level node hierarchy using BFS."""
        from collections import deque

        levels = []
        visited = set()
        queue = deque([(self.trap_map.root, 0)])

        while queue:
            node, level = queue.popleft()

            if id(node) in visited:
                continue
            visited.add(id(node))

            # Extend levels list if needed
            while len(levels) <= level:
                levels.append([])

            # Add node to this level
            levels[level].append(node)

            # Add children to queue
            if isinstance(node, utils.XNode):
                queue.append((node.left, level + 1))
                queue.append((node.right, level + 1))
            elif isinstance(node, utils.YNode):
                queue.append((node.above, level + 1))
                queue.append((node.below, level + 1))

        return levels

    def _layout_dag_nodes(self, levels):
        """Calculate positions for all nodes."""
        node_positions = {}

        for level_idx, level_nodes in enumerate(levels):
            y = len(levels) - level_idx - 1  # Top to bottom

            # Distribute nodes horizontally
            num_nodes = len(level_nodes)
            if num_nodes == 1:
                x_positions = [8]  # Center
            else:
                x_positions = np.linspace(1, 15, num_nodes)

            for i, node in enumerate(level_nodes):
                node_positions[id(node)] = (x_positions[i], y)

        return node_positions

    def _get_node_label(self, node):
        """Get label for a node."""
        if isinstance(node, utils.Leaf):
            return node.trap.label
        elif isinstance(node, utils.XNode):
            return node.point.label
        elif isinstance(node, utils.YNode):
            return node.seg.label
        return "?"

    def _get_node_color(self, node):
        """Get color for node type."""
        if isinstance(node, utils.Leaf):
            return 'lightgreen'
        elif isinstance(node, utils.XNode):
            return 'lightcoral'
        elif isinstance(node, utils.YNode):
            return 'lightblue'
        return 'white'

    def _draw_dag_nodes(self, ax, node_positions, levels):
        """Draw all DAG nodes."""
        for level_nodes in levels:
            for node in level_nodes:
                x, y = node_positions[id(node)]
                label = self._get_node_label(node)
                color = self._get_node_color(node)

                # Draw node
                if isinstance(node, utils.Leaf):
                    # Trapezoids as rectangles
                    box = FancyBboxPatch(
                        (x - 0.4, y - 0.3), 0.8, 0.6,
                        boxstyle="round,pad=0.05",
                        facecolor=color, edgecolor='black', linewidth=2
                    )
                else:
                    # XNode and YNode as circles
                    circle = plt.Circle((x, y), 0.35, color=color,
                                       ec='black', linewidth=2, zorder=10)
                    ax.add_patch(circle)
                    box = None

                if box:
                    ax.add_patch(box)

                # Add label
                ax.text(x, y, label, ha='center', va='center',
                       fontsize=11, fontweight='bold', zorder=11)

    def _draw_dag_edges(self, ax, node_positions):
        """Draw edges between nodes."""
        visited = set()
        queue = [self.trap_map.root]

        while queue:
            node = queue.pop(0)

            if id(node) in visited:
                continue
            visited.add(id(node))

            if id(node) not in node_positions:
                continue

            x1, y1 = node_positions[id(node)]

            # Draw edges to children
            children = []
            edge_labels = []

            if isinstance(node, utils.XNode):
                children = [node.left, node.right]
                edge_labels = ['x<', 'x≥']
            elif isinstance(node, utils.YNode):
                children = [node.above, node.below]
                edge_labels = ['above', 'below']

            for child, label in zip(children, edge_labels):
                if id(child) in node_positions:
                    x2, y2 = node_positions[id(child)]

                    # Draw arrow
                    arrow = FancyArrowPatch(
                        (x1, y1 - 0.35), (x2, y2 + 0.35),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=1.5, color='gray', zorder=1
                    )
                    ax.add_patch(arrow)

                    # Add edge label
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    ax.text(mid_x + 0.2, mid_y, label, fontsize=8,
                           style='italic', color='darkblue',
                           bbox=dict(boxstyle='round,pad=0.2',
                                   facecolor='white', alpha=0.7))

                    queue.append(child)

    def _add_dag_legend(self, ax):
        """Add legend explaining node types."""
        from matplotlib.patches import Circle, Rectangle

        # Create legend elements
        xnode = Circle((0, 0), 0.2, color='lightcoral', ec='black', linewidth=2)
        ynode = Circle((0, 0), 0.2, color='lightblue', ec='black', linewidth=2)
        leaf = Rectangle((0, 0), 0.4, 0.3, color='lightgreen', ec='black', linewidth=2)

        ax.legend([xnode, ynode, leaf],
                 ['XNode (Point split)', 'YNode (Segment split)', 'Leaf (Trapezoid)'],
                 loc='upper left', fontsize=11, framealpha=0.9)

    def generate_all_diagrams(self):
        """Generate both diagrams."""
        print("Generating diagrams...")
        self.generate_trapezoidal_map_diagram()
        self.generate_dag_diagram()
        print("\nBoth diagrams generated successfully!")
        print("\nNext steps:")
        print("1. Copy trapezoidal_map.png and dag_structure.png to your Overleaf project")
        print("2. Uncomment lines 67 and 78 in report.txt")
        print("3. Compile the LaTeX document")


if __name__ == '__main__':
    # Generate diagrams for benjamin.txt
    generator = DiagramGenerator('./input/benjamin.txt')
    generator.generate_all_diagrams()
