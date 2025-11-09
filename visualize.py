"""
Visualization utilities for trapezoidal maps.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def visualize_trapezoidal_map(trap_map, title: str = "Trapezoidal Map"):
    """Visualize trapezoidal map using matplotlib."""
    _, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')

    bbox = trap_map.bbox
    padding = 2.0
    ax.set_xlim(bbox[0] - padding, bbox[2] + padding)
    ax.set_ylim(bbox[1] - padding, bbox[3] + padding)

    # Draw bounding box
    from matplotlib.patches import Rectangle
    bbox_rect = Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=2, edgecolor='black', facecolor='none', linestyle='--'
    )
    ax.add_patch(bbox_rect)

    # Collect unique segments
    segments_set = set()
    for trap in trap_map.trapezoids:
        if trap.top:
            segments_set.add(trap.top)
        if trap.bottom:
            segments_set.add(trap.bottom)

    # Draw trapezoids
    colors = plt.cm.Set3(np.linspace(0, 1, len(trap_map.trapezoids)))
    for i, trap in enumerate(trap_map.trapezoids):
        xl, xr = trap.leftx, trap.rightx
        y_top_left = trap.y_top(xl, bbox[3])
        y_top_right = trap.y_top(xr, bbox[3])
        y_bot_left = trap.y_bottom(xl, bbox[1])
        y_bot_right = trap.y_bottom(xr, bbox[1])

        # Build trapezoid vertices (counter-clockwise)
        vertices = [
            [xl, y_bot_left],
            [xr, y_bot_right],
            [xr, y_top_right],
            [xl, y_top_left]
        ]

        poly = Polygon(vertices, alpha=0.3, facecolor=colors[i],
                      edgecolor='gray', linewidth=0.5)
        ax.add_patch(poly)

        # Draw vertical boundaries
        ax.plot([xl, xl], [y_bot_left, y_top_left], 'k-', linewidth=1, alpha=0.5)
        ax.plot([xr, xr], [y_bot_right, y_top_right], 'k-', linewidth=1, alpha=0.5)

        # Label trapezoid
        if trap.label:
            center_x = (xl + xr) / 2
            center_y = (y_top_left + y_top_right + y_bot_left + y_bot_right) / 4
            ax.text(center_x, center_y, trap.label,
                   fontsize=10, ha='center', va='center',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Draw segments
    for seg in segments_set:
        ax.plot([seg.left.x, seg.right.x], [seg.left.y, seg.right.y],
               'b-', linewidth=2, marker='o', markersize=5)
        ax.text(seg.left.x, seg.left.y, f'  {seg.left.label}',
               fontsize=8, ha='left', va='bottom')
        ax.text(seg.right.x, seg.right.y, f'  {seg.right.label}',
               fontsize=8, ha='left', va='bottom')

    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.text(0.02, 0.98, f'Trapezoids: {len(trap_map.trapezoids)}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()
