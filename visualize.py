#
# file:   visualize.py
# desc:   Visualization utilities for trapezoidal maps.
# author: Benjamin Piro (brp8396@rit.edu)
# date:   7 November 2025
#

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon


def visualize_trapezoidal_map(trap_map, title: str = "Trapezoidal Map"):
    """
    Visualize a trapezoidal map using matplotlib.
    Shows all trapezoids with their boundaries and the line segments.

    @param trap_map: TrapezoidalMap instance to visualize
    @param title: Title for the plot
    """
    # Convert set to list for visualization
    trapezoids_list = list(trap_map.trapezoids)

    _, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')

    padding = 1.0
    ax.set_xlim(trap_map.bbox_x_min - padding, trap_map.bbox_x_max + padding)
    ax.set_ylim(trap_map.bbox_y_min - padding, trap_map.bbox_y_max + padding)

    bbox_rect = Rectangle(
        (trap_map.bbox_x_min, trap_map.bbox_y_min),
        trap_map.bbox_x_max - trap_map.bbox_x_min,
        trap_map.bbox_y_max - trap_map.bbox_y_min,
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
        y_top_left = trap.y_top(xl, trap_map.bbox_y_max)
        y_top_right = trap.y_top(xr, trap_map.bbox_y_max)
        y_bot_left = trap.y_bottom(xl, trap_map.bbox_y_min)
        y_bot_right = trap.y_bottom(xr, trap_map.bbox_y_min)

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

        # Add trapezoid label at the center
        if trap.label:
            center_x = (xl + xr) / 2
            center_y = (y_top_left + y_top_right + y_bot_left + y_bot_right) / 4
            ax.text(center_x, center_y, trap.label,
                   fontsize=10, ha='center', va='center',
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

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
