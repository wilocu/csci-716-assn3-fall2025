#
# file:   utils.py
# desc:   Provides generic utilities for reading/writing files of points.
#         Note: this implementation is taken from Benjamin Piro's assignment 2 submission.
#         It has been adapted here for use in assignment 3.
# author: Benjamin Piro (brp8396@rit.edu)
# date:   8 October 2025
#

import os, sys
from dataclasses import dataclass
from typing import Optional, Tuple

#
# Read/write points from/to file utilities
#

def read_file(file_path: str) -> tuple[int, tuple, list[tuple]]:
    """
    @desc: Reads an input `.txt` file containing line segment endpoints.
        This function may behave unpredictably if the proper format is not used.
        Expected format:
            Line 1: Number of line segments (integer)
            Line 2: Bounding box as "x_min y_min x_max y_max" (lower-left and upper-right corners)
            Lines 3+: Line segments as "x1 y1 x2 y2"

    @type file_path: str
    @param file_path: The full path of the file being read.

    @rtype: tuple[int, tuple, list[tuple]]
    @returns: A tuple containing:
        - num_points: The number of line segments (integer)
        - bounding_box: (x_min, y_min, x_max, y_max) representing the bounding box (lower-left, upper-right)
        - points: A list of (x1, y1, x2, y2) point tuples, representing line segments
    """
    file_basename: str = os.path.basename(file_path)
    if len(file_basename) < 4 or file_path[-4:] != ".txt":
        raise ValueError("Invalid file type, expected `.txt`.")
    try:
        # read the file
        points: list[tuple[float]] = []
        with open(file_path, 'r') as f:
            # first line should be an integer
            num_points: int = int(f.readline().strip())

            # second line should be the bounding box: x_min y_min x_max y_max (lower-left, upper-right)
            bbox_vec: list[float] = [float(c) for c in f.readline().strip().split()]
            if len(bbox_vec) != 4:
                raise ValueError(
                    "Invalid bounding box format. Expected 4 values " \
                    f"(x_min y_min x_max y_max), but got {len(bbox_vec)} instead."
                )
            bounding_box: tuple[float, float, float, float] = tuple(bbox_vec)

            # read line segments
            for i in range(num_points):
                vec: list[float] = [float(c) for c in f.readline().strip().split()]
                val_check: int = len(vec)
                if val_check != 4:
                    raise ValueError(
                        "Invalid format. Expected 4 values, " \
                        f"but got {val_check} at {file_basename}:{i+3} instead."
                    )
                # order points by x coordinate
                x1, y1, x2, y2 = vec
                if x2 < x1 or (x1 == x2 and y2 < y1):
                    vec = [x2, y2, x1, y1]
                points.append(tuple(vec))
            np_check = len(points)
            if np_check != num_points:
                raise ValueError(
                    "Read an unexpected number of points. " \
                    f"Expected {num_points}, but got {np_check} instead."
                )
        return num_points, bounding_box, points
    except Exception as e:
        print(f"Failed to parse file at {file_path}", file=sys.stderr)
        raise e


def write_file(intersections, path: str) -> None:
    """
    Write intersections to a `.txt` file
    """
    raise NotImplementedError("write_file function is not yet implemented.")