# [CSCI-716] Assignment 3 - Trapezoidal Maps and Planar Point Location

Implemented by **Benjamin Piro (brp8396@rit.edu)** and **Matt Pasetto (mp2163@rit.edu)**.


## Setup Instructions

1. Install the required dependencies by executing the following:
```bash
pip install -r requirements.txt
```

## Included Files

### main.py
Main implementation file containing the randomized incremental trapezoidal map construction algorithm. Handles segment insertion, trapezoid creation and neighbor connections, DAG construction for point location queries, and adjacency matrix generation. The implementation supports shared endpoints between segments and properly handles triangle degeneration cases.

### utils.py
Contains the fundamental data structures and geometric primitives: Point, Segment, Trapezoid, and DAG node types (XNode, YNode, Leaf). Provides the building blocks for representing the trapezoidal map and its query structure.

### visualize.py
Visualization utility for rendering the trapezoidal map structure. Generates graphical representations to help verify correctness and debug the construction process.

## Usage

Run the program by providing an input file containing line segments:

```bash
python3 main.py <input_file> [-o <output_file>]
```

### Arguments

- `<input_file>`: Path to a text file containing segments (required)
- `-o, --output <output_file>`: Path for output file (optional, defaults to `out/out.txt`)

### Input File Format

The input file should contain segments in the following format:

```
<number_of_segments>
<bbox_xmin> <bbox_ymin> <bbox_xmax> <bbox_ymax>
<x1> <y1> <x2> <y2>
<x1> <y1> <x2> <y2>
...
```

Example (`input/shared_endpoints.txt`):

```
3
0 0 10 10
1 2 5 4
5 4 9 2
5 4 9 6
```

Note: The implementation supports shared endpoints between segments (e.g., segments 2, 3, and 4 all share the point (5, 4)).

### Example Usage

```bash
# Run with default output location
python3 main.py input/test.txt

# Run with custom output location
python3 main.py input/shared_endpoints.txt -o out/shared_endpoints.txt

# Run with visualization
python3 main.py input/benjamin.txt -o out/benjamin.txt
python3 visualize.py  # Visualize the constructed trapezoidal map
```

### Output

The program generates:

1. Console output showing construction progress and trapezoid information
2. An adjacency matrix file showing the DAG structure (at the specified output path)
   - Rows represent source nodes, columns represent target nodes
   - Matrix is transposed (rows show incoming edges)
   - Duplicate labels are merged using OR logic
   - Row and column sums are provided

The adjacency matrix uses the following node naming convention:
- `P1, P2, ...`: Point nodes (XNode) representing vertical line splits
- `Q1, Q2, ...`: Additional point nodes
- `S1, S2, ...`: Segment nodes (YNode) representing above/below splits
- `T1, T2, ...`: Trapezoid leaf nodes

## Algorithm Details

This implementation constructs a trapezoidal map using a randomized incremental algorithm:

1. Segments are added one at a time in random order
2. For each segment, the algorithm:
   - Finds all trapezoids intersected by the segment
   - Splits these trapezoids into new regions (above, below, and optional left/right remainders)
   - Updates neighbor relationships, handling shared endpoints and triangle degeneration
   - Updates the DAG for efficient point location queries

### General Position Assumptions

The algorithm assumes the input segments satisfy the following general position constraints:

- **No vertical segments**: All segments must have distinct x-coordinates for their endpoints
- **No coincident segments**: No two segments may overlap or be identical
- **No intersecting segments**: Segments may only meet at their endpoints, not in their interiors
- **No duplicate x-coordinates**: No two endpoints may have the same x-coordinate unless they share the same endpoint (i.e., coincide completely)
- **Limited endpoint sharing**: At most two segments may share the same endpoint

This implementation relaxes the strict general position assumption by allowing segments to share endpoints (with at most 2 segments sharing any single endpoint), properly handling the resulting triangle degeneration cases.