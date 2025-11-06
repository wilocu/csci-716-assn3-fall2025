import argparse
import os, sys
import utils


class Point:
    def __init__(self, x, y, label=""):
        self.x = x
        self.y = y
        self.label = label
    
    def __repr__(self):
        return f"{self.label}({self.x}, {self.y})"
    
    def __eq__(self, other):
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9
    

class Segment:
    def __init__(self, p1, p2, label=""):
        # Ensure left point has smaller x (or smaller y if vertical)
        if p1.x < p2.x or (p1.x == p2.x and p1.y < p2.y):
            self.left = p1
            self.right = p2
        else:
            self.left = p2
            self.right = p1
        self.label = label
    
    def is_above(self, point):
        """Check if a point is above this segment using cross product"""
        # Vector from left to right
        dx = self.right.x - self.left.x
        dy = self.right.y - self.left.y
        
        # Vector from left to point
        dpx = point.x - self.left.x
        dpy = point.y - self.left.y
        
        # Cross product (positive = above, negative = below)
        cross = dx * dpy - dy * dpx
        return cross > 1e-9

    def __repr__(self):
        return f"{self.label}[{self.left} -> {self.right}]"


class Trapezoid:
    """Represents a trapezoid in the map"""
    def __init__(self, top, bottom, leftp, rightp, label=""):
        self.top = top          # Segment 
        self.bottom = bottom    # Segment 
        self.leftp = leftp      # Point
        self.rightp = rightp    # Point
        self.label = label
        
        # Neighbors for traversing trapezoids
        self.upper_left = None
        self.lower_left = None
        self.upper_right = None
        self.lower_right = None
    
    def contains_point(self, point):
        """Check if a point is inside this trapezoid"""
        # Check x bounds
        if point.x < self.leftp.x or point.x > self.rightp.x:
            return False
        
        # Check if below top segment (if it exists)
        if self.top and self.top.is_above(point):
            return False  # Point is above the top, so not in trapezoid
        
        # Check if above bottom segment (if it exists)
        if self.bottom and not self.bottom.is_above(point):
            return False  # Point is below the bottom, so not in trapezoid
        
        return True
    
    def __repr__(self):
        return f"{self.label}"
    

class Node:
    """Node in the DAG structure"""
    def __init__(self, node_type, label=""):
        self.type = node_type  # 'X' (point), 'Y' (segment), or 'T' (trapezoid/leaf)
        self.label = label
        
        # For X and Y nodes
        self.left = None       # Left child (or below for Y-nodes)
        self.right = None      # Right child (or above for Y-nodes)
        
        # For different node types
        self.point = None      # For X-nodes
        self.segment = None    # For Y-nodes
        self.trapezoid = None  # For T-nodes (leaves)
    
    def __repr__(self):
        return f"Node({self.type}, {self.label})"


class TrapezoidalMap:
    """Main structure managing the trapezoidal map and DAG"""
    def __init__(self, bbox):
        self.bbox = bbox
        self.root = None
        self.trapezoids = []
        self.all_nodes = []
        
        # Initialize with bounding box trapezoid
        self._initialize_bounding_box()
    
    def _initialize_bounding_box(self):
        """Create initial trapezoid covering the entire bounding box"""
        # Create corner points for bounding box
        left_point = Point(self.bbox['min_x'], 
                          (self.bbox['min_y'] + self.bbox['max_y']) / 2, 
                          "BBox_left")
        right_point = Point(self.bbox['max_x'], 
                           (self.bbox['min_y'] + self.bbox['max_y']) / 2, 
                           "BBox_right")
        
        # Create initial trapezoid (no top/bottom segments, just bounding box)
        t1 = Trapezoid(None, None, left_point, right_point, "T1")
        self.trapezoids.append(t1)
        
        # Create root node (leaf node pointing to T1)
        root = Node('T', 'T1')
        root.trapezoid = t1
        self.root = root
        self.all_nodes.append(root)
        
        print(f"Initialized bounding box trapezoid: {t1}")
        

def parse_input_file():
    parser = argparse.ArgumentParser(description="CSCI-716 Assn3 Trapezoidal Maps & Planar Point Location")
    parser.add_argument("file", type=str, help="Input file containing the planar subdivision data")
    parser.add_argument("-o", "--output", type=str, default="./out/out.txt", help="Output file to save planar subdivision matrix")

    args = parser.parse_args()

    return args

def main():
    args = parse_input_file()
    input_path = args.file
    output_path = args.output

    print("="*60)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print("="*60)

    # Read input segments from file:
    try:
        num_segments, bbox, segments_data = utils.read_file(input_path)
        print(f"\nRead {num_segments} segments from {input_path}")
        min_x, min_y, max_x, max_y = bbox
        print(f"Bounding box: ({min_x}, {min_y}) to ({max_x}, {max_y})")
        print()
    except Exception as e:
        print(f"Error reading input file:", e, file=sys.stderr)
        sys.exit(1)
    
    # # Create Segment objects with proper labels
    # segments = []
    # for i, (x1, y1, x2, y2) in enumerate(segments_data):
    #     p1 = Point(x1, y1)
    #     p2 = Point(x2, y2)
    #     seg = Segment(p1, p2, f"S{i+1}")
        
    #     # Label endpoints
    #     seg.left.label = f"P{i+1}"
    #     seg.right.label = f"Q{i+1}"
        
    #     segments.append(seg)
    #     print(f"  {seg}")
    
    # print("\n" + "="*60)
    # print("BUILDING TRAPEZOIDAL MAP")
    # print("="*60)
    
    # # Initialize trapezoidal map with bounding box
    # trap_map = TrapezoidalMap(bbox)
    
    # # Add segments incrementally
    # for seg in segments:
    #     trap_map.add_segment(seg)
    
    # print("\n" + "="*60)
    # print("TRAPEZOIDAL MAP COMPLETE")
    # print("="*60)
    # print(f"Total trapezoids: {len(trap_map.trapezoids)}")
    # print(f"Total DAG nodes: {len(trap_map.all_nodes)}")
    
    # # Generate and save adjacency matrix
    # print(f"\nGenerating adjacency matrix...")
    # trap_map.generate_adjacency_matrix(output_path)
    # print(f"✓ Adjacency matrix saved to {output_path}")
    
    # # Interactive point query mode
    # print("\n" + "="*60)
    # print("POINT QUERY MODE")
    # print("="*60)
    # print("Enter point coordinates as 'x y' (or 'q' to quit):")
    
    # while True:
    #     try:
    #         user_input = input("> ").strip()
    #         if user_input.lower() == 'q':
    #             break
            
    #         x, y = map(float, user_input.split())
    #         path, trapezoid = trap_map.query_point(x, y)
    #         print(f"Path: {' '.join(path)}")
    #         print(f"Trapezoid: {trapezoid}")
            
    #     except (ValueError, EOFError):
    #         print("Invalid input. Enter 'x y' or 'q' to quit.")
    #     except KeyboardInterrupt:
    #         break
    
    # print("Done!")



if __name__ == "__main__":
    main()
