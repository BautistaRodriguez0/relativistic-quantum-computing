import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re

# ================== HEXAGON GRID ================== #

def hexagon_vertices(center, size=1.0):
    """
    Generate the coordinates of the vertices of a regular hexagon.

    Parameters:
        center : tuple or array-like
            The (x, y) coordinates of the hexagon center.
        size : float
            The distance from the center to a vertex.

    Returns:
        np.array of shape (6, 2): The vertices of the hexagon.
    """
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 vertices
    x = center[0] + size * np.cos(angles)
    y = center[1] + size * np.sin(angles)
    return np.column_stack([x, y])


def generate_hex_grid(radius, hex_size=1.0):
    """
    Generate a hexagonal grid of centers inside a circle of given radius.

    Parameters:
        radius : float
            The radius of the circular area to cover.
        hex_size : float
            The size of each hexagon.

    Returns:
        List of tuples: (x, y) coordinates of hexagon centers.
    """
    hex_height = hex_size * np.sqrt(3)/2  # vertical spacing between rows
    cols = int(np.ceil(radius / (hex_size * 1.5))) + 1
    rows = int(np.ceil(radius / hex_height)) + 1

    centers = []
    for i in range(-rows, rows + 1):
        for j in range(-cols, cols + 1):
            x = j * hex_size * 1.5
            y = i * hex_height * 2
            if j % 2 == 1:  # staggered row for hex grid
                y += hex_height
            if x**2 + y**2 <= radius**2:  # inside the circle
                centers.append((x, y))
    return centers


def find_nearest_vertex(point, centers, hex_size, tolerance=0.1):
    """
    Find the nearest hexagon vertex (or center) to a given point.

    Parameters:
        point : array-like
            The (x, y) coordinates of the target point.
        centers : list
            List of hexagon centers.
        hex_size : float
            Size of hexagons.
        tolerance : float
            Maximum distance to snap to a vertex.

    Returns:
        np.array: Coordinates of the nearest vertex.
    """
    all_vertices = []
    for center in centers:
        hex_verts = hexagon_vertices(center, hex_size)
        all_vertices.extend(hex_verts)
        all_vertices.append(center)  # include center itself
    all_vertices = np.array(all_vertices)
    distances = np.linalg.norm(all_vertices - point, axis=1)
    valid_indices = np.where(distances <= tolerance)[0]
    if len(valid_indices) > 0:
        return all_vertices[valid_indices[0]]
    else:
        return all_vertices[np.argmin(distances)]


# ================== ROTATIONS ================== #

def rotation_matrix(angle):
    """
    Generate a 2D rotation matrix for a given angle.

    Parameters:
        angle : float
            Rotation angle in radians.

    Returns:
        np.array : 2x2 rotation matrix.
    """
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


# ================== TRANSLATION GENERATION ================== #

def generate_n_level_translations(R, t1, t2, levels=1, current_level=1):
    """
    Recursively generate smaller translations using commutators.

    Parameters:
        R : np.array
            Rotation matrix used in the commutator.
        t1, t2 : np.array
            Base translations (vectors).
        levels : int
            Number of decomposition levels.
        current_level : int
            Current recursion depth.

    Returns:
        moves : list of np.array
            List of translation vectors.
        names : list of str
            Names of the corresponding moves.
    """
    if levels == 0:
        return [], []

    # Compute small translations from commutator formula
    small_t1 = R @ t1 - t1
    small_t2 = R @ t2 - t2

    name_t1 = f"T1({current_level})"
    name_t2 = f"T2({current_level})"

    print(f"Level {current_level}:")
    print(f"  {name_t1} = M·T1({current_level-1})·M⁻¹·T1⁻¹({current_level-1})")
    print(f"  {name_t2} = M·T2({current_level-1})·M⁻¹·T2⁻¹({current_level-1})")
    print(f"  small_t1: {small_t1} (norm: {np.linalg.norm(small_t1):.4f})")
    print(f"  small_t2: {small_t2} (norm: {np.linalg.norm(small_t2):.4f})")

    # Recursively compute smaller translations
    smaller_moves, smaller_names = generate_n_level_translations(R, small_t1, small_t2, levels-1, current_level+1)

    # Combine current level translations with recursive results
    moves = [small_t1, -small_t1, small_t2, -small_t2] + smaller_moves
    names = [name_t1, f"{name_t1}⁻¹", name_t2, f"{name_t2}⁻¹"] + smaller_names

    return moves, names


# ================== ELEMENTARY GATE EXPANSION ================== #

def expand_to_elementary_gates(operation):
    """
    Recursively expand a composite translation into elementary gates.

    Elementary gates: T1(0), T2(0), T1⁻¹(0), T2⁻¹(0), M, M⁻¹.

    Parameters:
        operation : str
            Name of translation or its inverse.

    Returns:
        list of str : Expanded sequence of elementary operations.
    """
    # Normalize base cases
    if operation == "T1(0)⁻¹": return ["T1⁻¹(0)"]
    if operation == "T2(0)⁻¹": return ["T2⁻¹(0)"]

    # Base elementary operations
    base = {"T1(0)", "T2(0)", "T1⁻¹(0)", "T2⁻¹(0)", "M", "M⁻¹"}
    if operation in base: return [operation]

    # Expand T(n)
    m = re.match(r'^(T[12])\((\d+)\)$', operation)
    if m:
        T, level = m.group(1), int(m.group(2))
        if level == 0: return [f"{T}(0)"]
        return (["M"]
                + expand_to_elementary_gates(f"{T}({level-1})")
                + ["M⁻¹"]
                + expand_to_elementary_gates(f"{T}⁻¹({level-1})"))

    # Expand T(n)⁻¹
    m = re.match(r'^(T[12])\((\d+)\)⁻¹$', operation)
    if m:
        T, level = m.group(1), int(m.group(2))
        if level == 0: return [f"{T}⁻¹(0)"]
        return (expand_to_elementary_gates(f"{T}({level-1})")
                + ["M"]
                + expand_to_elementary_gates(f"{T}⁻¹({level-1})")
                + ["M⁻¹"])

    # Expand T⁻¹(n)
    m = re.match(r'^(T[12])⁻¹\((\d+)\)$', operation)
    if m:
        T, level = m.group(1), int(m.group(2))
        if level == 0: return [f"{T}⁻¹(0)"]
        return (expand_to_elementary_gates(f"{T}({level-1})")
                + ["M"]
                + expand_to_elementary_gates(f"{T}⁻¹({level-1})")
                + ["M⁻¹"])

    return [operation]  # fallback


# ================== MAIN ================== #

# Parameters
radius = 3
hex_size = 1.0
tolerance = 1e-7
theta = np.sqrt(2)*np.pi/16  # irrational rotation angle
decomposition_levels = 30

# Generate hexagonal grid
centers = generate_hex_grid(radius, hex_size)

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 14))

# Plot hexagons and their triangles
for center in centers:
    hex_verts = hexagon_vertices(center, hex_size)
    ax.add_patch(plt.Polygon(hex_verts, edgecolor='gray', facecolor='none', alpha=0.3))
    for i in range(6):
        triangle = np.vstack([center, hex_verts[i], hex_verts[(i+1)%6]])
        ax.add_patch(plt.Polygon(triangle, edgecolor='blue', facecolor='none', alpha=0.2))

# Draw circular boundary
ax.add_patch(plt.Circle((0, 0), radius, edgecolor='red', facecolor='none', linewidth=2, linestyle='--'))

# Define target and find nearest vertex
target = np.array([1.9, 1.0])
nearest_vertex = find_nearest_vertex(target, centers, hex_size, tolerance)
ax.add_patch(plt.Circle(target, tolerance, edgecolor='purple', facecolor='none', linewidth=1.5, linestyle=':'))

# Plot target and nearest vertex
ax.plot(*target, 'ro', markersize=8, label='Target')
ax.plot(*nearest_vertex, 'go', markersize=8, label='Nearest vertex')

# Define base translations and rotation matrices
t1 = np.array([hex_size, 0])  # Right
t2 = np.array([-hex_size*0.5, hex_size*np.sqrt(3)/2])  # Up-right
R = rotation_matrix(theta)  # Irrational rotation
R_inv = rotation_matrix(-theta)  # Inverse rotation

# Generate small translations recursively
small_moves, small_move_names = generate_n_level_translations(R, t1, t2,
                                                              levels=decomposition_levels,
                                                              current_level=1)

# Combine all moves (base + small translations)
all_moves = [t1, -t1, t2, -t2] + small_moves
all_move_names = ["T1(0)", "T1⁻¹(0)", "T2(0)", "T2⁻¹(0)"] + small_move_names

# Map moves to names
move_to_name = {tuple(move): name for move, name in zip(all_moves, all_move_names)}

# Sort moves by size (descending)
move_name_pairs_sorted = sorted(zip(all_moves, all_move_names),
                                key=lambda x: -np.linalg.norm(x[0]))

# ================== PATHFINDING ================== #

current = np.array([0.0, 0.0])
path = [current.copy()]
step_sequence = []

while np.linalg.norm(current - target) > tolerance:
    # Try moves from largest to smallest
    for move, move_name in move_name_pairs_sorted:
        new_pos = current + move
        if np.linalg.norm(new_pos - target) < np.linalg.norm(current - target):
            current = new_pos
            path.append(current.copy())
            step_sequence.append(move_name)

            # Draw the move
            ax.quiver(path[-2][0], path[-2][1], move[0], move[1],
                      scale=1, scale_units='xy', angles='xy',
                      color='blue', width=0.005, headwidth=3, headlength=4)
            break

# Expand steps to elementary gates
elementary_sequence = []
for op in reversed(step_sequence):
    expanded = expand_to_elementary_gates(op)
    elementary_sequence.extend(expanded)

# Plot final path
path = np.array(path)
ax.plot(path[:,0], path[:,1], 'r-', linewidth=2, label='Path Taken')

# Create legend
legend_elements = [
    Line2D([0], [0], color='red', lw=2, label='Target'),
    Line2D([0], [0], color='green', lw=2, label='Nearest Vertex'),
    Line2D([0], [0], color='red', linestyle='--', lw=2, label='Grid Boundary'),
    Line2D([0], [0], color='purple', linestyle=':', lw=2, label='Tolerance'),
    Line2D([0], [0], color='blue', lw=2, label='Steps')
]
ax.legend(handles=legend_elements)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.show()

# ================== SUMMARY ================== #
print("\n=== Summary ===")
print(f"Target: {target}")
print(f"Nearest vertex: {nearest_vertex}")
print(f"Final position: {current}")
print(f"Distance to target: {np.linalg.norm(current - target):.4f}")
print(f"Total steps: {len(path)-1}")
print(f"Total gates: {len(elementary_sequence)}")
