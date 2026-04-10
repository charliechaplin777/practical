##1Practical
import networkx as nx
import matplotlib.pyplot as plt

# Kruskal's Algorithm
class UnionFind:
    # Manual Implementation
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]
    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            self.parent[root_v] = root_u
            return True
        return False

# Create Graph
G = nx.Graph()
edges = [
    (0, 1, 4), (0, 7, 8), (1, 7, 11), (1, 2, 8),
    (2, 8, 2), (2, 5, 4), (2, 3, 7), (3, 4, 9),
    (3, 5, 14), (4, 5, 10), (5, 6, 2), (6, 8, 6), (7, 8, 7)
]
G.add_weighted_edges_from(edges)

# STEP 1: Sort edges by weight
sorted_edges = sorted(edges, key=lambda x: x[2])

print("Sorted edges by weight:")
for e in sorted_edges:
    print(e)

# STEP 2: Initialize Union-Find
uf = UnionFind(G.nodes())
MST = []
total_weight = 0
print("\nStep-by-step Kruskal process:\n")

# STEP 3: Pick edges in sorted order
for u, v, w in sorted_edges:
    if uf.union(u, v):
        MST.append((u, v, w))
        total_weight += w
        print(f"Adding edge ({u}, {v}) with weight {w}")
    else:
        print(f"Skipping edge ({u}, {v}) with weight {w} forms cycle")

print("\nFinal MST edges:")
for edge in MST:
    print(edge)
print(f"\nTotal MST Weight: {total_weight}")

# Visualization
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos, with_labels=True, node_color="lightblue")
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))

# Highlight MST
mst_graph = nx.Graph()
mst_graph.add_weighted_edges_from(MST)
nx.draw_networkx_edges(mst_graph, pos, edge_color="green", width=3)
plt.title("Kruskal's Minimum Spanning Tree (Green Edges)")
plt.axis("off")
plt.show()

# Prim's Minimum Spanning Tree Visualization
import heapq

def prims_mst(graph, start_node):
    """Manually implements Prim's algorithm."""
    min_heap = []
    visited = set()
    mst_edges = []
    total_weight = 0
    # Start with the initial node
    visited.add(start_node)
    for neighbor, attrs in graph[start_node].items():
        heapq.heappush(min_heap, (attrs['weight'], start_node, neighbor))
    while min_heap:
        weight, u, v = heapq.heappop(min_heap)
        if v not in visited:
            visited.add(v)
            mst_edges.append((u, v, weight))
            total_weight += weight
            # Add neighbors of the newly visited node to the heap
            for neighbor, attrs in graph[v].items():
                if neighbor not in visited:
                    heapq.heappush(min_heap, (attrs['weight'], v, neighbor))
    return mst_edges, total_weight

# 1. Create a weighted graph
G_prim = nx.Graph()
edges_prim = [
    (0, 1, {"weight": 4}), (0, 7, {"weight": 8}), (1, 7, {"weight": 11}),
    (1, 2, {"weight": 8}), (2, 8, {"weight": 2}), (2, 5, {"weight": 4}),
    (2, 3, {"weight": 7}), (3, 4, {"weight": 9}), (3, 5, {"weight": 14}),
    (4, 5, {"weight": 10}), (5, 6, {"weight": 2}), (6, 8, {"weight": 6}),
    (7, 8, {"weight": 7})
]
G_prim.add_edges_from(edges_prim)

# 2. Run Prim's algorithm
start_node = 0
mst_edges_prim, total_weight_prim = prims_mst(G_prim, start_node)

print("\n\nPrim's Algorithm Results:")
print("Final MST edges:")
for edge in mst_edges_prim:
    print(edge)
print(f"\nTotal MST Weight: {total_weight_prim}")

# 3. Visualization for Prim's
pos_prim = nx.spring_layout(G_prim)
plt.figure()
nx.draw_networkx(G_prim, pos_prim, with_labels=True, node_color="lightblue")
nx.draw_networkx_edge_labels(G_prim, pos_prim, edge_labels=nx.get_edge_attributes(G_prim, 'weight'))

# Highlight MST edges
mst_graph_prim = nx.Graph()
mst_graph_prim.add_weighted_edges_from(mst_edges_prim)
nx.draw_networkx_edges(mst_graph_prim, pos_prim, edge_color="red", width=3)
plt.title("Prim's Minimum Spanning Tree (Red Edges)")
plt.axis("off")
plt.show()

# Dijkstra's Algorithm
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    path = {node: None for node in graph}
    pq = [(0, start)]
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        # Optimization: If the current distance is greater than the recorded distance, skip
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            new_distance = current_distance + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                path[neighbor] = current_node
                heapq.heappush(pq, (new_distance, neighbor))
    return distances, path

def get_shortest_path(path, start, end):
    """Reconstructs the shortest path from the predecessors dictionary."""
    route = []
    current = end
    while current is not None:
        route.append(current)
        current = path[current]
    route.reverse()
    # Check if a path was found
    if route[0] == start:
        return route
    else:
        return [] # Return empty list if no path exists

def draw_graph_with_path(graph, shortest_path):
    """Draws the graph using NetworkX and Matplotlib, highlighting the shortest path."""
    G = nx.Graph()
    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            G.add_edge(node, neighbor, weight=weight)
    pos = nx.spring_layout(G) # Position nodes
    plt.figure(figsize=(10, 7))
    # Draw all edges and nodes
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray', width=1.0)
    # Highlight the shortest path edges
    sp_edges = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]
    nx.draw_networkx_edges(G, pos, edgelist=sp_edges, edge_color='red', width=3.0)
    # Add edge labels (weights)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='blue')
    plt.title("Weighted Graph with Highlighted Shortest Path")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    graph = {
        'A': {'B': 4, 'C': 1},
        'B': {'A': 4, 'D': 7, 'E': 3},
        'C': {'A': 1, 'D': 2, 'F': 8},
        'D': {'B': 7, 'C': 2, 'E': 2, 'G': 6},
        'E': {'B': 3, 'D': 2, 'H': 5},
        'F': {'C': 8, 'G': 3},
        'G': {'D': 6, 'F': 3, 'H': 4},
        'H': {'E': 5, 'G': 4}
    }
    start = 'A'
    end = 'H'
    distances, path = dijkstra(graph, start)
    shortest_path = get_shortest_path(path, start, end)
    print(f"Shortest distances from {start}: {distances}")
    print(f"Shortest path {start} → {end}: {shortest_path}")
    try:
        draw_graph_with_path(graph, shortest_path)
    except ImportError:
        print("\nInstall networkx and matplotlib to visualize the graph:")
        print("pip install networkx matplotlib")

#2Practical - Implement Random Graph Showing Large Graph And Cycles
import networkx as nx
import random
import matplotlib.pyplot as plt

# 1. Generate a large random graph
n_nodes = 1000
n_edges = 2000
G = nx.gnm_random_graph(n_nodes, n_edges)

# 2. Add guaranteed cycles
num_cycles = 10
for _ in range(num_cycles):
    cycle_len = random.randint(3, 10)
    cycle_nodes = random.sample(range(n_nodes), cycle_len)
    for i in range(cycle_len):
        G.add_edge(cycle_nodes[i], cycle_nodes[(i + 1) % cycle_len])

# 3. Detect cycles
cycles = nx.cycle_basis(G)

print("Graph Statistics")
print("----------------")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Number of cycles detected:", len(cycles))

# 4. Identify cycle edges
cycle_edges = set()
for cycle in cycles:
    for i in range(len(cycle)):
        edge = tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)])))
        cycle_edges.add(edge)

# 5. Assign edge colors
edge_colors = []
for edge in G.edges():
    if tuple(sorted(edge)) in cycle_edges:
        edge_colors.append("red") # cycle edges
    else:
        edge_colors.append("gray")

# 6. Compute layout (optimized)
pos = nx.spring_layout(G, seed=42, iterations=50)

# 7. Draw graph
plt.figure(figsize=(12, 10))
nx.draw(
    G, pos,
    node_size=8,
    node_color="blue",
    edge_color=edge_colors,
    alpha=0.7,
    width=0.6
)
plt.title("Large Random Graph (1000 Nodes) with Cycles Highlighted")
plt.show()

#3Practical 3: Implement Non-Uniform Model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load iris dataset
iris = load_iris()

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Logistic Regression with L2 regularization (Non-Uniform Model)
model = LogisticRegression(solver='lbfgs', max_iter=42)

# Train the model
model.fit(x_train, y_train)

# Evaluate model
test_accuracy = model.score(x_test, y_test)
print("Test Accuracy:", test_accuracy)
print("\nClassification Report")
print(classification_report(y_test, model.predict(x_test)))

#
Here is the complete code for all 8 practicals, exactly as provided in the document.

Practical 1: Implement minimum spanning tree
Python
import networkx as nx
import matplotlib.pyplot as plt

# Kruskal's Algorithm
class UnionFind:
    # Manual Implementation
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]
    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            self.parent[root_v] = root_u
            return True
        return False

# Create Graph
G = nx.Graph()
edges = [
    (0, 1, 4), (0, 7, 8), (1, 7, 11), (1, 2, 8),
    (2, 8, 2), (2, 5, 4), (2, 3, 7), (3, 4, 9),
    (3, 5, 14), (4, 5, 10), (5, 6, 2), (6, 8, 6), (7, 8, 7)
]
G.add_weighted_edges_from(edges)

# STEP 1: Sort edges by weight
sorted_edges = sorted(edges, key=lambda x: x[2])

print("Sorted edges by weight:")
for e in sorted_edges:
    print(e)

# STEP 2: Initialize Union-Find
uf = UnionFind(G.nodes())
MST = []
total_weight = 0
print("\nStep-by-step Kruskal process:\n")

# STEP 3: Pick edges in sorted order
for u, v, w in sorted_edges:
    if uf.union(u, v):
        MST.append((u, v, w))
        total_weight += w
        print(f"Adding edge ({u}, {v}) with weight {w}")
    else:
        print(f"Skipping edge ({u}, {v}) with weight {w} forms cycle")

print("\nFinal MST edges:")
for edge in MST:
    print(edge)
print(f"\nTotal MST Weight: {total_weight}")

# Visualization
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos, with_labels=True, node_color="lightblue")
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))

# Highlight MST
mst_graph = nx.Graph()
mst_graph.add_weighted_edges_from(MST)
nx.draw_networkx_edges(mst_graph, pos, edge_color="green", width=3)
plt.title("Kruskal's Minimum Spanning Tree (Green Edges)")
plt.axis("off")
plt.show()

# Prim's Minimum Spanning Tree Visualization
import heapq

def prims_mst(graph, start_node):
    """Manually implements Prim's algorithm."""
    min_heap = []
    visited = set()
    mst_edges = []
    total_weight = 0
    # Start with the initial node
    visited.add(start_node)
    for neighbor, attrs in graph[start_node].items():
        heapq.heappush(min_heap, (attrs['weight'], start_node, neighbor))
    while min_heap:
        weight, u, v = heapq.heappop(min_heap)
        if v not in visited:
            visited.add(v)
            mst_edges.append((u, v, weight))
            total_weight += weight
            # Add neighbors of the newly visited node to the heap
            for neighbor, attrs in graph[v].items():
                if neighbor not in visited:
                    heapq.heappush(min_heap, (attrs['weight'], v, neighbor))
    return mst_edges, total_weight

# 1. Create a weighted graph
G_prim = nx.Graph()
edges_prim = [
    (0, 1, {"weight": 4}), (0, 7, {"weight": 8}), (1, 7, {"weight": 11}),
    (1, 2, {"weight": 8}), (2, 8, {"weight": 2}), (2, 5, {"weight": 4}),
    (2, 3, {"weight": 7}), (3, 4, {"weight": 9}), (3, 5, {"weight": 14}),
    (4, 5, {"weight": 10}), (5, 6, {"weight": 2}), (6, 8, {"weight": 6}),
    (7, 8, {"weight": 7})
]
G_prim.add_edges_from(edges_prim)

# 2. Run Prim's algorithm
start_node = 0
mst_edges_prim, total_weight_prim = prims_mst(G_prim, start_node)

print("\n\nPrim's Algorithm Results:")
print("Final MST edges:")
for edge in mst_edges_prim:
    print(edge)
print(f"\nTotal MST Weight: {total_weight_prim}")

# 3. Visualization for Prim's
pos_prim = nx.spring_layout(G_prim)
plt.figure()
nx.draw_networkx(G_prim, pos_prim, with_labels=True, node_color="lightblue")
nx.draw_networkx_edge_labels(G_prim, pos_prim, edge_labels=nx.get_edge_attributes(G_prim, 'weight'))

# Highlight MST edges
mst_graph_prim = nx.Graph()
mst_graph_prim.add_weighted_edges_from(mst_edges_prim)
nx.draw_networkx_edges(mst_graph_prim, pos_prim, edge_color="red", width=3)
plt.title("Prim's Minimum Spanning Tree (Red Edges)")
plt.axis("off")
plt.show()

# Dijkstra's Algorithm
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    path = {node: None for node in graph}
    pq = [(0, start)]
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        # Optimization: If the current distance is greater than the recorded distance, skip
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            new_distance = current_distance + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                path[neighbor] = current_node
                heapq.heappush(pq, (new_distance, neighbor))
    return distances, path

def get_shortest_path(path, start, end):
    """Reconstructs the shortest path from the predecessors dictionary."""
    route = []
    current = end
    while current is not None:
        route.append(current)
        current = path[current]
    route.reverse()
    # Check if a path was found
    if route[0] == start:
        return route
    else:
        return [] # Return empty list if no path exists

def draw_graph_with_path(graph, shortest_path):
    """Draws the graph using NetworkX and Matplotlib, highlighting the shortest path."""
    G = nx.Graph()
    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            G.add_edge(node, neighbor, weight=weight)
    pos = nx.spring_layout(G) # Position nodes
    plt.figure(figsize=(10, 7))
    # Draw all edges and nodes
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray', width=1.0)
    # Highlight the shortest path edges
    sp_edges = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]
    nx.draw_networkx_edges(G, pos, edgelist=sp_edges, edge_color='red', width=3.0)
    # Add edge labels (weights)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='blue')
    plt.title("Weighted Graph with Highlighted Shortest Path")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    graph = {
        'A': {'B': 4, 'C': 1},
        'B': {'A': 4, 'D': 7, 'E': 3},
        'C': {'A': 1, 'D': 2, 'F': 8},
        'D': {'B': 7, 'C': 2, 'E': 2, 'G': 6},
        'E': {'B': 3, 'D': 2, 'H': 5},
        'F': {'C': 8, 'G': 3},
        'G': {'D': 6, 'F': 3, 'H': 4},
        'H': {'E': 5, 'G': 4}
    }
    start = 'A'
    end = 'H'
    distances, path = dijkstra(graph, start)
    shortest_path = get_shortest_path(path, start, end)
    print(f"Shortest distances from {start}: {distances}")
    print(f"Shortest path {start} → {end}: {shortest_path}")
    try:
        draw_graph_with_path(graph, shortest_path)
    except ImportError:
        print("\nInstall networkx and matplotlib to visualize the graph:")
        print("pip install networkx matplotlib")
//
//
//
//

Practical 2: Implement Random Graph Showing Large Graph And Cycles
Python
import networkx as nx
import random
import matplotlib.pyplot as plt

# 1. Generate a large random graph
n_nodes = 1000
n_edges = 2000
G = nx.gnm_random_graph(n_nodes, n_edges)

# 2. Add guaranteed cycles
num_cycles = 10
for _ in range(num_cycles):
    cycle_len = random.randint(3, 10)
    cycle_nodes = random.sample(range(n_nodes), cycle_len)
    for i in range(cycle_len):
        G.add_edge(cycle_nodes[i], cycle_nodes[(i + 1) % cycle_len])

# 3. Detect cycles
cycles = nx.cycle_basis(G)

print("Graph Statistics")
print("----------------")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Number of cycles detected:", len(cycles))

# 4. Identify cycle edges
cycle_edges = set()
for cycle in cycles:
    for i in range(len(cycle)):
        edge = tuple(sorted((cycle[i], cycle[(i + 1) % len(cycle)])))
        cycle_edges.add(edge)

# 5. Assign edge colors
edge_colors = []
for edge in G.edges():
    if tuple(sorted(edge)) in cycle_edges:
        edge_colors.append("red") # cycle edges
    else:
        edge_colors.append("gray")

# 6. Compute layout (optimized)
pos = nx.spring_layout(G, seed=42, iterations=50)

# 7. Draw graph
plt.figure(figsize=(12, 10))
nx.draw(
    G, pos,
    node_size=8,
    node_color="blue",
    edge_color=edge_colors,
    alpha=0.7,
    width=0.6
)
plt.title("Large Random Graph (1000 Nodes) with Cycles Highlighted")
plt.show()
//
//
//
//
Practical 3: Implement Non-Uniform Model
Python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load iris dataset
iris = load_iris()

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Logistic Regression with L2 regularization (Non-Uniform Model)
model = LogisticRegression(solver='lbfgs', max_iter=42)

# Train the model
model.fit(x_train, y_train)

# Evaluate model
test_accuracy = model.score(x_test, y_test)
print("Test Accuracy:", test_accuracy)
print("\nClassification Report")
print(classification_report(y_test, model.predict(x_test)))
.//
../
..//
../

#Practical 4: Implement and discuss working of SVD
import numpy as np
# Generate a random matrix
np.random.seed(42) # for reproducibility
A = np.random.randint(0, 10, size=(3, 2))

# Apply SVD
U, S, VT = np.linalg.svd(A)

# Convert singular values to diagonal matrix
Sigma = np.diag(S)

# Reconstruct the matrix
A_reconstructed = U[:, :2] @ Sigma @ VT

# Display results
print("Random Matrix A:\n", A)
print("\nU Matrix:\n", U)
print("\nSigma Matrix:\n", Sigma)
print("\nV Transpose Matrix:\n", VT)
print("\nReconstructed Matrix:\n", A_reconstructed)
./.
./.
./.
../

#Practical 5: Implement best rank k approximation
import numpy as np

def best_rank_k_approx(A, k):
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    S_k = np.zeros_like(S)
    S_k[:k] = S[:k]
    return U @ np.diag(S_k) @ VT

# Input matrix size
rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))
print("Enter matrix row-wise:")
A = []
for i in range(rows):
    row = list(map(float, input().split()))
    A.append(row)
A = np.array(A)
k = int(input("Enter value of k: "))

A_k = best_rank_k_approx(A, k)
print("\nOriginal Matrix:\n", A)
print("\nBest Rank-k Approximation:\n", A_k)
./
./
./
./

#Practical 6: SHOW IMPLEMENTATION OF SPHERE AND CUBE IN HIGH DIMENSIONAL SPACE

import numpy as np

def generate_sphere_points(n_points, dimensions, radius):
    points = np.random.rand(n_points, dimensions)
    points = points / np.linalg.norm(points, axis=1)[:,np.newaxis]
    r = np.random.rand(n_points,1)**(1/dimensions)
    return radius * r * points

def generate_cube_points(n_points, dimensions, size):
    return np.random.rand(n_points, dimensions) * size

dimension = 10
n_points = 1000
radius = 5
size = 10

sphere_points = generate_sphere_points(n_points, dimension, radius)
cube_points = generate_cube_points(n_points, dimension, size)

print("Sphere points shape", sphere_points.shape)
print("Cube points shape", cube_points.shape)
print("Max Distance from origin (sphere):", np.max(np.linalg.norm(sphere_points, axis=1)))
print("Min & Max values (Cube):", np.min(cube_points), np.max(cube_points))
.
.
.
.
#Practical 7: IMPLEMENT RANDOM PROJECTION IN HIGH DIMENSIONAL SPACE
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
x = iris.data
y = iris.target
print("Original Dataset Shape", x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Training Shape", x_train.shape)
print("Testing Shape", x_test.shape)

random_proj = GaussianRandomProjection(n_components=3, random_state=42)
x_train_proj = random_proj.fit_transform(x_train)
x_test_proj = random_proj.transform(x_test)

print("\nProjected Training Shape", x_train_proj.shape)
print("Projected Testing Shape", x_test_proj.shape)
print("\nSample Projected Data (First 5 rows):\n")
print(np.round(x_train_proj[:5], 3))

original_dist = np.linalg.norm(x_train[0] - x_train[1])
projected_dist = np.linalg.norm(x_train_proj[0] - x_train_proj[1])
print("\nDistance Between First Two Points:")
print("Original Distance:", round(original_dist, 3))
print("Projected Distance:", round(projected_dist, 3))
print("\nProjection Matrix Shape:", random_proj.components_.shape)

