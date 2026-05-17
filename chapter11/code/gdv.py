import networkx as nx
from graphlet import Graphlet

# Create a sample graph
G = nx.karate_club_graph()

# Initialize the Graphlet class with the graph
graphlet_instance = Graphlet(G)

# Compute the GDVs for all nodes in the graph
gdvs = graphlet_instance.compute_graphlets()

# Print the GDV vectors
for node, gdv in gdvs.items():
    print(f"Node {node}: GDV {gdv}")
