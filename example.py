import networkx as nx
import my_networkx as my_nx
import matplotlib.pyplot as plt


G = nx.DiGraph()
edge_list = [(1,2,{'w':'A1'}),(2,1,{'w':'A2'}),(2,3,{'w':'B'}),(3,1,{'w':'C'}),
            (3,4,{'w':'D1'}),(4,3,{'w':'D2'}),(1,5,{'w':'E1'}),(5,1,{'w':'E2'}),
            (3,5,{'w':'F'}),(5,4,{'w':'G'})]
G.add_edges_from(edge_list)
print(G.edges())
pos=nx.spring_layout(G,seed=5)
fig, ax = plt.subplots()
nx.draw_networkx_nodes(G, pos, ax=ax)
nx.draw_networkx_labels(G, pos, ax=ax)
edge_weights = nx.get_edge_attributes(G,'w')
curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
straight_edges = list(set(G.edges()) - set(curved_edges))
arc_rad = 0.25
curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
print(curved_edge_labels)
straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges)
nx.draw_networkx_edges(G, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')
my_nx.my_draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=curved_edge_labels,rotate=False,rad = arc_rad)
nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=straight_edge_labels,rotate=False)
plt.show()