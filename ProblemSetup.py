import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

# Set of total resources
resource_set = {1, 2, 3, 4, 5, 6, 7}

# Number of nodes in the graph
num_nodes = 300

# Minimum and maximum number of resources per node
min_resources = 2
max_resources = 4

# Create a random graph
G = nx.erdos_renyi_graph(num_nodes, 0.004)

# Assign resources to each node within the specified range
for node in G.nodes():
    num_resources = random.randint(min_resources, max_resources)
    G.nodes[node]['resource'] = random.sample(resource_set, num_resources)
    G.nodes[node]['value'] = len(G.nodes[node]['resource'])
    G.nodes[node]['sharing'] = [node]

# Print the resources assigned to each node
for node, data in G.nodes(data=True):
    print(f"Node {node}: Resources {data['resource']}")

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

pick_count = {node: 0 for node in G.nodes()}

prev_sum_val = None

for i in range(500000):

    sum_val = 0

    for node in G.nodes():
        sum_val += G.nodes[node]['value']
    
    if prev_sum_val is None or sum_val != prev_sum_val:
        print("+++++++++++")
        print(sum_val)
        print("+++++++++++")

    prev_sum_val = sum_val

    node_probs = softmax([pick_count[node] for node in G.nodes()])
    node = np.random.choice(list(G.nodes()), p=node_probs)

    for neighbor in list(G.neighbors(node)):

        if neighbor not in G.nodes[node]['sharing']:

            node_loss = len(G.nodes[node]['resource'])/(len(G.nodes[node]['sharing'])*(len(G.nodes[node]['sharing']) + 1))
            node_gain = len(G.nodes[neighbor]['resource'])/(len(G.nodes[neighbor]['sharing']) + 1)
            node_multiplier = 0

            neighbor_loss = len(G.nodes[neighbor]['resource'])/(len(G.nodes[neighbor]['sharing'])*(len(G.nodes[neighbor]['sharing']) + 1))
            neighbor_gain = len(G.nodes[node]['resource'])/(len(G.nodes[node]['sharing']) + 1)
            neighbor_multiplier = 0

            neighbor_set = set(G.nodes[neighbor]['resource'])
            node_set = set(G.nodes[node]['resource'])

            for n in G.nodes[node]['sharing']:

                node_set = set.union(node_set, G.nodes[n]['resource'])
            
            for n in G.nodes[neighbor]['sharing']:

                neigbor_set = set.union(neighbor_set, G.nodes[n]['resource'])

            node_multiplier = 1 + (len(set(G.nodes[neighbor]['resource']) - node_set)/len(resource_set))
            neighbor_multiplier = 1 + (len(set(G.nodes[node]['resource']) - neighbor_set)/len(resource_set))

            if ((G.nodes[node]['value'] + node_gain - node_loss) * node_multiplier) > G.nodes[node]['value'] and ((G.nodes[neighbor]['value'] + neighbor_gain - neighbor_loss) * neighbor_multiplier) > G.nodes[neighbor]['value']:
                G.nodes[node]['value'] = G.nodes[node]['value'] + node_gain - node_loss * node_multiplier
                G.nodes[neighbor]['value'] = G.nodes[neighbor]['value'] + neighbor_gain - neighbor_loss * neighbor_multiplier
                G.nodes[node]['sharing'].append(neighbor)
                G.nodes[neighbor]['sharing'].append(node)
                print("added")
        
        elif neighbor in G.nodes[node]['sharing'] and neighbor != node:

            node_gain = len(G.nodes[node]['resource'])/(len(G.nodes[node]['sharing'])*(len(G.nodes[node]['sharing']) - 1))
            node_loss = len(G.nodes[neighbor]['resource'])/(len(G.nodes[neighbor]['sharing']))
            node_multiplier = 0
            node_set = set()

            for n in G.nodes[node]['sharing']:
                if n != neighbor:
                    node_set = set.union(node_set, G.nodes[n]['resource'])
            
            node_multiplier = 1 + (len(set(G.nodes[neighbor]['resource']) - node_set)/len(resource_set))

            if ((G.nodes[node]['value'] + node_gain - node_loss) / node_multiplier) > G.nodes[node]['value']:
                G.nodes[node]['value'] = (G.nodes[node]['value'] + node_gain - node_loss) / node_multiplier
                G.nodes[node]['sharing'].remove(neighbor)
                G.nodes[neighbor]['sharing'].remove(node)
                print("removed")
 
#sum_val = 0

#for node in G.nodes():
    #sum_val += G.nodes[node]['value']
    #print("----------")
    #print(list(G.neighbors(node)) + [node])
    #print(G.nodes[node]['sharing'])
    #print("----------")

print()
print(f"Fitness of the Brute Force Solution is : {sum_val}")
print()

'''
pos = nx.spring_layout(G)  

node_size = 750
nx.draw(G, pos, with_labels=False, node_size=node_size)

# Add resource information as node labels with braces, offset to the right
offset = 0.0
resource_labels = {node: f"{{{', '.join(map(str, G.nodes[node]['resource']))}}}" for node in G.nodes}
resource_pos = {k: (v[0] + offset, v[1]) for k, v in pos.items()}  # Offset to the right
nx.draw_networkx_labels(G, resource_pos, labels=resource_labels, font_color='black', font_size=5)

plt.show()

'''
'''
#DO NOT USE GRAPHING FOR LARGE GRAPHS, THE GRAPH WILL BE TOO CLUTERED TO VIEW ANYTHING AND TAKE VERY LONG TO GENERATE
#------------------------------------------------------------------------------------------------------------------

edges_to_draw = [(node, neighbor) for neighbor in list(G.neighbors(node)) if neighbor in G.nodes[node]['sharing']]
H = G.edge_subgraph(edges_to_draw)

pos = nx.spring_layout(G)  

node_size = 750
nx.draw(G, pos, with_labels=False, node_size=node_size)

# Draw edges only between nodes in G.nodes[node]['sharing']
nx.draw_networkx_edges(H, pos, edgelist=edges_to_draw)

# Add resource information as node labels with braces, offset to the right
offset = 0.0
resource_labels = {node: f"{{{', '.join(map(str, G.nodes[node]['resource']))}}}" for node in list(G.nodes)}
resource_pos = {k: (v[0] + offset, v[1]) for k, v in pos.items()}  # Offset to the right
nx.draw_networkx_labels(G, resource_pos, labels=resource_labels, font_color='black', font_size=5)

plt.show()



edges_to_draw = []

for node in G.nodes():
    for next in G.nodes[node]['sharing']:
        if node != next:
            edges_to_draw.append((node,next))

edges_list = list(G.edges())

def remove_duplicate_edges(edges_list):
    unique_edges = set()

    for edge in edges_list:
        
        sorted_edge = tuple(sorted(edge))
        unique_edges.add(sorted_edge)

    return list(unique_edges)


print(edges_list)
print(remove_duplicate_edges(edges_to_draw))

H = G.edge_subgraph(remove_duplicate_edges(edges_to_draw))
pos = nx.spring_layout(G)  

node_size = 750
nx.draw_networkx(G, pos, with_labels=False, node_size=node_size, edgelist=edges_to_draw)

offset = 0.0
resource_labels = {node: f"{{{', '.join(map(str, G.nodes[node]['resource']))}}}" for node in G.nodes}
resource_pos = {k: (v[0] + offset, v[1]) for k, v in pos.items()}  # Offset to the right
nx.draw_networkx_labels(G, resource_pos, labels=resource_labels, font_color='black', font_size=5)

plt.axis('off')

# Adjust the figure size (optional)
plt.gcf().set_size_inches(7, 5)

plt.show()

'''
