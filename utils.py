import networkx as nx
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

#########################     GRAPH INITIALIZATION  ####################################
def create_and_display_graph(total_resources, num_nodes=10, p=0.01, ba_graph=True, m=3, display_graph = True, display_nodes = True):
    if ba_graph:
        G = nx.barabasi_albert_graph(num_nodes, m)
    else:
        G = nx.fast_gnp_random_graph(num_nodes, p)

    for node in G.nodes():
        subset_size = random.randint(1, 6)
        node_resources = set(random.sample(total_resources, subset_size))
        G.nodes[node]['resources'] = node_resources

    if display_nodes:
        for node in G.nodes():
            print(f"Node {node}: Resources = {G.nodes[node]['resources']}")

    if display_graph:
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue')

        # Title for the entire figure
        plt.suptitle('Initial Graph')

        plt.show()

    return G


####################  SCORING_FUNCTION  #############################
def scoring_function(node, graph, total_resources):
    node_resources = set(graph.nodes[node].get('resources', set()))
    neighbors = list(graph.neighbors(node))
    shared_with_neighbors = 0  # Initialize the count of neighbors shared with
    total_neighbor_resources = set()

    for neighbor in neighbors:
        # Consider only activated edges
        if graph[node][neighbor]['activated']:
          neighbor_resources = graph.nodes[neighbor].get('resources', set())
          total_neighbor_resources.update(neighbor_resources)

    own_shared_resources = len(node_resources.intersection(total_neighbor_resources))
    neighbor_shared_resources = len(total_neighbor_resources)

    total_shared_resources = own_shared_resources + neighbor_shared_resources
    fraction_total_resources = total_shared_resources / len(total_resources) if len(total_resources) > 0 else 0

    return fraction_total_resources


############  CALCULATES SCORE FOR EACH SOLUTION  #################
def fitness(solution, graph, total_resources):
    for i, edge in enumerate(graph.edges()):
        graph[edge[0]][edge[1]]['activated'] = bool(solution[i])

    scores = []
    for node in graph.nodes():
        score = scoring_function(node, graph, total_resources)
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    return avg_score


########## ACTIVATE EDGES AND PLOT THEM ####################
def activate_edges_and_plot(G, best_individual, display_graph=True):
    # Create a copy of the original graph
    G_copy = G.copy()

    # Activate edges based on the best individual
    for i, edge in enumerate(G_copy.edges()):
        G_copy[edge[0]][edge[1]]['activated'] = bool(best_individual[i])

    # Highlight activated edges in a different color (e.g., red)
    activated_edges = [(u, v) for u, v, data in G_copy.edges(data=True) if data.get('activated')]

    if display_graph:
        # Plot the graph with activated edges
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G_copy)  # Layout for the nodes
        nx.draw(G_copy, pos, with_labels=True, node_size=300, node_color='skyblue')

        nx.draw_networkx_edges(G_copy, pos, edgelist=activated_edges, edge_color='red', width=2)

        plt.title('Network with Activated Edges (Best Individual)')
        plt.show()

    return activated_edges


##########  COMPUTES AND DISPLAYS ACCESSIBLE RESOURCES FOR EACH NODE  ###########
def compute_accessible_resources(activated_edges, G):
    # Initialize a dictionary to store the number of accessible resources for each node
    accessible_resources = {node: set() for node in G.nodes()}

    # Check activated edges and update accessible resources for connected nodes
    for u, v in activated_edges:
        resources_u = G.nodes[u]['resources']
        resources_v = G.nodes[v]['resources']

        # Update accessible resources for both nodes based on the activated edge
        accessible_resources[u].update(resources_v)
        accessible_resources[v].update(resources_u)

    # Convert the accessible_resources dictionary to an array
    accessible_resources_array = [list(resources) for _, resources in accessible_resources.items()]

    # Display the number of accessible resources for each node
    for i, resources in enumerate(accessible_resources_array):
        print(f"Node {i}: Accessible Resources = {resources}")

    return accessible_resources_array

######  CALCULATES THE AVERAGE RESOURCES ACCESSIBLE BY EACH NODE  ##########
def calculate_average_resources(activated_edges, G):
    # Initialize a dictionary to store the aggregated accessible resources for each node
    accessible_resources = defaultdict(set)

    # Check activated edges and update accessible resources for connected nodes
    for u, v in activated_edges:
        resources_u = G.nodes[u]['resources']
        resources_v = G.nodes[v]['resources']

        combined_resources = set(resources_u) | set(resources_v)
        accessible_resources[u] |= combined_resources
        accessible_resources[v] |= combined_resources  # Assuming it's undirected

    # Calculate the total number of resources across all nodes
    total_resources = sum(len(resources) for resources in accessible_resources.values())

    # Calculate the average number of resources per node
    average_resources_per_node = total_resources / len(G.nodes)

    return average_resources_per_node


def save_simulation_results(G, best_sol_per_gen, best_fitness, average_resources_per_node, activated_edges, execution_time, filename):
    # Create a figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Subplot 1: Network with Activated Edges
    axs[0].set_title('Network with Activated Edges (Best Individual)')
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=axs[0], with_labels=True, node_size=300, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, edgelist=activated_edges, ax=axs[0], edge_color='red', width=2)

    # Subplot 2: Fitness Graph
    generations = range(1, len(best_sol_per_gen) + 1)
    axs[1].plot(generations, best_sol_per_gen)
    axs[1].set_xlabel('Generation')
    axs[1].set_ylabel('Fitness Score')
    axs[1].set_title('Fitness per Generation')

    # Subplot 3: Table with multiple rows and columns
    table_data = [
        ['Best Fitness', 'Average Resource'],
        [f'{best_fitness:.3f}', f'{average_resources_per_node:.3f}'],
        ['Num of activated edges', 'Total number of edges in graph'],
        [len(activated_edges), G.number_of_edges()],
        ['Execution Time', f'{execution_time:.4f} seconds']  # Include execution time in the table
    ]

    # Create a pandas DataFrame for formatting and displaying the table
    df = pd.DataFrame(table_data)
    table = axs[2].table(cellText=df.values, loc='center', cellLoc='center', colLabels=None)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)  # Adjust the scaling as needed

    # Limit the width of the table
    axs[2].set_xlim(0, 2)
    
    # Hide axes for the table subplot
    axs[2].axis('off')

    # Title for the entire figure
    fig.suptitle('GA Simulation Results')

    # Save the figure to the specified filename
    plt.tight_layout()
    plt.savefig("results/" + filename)
    plt.close()  # Close the figure to prevent display if not needed



# Example usage:
# display_simulation_results(G, best_sol_per_gen, best_fitness, average_resources_per_node, activated_edges)











# def fitness(solution, graph, total_resources):
#     # Activate edges based on the solution
#     for i, edge in enumerate(graph.edges()):
#         graph[edge[0]][edge[1]]['activated'] = bool(solution[i])

#     completeness_scores = []
#     for node in graph.nodes():
#         node_resources = set()
#         # Gather resources of the current node and its activated neighbors
#         neighbors = list(graph.neighbors(node))
#         node_resources.update(graph.nodes[node].get('resources', set()))

#         for neighbor in neighbors:
#             # Consider only activated edges
#             if graph[node][neighbor]['activated']:
#                 neighbor_resources = graph.nodes[neighbor].get('resources', set())
#                 node_resources.update(neighbor_resources)

#         completeness = len(node_resources.intersection(total_resources)) / len(total_resources)
#         completeness_scores.append(completeness)

#     avg_completeness = sum(completeness_scores) / len(completeness_scores)
#     # print(avg_completeness)
#     return avg_completeness