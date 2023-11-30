import random
import time

class TabuSearch:
   def __init__(self, graph, initial_solution, tabu_list_size, iterations, total_res):
       self.initial_solution = initial_solution
       self.tabu_list_size = tabu_list_size
       self.iterations = iterations
       self.graph = graph
       self.total_resources = total_res

   def find_best_solution(self):
        best_solution = self.initial_solution
        best_fitness = self.fitness(best_solution)
        best_solutions_per_generation = []

        tabu_list = []
        start_time = time.time()

        for _ in range(self.iterations):
            current_solution = [random.choice([0, 1]) for _ in range(self.graph.number_of_edges())]
            current_fitness = self.fitness(current_solution)

            while current_fitness > best_fitness:
                neighbor = self.find_neighbor(current_solution)
                neighbor_fitness = self.fitness(neighbor)

                if neighbor_fitness > current_fitness and neighbor not in tabu_list:
                    current_solution = neighbor
                    current_fitness = neighbor_fitness

                    if neighbor_fitness > best_fitness:
                        best_solution = neighbor
                        best_fitness = neighbor_fitness

            tabu_list.append(current_solution)
            # Store the best solution of the current generation
            best_solutions_per_generation.append(self.fitness(current_solution))

            if len(tabu_list) > self.tabu_list_size:
                tabu_list.pop(0)

        end_time = time.time()
        exec_time = end_time - start_time

        print("Tabu Search Completed!")

        return best_solution, best_solutions_per_generation, exec_time


   def scoring_function(self, node):
        node_resources = set(self.graph.nodes[node].get('resources', set()))
        neighbors = list(self.graph.neighbors(node))
        shared_with_neighbors = 0  # Initialize the count of neighbors shared with
        total_neighbor_resources = set()

        for neighbor in neighbors:
            # Consider only activated edges
            if self.graph[node][neighbor]['activated']:
                neighbor_resources = self.graph.nodes[neighbor].get('resources', set())
                total_neighbor_resources.update(neighbor_resources)

        own_shared_resources = len(node_resources.intersection(total_neighbor_resources))
        neighbor_shared_resources = len(total_neighbor_resources)

        total_shared_resources = own_shared_resources + neighbor_shared_resources
        fraction_total_resources = total_shared_resources / len(self.total_resources) if len(
            self.total_resources) > 0 else 0

        return fraction_total_resources

   def fitness(self, solution):
        for i, edge in enumerate(self.graph.edges()):
            self.graph[edge[0]][edge[1]]['activated'] = bool(solution[i])

        scores = []
        for node in self.graph.nodes():
            score = self.scoring_function(node)
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        return avg_score
   
   def find_neighbor(self, solution):
        # Copy the current solution
        neighbor = solution.copy()

        # Randomly select one or two edges to flip
        num_edges_to_flip = random.randint(20, 50)
        for _ in range(num_edges_to_flip):
            # Randomly select an edge
            edge_index = random.randint(0, len(neighbor) - 1)
            # Flip the state of the edge
            neighbor[edge_index] = 1 - neighbor[edge_index]

        return neighbor
