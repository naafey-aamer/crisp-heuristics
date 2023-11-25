import random
import math
import time

class SimulatedAnnealing:
    def __init__(self, graph, total_resources, initial_temperature, cooling_rate, iteration_limit):
        self.graph = graph
        self.total_resources = total_resources
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.iteration_limit = iteration_limit
        self.best_solution = None

    def mutate(self, solution, mutation_rate=0.1):
        mutated_solution = solution[:]  # Create a copy to avoid changing the original solution
        for i in range(len(mutated_solution)):
            if random.random() < mutation_rate:
                mutated_solution[i] = 1 - mutated_solution[i]  # Flip the bit
        return mutated_solution

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

    def simulated_annealing(self):
        all_sols = []
    
        start_time = time.time() 

        # Initialize current solution
        current_solution = [random.choice([0, 1]) for _ in range(len(self.graph.edges()))]

        current_fitness = self.fitness(current_solution)

        for iteration in range(self.iteration_limit):
            # Generate a new solution
            new_solution = self.mutate(current_solution.copy())
            new_fitness = self.fitness(new_solution)

            # Calculate difference in fitness
            diff = new_fitness - current_fitness

            # Calculate temperature for current iteration
            t = self.initial_temperature * (1 - self.cooling_rate) ** iteration

            # If the new solution is better or the difference is accepted probabilistically, update current solution
            if diff > 0 or random.random() < math.exp(diff / t):
                current_solution, current_fitness = new_solution, new_fitness
            
            all_sols.append(current_fitness)

        end_time = time.time()  # Record the end time
        self.best_solution = current_solution

        execution_time = end_time - start_time  # Calculate the execution time
        
        print("Simulated Annealing Completed!")

        return all_sols, current_solution, execution_time
