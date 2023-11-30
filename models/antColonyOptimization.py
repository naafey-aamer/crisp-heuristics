import numpy as np
import random
import time

class AntColonyOptimization:
    def __init__(self, graph, total_resources, num_ants, num_iterations, decay_rate, alpha, beta, Q):
        self.graph = graph
        self.total_resources = total_resources
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        self.pheromones = np.ones((len(self.graph.edges()), len(self.graph.edges())))


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
        fraction_total_resources = total_shared_resources / len(self.total_resources) if len(self.total_resources) > 0 else 0

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


    def ant_colony_optimization(self):
        best_solution = None
        best_fitness = float('-inf')
        best_fitnesses = []

        start_time = time.time()

        for iteration in range(self.num_iterations):
            solutions = []
            fitnesses = []

            for ant in range(self.num_ants):
                solution = self.generate_solution()
                fitness = self.fitness(solution)
                solutions.append(solution)
                fitnesses.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution

            best_fitnesses.append(best_fitness)
            self.update_pheromones(solutions, fitnesses)

        end_time = time.time()
        execution_time = end_time - start_time

        return best_solution, best_fitness, best_fitnesses, execution_time


    def generate_solution(self):
        num_edges = len(self.graph.edges())
        solution = [random.choice([0]) for _ in range(num_edges)]
        for i in range(num_edges):
            # Calculate the probability of activating the edge based on the pheromone level
            probability = self.pheromones[i]**self.alpha / (self.fitness(solution) + 1e-10)**self.beta
            # Normalize the probability to be between 0 and 1
            probability /= np.sum(self.pheromones**self.alpha / (self.fitness(solution) + 1e-10)**self.beta)
            # Activate the edge with the calculated probability
            if random.random() < probability[0]:
                solution[i] = 1
            else:
                solution[i] = 0
        return solution

    def update_pheromones(self, solutions, fitnesses):
        self.pheromones *= self.decay_rate
        for solution, fitness in zip(solutions, fitnesses):
            for i in range(len(self.graph.edges())):
                if solution[i] == 1: # If the edge is activated in the solution
                    self.pheromones[i] += self.Q / fitness
