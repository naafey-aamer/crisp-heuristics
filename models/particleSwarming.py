import numpy as np
import networkx as nx
import time

class ParticleSwarmOptimization:
    def __init__(self, graph, total_resources, num_particles=40, max_iter=100, inertia=0.7,
                 cognitive_coef=1.5, social_coef=1.5, max_velocity=0.5, min_velocity=-0.5):
        self.graph = graph
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.inertia = inertia
        self.cognitive_coef = cognitive_coef
        self.social_coef = social_coef
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.total_resources = total_resources
        self.global_best_position = None
        self.global_best_fitness = 0
        self.particles = self.initialize_particles()

    def scoring_function(self, node):
        # Define your scoring_function logic here using self.graph
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


    def initialize_particles(self):
        particles = []
        for _ in range(self.num_particles):
            position = np.random.randint(0, 2, len(self.graph.edges()))
            velocity = np.random.uniform(self.min_velocity, self.max_velocity, len(self.graph.edges()))
            particles.append({'position': position, 'velocity': velocity, 'best_position': None, 'best_fitness': float('-inf')})
        return particles

    def optimize(self):
        all_sols = []
        start_time = time.time()
        for _ in range(self.max_iter):
            for i, particle in enumerate(self.particles):
                fitness_value = self.fitness(particle['position'])

                if fitness_value > particle['best_fitness']:
                    particle['best_fitness'] = fitness_value
                    particle['best_position'] = particle['position'].copy()

                if fitness_value > self.global_best_fitness:
                    self.global_best_fitness = fitness_value
                    self.global_best_position = particle['position'].copy()
                
                all_sols.append(self.global_best_fitness)

                new_velocity = (self.inertia * particle['velocity']) + \
                               (self.cognitive_coef * np.random.rand() * (particle['best_position'] - particle['position'])) + \
                               (self.social_coef * np.random.rand() * (self.global_best_position - particle['position']))

                new_velocity = np.clip(new_velocity, self.min_velocity, self.max_velocity)
                particle['velocity'] = new_velocity

                new_position = np.where(np.random.rand(len(self.graph.edges())) < 1 / (1 + np.exp(-new_velocity)), 1, 0)
                particle['position'] = new_position

        end_time = time.time()
        exec_time = end_time - start_time

        print("Particle Swarming Completed!")

        return all_sols, self.global_best_position, exec_time

# Example usage:
# Assuming G is an instance of a NetworkX graph
# G = nx.Graph()
# ... (Add nodes and edges to G)

# Create an instance of ParticleSwarmOptimization
# pso = ParticleSwarmOptimization(G)

# Run the optimization
# best_solution = pso.optimize()
