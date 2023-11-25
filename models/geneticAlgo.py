import random
import copy
import time

class GeneticAlgorithm:
    def __init__(self, graph, total_resources):
        self.graph = graph
        self.total_resources = total_resources
        self.best_individual = None

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

    def calculate_fitness(self, solution):
        for i, edge in enumerate(self.graph.edges()):
            self.graph[edge[0]][edge[1]]['activated'] = bool(solution[i])

        scores = []
        for node in self.graph.nodes():
            score = self.scoring_function(node)
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        return avg_score

    def evolve(self, tournament_size=5, initial_population_size=100, max_population_size=5000,
               num_iterations=2000, mutation_rate=0.01, fitness_threshold=0.3, display_gens = False):
        num_edges = len(self.graph.edges())
        population = [[random.choice([0, 1]) for _ in range(num_edges)] for _ in range(initial_population_size)]

        best_individuals_per_generation = []  # Store best individuals of each generation
        best_overall_individual = 0
        best_overall_score = 0

        start_time = time.time()  # Record the start time

        for iteration in range(num_iterations):
            # Gradually increase population size
            population_size = min(initial_population_size + iteration, max_population_size)
            while len(population) < population_size:
                population.append([random.choice([0, 1]) for _ in range(num_edges)])

            fitnesses = [self.calculate_fitness(solution) for solution in population]

            # Tournament selection
            selected_parents = self.tournament_selection(population, fitnesses, tournament_size)

            new_population = []
            for i in range(0, len(selected_parents), 2):
                parent1 = selected_parents[i]
                parent2 = selected_parents[i + 1] if i + 1 < len(selected_parents) else selected_parents[i]

                # Crossover
                offspring1, offspring2 = self.crossover(parent1, parent2)

                # Mutation
                offspring1 = self.mutate(offspring1, mutation_rate)
                offspring2 = self.mutate(offspring2, mutation_rate)

                new_population.extend([offspring1, offspring2])

            population = new_population

            # Remove individuals with fitness below the threshold
            fitnesses = [self.calculate_fitness(solution) for solution in population]
            below_threshold_indices = [i for i, f in enumerate(fitnesses) if f < fitness_threshold]
            for index in sorted(below_threshold_indices, reverse=True):
                del population[index]

            # Calculate fitnesses for the final population
            final_fitnesses = [self.calculate_fitness(individual) for individual in population]

            # Find the index of the best individual
            best_index = final_fitnesses.index(max(final_fitnesses))

            # Get the best individual and its score
            best_individual = population[best_index]
            best_score = max(final_fitnesses)

            best_individuals_per_generation.append(best_score)  # Store best of this generation

            if best_score > best_overall_score:
                best_overall_individual = best_individual
                best_overall_score = best_score

            if display_gens:
                # Print the score of the best individual for each generation
                print(f"Generation {iteration + 1}: Best Score - {best_score}")

        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the execution time

        # Calculate fitnesses for the final population
        final_fitnesses = [self.calculate_fitness(individual) for individual in population]

        # Find the index of the best individual
        best_index = final_fitnesses.index(max(final_fitnesses))

        # Get the best individual
        best_individual = population[best_index]

        self.best_individual = best_individual

        print("Gentic Evolution Completed!")

        return best_individuals_per_generation, best_overall_individual, execution_time

    def tournament_selection(self, population, fitnesses, tournament_size):
        selected_parents = []
        population_size = len(population)
        for _ in range(population_size):
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_index = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
            selected_parents.append(population[winner_index])
        return selected_parents

    def crossover(self, parent1, parent2):
        # Perform crossover between two solutions
        crossover_point = random.randint(1, len(parent1) - 2)
        return parent1[:crossover_point] + parent2[crossover_point:], parent2[:crossover_point] + parent1[crossover_point:]

    def mutate(self, solution, mutation_rate=0.1):
        mutated_solution = solution[:]  # Create a copy to avoid changing the original solution
        for i in range(len(mutated_solution)):
            if random.random() < mutation_rate:
                mutated_solution[i] = 1 - mutated_solution[i]  # Flip the bit
        return mutated_solution
