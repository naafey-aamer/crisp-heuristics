
from models.geneticAlgo import GeneticAlgorithm
from utils import create_and_display_graph
from utils import compute_accessible_resources
from utils import activate_edges_and_plot
from utils import calculate_average_resources
from utils import fitness
from utils import save_simulation_results
from models.simAnnealing import SimulatedAnnealing 
from models.particleSwarming import ParticleSwarmOptimization
from models.tabuSearch import TabuSearch
import random
# from models.antColonyOptimization import AntColonyOptimization



total_res = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
G = create_and_display_graph(total_res, num_nodes=550, p=0.013, ba_graph=False, m=3, display_graph=False, display_nodes=False)
num_edges_at_end = G.number_of_edges()
print(f"The number of edges in the initial graph are: {num_edges_at_end}")


# Creating an instance of the GeneticAlgorithm class
gen_algo = GeneticAlgorithm(G, total_res)
best_sol_per_gen, best_solution, ga_time = gen_algo.evolve(tournament_size=5, initial_population_size=100, max_population_size=5000, num_iterations=150, mutation_rate=0.01, fitness_threshold=0.3, display_gens=False)
activated_edges = activate_edges_and_plot(G.copy(), best_solution, display_graph=False)
average_resources_per_node = calculate_average_resources(activated_edges, G)
best_fitness = fitness(best_solution, G.copy(), total_res)
save_simulation_results(G, best_sol_per_gen, best_fitness, average_resources_per_node, activated_edges, ga_time, "GA_sim1")


# Create an instance of the SimulatedAnnealing class
initial_temperature = 1000
cooling_rate = 0.01
iteration_limit = 38000
sa = SimulatedAnnealing(G, total_res, initial_temperature, cooling_rate, iteration_limit)
all_solutions, best_solution, sa_time = sa.simulated_annealing()
activated_edges = activate_edges_and_plot(G.copy(), best_solution, display_graph=False)
average_resources_per_node = calculate_average_resources(activated_edges, G)
best_fitness = fitness(best_solution, G.copy(), total_res)
save_simulation_results(G, all_solutions, best_fitness, average_resources_per_node, activated_edges, sa_time, "SA_sim1")

# Create an instance of ParticleSwarmOptimization
pso = ParticleSwarmOptimization(G, total_res, num_particles=55, max_iter=455, inertia=0.7, cognitive_coef=1.5, social_coef=1.5, max_velocity=0.5, min_velocity=-0.5)
all_sols, best_solution, pso_time = pso.optimize()
activated_edges = activate_edges_and_plot(G.copy(), best_solution, display_graph=False)
average_resources_per_node = calculate_average_resources(activated_edges, G)
best_fitness = fitness(best_solution, G.copy(), total_res)
save_simulation_results(G, all_sols, best_fitness, average_resources_per_node, activated_edges, pso_time, "PSO_sim1")


# Create an instance of the TabuSearch class
population = [random.choice([0, 1]) for _ in range(G.number_of_edges())]
ts = TabuSearch(G, population, 55000, 55000, total_res)
best_solution, best_solutions_per_generation, exec_time = ts.find_best_solution()
activated_edges = activate_edges_and_plot(G.copy(), best_solution, display_graph=False)
average_resources_per_node = calculate_average_resources(activated_edges, G)
best_fitness = fitness(best_solution, G.copy(), total_res)
save_simulation_results(G, best_solutions_per_generation, best_fitness, average_resources_per_node, activated_edges, exec_time, "TS_sim1")

# # Create an instance of the AntColonyOptimization class
# aco = AntColonyOptimization(G, total_res, num_ants=10, num_iterations=100, decay_rate=0.5, alpha=1, beta=1, Q=1)
# best_solution, best_fitness, best_fitnesses, execution_time = aco.ant_colony_optimization()
# activated_edges = activate_edges_and_plot(G.copy(), best_solution, display_graph=False)
# average_resources_per_node = calculate_average_resources(activated_edges, G)
# best_fitness = fitness(best_solution, G.copy(), total_res)
# save_simulation_results(G, best_fitnesses, best_fitness, average_resources_per_node, activated_edges, execution_time, "ACO_sim1")
