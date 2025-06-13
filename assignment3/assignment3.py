#%% Imports and global variables
import math
import random
import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from queue import PriorityQueue
import re

TIME_LIMIT = 0
POPULATION_SIZE = 512
LOCAL_OPTIMUM_THRESHOLD = 100
MAX_ITERATIONS = 1000

PROBLEM = "CVRP" # (CVRP, ACKLEY)
ALGORITHM = "BB" # (MULTI_STAGE_HEURISTIC, ILS, GA, ALNS, BB)

#ILS Parameters
ILS_META_HEURISTIC = "None" # (None, SA, TS, ACO)
CURRENT_TEMPERATURE = 5
COOLING_RATE = 0.93
ACO_EVAPORATION_RATE = 0.7
ACO_Q = 150
ACO_ALPHA = 1.0
ACO_BETA = 4.5

# GA Parameters
PARENT_SELECTION_METHOD = "TOURNAMENT_DET" # (TOP_HALF_UNIFORM, RWS, TOURNAMENT_DET, TOURNAMENT_STOCH)
TOURNAMENT_K = 15
TOURNAMENT_P = 0.86
MIGRATION_RATE = 0.05
ISLANDS_COUNT = 7
MIGRATION_INTERVAL = 25
ELITISM_RATE = 0.1
MUTATION_RATE = 0.25
CROSSOVER_TYPE = "OX" # (OX, PMX, CX, ER, arithmetic, uniform)
MUTATION_TYPE = "insertion" # (displacement, swap, insertion, simple_inversion, inversion, scramble)

#ALNS Parameters
ALNS_PREV_WEIGHT = 0.6 #Operator previous value weight in its new value

#%% Population and Individual classes
class CVRPPopulation:
    def __init__(self, filepath):
        self.truck_capacity = None
        self.trucks_count = None
        self.coords = {}
        self.demands = {}
        self.depot = None
        self.optimal_solution = 0
        self._parse_file(filepath)
        self.dist_matrix = self._compute_distance_matrix()
        self.individuals = []
        self.best_fitness = []
        self.avg_fitness = []
        self.variance = []

    #A function that extracts the problem inputs
    def _parse_file(self, path):
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            section = None
            depot_idx = None
            customer_idx = 1
            if re.search(r'n(\d+)-k(\d+)', lines[0]): 
                self.trucks_count = int(re.search(r'n(\d+)-k(\d+)', lines[0]).group(2))
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith("CAPACITY"): #the truck capacity
                    self.truck_capacity = int(line.split(":")[1])
                elif line.startswith("COMMENT"): #number of trucks
                    if "of trucks" in line:
                        trucks_count = line.split("of trucks:")[-1].split(",")[0]
                        self.trucks_count = int(trucks_count.strip())
                    if "Optimal value:" in line:
                        optimal_solution = line.split("Optimal value:")[-1].split(")")[0]
                        self.optimal_solution = float(optimal_solution.strip())
                elif line == "NODE_COORD_SECTION":
                    section = "coords"
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip() == "DEPOT_SECTION":
                            depot_idx = lines[j+1].strip()
                            break
                elif line == "DEMAND_SECTION":
                    section = "demands"
                    customer_idx = 1
                elif line == "DEPOT_SECTION":
                    break
                elif section == "coords":
                    parts = line.split()
                    if parts[0] == depot_idx:
                        self.depot = (int(parts[1]), int(parts[2]))
                    else:
                        self.coords[customer_idx] = (int(parts[1]), int(parts[2]))
                        customer_idx += 1
                elif section == "demands":
                    parts = line.split()
                    if parts[0] == depot_idx:
                        continue
                    else:
                        self.demands[customer_idx] = int(parts[1])
                        customer_idx += 1

            if not self.truck_capacity or not self.trucks_count or not self.coords or not self.demands:
                raise ValueError("The file is not of the correct format or it missing data.")
        except Exception as e:
            print(f"Error reading input file: {e}")
            sys.exit(1)

    # A function computing the distance matrix
    def _compute_distance_matrix(self):
        n = len(self.coords)
        matrix = np.zeros((n + 1, n + 1))
        for i in self.coords:
            for j in self.coords:
                matrix[i][j] = np.linalg.norm(np.array(self.coords[i]) - np.array(self.coords[j]))
        return matrix

class CVRPIndividual:
    def __init__(self, routes, population):
        self.routes = routes
        self.fitness = 0
        self.population = population
        self.rank = None

    # A function that evaluates the fitness of the individual as the total distance of all routes
    def evaluate(self):
        total_length = 0
        for route in self.routes:
            if not route:
                continue
            route_length = np.linalg.norm(np.array(self.population.coords[route[0]]) - np.array(self.population.depot))
            route_length += sum(self.population.dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
            route_length += np.linalg.norm(np.array(self.population.coords[route[-1]]) - np.array(self.population.depot))
            total_length += route_length
        self.fitness = total_length

class AckleyPopulation:
    def __init__(self):
        self.dim = 10
        self.lower_bound = -32.768
        self.upper_bound = 32.768
        self.a = 20
        self.b = 0.2
        self.c = 2 * math.pi
        self.individuals = []
        self.best_fitness = []
        self.avg_fitness = []
        self.variance = []

class AckleyIndividual:
    def __init__(self, vector, population):
        self.population = population
        self.routes = vector
        self.fitness = 0
        self.rank = None

    def evaluate(self, input_vector = None):
        if input_vector is None:
            vector = self.routes
        else:
            vector = input_vector
        sum_sq = np.sum(vector ** 2)
        sum_cos = np.sum(np.cos(self.population.c * vector))
        term1 = -self.population.a * np.exp(-self.population.b * np.sqrt(sum_sq / self.population.dim))
        term2 = -np.exp(sum_cos / self.population.dim)
        result = term1 + term2 + self.population.a + math.e
        if input_vector is None:
            self.fitness = result
        else:
            return result

#%% Multi-Stage Heuristic Algorithm
class MSHeuristicsAlgorithm:
    def __init__(self, population):
        self.population = population

    def solve(self):
        #initializing timer
        elapsed_start_time = time.time()
        cpu_start_time = time.process_time()

        #generating the population until population size is reached or no time left
        while time.time() - elapsed_start_time < TIME_LIMIT and len(self.population.individuals) < POPULATION_SIZE:
            if PROBLEM == "CVRP":
                assignment = cvrp_generate_assignment(self.population) #first stage
                if assignment:
                    new_individual = CVRPIndividual(assignment, self.population)
                    new_individual.evaluate() 
                    self.tsp_solve(new_individual) #second stage
                    self.population.individuals.append(new_individual)
            elif PROBLEM == "ACKLEY":
                assignment = np.random.uniform(self.population.lower_bound, self.population.upper_bound, self.population.dim) #first stage
                new_individual = AckleyIndividual(assignment, self.population)
                self.local_search_optimize(new_individual) #second stage
                self.population.individuals.append(new_individual)

        elapsed_end_time = time.time()
        cpu_end_time = time.process_time()

        #printing the best result found
        best_individual(self.population, elapsed_end_time - elapsed_start_time, cpu_end_time - cpu_start_time)
        if PROBLEM == "CVRP":
            plot_routes(self.population, min(self.population.individuals, key=lambda ind: ind.fitness).routes)

    #A function that order the customers in each route
    def tsp_solve(self, individual):
        optimized = []
        total_cost = 0
        for route in individual.routes:
            ordered, cost = self.nn_route_reorder(route)
            optimized.append(ordered)
            total_cost += cost
        individual.routes = optimized
        individual.fitness = total_cost

    #A function that reorders the route using nearest neighbor heuristic
    def nn_route_reorder(self, route):
        if not route:
            return [], 0
        current, cost = min(((node, np.linalg.norm(np.array(self.population.coords[node]) - np.array(self.population.depot))) for node in route), key=lambda x: x[1])
        ordered = [current] #initializing the route with the closest customer to the depot
        unvisited = set(route) 
        unvisited.remove(current)

        while unvisited:
            nearest = None
            min_dist = float('inf')
            for node in unvisited:
                d = self.population.dist_matrix[current][node]
                if d < min_dist:
                    min_dist = d
                    nearest = node
            cost += min_dist
            ordered.append(nearest) #in each iteration the nearest customer is added to the route
            current = nearest
            unvisited.remove(nearest)

        cost += np.linalg.norm(np.array(self.population.coords[current]) - np.array(self.population.depot)) #adding the cost of returning to the depot
        return ordered, cost

    def local_search_optimize(self, individual):
        best_vector = individual.routes.copy()
        best_fitness = individual.evaluate(best_vector)

        for _ in range(1000):
            new_vector = best_vector + np.random.normal(0, 1, self.population.dim)
            new_vector = np.clip(new_vector, self.population.lower_bound, self.population.upper_bound)
            new_fitness = individual.evaluate(new_vector)
            if new_fitness < best_fitness:
                best_vector = new_vector
                best_fitness = new_fitness

        individual.routes = best_vector
        individual.fitness = best_fitness

#%% ILS Algorithm
class ILSAlgorithm:
    def __init__(self, population):
        self.population = population
        self.tabu_list = [] #for Tabu Search use 
        self.tabu_set = set() #for Tabu Search use 
        self.pheromone = {} #for ACO use

    def solve(self):
        elapsed_start_time = time.time()
        cpu_start_time = time.process_time()

        while time.time() - elapsed_start_time < TIME_LIMIT and len(self.population.individuals) < POPULATION_SIZE:
            if PROBLEM == "CVRP":    
                assignment = cvrp_generate_assignment(self.population)
                if assignment:
                    new_ind = CVRPIndividual(assignment, self.population)
                    new_ind.evaluate()
                    self.population.individuals.append(new_ind)
            elif PROBLEM == "ACKLEY":
                assignment = np.random.uniform(self.population.lower_bound, self.population.upper_bound, self.population.dim)
                new_ind = AckleyIndividual(assignment, self.population)
                new_ind.fitness = new_ind.evaluate(new_ind.routes)
                self.population.individuals.append(new_ind)

        iter_count = 0
        no_improvement_count = 0
        best_fitness_found = float('inf')
        best_solution_found = None
        while time.time() - elapsed_start_time < TIME_LIMIT and no_improvement_count < LOCAL_OPTIMUM_THRESHOLD and iter_count < MAX_ITERATIONS:
            update_temperature()
            for i, individual in enumerate(self.population.individuals):
                if ILS_META_HEURISTIC == "None":
                    if PROBLEM == "CVRP":
                        neighbor = cvrp_find_neighbor(self.population, individual) 
                    elif PROBLEM == "ACKLEY":
                        neighbor = ackley_find_neighbor(self.population, individual) 
                    if neighbor.fitness < individual.fitness:
                        individual.routes = neighbor.routes
                        individual.fitness = neighbor.fitness
                elif ILS_META_HEURISTIC == "SA":
                    if PROBLEM == "CVRP":
                        neighbor = cvrp_find_neighbor(self.population, individual) 
                    elif PROBLEM == "ACKLEY":
                        neighbor = ackley_find_neighbor(self.population, individual) 
                    if neighbor.fitness < individual.fitness:
                        individual.routes = neighbor.routes
                        individual.fitness = neighbor.fitness
                    else:
                        toReplace = simulated_annealing(individual.fitness, neighbor.fitness)
                        if toReplace:
                            individual.routes = neighbor.routes
                            individual.fitness = neighbor.fitness
                elif ILS_META_HEURISTIC == "TS":
                    if not self.tabu_list:
                        self.tabu_hash_initiallize()
                    if PROBLEM == "CVRP":
                        neighbors = [cvrp_find_neighbor(self.population, individual, method) for method in ["2-opt", "relocate", "reposition", "swap", "shuffle"]]
                    elif PROBLEM == "ACKLEY":
                        neighbors = [ackley_find_neighbor(self.population, individual, method) for method in ["shift_one", "shift_all", "set_random"]]
                    neighbors.sort(key=lambda ind: ind.fitness)
                    for neighbor in neighbors:
                        if PROBLEM == "CVRP":
                            neighbor_hash = str(sorted([tuple(route) for route in neighbor.routes]))
                        elif PROBLEM == "ACKLEY":
                            neighbor_hash = str(np.round(neighbor.routes, decimals=2).tolist())
                        if neighbor_hash not in self.tabu_set:
                            individual.routes = neighbor.routes
                            individual.fitness = neighbor.fitness
                            self.tabu_list.append(neighbor_hash)
                            self.tabu_set.add(neighbor_hash)
                            if len(self.tabu_list) > POPULATION_SIZE * 3.5:
                                oldest = self.tabu_list.pop(0)
                                self.tabu_set.discard(oldest)
                            break
                elif ILS_META_HEURISTIC == "ACO":
                    if not self.pheromone:
                        self.pheromone_initialize()
                        self.pheromone_update()
                    new_individual = self.aco_solution_construct()
                    individual.routes = new_individual.routes
                    individual.fitness = new_individual.fitness
                    
                    if i == len(self.population.individuals) - 1:
                        self.pheromone_update()

            #best individual and non-improvement updates
            best_solution_iter = min(self.population.individuals, key=lambda ind: ind.fitness)
            if best_solution_iter.fitness < best_fitness_found:
                best_fitness_found = best_solution_iter.fitness
                best_solution_found = best_solution_iter.routes
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            iteration_statistics(self.population, iter_count, best_solution_found, best_fitness_found)
            iter_count += 1
            
        elapsed_end_time = time.time()
        cpu_end_time = time.process_time()

        best_individual(self.population, elapsed_end_time - elapsed_start_time, cpu_end_time - cpu_start_time, best_solution_found, best_fitness_found)
        if PROBLEM == "CVRP":
            plot_routes(self.population, best_solution_found)
        plot_iterations_statistics(self.population)

    #A function that initializes the tabu list and set with the hash of current population individuals
    def tabu_hash_initiallize(self):
        for ind in self.population.individuals:
            if PROBLEM == "CVRP":
                ind_hash = str(sorted([tuple(route) for route in ind.routes]))
            elif PROBLEM == "ACKLEY":
                ind_hash = str(np.round(ind.routes, decimals=2).tolist())
            self.tabu_list.append(ind_hash)
            self.tabu_set.add(ind_hash)
    
    #A function that initializes the pheromone matrix
    def pheromone_initialize(self):
        for i in self.population.coords:
            #self.pheromone[(i, 0)] = 1.0
            #self.pheromone[(0, i)] = 1.0
            for j in self.population.coords:
                if i != j:
                    self.pheromone[(i, j)] = 1.0 
    
    #A function that constructs a solution using ACO
    def aco_solution_construct(self):
        routes = []
        unvisited = set(self.population.coords.keys())
        current = random.choice(list(unvisited)) #start from a random customer
        load = self.population.demands[current]
        curr_route = [current]
        unvisited.remove(current)
        while unvisited:
            pheromone_prob = []
            for customer in unvisited:
                if load + self.population.demands[customer] <= self.population.truck_capacity:
                    tau = self.pheromone.get((current, customer), 1.0)
                    eta = 1.0 / (self.population.dist_matrix[current][customer] + 1e-6) #the small value to avoid division by zero
                    pheromone = (tau ** ACO_ALPHA) * (eta ** ACO_BETA)
                    pheromone_prob.append((customer, pheromone))
            # tau_to_depot = self.pheromone.get((current, 0), 1.0)
            # eta_to_depot = 1.0 / (self.population.dist_matrix[current][0] + 1e-6)
            # depot_pheromone = (tau_to_depot ** ACO_ALPHA) * (eta_to_depot ** ACO_BETA)
            if pheromone_prob:
                next_customer = None
                pheromone_total = sum(p[1] for p in pheromone_prob)
                if pheromone_total > 0:
                    pheromone_prob = [(cust, prob / pheromone_total) for cust, prob in pheromone_prob]
                    random_choice = random.random()
                    acc = 0
                    for cust, prob in pheromone_prob:
                        acc += prob
                        if acc >= random_choice:
                            next_customer = cust
                            break
                    curr_route.append(next_customer)
                    load += self.population.demands[next_customer]
                    unvisited.remove(next_customer)
                    current = next_customer
            else: # or depot_pheromone > max_pheromone:
                routes.append(curr_route)
                if unvisited:
                    current = random.choice(list(unvisited))
                    load = self.population.demands[current]
                    curr_route = [current]
                    unvisited.remove(current)

        #add the last route if it has customers
        if curr_route:
            routes.append(curr_route)
        new_individual = CVRPIndividual(routes, self.population)
        new_individual.evaluate()
        return new_individual
    
    #A function that updates the pheromones after each iteration
    def pheromone_update(self):

        # Evaporation phase
        for key in self.pheromone.keys():
            self.pheromone[key] *= ACO_EVAPORATION_RATE 
        
        # Deposit phase
        for individual in self.population.individuals:
            for route in individual.routes:
                # key = (0, route[0])
                # self.pheromone[key] += ACO_Q / individual.fitness
                # key = (route[-1], 0)
                # self.pheromone[key] += ACO_Q / individual.fitness
                for i in range(len(route) - 1):
                    key = (route[i], route[i + 1])
                    if key[0] != key[1]:
                        self.pheromone[key] += ACO_Q / individual.fitness

#%% GA with Island Model
class GAAlgorithm:
    def __init__(self, population):
        self.population = population
        self.islands = [[] for _ in range(ISLANDS_COUNT)]
        self.generation = 0

    def solve(self):
        elapsed_start_time = time.time()
        cpu_start_time = time.process_time()
        self.individual_islands_initialize()

        best_fitness_found = float('inf')
        best_solution_found = None
        no_improvement_count = 0
        while time.time() - elapsed_start_time < TIME_LIMIT and no_improvement_count < LOCAL_OPTIMUM_THRESHOLD and self.generation < MAX_ITERATIONS:
            for i in range(ISLANDS_COUNT):
                self.islands[i] = self.evolve_island(self.islands[i])
            self.generation += 1
            if self.generation % MIGRATION_INTERVAL == 0:
                self.migrate()

            self.population.individuals = [ind for island in self.islands for ind in island]

            #best individual and non-improvement updates
            best_solution_iter = min(self.population.individuals, key=lambda ind: ind.fitness)
            if best_solution_iter.fitness < best_fitness_found:
                best_fitness_found = best_solution_iter.fitness
                best_solution_found = best_solution_iter.routes
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            iteration_statistics(self.population, self.generation, best_solution_found, best_fitness_found)

        elapsed_end_time = time.time()
        cpu_end_time = time.process_time()

        best_individual(self.population, elapsed_end_time - elapsed_start_time, cpu_end_time - cpu_start_time, best_solution_found, best_fitness_found)
        if PROBLEM == "CVRP":    
            plot_routes(self.population, best_solution_found)
        plot_iterations_statistics(self.population)

    # A function that initializes the individuals and the islands 
    def individual_islands_initialize(self):
        individual_idx = 0
        while individual_idx < POPULATION_SIZE:
            if PROBLEM == "CVRP":
                assignment = cvrp_generate_assignment(self.population)
                if assignment is not None:
                    new_ind = CVRPIndividual(assignment, self.population)
            elif PROBLEM == "ACKLEY":
                assignment = np.random.uniform(self.population.lower_bound, self.population.upper_bound, self.population.dim)
                if assignment is not None:
                    new_ind = AckleyIndividual(assignment, self.population)
            if assignment is not None:
                new_ind.evaluate()
                island_idx = individual_idx % ISLANDS_COUNT
                self.islands[island_idx].append(new_ind)
                individual_idx += 1

    # A function that performs the migration process
    def migrate(self):
        migrants = []
        for i, island in enumerate(self.islands):
            migrants_num = int(len(island) * MIGRATION_RATE * 0.5)
            self.islands[i].sort(key=lambda ind: ind.fitness)
            island_best_migrants = island[:migrants_num] #choose migrants out of the best
            self.islands[i] = self.islands[i][migrants_num:] #remove the best individuals from the island
            random.shuffle(self.islands[i])
            island_random_migrants = self.islands[i][:migrants_num] #choose random migrants
            self.islands[i] = self.islands[i][migrants_num:] #remove the random individuals from the island
            migrants.append(island_random_migrants + island_best_migrants)

        for i in range(ISLANDS_COUNT):
            for m in migrants[i]:
                target = (i + 1) % ISLANDS_COUNT
                self.islands[target].append(m)

    # A function that generate the next generation of individuals for the island
    def evolve_island(self, island):
        next_gen = sorted(island, key=lambda ind: ind.fitness)[:int(ELITISM_RATE * len(island))] #keep the elite for the next generation
        while len(next_gen) < len(island):
            _, p1 = self.select_parents(island)
            _, p2 = self.select_parents(island)
            child = self.crossover(p1, p2)
            if child is None: #if crossover failed
                continue
            if random.random() < MUTATION_RATE:
                child = self.mutate(child)
            if child is None: #if mutation failed
                continue
            child = CVRPIndividual(child, self.population) if PROBLEM == "CVRP" else AckleyIndividual(child, self.population)
            child.evaluate()
            next_gen.append(child)
        return sorted(next_gen, key=lambda ind: ind.fitness)

    def select_parents(self, island):
        if PARENT_SELECTION_METHOD == "TOP_HALF_UNIFORM":
            return self.top_half_uniform_selection(island)
        elif PARENT_SELECTION_METHOD == "RWS":
            return self.rws_selection(island)
        elif PARENT_SELECTION_METHOD == "TOURNAMENT_DET":
            self.fitness_ranking(island)
            return self.tournament_selection_deter(island)
        elif PARENT_SELECTION_METHOD == "TOURNAMENT_STOCH":
            self.fitness_ranking(island)
            return self.tournament_selection_stoch(island)

    #A function that selects a parent using Top Half Uniform
    def top_half_uniform_selection(self, island):
        rand = random.randint(0, len(island) // 2 - 1)
        return rand, island[rand]

    #A function that selects a parent using RWS
    def rws_selection(self, island):

        #Scaling the fitness values using linear scaling
        max_fitness = max(ind.fitness for ind in island)
        scaled_fitnesses = self.linear_scaling(island, -1, max_fitness)

        #Converting fitness values to probabilities
        total_scaled = sum(scaled_fitnesses)
        selection_probs = [fit / total_scaled for fit in scaled_fitnesses]
        if total_scaled == 0:
            random_choice = random.randint(0, len(island) - 1)
            return random_choice, island[random_choice]

        #Building the "roulete"
        cumulative_probs = []
        cumsum = 0
        for prob in selection_probs:
            cumsum += prob
            cumulative_probs.append(cumsum)

        #Selecting a random individual
        pick = random.random()
        for i, prob in enumerate(cumulative_probs):
            if pick < prob:
                return i, island[i]
        return len(island) - 1, island[-1]

    #A function that selects a parent using Deterministic Tournament Selection
    def tournament_selection_deter(self, island):
        k = min(TOURNAMENT_K, len(island))
        tournament = random.sample(range(len(island)), k) #choosing K random indices
        tournament.sort(key=lambda ind: island[ind].rank) #sorting the corresponding individuals
        return tournament[0], island[tournament[0]]

    #A function that selects a parent using Stochastic Tournament Selection
    def tournament_selection_stoch(self, island):
        k = min(TOURNAMENT_K, len(island))
        tournament = random.sample(range(len(island)), k) #choosing K random indices
        tournament.sort(key=lambda ind: island[ind].rank) #sorting the corresponding individuals
        for i in range(TOURNAMENT_K):
            if random.random() < TOURNAMENT_P:
                return tournament[i], island[tournament[i]]
        return tournament[-1], island[tournament[-1]]

    # A function that performs crossover between two parents
    def crossover(self, parent1, parent2):
        if PROBLEM == "CVRP":
            genome1 = self.routes_to_permutation(parent1.routes)
            genome2 = self.routes_to_permutation(parent2.routes)
        elif PROBLEM == "ACKLEY":
            genome1 = parent1.routes
            genome2 = parent2.routes

        if CROSSOVER_TYPE == "PMX":
            child = self.pmx_crossover(genome1, genome2)
        elif CROSSOVER_TYPE == "OX":
            child = self.order_crossover(genome1, genome2)
        elif CROSSOVER_TYPE == "CX":
            child = self.cx_crossover(genome1, genome2)
        elif CROSSOVER_TYPE == "ER":
            child = self.er_crossover(genome1, genome2)
        elif CROSSOVER_TYPE == "arithmetic":
            child = self.arithmetic_crossover(genome1, genome2)
        elif CROSSOVER_TYPE == "uniform":
            child = self.uniform_crossover(genome1, genome2)

        if PROBLEM == "CVRP":
            return self.permutation_to_routes(child)
        elif PROBLEM == "ACKLEY":
            return child

    def pmx_crossover(self, p1, p2):
        size = len(p1)
        indices_to_copy = random.sample(range(size), size // 2)
        child = [p1[i] if i in indices_to_copy else None for i in range(size)]
        for i in range(size):
            if child[i] is None:
                gene = p2[i]
                while gene in child:
                    index = p1.index(gene)
                    gene = p2[index]
                child[i] = gene
        return (child)

    def order_crossover(self, p1, p2):
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        middle = p1[a:b]
        if PROBLEM == "CVRP":
            remaining = [item for item in p2 if item not in middle]
            return remaining[:a] + middle + remaining[a:]
        elif PROBLEM == "ACKLEY":
            child = p2.copy()
            child[a:b] = middle
            return child

    def cx_crossover(self, p1, p2):
        size = len(p1)
        child = [None] * size
        indices = list(range(size))
        cycle = 0
        while None in child:
            start = indices[0]
            idx = start
            while True:
                if cycle % 2 == 0:
                    child[idx] = p1[idx]
                else:
                    child[idx] = p2[idx]
                indices.remove(idx)
                idx = p1.index(p2[idx])
                if idx == start:
                    break
            cycle += 1
        return (child)

    def er_crossover(self, p1, p2):
        size = len(p1)
        child = []
        used = set()
        def get_neighbors(p):
            neighbors = {i: set() for i in p}
            for i in range(len(p)):
                a, b = p[i], p[(i + 1) % size]
                neighbors[a].add(b)
                neighbors[b].add(a)
            return neighbors
        n1 = get_neighbors(p1)
        n2 = get_neighbors(p2)
        neighbors = {k: n1[k].union(n2[k]) for k in n1}
        current = random.choice(p1)
        child.append(current)
        used.add(current)
        while len(child) < size:
            common = n1[current].intersection(n2[current]) - used
            if common:
                next_city = min(common)
            else:
                candidates = neighbors[current] - used
                if candidates:
                    next_city = min(candidates)
                else:
                    remaining = [c for c in p1 if c not in used]
                    next_city = min(remaining)
            child.append(next_city)
            used.add(next_city)
            current = next_city
        return (child)

    # A function that performs arithmetic crossover, used for Ackley problem
    def arithmetic_crossover(self, p1, p2):
        lower_bound = -32.768
        upper_bound = 32.768
        alpha = random.random()
        child = alpha * p1 + (1 - alpha) * p2
        child = np.clip(child, lower_bound, upper_bound)
        return child
    
    # A function that performs uniform crossover, used for Ackley problem
    def uniform_crossover(self, p1, p2):
        lower_bound = -32.768
        upper_bound = 32.768
        mask = np.random.rand(len(p1)) < 0.5
        child_vector = np.where(mask, p1, p2)
        child = np.clip(child_vector, lower_bound, upper_bound)
        return child


    #A function that mutates the individual
    def mutate(self, individual):
        if PROBLEM == "CVRP":
            individual_genome = self.routes_to_permutation(individual)
        elif PROBLEM == "ACKLEY":
            individual_genome = individual.tolist() 
        if MUTATION_TYPE == "displacement":
            individual_genome = self.displacement_mutate(individual_genome)
        elif MUTATION_TYPE == "swap":
            individual_genome =self.swap_mutate(individual_genome)
        elif MUTATION_TYPE == "insertion":
            individual_genome = self.insertion_mutate(individual_genome)
        elif MUTATION_TYPE == "simple_inversion":
            individual_genome = self.simple_inversion_mutate(individual_genome)
        elif MUTATION_TYPE == "inversion":
            individual_genome = self.inversion_mutate(individual_genome)
        elif MUTATION_TYPE == "scramble":
            individual_genome = self.scramble_mutate(individual_genome)

        if PROBLEM == "CVRP":
            ind_genome = self.permutation_to_routes(individual_genome)
            if ind_genome is None:
                return None
            else:
                return self.permutation_to_routes(individual_genome)
        elif PROBLEM == "ACKLEY":
            return np.array(individual_genome)


    def displacement_mutate(self, genome):
        i, j = random.sample(range(len(genome)), 2)
        if i > j:
            i, j = j, i
        segment = genome[i:j + 1]
        remainder = genome[:i] + genome[j + 1:]
        insertion_place = random.randint(0, len(remainder))
        genome = remainder[:insertion_place] + segment + remainder[insertion_place:]
        return genome

    def swap_mutate(self, genome):
        i, j = random.sample(range(len(genome)), 2)
        genome[i], genome[j] = genome[j], genome[i]
        return genome

    def insertion_mutate(self, genome):
        i, j = random.sample(range(len(genome)), 2)
        gene = genome.pop(i)
        genome.insert(j, gene)
        return genome

    def simple_inversion_mutate(self, genome):
        i, j = random.sample(range(len(genome)), 2)
        if i > j:
            i, j = j, i
        segment = genome[i:j + 1]
        segment.reverse()
        genome = genome[:i] + segment + genome[j + 1:]
        return genome

    def inversion_mutate(self, genome):
        i, j = random.sample(range(len(genome)), 2)
        if i > j:
            i, j = j, i
        segment = genome[i:j + 1]
        segment.reverse()
        remainder = genome[:i] + genome[j + 1:]
        insertion_place = random.randint(0, len(remainder))
        genome = remainder[:insertion_place] + segment + remainder[insertion_place:]
        return genome

    def scramble_mutate(self, genome):
        i, j = random.sample(range(len(genome)), 2)
        if i > j:
            i, j = j, i
        segment = genome[i:j + 1]
        random.shuffle(segment)
        genome = genome[:i] + segment + genome[j + 1:]
        return genome

    #A function that converts individual representation to permutation
    def routes_to_permutation(self, individual):
        permutation = []
        for route in individual:
            permutation.extend(route)
        return permutation
        
    #A function that converts permutation representation to routes
    def permutation_to_routes(self, permutation, try_count = 0):
        routes = []
        current_route = [permutation[0]]
        demand = self.population.demands[permutation[0]]
        for cid in permutation[1:]:
            if demand + self.population.demands[cid] > self.population.truck_capacity:
                if len(routes) > self.population.trucks_count:
                    return None
                routes.append(current_route)
                current_route = [cid]
                demand = self.population.demands[cid]
            else:
                if len(routes) < self.population.trucks_count:
                    if (2* (np.linalg.norm(np.array(self.population.coords[cid]) - np.array(self.population.depot)))) < np.linalg.norm(np.array(self.population.coords[current_route[-1]]) - np.array(self.population.depot)):
                        routes.append(current_route)
                        current_route = [cid]
                        demand = self.population.demands[cid]
                        continue
                current_route.append(cid)
                demand += self.population.demands[cid]
        if current_route:
            routes.append(current_route)
        return routes

    def initiate_random_permutate(self):
        permutation = list(self.population.coords.keys())
        random.shuffle(permutation)
        return self.permutation_to_routes(permutation)

    #A function that linearly scales the fitness values
    def linear_scaling(self, individuals, a, b):
        scaled_fitness = [a * ind.fitness + b for ind in individuals]
        return scaled_fitness

    def fitness_ranking(self, island):
        sorted_inds = sorted(island, key=lambda ind: ind.fitness)
        for rank, ind in enumerate(sorted_inds):
            ind.rank = rank + 1


#%% ALNS Algorithm
class ALNSAlgorithm:
    def __init__(self, population):
        self.population = population 
        self.operators = ["2-opt", "relocate", "reposition", "swap", "shuffle"] if PROBLEM == "CVRP" else ["shift_one", "shift_all", "set_random"]
        self.operator_weights = [1.0] * len(self.operators)
        self.scores = [0.0] * len(self.operators)
        self.uses = [1] * len(self.operators)

    def solve(self):
        elapsed_start_time = time.time()
        cpu_start_time = time.process_time()

        while len(self.population.individuals) < POPULATION_SIZE and time.time() - elapsed_start_time < TIME_LIMIT:
            if PROBLEM == "CVRP":
                assignment = cvrp_generate_assignment(self.population)
                if assignment:
                    new_ind = CVRPIndividual(assignment, self.population)
                    new_ind.evaluate()
                    self.population.individuals.append(new_ind)
            elif PROBLEM == "ACKLEY":
                assignment = np.random.uniform(self.population.lower_bound, self.population.upper_bound, self.population.dim)
                new_ind = AckleyIndividual(assignment, self.population)
                new_ind.fitness = new_ind.evaluate(new_ind.routes)
                self.population.individuals.append(new_ind)

        best_fitness_found = float('inf')
        best_solution_found = None
        no_improvement_count = 0
        iter_count = 0
        while time.time() - elapsed_start_time < TIME_LIMIT and no_improvement_count < LOCAL_OPTIMUM_THRESHOLD and iter_count < MAX_ITERATIONS:
            update_temperature()
            for individual in self.population.individuals:
                op_idx = self.select_operator()
                self.uses[op_idx] += 1
                if PROBLEM == "CVRP":
                    new_ind = cvrp_find_neighbor(self.population, individual, method=self.operators[op_idx])
                elif PROBLEM == "ACKLEY":
                    new_ind = ackley_find_neighbor(self.population, individual, method=self.operators[op_idx])
                if new_ind.fitness < individual.fitness:
                    individual.routes = new_ind.routes
                    individual.fitness = new_ind.fitness
                    self.scores[op_idx] += 1
                else:
                    if simulated_annealing(individual.fitness, new_ind.fitness):
                        individual.routes = new_ind.routes
                        individual.fitness = new_ind.fitness
            
            self.update_weights()

            #best individual and non-improvement updates
            best_solution_iter = min(self.population.individuals, key=lambda ind: ind.fitness)
            if best_solution_iter.fitness < best_fitness_found:
                best_fitness_found = best_solution_iter.fitness
                best_solution_found = best_solution_iter.routes
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            iteration_statistics(self.population, iter_count, best_solution_found, best_fitness_found)
            iter_count += 1

        elapsed_end_time = time.time()
        cpu_end_time = time.process_time()
        
        best_individual(self.population, elapsed_end_time - elapsed_start_time, cpu_end_time - cpu_start_time, best_solution_found, best_fitness_found)
        if PROBLEM == "CVRP":
            plot_routes(self.population, best_solution_found)
        plot_iterations_statistics(self.population)

    # A function that selects an operator, using a weighted random choice
    def select_operator(self):
        total = sum(self.operator_weights)
        probs = [w / total for w in self.operator_weights]
        r = random.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i
        return len(probs) - 1

    # A function that updates the operator weights based on their uses and scores
    def update_weights(self):
        for i in range(len(self.operator_weights)):
            self.operator_weights[i] = (ALNS_PREV_WEIGHT * self.operator_weights[i]) + ((1-ALNS_PREV_WEIGHT) * (self.scores[i] / self.uses[i]))

#%% Branch and Bound Algorithm for CVRP
class BranchAndBoundAlgorithm:
    def __init__(self, population):
        self.population = population
        self.best_solution = None
        self.best_cost = float('inf')
        self.k_nearest_neighbors = self.compute_k_nearest_neighbors(k=3) if PROBLEM == "CVRP" else None

    def solve(self):
        if PROBLEM == "CVRP":
            self.cvrp_solve()
        elif PROBLEM == "ACKLEY":
            self.ackley_solve()

    def ackley_solve(self):
        elapsed_start_time = time.time()
        cpu_start_time = time.process_time()
        iteration = 0

        dim = self.population.dim
        lower_bound = self.population.lower_bound
        upper_bound = self.population.upper_bound

        dummy_individual = AckleyIndividual(np.zeros(dim), self.population)

        #Initialize the priority queue 
        initial_state = (0, [], 0)  #(bound, vector_so_far, variable_index)
        queue = PriorityQueue()
        queue.put(initial_state)

        while not queue.empty() and time.time() - elapsed_start_time < TIME_LIMIT:
            bound, vector_so_far, var_idx = queue.get()
            iteration += 1

            #If the vector is full, evaluate it 
            if var_idx == dim:
                vector = np.array(vector_so_far)
                fitness = dummy_individual.evaluate(vector)
                if fitness < self.best_cost: #if it better than the current best update the best
                    self.best_cost = fitness
                    self.best_solution = vector.copy()
                continue

            
            subdivisions = 100
            step = (upper_bound - lower_bound) / subdivisions

            for i in range(subdivisions + 1):

                #generate the next dimensions value randomly from the subdivision
                xi = random.uniform(lower_bound + i * step, lower_bound + (i + 1) * step) 
                new_vector = vector_so_far + [xi]

                #lower bound estimate: current partial vector padded with 0
                partial_vector = np.array(new_vector + [0] * (dim - len(new_vector)))
                partial_cost = dummy_individual.evaluate(partial_vector)

                #Prune it if it's not promising
                if partial_cost < self.best_cost:
                    queue.put((partial_cost, new_vector, var_idx + 1))

            if iteration % 100 == 0: #printing the promising solution every 1000 iterations
                best_ind = queue.queue[0] if queue.queue else None
                if best_ind:
                    best_solution = best_ind[1] 
                    best_fitness = best_ind[0] 
                    self.iteration_partial_statistics(iteration, best_solution, best_fitness, best_fitness)

        elapsed_end_time = time.time()
        cpu_end_time = time.process_time()

        best_individual(self.population, elapsed_end_time - elapsed_start_time, cpu_end_time - cpu_start_time, self.best_solution, self.best_cost)
        if PROBLEM == "CVRP":
            plot_routes(self.population, self.best_solution)

    def cvrp_solve(self):
        elapsed_start_time = time.time()
        cpu_start_time = time.process_time()
        iteration = 0

        customers = list(self.population.coords.keys())
        customers.sort(key=lambda cid: np.linalg.norm(np.array(self.population.coords[cid]) - np.array(self.population.depot))) #the sorting meant to take the nearest remained city to the depot each time a new route is started
        initial_state = (0, [], customers, 0, [])  #(priority, current_route, remaining_customers, current_cost, all_routes)

        queue = PriorityQueue()
        queue.put(initial_state)

        while not queue.empty() and time.time() - elapsed_start_time < TIME_LIMIT:
            _, route, remaining, cost, all_routes = queue.get()
            iteration += 1
            #if no customer remained, evaluate the solution
            if not remaining:
                total_routes = all_routes + [route] if route else all_routes
                total_cost = sum(self.estimate_cost(route) for route in total_routes)
                if total_cost < self.best_cost: #if it's better than the currently best, replace it 
                    self.best_cost = total_cost
                    self.best_solution = total_routes
                continue

            if route:
                last_customer = route[-1]
                candidates = [c for c in self.k_nearest_neighbors[last_customer] if c in remaining] #candidates are the k nearest cities
                if not candidates or len(candidates) == 1:
                    candidates = self.compute_k_nearest_neighbors(k=2, customer=last_customer, other_customers=remaining)[last_customer]
            else:
                candidates = remaining

            for customer in candidates:
                i = remaining.index(customer)
                new_route = route + [customer]
                new_demand = sum(self.population.demands[c] for c in new_route)
                new_remaining = remaining[:i] + remaining[i+1:]

                if new_demand <= self.population.truck_capacity: #truck capacity exceed check
                    partial_routes = all_routes.copy()
                    partial_cost = cost
                    lower_bound = partial_cost + self.estimate_cost(new_route) + self.mst_estimate(new_remaining)
                    if lower_bound < self.best_cost: #if the state is promising insert it into queue
                        queue.put((lower_bound, new_route, new_remaining, partial_cost, partial_routes))
                else: #start a new route if the current one exceeds the capacity
                    if route:
                        partial_routes = all_routes + [route]
                        partial_cost = cost + self.estimate_cost(route)
                        lower_bound = partial_cost + self.estimate_cost([customer]) + self.mst_estimate(new_remaining)
                        if lower_bound < self.best_cost: #if the state is promising insert it into queue
                            queue.put((lower_bound, [customer], new_remaining, partial_cost, partial_routes))
            
            if iteration % 100 == 0: #printing the promising solution every 1000 iterations
                best_ind = queue.queue[0] if queue.queue else None
                if best_ind:
                    best_solution = best_ind[4] 
                    best_fitness = best_ind[3] 
                    best_lower_bound = best_ind[0] 
                    self.iteration_partial_statistics(iteration, best_solution, best_fitness, best_lower_bound)
        
        elapsed_end_time = time.time()
        cpu_end_time = time.process_time()

        if self.best_solution:
            best_ind = CVRPIndividual(self.best_solution, self.population)
            best_ind.fitness = self.best_cost
            best_individual(self.population, elapsed_end_time - elapsed_start_time, cpu_end_time - cpu_start_time, best_ind.routes, best_ind.fitness)
            plot_routes(self.population, best_ind.routes)

    # A function that computes the k nearest neighbors for each customer
    def compute_k_nearest_neighbors(self, k=3, customer=None, other_customers = None):
        neighbors = {}
        all_customers = list(self.population.coords.keys()) if customer is None else [customer]
        all_other_customers = list(self.population.coords.keys()) if other_customers is None else other_customers
        for i in all_customers:
            distances = []
            for j in all_other_customers:
                if i != j:
                    dist = self.population.dist_matrix[i][j]
                    distances.append((dist, j))
            distances.sort()
            neighbors[i] = [j for _, j in distances[:k]]
        return neighbors

    #A function that estimates the cost of a given route
    def estimate_cost(self, route):
        if not route:
            return 0
        cost = np.linalg.norm(np.array(self.population.coords[route[0]]) - np.array(self.population.depot))
        cost += sum(self.population.dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
        cost += np.linalg.norm(np.array(self.population.coords[route[-1]]) - np.array(self.population.depot))
        return cost

    #A function that estimates the MST cost for the remaining customers, which used as a lower bound estimate
    def mst_estimate(self, remaining_customers):
        if not remaining_customers:
            return 0
        coords = [self.population.coords[cid] for cid in remaining_customers]
        n = len(coords)
        visited = [False] * n
        min_edge = [float('inf')] * n
        min_edge[0] = 0
        total = 0
        for _ in range(n):
            u = -1
            for i in range(n):
                if not visited[i] and (u == -1 or min_edge[i] < min_edge[u]):
                    u = i
            visited[u] = True
            total += min_edge[u]
            for v in range(n):
                if not visited[v]:
                    dist = np.linalg.norm(np.array(coords[u]) - np.array(coords[v]))
                    if dist < min_edge[v]:
                        min_edge[v] = dist
        return total
    
    # A function that prints the statistics of the current iteration promising solution
    def iteration_partial_statistics(self, iter_no, best_solution, current_fitness, lower_bound):
        print(f"Iteration {iter_no} Best Promising Solution:", end=' ')
        if PROBLEM == "CVRP":
            routes_str = [f"[0 {' '.join(map(str, route))} 0]" for route in best_solution]
            print(' '.join(routes_str), end=' ')
        elif PROBLEM == "ACKLEY":
            print(f"{best_solution}", end=' ')
        print(f"(Current Fitness: {int(current_fitness)}, Lower Bound: {int(lower_bound)})")
        print("")

#%% General Functions
#A function that generates an assignment of customers to vehicles using k-means clustering
def cvrp_generate_assignment(population, max_iter = 4, try_count = 0):
    customers = [cid for cid in population.coords]
    random.shuffle(customers)
    customer_coords = {cid: np.array(population.coords[cid]) for cid in customers}
    customer_demands = population.demands
    centroids = [customers[0]] #choose initial centroids randomly
    assignment = [[c] for c in centroids] #vehicles initially includes only centroids
    loads = [customer_demands[c] for c in centroids] #current loads for each vehicle initiallized as the centroid demand
    
    for _ in range(max_iter):
        #choose centroid representative as the closest one to the centroid
        new_centroids = []
        for group in assignment:
            coords = np.array([customer_coords[cid] for cid in group])
            center = np.mean(coords, axis=0)
            new_representive = min(group, key=lambda cid: np.linalg.norm(customer_coords[cid] - center))
            new_centroids.append(new_representive)
        centroids = new_centroids
        assignment = [[] for _ in centroids] 
        loads = [0 for _ in centroids]

        #assign customers to the closes centroid
        for cid in customers:
            best_idx = -1
            min_dist = float('inf')
            for idx, centroid in enumerate(centroids):
                if loads[idx] + customer_demands[cid] <= population.truck_capacity:
                    dist = np.linalg.norm(customer_coords[cid] - customer_coords[centroid])
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx

            #the case where the customer is closest to the depot than other centroids and there's available vehicle            
            if len(assignment) < population.trucks_count:
                dist_from_depot = 2 * (np.linalg.norm(customer_coords[cid] - np.array(population.depot)))
                if dist_from_depot < min_dist or best_idx == -1:
                    assignment.append([cid]) 
                    loads.append(customer_demands[cid])
                    centroids.append(cid) 
                    continue
            
            #the case where a customer is assigned to already existing route/vehicle
            if best_idx != -1:
                assignment[best_idx].append(cid) #assigns customer to the best centroid found
                loads[best_idx] += customer_demands[cid]

            #the case where no vehicle have enough capacity for the customer
            else:
                if try_count < 3: #try to generate a new assignment
                    return cvrp_generate_assignment(population, max_iter, try_count + 1)
                else: #if the assignment failed 3 times, try best-fit approach
                    return cvrp_best_fit_generate_assignment(population) 
    return assignment

#A function that generates an assignment of customers to vehicles using a best-fit approach
def cvrp_best_fit_generate_assignment(population):
    customers = [cid for cid in population.coords]
    random.shuffle(customers)
    customer_demands = population.demands
    centroids = [customers[0]]  #choose initial centroids randomly
    assignment = [[c] for c in centroids]  #vehicles initially includes only centroids
    loads = [customer_demands[c] for c in centroids]  #current loads for each vehicle initialized as the centroid demand

    for cid in customers:
        best_idx = -1
        max_load = -float('inf') 

        #looking for the truck with the most load that can still accommodate the customer
        for idx, load in enumerate(loads):
            if load + customer_demands[cid] <= population.truck_capacity: 
                if load > max_load:
                    max_load = load
                    best_idx = idx

        if best_idx != -1: #if a route found 
            assignment[best_idx].append(cid)
            loads[best_idx] += customer_demands[cid]
        elif len(assignment) < population.trucks_count: #if no route found but there's available vehicle
            assignment.append([cid])
            loads.append(customer_demands[cid])
        else: #no valid assignment 
            return None

    return assignment

#A function that finds a random neighbor of an individual
def cvrp_find_neighbor(population, individual, method = None):
    if not method:
        neighborhood_methods = ["2-opt", "relocate", "reposition", "swap", "shuffle"]
        method = random.choice(neighborhood_methods)
    routes = [r[:] for r in individual.routes]
    if method == "2-opt": #reversing a random segment of a random route
        route_idx = random.randint(0, len(routes) - 1)
        route = routes[route_idx]
        if len(route) < 3:
            return individual
        i,j = sorted(random.sample(range(len(route)), 2))
        new_route = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
        new_routes = individual.routes[:route_idx] + [new_route] + individual.routes[route_idx + 1:]
        routes = new_routes
    elif method == "relocate": #taking a random customer from a random route and placing it in another route
        if len(routes) < 2: #not enough routes for reposition
            return individual
        r1 = random.randint(0, len(routes) - 1)
        r2 = random.randint(0, individual.population.trucks_count - 1)
        customer_idx = random.randint(0, len(routes[r1]) - 1)
        customer = routes[r1].pop(customer_idx)
        if r2 >= len(routes):
            routes.append([customer])
            r2 = len(routes) - 1 
        else:
            customer_new_idx = random.randint(0, len(routes[r2]))
            routes[r2].insert(customer_new_idx, customer)
        if sum(population.demands[cid] for cid in routes[r2]) > population.truck_capacity:
            return individual  
        if not routes[r1]:
            routes.pop(r1)
    elif method == "reposition": #re-placing a random customer in a random position inside its route
        route = random.choice(routes)
        if len(route) < 2:
            return individual
        curr_idx, new_idx = random.sample(range(len(route)), 2)
        customer = route.pop(curr_idx)
        if new_idx > curr_idx:
            new_idx -= 1
        route.insert(new_idx, customer)
    elif method == "swap": #swapping two random customers between two random routes
        if len(routes) < 2: #not enough routes for reposition
            return individual
        r1, r2 = random.sample(range(len(routes)), 2)
        cus1_idx = random.randint(0, len(routes[r1]) - 1)
        cus2_idx = random.randint(0, len(routes[r2]) - 1)
        routes[r1][cus1_idx], routes[r2][cus2_idx] = routes[r2][cus2_idx], routes[r1][cus1_idx]
        if sum(population.demands[cid] for cid in routes[r1]) > population.truck_capacity or \
            sum(population.demands[cid] for cid in routes[r2]) > population.truck_capacity:
            return individual  
    elif method == "shuffle": #choose a random route and shuffling its customers
        routes = [r[:] for r in individual.routes]
        route_idx = random.randint(0, len(routes) - 1)
        random.shuffle(routes[route_idx])
    new_individual = CVRPIndividual(routes, population)
    new_individual.evaluate()
    return new_individual

def ackley_find_neighbor(population, individual, method=None):
    if not method:
        neighborhood_methods = ["shift_one", "shift_all", "set_random"]
        method = random.choice(neighborhood_methods)
    new_vector = individual.routes.copy()

    if method == "shift_one": #shifting one dimensions value by adding a small noise 
        i = random.randint(0, population.dim - 1)
        delta = np.random.uniform(-0.1, 0.1) 
        new_vector[i] += delta
        new_vector[i] = np.clip(new_vector[i], population.lower_bound, population.upper_bound)
    
    elif method == "shift_all": #shifting all dimensions values by adding a small uniform noise
        noise = np.random.uniform(-0.05, 0.05, population.dim)
        new_vector += noise
        new_vector = np.clip(new_vector, population.lower_bound, population.upper_bound)
    
    elif method == "set_random": #changing one dimensions value by a new random value
        i = random.randint(0, population.dim - 1)
        new_vector[i] = np.random.uniform(population.lower_bound, population.upper_bound)
    
    #Creating a new individual with the modified vector
    new_individual = AckleyIndividual(new_vector, population)
    new_individual.routes = new_vector
    new_individual.fitness = new_individual.evaluate(new_vector)
    return new_individual

#A function that uses simulated annealing mechanism to decide whether to accept a neighbor solution or not
def simulated_annealing(individual_fitness, neighbor_fitness):
    delta = neighbor_fitness - individual_fitness
    probability = math.exp(-delta / CURRENT_TEMPERATURE)
    if random.random() < probability:
        return True
    return False

# A function that updates the current temperature for simulated annealing as the algorithm progresses
def update_temperature():
    global CURRENT_TEMPERATURE, COOLING_RATE
    CURRENT_TEMPERATURE = COOLING_RATE * CURRENT_TEMPERATURE

#A function that prints the current best solution after each iteration
def iteration_statistics(population, iter_no, best_solution=None, best_fitness_given=None):
    if best_solution is None:
        best_ind = min(population.individuals.routes, key=lambda ind: ind.fitness)
        best_fitness = best_ind.fitness
    else:
        best_ind = best_solution
        best_fitness = best_fitness_given
    population.best_fitness.append(best_fitness)
    population.avg_fitness.append(np.mean([ind.fitness for ind in population.individuals]))
    population.variance.append(np.var([ind.fitness for ind in population.individuals]))

    print(f"Iteration {iter_no} Best:", end=' ')
    if PROBLEM == "CVRP":
        routes_str = [f"[0 {' '.join(map(str, route))} 0]" for route in best_ind]
        print(' '.join(routes_str), end=' ')
    elif PROBLEM == "ACKLEY":
        print(f"{best_ind}", end=' ')
    print(f"({int(best_fitness)})")
    print("")

#A function that prints the best solution found after the algorithm finishes
def best_individual(population, elapsed_time, cpu_time, best_solution = None, best_fitness = None):
    if best_solution is None:
        best_ind = min(population.individuals, key=lambda ind: ind.fitness)
        population.best_fitness.append(best_ind.fitness)
        population.avg_fitness.append(np.mean([ind.fitness for ind in population.individuals]))
        population.variance.append(np.var([ind.fitness for ind in population.individuals]))

        print("Best Solution Found:")
        if PROBLEM == "CVRP":
            for i, route in enumerate(best_ind.routes):
                print(f"Route #{i+1}: 0 {' '.join(map(str, route))} 0")
        elif PROBLEM == "ACKLEY":
            print(f"{best_ind.routes}")
        print(f"Cost: {best_ind.fitness:.2f}")
    else:
        if PROBLEM == "CVRP":
            for i, route in enumerate(best_solution):
                print(f"Route #{i+1}: 0 {' '.join(map(str, route))} 0")
        elif PROBLEM == "ACKLEY":
            print(f"{best_solution}")
        print(f"Cost: {best_fitness:.2f}")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"CPU Time: {cpu_time:.2f} seconds")

# A function that plots the routes of the best solution found
def plot_routes(population, routes):
    depot_x, depot_y = population.depot
    plt.figure(figsize=(8, 8))
    plt.plot(depot_x, depot_y, 'rs', markersize=10, label="Depot") #depot marker
    for cid, (x, y) in population.coords.items(): #customer markers
        plt.plot(x, y, 'bo')
        plt.text(x + 0.5, y + 0.5, str(cid), fontsize=9)
    for route in routes: 
        if not route:
            continue
        path_x = [depot_x]
        path_y = [depot_y]
        for cid in route:
            x, y = population.coords[cid]
            path_x.append(x)
            path_y.append(y)
        path_x.append(depot_x)
        path_y.append(depot_y)
        plt.plot(path_x, path_y, linestyle='-', marker='o')

    plt.title("Vehicle Routes")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

# A function that plots the best and avg fitness over iterations
def plot_iterations_statistics(population):
    plt.figure(figsize=(12, 6))
    plt.plot(population.best_fitness, label='Best Fitness', color='blue')
    plt.plot(population.avg_fitness, label='Average Fitness', color='orange')
    plt.title('Fitness Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_full_comparison(input_file):
    import copy
    global ALGORITHM, ILS_META_HEURISTIC
    algorithms = ["MULTI_STAGE_HEURISTIC", ("ILS", "None"), ("ILS", "SA"), ("ILS", "TS"), ("ILS", "ACO"), "GA", "ALNS", "BB"]
    results = {}
    original_population = CVRPPopulation(input_file) if PROBLEM == "CVRP" else AckleyPopulation()

    for algo in algorithms:
        print(f"\n{algo}..\n")
        if isinstance(algo, tuple) and algo[0] == "ILS":
            ALGORITHM = algo
            ILS_META_HEURISTIC = algo[1]
        else:
            ALGORITHM = algo

        population = copy.deepcopy(original_population)

        if ALGORITHM == "MULTI_STAGE_HEURISTIC":
            solver = MSHeuristicsAlgorithm(population)
        elif ALGORITHM == "ILS":
            solver = ILSAlgorithm(population)
        elif ALGORITHM == "GA":
            solver = GAAlgorithm(population)
        elif ALGORITHM == "ALNS":
            solver = ALNSAlgorithm(population)
        elif ALGORITHM == "BB":
            solver = BranchAndBoundAlgorithm(population)

        start_elapsed = time.time()
        start_cpu = time.process_time()
        solver.solve()
        end_elapsed = time.time()
        end_cpu = time.process_time()

        results[algo] = {
            'best_fitness': population.best_fitness,
            'avg_fitness': population.avg_fitness,
            'variance': population.variance,
            'elapsed_time': end_elapsed - start_elapsed,
            'cpu_time': end_cpu - start_cpu
        }

    #Best fitness plot
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        plt.plot(results[algo]['best_fitness'], label=algo)
    plt.title('Best Fitness Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()

    #Average fitness plot
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        plt.plot(results[algo]['avg_fitness'], label=algo)
    plt.title('Average Fitness Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Average Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()

    #Variance plot
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        plt.plot(results[algo]['variance'], label=algo)
    plt.title('Variance Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True)
    plt.show()

    #Elapsed time plot 
    plt.figure(figsize=(8, 6))
    elapsed_times = [results[algo]['elapsed_time'] for algo in algorithms]
    plt.bar(algorithms, elapsed_times)
    plt.title('Total Elapsed Time per Algorithm')
    plt.ylabel('Elapsed Time (seconds)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

    #CPU time plot
    plt.figure(figsize=(8, 6))
    cpu_times = [results[algo]['cpu_time'] for algo in algorithms]
    plt.bar(algorithms, cpu_times)
    plt.title('Total CPU Time per Algorithm')
    plt.ylabel('CPU Time (seconds)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()


#%% Main Function
def main(input_file):
    population = CVRPPopulation(input_file) if PROBLEM == "CVRP" else AckleyPopulation()
    if ALGORITHM == "MULTI_STAGE_HEURISTIC":
        solver = MSHeuristicsAlgorithm(population)
    elif ALGORITHM == "ILS":
        solver = ILSAlgorithm(population)
    elif ALGORITHM == "GA":
        solver = GAAlgorithm(population)
    elif ALGORITHM == "ALNS":
        solver = ALNSAlgorithm(population)
    elif ALGORITHM == "BB":
        solver = BranchAndBoundAlgorithm(population)
    solver.solve()

if __name__ == "__main__":

    #checking command line and its arguments validity
    if len(sys.argv) not in [3,4]:
        print("Usage: python cvrp_solver.py <time_limit> <problem> <input_file (only for CVRP)>")
        sys.exit(1)

    try:
        TIME_LIMIT = int(sys.argv[1])
    except ValueError:
        print("Time limit must be an integer.")
        sys.exit(1)
    if TIME_LIMIT <= 0:
        print("Time limit must be a positive integer.")
        sys.exit(1)

    PROBLEM = sys.argv[2].upper()
    if PROBLEM not in ["CVRP", "ACKLEY"]:
        print("Problem must be either 'CVRP' or 'ACKLEY'.")
        sys.exit(1)
    if PROBLEM == "ACKLEY":
        if ALGORITHM == "ILS" and ILS_META_HEURISTIC == "ACO":
            print("ACO is not supported for ACKLEY problem.")
            sys.exit(1)
        CURRENT_TEMPERATURE = 1
        COOLING_RATE = 0.95
        CROSSOVER_TYPE = "arithmetic"
        MUTATION_TYPE = "simple_inversion"
        MIGRATION_RATE = 0.1
        MIGRATION_INTERVAL = 30
        ISLANDS_COUNT = 6
        ALNS_PREV_WEIGHT = 0.8

    if PROBLEM == "CVRP":
        if len(sys.argv) != 4:
            print("Input file is required for CVRP problem.")
            sys.exit(1)
        input_file = sys.argv[3]
        if not os.path.isfile(input_file):
            print(f"Input file '{input_file}' does not exist.")
            sys.exit(1)
    else:
        input_file = None
    #main(input_file)
    run_full_comparison(input_file)