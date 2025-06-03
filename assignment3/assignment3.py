#%% Imports and global variables
import math
import random
import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt


TIME_LIMIT = 0
POPULATION_SIZE = 512
LOCAL_OPTIMUM_THRESHOLD = 50

PROBLEM = "CVRP"
ALGORITHM = "ILS" # ("MULTI_STAGE_HEURISTIC", "ILS")

# ILS Parameters
ILS_META_HEURISTIC = "ACO" # ("SA", "TS", "ACO")
CURRENT_TEMPERATURE = 500
COOLING_RATE = 0.9
ACO_EVAPORATION_RATE = 0.5
ACO_Q = 100

#%% Population and Individual classes
class Population:
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

    #A function that extracts the problem inputs
    def _parse_file(self, path):
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            section = None
            depot_idx = None
            customer_idx = 1
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith("CAPACITY"): #the truck capacity
                    self.truck_capacity = int(line.split(":")[1])
                elif line.startswith("COMMENT"): #number of trucks
                    if "No of trucks" in line:
                        trucks_count = line.split("No of trucks:")[-1].split(",")[0]
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

class Individual:
    def __init__(self, routes, population):
        self.routes = routes
        self.fitness = 0
        self.population = population

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
            assignment = generate_assignment(self.population) #first stage
            if assignment:
                new_individual = Individual(assignment, self.population)
                self.tsp_solve(new_individual) #second stage
                self.population.individuals.append(new_individual)

        elapsed_end_time = time.time()
        cpu_end_time = time.process_time()

        #printing the best result found
        best_individual(self.population, elapsed_end_time - elapsed_start_time, cpu_end_time - cpu_start_time)
        plot_routes(self.population, min(self.population.individuals, key=lambda ind: ind.fitness))

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

#%% ILS Algorithm
class ILSAlgorithm:
    def __init__(self, population):
        self.population = population
        self.tabu_list = [] #for Tabu Search use 
        self.tabu_set = set() #for Tabu Search use 
        self.pheromone = {} #for ACO use

    def solve(self):
        global CURRENT_TEMPERATURE, COOLING_RATE
        elapsed_start_time = time.time()
        cpu_start_time = time.process_time()

        while time.time() - elapsed_start_time < TIME_LIMIT and len(self.population.individuals) < POPULATION_SIZE:
            assignment = generate_assignment(self.population)
            if assignment:
                new_ind = Individual(assignment, self.population)
                new_ind.evaluate()
                self.population.individuals.append(new_ind)

        iter_count = 0
        no_improvement_count = 0
        best_fitness_found = float('inf')
        while time.time() - elapsed_start_time < TIME_LIMIT and no_improvement_count < LOCAL_OPTIMUM_THRESHOLD:
            CURRENT_TEMPERATURE = COOLING_RATE * CURRENT_TEMPERATURE
            for i, individual in enumerate(self.population.individuals):
                # local search
                neighbor = self.find_neighbor(individual)
                if neighbor.fitness < individual.fitness:
                    individual.routes = neighbor.routes
                    individual.fitness = neighbor.fitness
                else:
                    if ILS_META_HEURISTIC == "SA":
                        toReplace = self.simulated_annealing(individual.fitness, neighbor.fitness)
                        if toReplace:
                            individual.routes = neighbor.routes
                            individual.fitness = neighbor.fitness
                    elif ILS_META_HEURISTIC == "TS":
                        if not self.tabu_list:
                            self.tabu_hash_initiallize()
                        neighbors = [self.find_neighbor(individual, method) for method in ["2-opt", "relocate", "reposition", "swap", "shuffle"]]
                        neighbors.sort(key=lambda ind: ind.fitness)
                        for neighbor in neighbors:
                            neighbor_hash = str(sorted([tuple(route) for route in neighbor.routes]))
                            if neighbor_hash not in self.tabu_set:
                                individual.routes = neighbor.routes
                                individual.fitness = neighbor.fitness
                                self.tabu_list.append(neighbor_hash)
                                self.tabu_set.add(neighbor_hash)
                                if len(self.tabu_list) > POPULATION_SIZE * 10:
                                    oldest = self.tabu_list.pop(0)
                                    self.tabu_set.remove(oldest)
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
            best_fitness_iter = min(self.population.individuals, key=lambda ind: ind.fitness).fitness
            if best_fitness_iter < best_fitness_found:
                best_fitness_found = best_fitness_iter
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            iteration_statistics(self.population, iter_count)
            iter_count += 1
            
        elapsed_end_time = time.time()
        cpu_end_time = time.process_time()
        best_individual(self.population, elapsed_end_time - elapsed_start_time, cpu_end_time - cpu_start_time)
        plot_routes(self.population, min(self.population.individuals, key=lambda ind: ind.fitness))

    #A function that finds a random neighbor of an individual
    def find_neighbor(self, individual, method = None):
        neighborhood_methods = ["2-opt", "relocate", "reposition", "swap", "shuffle"]
        method = random.choice(neighborhood_methods)
        if method == "2-opt": #reversing a random segment of a random route
            route_idx = random.randint(0, len(individual.routes) - 1)
            route = individual.routes[route_idx]
            if len(route) < 3:
                return individual
            i,j = sorted(random.sample(range(len(route)), 2))
            new_route = route[:i] + list(reversed(route[i:j+1])) + route[j+1:]
            new_routes = individual.routes[:route_idx] + [new_route] + individual.routes[route_idx + 1:]
            new_individual = Individual(new_routes, self.population)
            new_individual.evaluate()
            return new_individual
        elif method == "relocate": #taking a random customer from a random route and placing it in another route
            routes = [r[:] for r in individual.routes] 
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
            if sum(self.population.demands[cid] for cid in routes[r2]) > self.population.truck_capacity:
                return individual  
            if not routes[r1]:
                routes.pop(r1)
            new_individual = Individual(routes, self.population)
            new_individual.evaluate()
            return new_individual
        elif method == "reposition": #re-placing a random customer in a random position inside its route
            routes = [r[:] for r in individual.routes]
            route = random.choice(routes)
            if len(route) < 2:
                return individual
            curr_idx, new_idx = random.sample(range(len(route)), 2)
            customer = route.pop(curr_idx)
            if new_idx > curr_idx:
                new_idx -= 1
            route.insert(new_idx, customer)
            new_individual = Individual(routes, self.population)
            new_individual.evaluate()
            return new_individual
        elif method == "swap":
            routes = [r[:] for r in individual.routes] 
            if len(routes) < 2: #not enough routes for reposition
                return individual
            r1, r2 = random.sample(range(len(routes)), 2)
            cus1_idx = random.randint(0, len(routes[r1]) - 1)
            cus2_idx = random.randint(0, len(routes[r2]) - 1)
            routes[r1][cus1_idx], routes[r2][cus2_idx] = routes[r2][cus2_idx], routes[r1][cus1_idx]
            if sum(self.population.demands[cid] for cid in routes[r1]) > self.population.truck_capacity or \
               sum(self.population.demands[cid] for cid in routes[r2]) > self.population.truck_capacity:
                return individual  
            new_individual = Individual(routes, self.population)
            new_individual.evaluate()
            return new_individual
        elif method == "shuffle": #choose a random route and shuffling its customers
            routes = [r[:] for r in individual.routes]
            route_idx = random.randint(0, len(routes) - 1)
            random.shuffle(routes[route_idx])
            new_individual = Individual(routes, self.population)
            new_individual.evaluate()
            return new_individual

    #A function that uses simulated annealing mechanism to decide whether to accept a neighbor solution or not
    def simulated_annealing(self, individual_fitness, neighbor_fitness):
        delta = neighbor_fitness - individual_fitness
        probability = math.exp(-delta / CURRENT_TEMPERATURE)
        if random.random() < probability:
            return True
        return False

    def tabu_hash_initialize(self):
        for ind in self.population.individuals:
            ind_hash = str(sorted([tuple(route) for route in ind.routes]))
            self.tabu_list.append(ind_hash)
            self.tabu_set.add(ind_hash)
    
    def pheromone_initialize(self):
        for i in self.population.coords:
            for j in self.population.coords:
                if i != j:
                    self.pheromone[(i, j)] = 1.0 
    
    def aco_solution_construct(self):
        routes = []
        unvisited = set(self.population.coords.keys())
        current = random.choice(list(unvisited)) #start from a random customer
        load = self.population.demands[current]
        curr_route = [current]
        unvisited.remove(current)
        while unvisited:
            next_customer = None
            max_pheromone = -1
            for customer in unvisited:
                if load + self.population.demands[customer] <= self.population.truck_capacity:
                    pheromone = self.pheromone.get((current, customer), 0)
                    if pheromone > max_pheromone:
                        max_pheromone = pheromone
                        next_customer = customer
            if next_customer is None:
                routes.append(curr_route)
                if unvisited:
                    current = random.choice(list(unvisited))
                    load = self.population.demands[current]
                    curr_route = [current]
                    unvisited.remove(current)
                else:
                    break
            else:
                curr_route.append(next_customer)
                load += self.population.demands[next_customer]
                unvisited.remove(next_customer)
                current = next_customer

        #add the last route if it has customers
        if curr_route:
            routes.append(curr_route)
        new_individual = Individual(routes, self.population)
        new_individual.evaluate()
        return new_individual
    
    def pheromone_update(self):
        for key in self.pheromone.keys():
            self.pheromone[key] *= ACO_EVAPORATION_RATE 
        
        for individual in self.population.individuals:
            for route in individual.routes:
                if len(route) <= 1:
                    continue
                for i in range(len(route) - 1):
                    key = (route[i], route[i + 1])
                    self.pheromone[key] += ACO_Q / individual.fitness

#%% General Functions
#A function that generates an assignment of customers to vehicles using k-means clustering
def generate_assignment(population, max_iter = 4):
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
                dist_from_depot = np.linalg.norm(customer_coords[cid] - np.array(population.depot))
                if dist_from_depot < min_dist:
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
                return None #No valid assignment found, try failed
    return assignment

#A function that prints the current best solution after each iteration
def iteration_statistics(population, iter_no):
    best_ind = min(population.individuals, key=lambda ind: ind.fitness)
    print(f"Iteration {iter_no} Best:", end=' ')
    routes_str = [f"[0 {' '.join(map(str, route))} 0]" for route in best_ind.routes]
    print(' '.join(routes_str), end=' ')
    print(f"({int(best_ind.fitness)})")
    print("")

#A function that prints the best solution found after the algorithm finishes
def best_individual(population, elapsed_time, cpu_time):
    best_ind = min(population.individuals, key=lambda ind: ind.fitness)
    for i, route in enumerate(best_ind.routes):
        print(f"Route #{i+1}: 0 {' '.join(map(str, route))} 0")
    print(f"Cost: {best_ind.fitness:.2f}")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"CPU Time: {cpu_time:.2f} seconds")

# A function that plots the routes of the best solution found
def plot_routes(population, individual):
    depot_x, depot_y = population.depot
    plt.figure(figsize=(8, 8))
    plt.plot(depot_x, depot_y, 'rs', markersize=10, label="Depot") #depot marker
    for cid, (x, y) in population.coords.items(): #customer markers
        plt.plot(x, y, 'bo')
        plt.text(x + 0.5, y + 0.5, str(cid), fontsize=9)
    for route in individual.routes: #routes
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

#%% Main Function
def main(input_file):
    population = Population(input_file)
    if ALGORITHM == "MULTI_STAGE_HEURISTIC":
        solver = MSHeuristicsAlgorithm(population)
    elif ALGORITHM == "ILS":
        solver = ILSAlgorithm(population)
    solver.solve()

if __name__ == "__main__":

    #checking command line and its arguments validity
    if len(sys.argv) != 3:
        print("Usage: python cvrp_solver.py <time_limit_in_seconds> <input_file>")
        sys.exit(1)
    try:
        TIME_LIMIT = int(sys.argv[1])
    except ValueError:
        print("Time limit must be an integer.")
        sys.exit(1)
    if TIME_LIMIT <= 0:
        print("Time limit must be a positive integer.")
        sys.exit(1)
    input_file = sys.argv[2]
    if not os.path.isfile(input_file):
        print(f"Input file '{input_file}' does not exist.")
        sys.exit(1)

    main(input_file)

