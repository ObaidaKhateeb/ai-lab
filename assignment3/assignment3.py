import math
import random
import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt


TIME_LIMIT = 0
POPULATION_SIZE = 512

PROBLEM = "CVRP"
ALGORITHM = "MULTI_STAGE_HEURISTIC"


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
    def __init__(self, routes):
        self.routes = routes
        self.fitness = 0

    # def evaluate(self):
    #     total = 0
    #     first = (0, 0)  #depot coordinates
    #     for route in self.routes:
    #         if not route:
    #             continue
    #         route_length = self.dist_matrix[first][route[0]]
    #         route_length += sum(self.dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
    #         route_length += self.dist_matrix[route[-1]][first]
    #         total += route_length
    #     return total   

class MSHeuristicsAlgorithm:
    def __init__(self, population):
        self.population = population

    def solve(self):
        #initializing timer
        elapsed_start_time = time.time()
        cpu_start_time = time.process_time()

        #generating the population until population size is reached or no time left
        while time.time() - elapsed_start_time < TIME_LIMIT and len(self.population.individuals) < POPULATION_SIZE:
            assignment = self.generate_assignment() #first stage
            if assignment:
                new_individual = Individual(assignment) 
                self.tsp_solve(new_individual) #second stage
                self.population.individuals.append(new_individual)

        elapsed_end_time = time.time()
        cpu_end_time = time.process_time()

        #printing the best result found
        best_individual(self.population, elapsed_end_time - elapsed_start_time, cpu_end_time - cpu_start_time)
        plot_routes(self.population, min(self.population.individuals, key=lambda ind: ind.fitness))

    #A function that generates an assignment of customers to vehicles
    def generate_assignment(self, max_iter=10):
        customers = [cid for cid in self.population.coords]
        customer_coords = {cid: np.array(self.population.coords[cid]) for cid in customers}
        customer_demands = self.population.demands
        centroids = random.sample(customers, self.population.trucks_count)
        assignment = [[c] for c in centroids]
        loads = [customer_demands[c] for c in centroids]

        for _ in range(max_iter):
            new_centroids = []
            for group in assignment:
                coords = np.array([customer_coords[cid] for cid in group])
                center = np.mean(coords, axis=0)
                new_representive = min(group, key=lambda cid: np.linalg.norm(customer_coords[cid] - center))
                new_centroids.append(new_representive)
            
            centroids = new_centroids
            assignment = [[] for _ in centroids]
            loads = [0 for _ in centroids]

            # Assign customers to nearest centroid if within capacity
            for cid in customers:
                best_idx = -1
                min_dist = float('inf')
                for idx, centroid in enumerate(centroids):
                    if loads[idx] + customer_demands[cid] <= self.population.truck_capacity:
                        dist = np.linalg.norm(customer_coords[cid] - customer_coords[centroid])
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = idx

                if best_idx != -1:
                    assignment[best_idx].append(cid)
                    loads[best_idx] += customer_demands[cid]
                else:
                    return None

        return assignment

    def tsp_solve(self, individual):
        optimized = []
        total_cost = 0
        for route in individual.routes:
            ordered, cost = self.nn_route_reorder(route)
            optimized.append(ordered)
            total_cost += cost
        individual.routes = optimized
        individual.fitness = total_cost

    def nn_route_reorder(self, route):
        if not route:
            return [], 0
        current, cost = min( ((node, np.linalg.norm(np.array(self.population.coords[node]) - np.array(self.population.depot))) for node in route), key=lambda x: x[1])
        ordered = [current]
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
            ordered.append(nearest)
            current = nearest
            unvisited.remove(nearest)

        cost += np.linalg.norm(np.array(self.population.coords[current]) - np.array(self.population.depot))
        return ordered, cost

class Algorithm:
    def __init__(self, population: Population, time_limit: int, T_init=100, alpha=0.95):
        self.population = population
        self.T = T_init
        self.alpha = alpha
        self.time_limit = time_limit

    def solve(self):
        start_time = time.time()
        current = Individual(self.population)
        best = current

        while time.time() - start_time < self.time_limit:
            neighbor = current.neighbor()
            delta = neighbor.fitness - current.fitness
            if delta < 0 or random.random() < math.exp(-delta / self.T):
                current = neighbor
                if current.fitness < best.fitness:
                    best = current
            self.T *= self.alpha

        return best

def best_individual(population, elapsed_time, cpu_time):
    best_ind = min(population.individuals, key=lambda ind: ind.fitness)
    for i, route in enumerate(best_ind.routes):
        print(f"Route #{i+1}: {' '.join(map(str, route))}")
    print(f"Cost {best_ind.fitness:.0f}")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"CPU Time: {cpu_time:.2f} seconds")

def plot_routes(population, individual):
    depot_x, depot_y = population.depot
    plt.figure(figsize=(8, 8))
    plt.plot(depot_x, depot_y, 'rs', markersize=10, label="Depot") #depot marker
    for cid, (x, y) in population.coords.items(): # customer markers
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



def main(input_file):
    population = Population(input_file)
    if ALGORITHM == "MULTI_STAGE_HEURISTIC":
        solver = MSHeuristicsAlgorithm(population)
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

