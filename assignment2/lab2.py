import random
import math
import os 
import time

#GA Parameters
GA_POPSIZE = 2048 #Population size
GA_MAXITER = 3000
GA_TIMELIMIT = None
GA_ELITRATE = 0.05 #Elitism rate
GA_MUTATIONRATE = 0.25 #Mutation rate
NO_IMPROVEMENT_LIMIT = 50  #Local optimum threshold

#Problem (TSP, BIN_PACK)
PROBLEM = "BIN_PACK"

#Crossover mode (options: PMX, OX, CX, ER)
CROSSOVER_TYPE = "PMX"

#Parent selection method (TOP_HALF_UNIFORM ,RWS, SUS, TOURNAMENT_DET, TOURNAMENT_STOCH)
PARENT_SELECTION_METHOD = "TOURNAMENT_DET"

#Tournament Parameters
TOURNAMENT_K = 49
TOURNAMENT_P = 0.86

#Mutation types (displacement, swap, insertion, simple_inversion, inversion, scramble)
MUTATION_TYPE = "simple_inversion" 

#A function to extract the data from a csv file 
def read_tsp_file(filepath):
    coords = []
    with open(filepath, 'r') as file:
        relevant_line = False
        for line in file:
            if "NODE_COORD_SECTION" in line: 
                relevant_line = True #start reading coordinates beginning from the next line
                continue
            if "EOF" in line: 
                break #stop reading coordinates
            if relevant_line:
                parts = line.strip().split()
                if len(parts) == 3:
                    _, x, y = parts
                    coords.append((float(x), float(y)))
    return coords

def read_binpack_file(filepath):
    weights = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
    try: 
        bin_max = int(lines[2].strip().split()[0])
    except ValueError:
        print("Error: Invalid Format")
        return None
    for line in lines[3:]:
        line = line.strip()
        if line.startswith('u'):
            break  
        if line: 
            weights.append(int(line))

    return bin_max, weights

def read_opt_tour(filepath, coords):
    opt_path = filepath.replace(".tsp", ".opt.tour")

    if not os.path.exists(opt_path):
        return None

    with open(opt_path, 'r') as file:
        for line in file:
            if line.startswith("COMMENT"):
                start = line.find('(')
                end = line.find(')', start)
                if start != -1 and end != -1:
                    num_str = line[start+1:end]
                    if num_str.isdigit():
                        return int(num_str)
    return None

def read_opt_binpack(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    if len(lines) < 3:
        return None

    third_line = lines[2].strip()
    parts = third_line.split()

    if len(parts) >= 3 and parts[2].isdigit():
        return int(parts[2])

    return None 


class TSPIndividual:
    def __init__(self, genome):
        self.genome = genome  
        self.fitness = None
        self.rank = None

    def calculate_fitness(self, dist_matrix, optimal_given =None, bin_size = None):
        path = self.genome
        optimal = optimal_given if optimal_given is not None else 0
        if PROBLEM == "TSP":
            self.fitness = sum(dist_matrix[path[i]][path[(i+1) % len(path)]]
                        for i in range(len(path))) - optimal       
        elif PROBLEM == "BIN_PACK":
            total_bins = 0
            bin_sum = 0
            for element in self.genome:
                weight = element[1] 
                bin_sum += weight
                if bin_sum > bin_size:
                    total_bins += 1
                    bin_sum = weight
            self.fitness = total_bins - optimal         

    def mutate(self):
        if MUTATION_TYPE == "displacement":
            self.displacement_mutate()
        elif MUTATION_TYPE == "swap":
            self.swap_mutate()
        elif MUTATION_TYPE == "insertion":
            self.insertion_mutate()
        elif MUTATION_TYPE == "simple_inversion":
            self.simple_inversion_mutate()
        elif MUTATION_TYPE == "inversion":
            self.inversion_mutate()
        elif MUTATION_TYPE == "scramble":
            self.scramble_mutate()
        else:
            raise ValueError(f"Wrong mutation type: {MUTATION_TYPE}")

    def displacement_mutate(self):
        i, j = random.sample(range(len(self.genome)), 2)
        if i > j:
            i, j = j, i
        segment = self.genome[i:j + 1]
        reimaner = self.genome[:i] + self.genome[j + 1:]
        insertion_place = random.randint(0, len(reimaner))
        self.genome = reimaner[:insertion_place] + segment + reimaner[insertion_place:]

    def swap_mutate(self):
        i, j = random.sample(range(len(self.genome)), 2)
        self.genome[i], self.genome[j] = self.genome[j], self.genome[i]

    def insertion_mutate(self):
        i, j = random.sample(range(len(self.genome)), 2)
        gene = self.genome.pop(i)
        self.genome.insert(j, gene)    
    
    def simple_inversion_mutate(self):
        i, j = random.sample(range(len(self.genome)), 2)
        if i > j:
            i, j = j, i
        segment = self.genome[i:j + 1]
        segment.reverse()
        self.genome = self.genome[:i] + segment + self.genome[j + 1:]
    
    def inversion_mutate(self): #it includes placement of the segment in a random place 
        i, j = random.sample(range(len(self.genome)), 2)
        if i > j:
            i, j = j, i
        segment = self.genome[i:j + 1]
        segment.reverse()
        reimaner = self.genome[:i] + self.genome[j + 1:]
        insertion_place = random.randint(0, len(reimaner))
        self.genome = reimaner[:insertion_place] + segment + reimaner[insertion_place:]
    
    def scramble_mutate(self):
        i, j = random.sample(range(len(self.genome)), 2)
        if i > j:
            i, j = j, i
        segment = self.genome[i:j + 1]
        random.shuffle(segment)
        self.genome = self.genome[:i] + segment + self.genome[j + 1:]

class TSPPopulation:
    def __init__(self, coords, size, optimal_distance = None):
        self.dist_matrix = compute_distance_matrix(coords)
        self.size = size
        self.individuals = self.init_population(coords)
        self.optimal = optimal_distance
        self.parents = None #to store the parents selected by the SUS method

    def init_population(self, items):
        base = list(range(len(items)))
        return [TSPIndividual(random.sample(base, len(base))) for _ in range(self.size)] #randomly shuffle the items

    def evaluate_fitness(self):
        for ind in self.individuals:
            ind.calculate_fitness(self.dist_matrix, self.optimal)

    def select_parents(self):
        if PARENT_SELECTION_METHOD == "TOP_HALF_UNIFORM":
            return self.top_half_uniform_selection()
        elif PARENT_SELECTION_METHOD == "RWS":
            return self.rws_selection()
        elif PARENT_SELECTION_METHOD == "TOURNAMENT_DET":
            self.fitness_ranking()
            return self.tournament_selection_deter()
        elif PARENT_SELECTION_METHOD == "TOURNAMENT_STOCH":
            self.fitness_ranking()
            return self.tournament_selection_stoch()
        elif PARENT_SELECTION_METHOD == "SUS":
            return self.sus_selection()
        else:
            raise ValueError(f"Wrong parent selection method: {PARENT_SELECTION_METHOD}")

    #A function that selects a parent using Top Half Uniform  
    def top_half_uniform_selection(self):
        rand = random.randint(0, len(self.individuals) // 2 - 1)
        return rand, self.individuals[rand].genome

    #A function that selects a parent using RWS 
    def rws_selection(self):
        
        #Scaling the fitness values using linear scaling    
        max_fitness = max(ind.fitness for ind in self.individuals)
        scaled_fitnesses = linear_scaling(self.individuals, -1, max_fitness)
        
        #Converting fitness values to probabilities
        total_scaled = sum(scaled_fitnesses)
        selection_probs = [fit / total_scaled for fit in scaled_fitnesses]
        
        if total_scaled == 0:
            random_choice = random.randint(0, len(self.individuals) - 1)
            return random_choice, self.individuals[random_choice].genome
        
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
                return i, self.individuals[i].genome
        return len(self.individuals) - 1, self.individuals[-1].genome 

    #A function that selects a list of parents using SUS 
    def sus_selection(self):
        if not self.parents:
            #Scaling the fitness values using linear scaling
            max_fitness = max(ind.fitness for ind in self.individuals)
            scaled_fitness = linear_scaling(self.individuals, -1, max_fitness)

            #Converting fitness values to probabilities
            total_fitness = sum(scaled_fitness)
            if total_fitness == 0:
                random_choice = random.randint(0, len(self.individuals) - 1)
                return random_choice, self.individuals[random_choice].genome
            
            selection_probs = [fit / total_fitness for fit in scaled_fitness]
            
            #Building the "roulete"
            cumulative_probs = []
            cumsum = 0
            for prob in selection_probs:
                cumsum += prob
                cumulative_probs.append(cumsum)

            #Selecting a random starting point, then parents
            target = random.random() 
            self.parents = []
            for j, prob in enumerate(cumulative_probs):
                if target < prob:
                    self.parents.append((j, self.individuals[j].genome))
                    break

        i1, p1 = self.parents[0]
        self.parents = self.parents[1:] #removing the two chosen parents from the list
        return i1, p1


    #A function that selects a parent using Deterministic Tournament Selection 
    def tournament_selection_deter(self):
        tournament = random.sample(range(len(self.individuals)), TOURNAMENT_K) #choosing K random indices 
        tournament.sort(key=lambda ind: self.individuals[ind].rank) #sorting the corresponding individuals
        return tournament[0], self.individuals[tournament[0]].genome

    #A function that selects a parent using Stochastic Tournament Selection 
    def tournament_selection_stoch(self):
        tournament = random.sample(range(len(self.individuals)), TOURNAMENT_K) #choosing K random indices
        tournament.sort(key=lambda ind: self.individuals[ind].rank) #sorting the corresponding individuals
        for i in range(TOURNAMENT_K):
            if random.random() < TOURNAMENT_P: 
                return tournament[i], self.individuals[tournament[i]].genome
        return tournament[-1], self.individuals[tournament[-1]].genome

    def fitness_ranking(self):
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness)
        for rank, ind in enumerate(sorted_inds):
           ind.rank = rank + 1

    def crossover(self, p1, p2):
        if CROSSOVER_TYPE == "PMX":
            return self.pmx_crossover(p1, p2)
        elif CROSSOVER_TYPE == "OX":
            return self.order_crossover(p1, p2)
        elif CROSSOVER_TYPE == "CX":
            return self.cx_crossover(p1, p2)
        elif CROSSOVER_TYPE == "ER":
            return self.er_crossover(p1, p2)
        else:
            raise ValueError(f"Wrong crossover type: {CROSSOVER_TYPE}")

    def order_crossover(self, p1, p2):
        size = len(p1)
        indices_to_copy = random.sample(range(size), size // 2)
        child = [p1[i] if i in indices_to_copy else None for i in range(size)]
        remaining_items = [p1[i] for i in range(size) if i not in indices_to_copy]
        pos = 0
        for i in range(size):
            if child[i] is None:
                while p2[pos] not in remaining_items:
                    pos += 1
                child[i] = p2[pos]
                remaining_items.remove(p2[pos])
                pos += 1
        return TSPIndividual(child)
    
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
        return TSPIndividual(child)

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

        return TSPIndividual(child)
    
    def er_crossover(self, p1, p2):
        size = len(p1)
        child = []
        used = set()

        # Build a neighbor dictionary from both parents
        def get_neighbors(p):
            neighbors = {i: set() for i in p}
            for i in range(len(p)):
                a, b = p[i], p[(i+1) % size]
                neighbors[a].add(b)
                neighbors[b].add(a)
            return neighbors

        n1 = get_neighbors(p1)
        n2 = get_neighbors(p2)

        # Union both neighbor maps
        neighbors = {k: n1[k].union(n2[k]) for k in n1}

        # Start from a random city
        current = random.choice(p1)
        child.append(current)
        used.add(current)

        while len(child) < size:
            common = n1[current].intersection(n2[current]) - used
            if common:
                next_city = min(common)
            else:
                # All neighbors from both parents excluding used ones
                candidates = neighbors[current] - used
                if candidates:
                    next_city = min(candidates)
                else:
                    # Fallback: choose smallest unused city
                    remaining = [c for c in p1 if c not in used]
                    next_city = min(remaining)

            child.append(next_city)
            used.add(next_city)
            current = next_city

        return TSPIndividual(child)

    def mate(self):
        self.evaluate_fitness()
        new_pop = sorted(self.individuals, key=lambda x: x.fitness)[:int(GA_ELITRATE * self.size)]

        while len(new_pop) < self.size:
            _, p1 = self.select_parents()
            _, p2 = self.select_parents()
            child = self.crossover(p1, p2)
            if random.random() < GA_MUTATIONRATE:
                child.mutate()
            child.calculate_fitness(self.dist_matrix, self.optimal)
            new_pop.append(child)
        self.individuals = new_pop

    def best_individual(self):
        routes = [(ind.genome, ind.fitness) for ind in self.individuals]
        routes.sort(key=lambda x: x[1])
        #intialize it as a list of one element which is the set of the first element routes (paths between each two cities)
        edges_to_check_wtih = [set((routes[0][0][i], routes[0][0][(i + 1) % len(routes[0][0])]) for i in range(len(routes[0][0])))]

        #to find the first individual that not disjoint with the rest of the individuals
        for i in range(1, len(routes)):
            route_edges = set((routes[i][0][j], routes[i][0][(j + 1) % len(routes[i][0])]) for j in range(len(routes[i][0])))
            if all(route_edges.isdisjoint(edges) for edges in edges_to_check_wtih):
                edges_to_check_wtih.append(route_edges)
            else:
                return routes[i][0], routes[i][1]    

class BinPackPopulation:
    def __init__(self, items, size, optimal, bin_max):
        self.items = [(index, weight) for index, weight in enumerate(items)]
        self.size = size
        self.individuals = self.init_population()
        self.optimal = optimal
        self.bin_max = bin_max
        self.parents = None #to store the parents selected by the SUS method

    def init_population(self):
        return [TSPIndividual(random.sample(self.items, len(self.items))) for _ in range(self.size)]

    def evaluate_fitness(self):
        for ind in self.individuals:
            ind.calculate_fitness(None, self.optimal, self.bin_max)

    def select_parents(self):
        if PARENT_SELECTION_METHOD == "TOP_HALF_UNIFORM":
            return self.top_half_uniform_selection()
        elif PARENT_SELECTION_METHOD == "RWS":
            return self.rws_selection()
        elif PARENT_SELECTION_METHOD == "TOURNAMENT_DET":
            self.fitness_ranking()
            return self.tournament_selection_deter()
        elif PARENT_SELECTION_METHOD == "TOURNAMENT_STOCH":
            self.fitness_ranking()
            return self.tournament_selection_stoch()
        elif PARENT_SELECTION_METHOD == "SUS":
            return self.sus_selection()
        else:
            raise ValueError(f"Wrong parent selection method: {PARENT_SELECTION_METHOD}")
    
    #A function that selects a parent using Top Half Uniform
    def top_half_uniform_selection(self):
        rand = random.randint(0, len(self.individuals) // 2 - 1)
        return rand, self.individuals[rand].genome
    
    #A function that selects a parent using RWS
    def rws_selection(self):
        #Scaling the fitness values using linear scaling
        max_fitness = max(ind.fitness for ind in self.individuals)
        scaled_fitnesses = linear_scaling(self.individuals, -1, max_fitness)
        
        #Converting fitness values to probabilities
        total_scaled = sum(scaled_fitnesses)
        selection_probs = [fit / total_scaled for fit in scaled_fitnesses]
        
        if total_scaled == 0:
            random_choice = random.randint(0, len(self.individuals) - 1)
            return random_choice, self.individuals[random_choice].genome
        
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
                return i, self.individuals[i].genome
        return len(self.individuals) - 1, self.individuals[-1].genome
    
    #A function that selects a list of parents using SUS
    def sus_selection(self):
        if not self.parents:
            #Scaling the fitness values using linear scaling
            max_fitness = max(ind.fitness for ind in self.individuals)
            scaled_fitness = linear_scaling(self.individuals, -1, max_fitness)

            #Converting fitness values to probabilities
            total_fitness = sum(scaled_fitness)
            if total_fitness == 0:
                random_choice = random.randint(0, len(self.individuals) - 1)
                return random_choice, self.individuals[random_choice].genome
            
            selection_probs = [fit / total_fitness for fit in scaled_fitness]
            
            #Building the "roulete"
            cumulative_probs = []
            cumsum = 0
            for prob in selection_probs:
                cumsum += prob
                cumulative_probs.append(cumsum)

            #Selecting a random starting point, then parents
            target = random.random() 
            self.parents = []
            for j, prob in enumerate(cumulative_probs):
                if target < prob:
                    self.parents.append((j, self.individuals[j].genome))
                    break

        i1, p1 = self.parents[0]
        self.parents = self.parents[1:] #removing the two chosen parents from the list
        return i1, p1
    
    #A function that selects a parent using Deterministic Tournament Selection
    def tournament_selection_deter(self):
        tournament = random.sample(range(len(self.individuals)), TOURNAMENT_K) #choosing K random indices 
        tournament.sort(key=lambda ind: self.individuals[ind].rank) #sorting the corresponding individuals
        return tournament[0], self.individuals[tournament[0]].genome
    
    #A function that selects a parent using Stochastic Tournament Selection
    def tournament_selection_stoch(self):
        tournament = random.sample(range(len(self.individuals)), TOURNAMENT_K) #choosing K random indices
        tournament.sort(key=lambda ind: self.individuals[ind].rank) #sorting the corresponding individuals
        for i in range(TOURNAMENT_K):
            if random.random() < TOURNAMENT_P: 
                return tournament[i], self.individuals[tournament[i]].genome
        return tournament[-1], self.individuals[tournament[-1]].genome
    
    def fitness_ranking(self):
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness)
        for rank, ind in enumerate(sorted_inds):
           ind.rank = rank + 1

    def crossover(self, p1, p2):
        if CROSSOVER_TYPE == "PMX":
            return self.pmx_crossover(p1, p2)
        elif CROSSOVER_TYPE == "OX":
            return self.order_crossover(p1, p2)
        elif CROSSOVER_TYPE == "CX":
            return self.cx_crossover(p1, p2)
        elif CROSSOVER_TYPE == "ER":
            return self.er_crossover(p1, p2)
        else:
            raise ValueError(f"Wrong crossover type: {CROSSOVER_TYPE}")
        
    def order_crossover(self, p1, p2):
        size = len(p1)
        indices_to_copy = random.sample(range(size), size // 2)
        child = [p1[i] if i in indices_to_copy else None for i in range(size)]
        remaining_items = [p1[i] for i in range(size) if i not in indices_to_copy]
        pos = 0
        for i in range(size):
            if child[i] is None:
                while p2[pos] not in remaining_items:
                    pos += 1
                child[i] = p2[pos]
                remaining_items.remove(p2[pos])
                pos += 1
        return TSPIndividual(child)
    
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
        return TSPIndividual(child)
    
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

        return TSPIndividual(child)
    
    def er_crossover(self, p1, p2):
        size = len(p1)
        child = []
        used = set()

        # Build a neighbor dictionary from both parents
        def get_neighbors(p):
            neighbors = {i: set() for i in p}
            for i in range(len(p)):
                a, b = p[i], p[(i+1) % size]
                neighbors[a].add(b)
                neighbors[b].add(a)
            return neighbors

        n1 = get_neighbors(p1)
        n2 = get_neighbors(p2)

        # Union both neighbor maps
        neighbors = {k: n1[k].union(n2[k]) for k in n1}

        # Start from a random city
        current = random.choice(p1)
        child.append(current)
        used.add(current)

        while len(child) < size:
            common = n1[current].intersection(n2[current]) - used
            if common:
                next_city = min(common)
            else:
                # All neighbors from both parents excluding used ones
                candidates = neighbors[current] - used
                if candidates:
                    next_city = min(candidates)
                else:
                    # Fallback: choose smallest unused city
                    remaining = [c for c in p1 if c not in used]
                    next_city = min(remaining)

            child.append(next_city)
            used.add(next_city)
            current = next_city

        return TSPIndividual(child)
    
    def mate(self):
        self.evaluate_fitness()
        new_pop = sorted(self.individuals, key=lambda x: x.fitness)[:int(GA_ELITRATE * self.size)]

        while len(new_pop) < self.size:
            _, p1 = self.select_parents()
            _, p2 = self.select_parents()
            child = self.crossover(p1, p2)
            if random.random() < GA_MUTATIONRATE:
                child.mutate()
            child.calculate_fitness(None, self.optimal, self.bin_max)
            new_pop.append(child)
        self.individuals = new_pop

    def best_individual(self):
        best = min(self.individuals, key=lambda ind: ind.fitness)
        bins = []
        current_bin = []
        current_weight = 0
        for item in best.genome:
            item_weight = item[1]
            if current_weight + item_weight <= self.bin_max:
                current_bin.append(item_weight)
                current_weight += item_weight
            else:
                bins.append(current_bin)
                current_bin = [item_weight]
                current_weight = item_weight
        if current_bin:
            bins.append(current_bin)
        return bins, best.fitness
        


def euclidean_distance(a, b):
    return round(math.hypot(a[0] - b[0], a[1] - b[1]), 2)

def compute_distance_matrix(coords):
    n = len(coords)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = euclidean_distance(coords[i], coords[j])
    return matrix





#A function that linearly scales the fitness values 
def linear_scaling(individuals, a,b):
    scaled_fitness = [a * ind.fitness + b for ind in individuals]
    return scaled_fitness
# -------- Main Runner --------

def main(filepath):
    #Initializing the time
    start_time = time.time()

    #Extracting the coordinates from the file
    if PROBLEM == "TSP":
        items = read_tsp_file(filepath)
    elif PROBLEM == "BIN_PACK":
        bin_max, items = read_binpack_file(filepath)

    optimal = read_opt_tour(filepath, items) if PROBLEM == "TSP" else read_opt_binpack(filepath)

    #Initializing the population
    population = TSPPopulation(items, GA_POPSIZE, optimal) if PROBLEM == "TSP" else BinPackPopulation(items, GA_POPSIZE, optimal, bin_max)

    #Initiallizing variables to detect local convergence
    best_fit_so_far = float('inf')
    no_improvement_count = 0

    for gen in range(GA_MAXITER):
        population.mate()
        best = population.best_individual()
        
        #Check for convergence
        #global optimum check
        if best[1] == 0:
            print("Global optimum convergence.")
            break
        #local optimum check
        if best[1] < best_fit_so_far:
            best_fit_so_far = best[1]
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if no_improvement_count > NO_IMPROVEMENT_LIMIT:
            print("No improvement => Local optimum convergence.")
            break
        #time exceeded check
        if time.time() - start_time > GA_TIMELIMIT:
            print("Time limit exceeded.")
            break


        print(f"Gen {gen:3}. Best = {best[0]} ({best[1]:.2f})")

    best = population.best_individual()
    print("\nBest tour:")
    print(best[0])
    print(f"Best Fitness Achieved: {best[1]:.2f}")
    print(f"Best Distance Achieved: {best[1] + optimal:.2f}")

if __name__ == "__main__":

    #Check validity of the number of arguments
    if len(os.sys.argv) != 3:
        print("Usage: python lab2.py <time_limit> <tsp_file>")
        exit(1)

    #Check validity of the time limit (it should be a positive integer)
    try:
        GA_TIMELIMIT = int(os.sys.argv[1])
        if GA_TIMELIMIT <= 0:
            raise ValueError
    except ValueError:
        print("Time limit should be a positive integer.")
        exit(1)

    #Extracting the tsp file path and checking its validity
    file_path = os.sys.argv[2]
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        exit(1)

    main(file_path)
