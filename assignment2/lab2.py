import math
import os
import time
from scipy.stats import kendalltau
from scipy.optimize import linear_sum_assignment
import numpy as np
import random

#GA Parameters
GA_POPSIZE = 512 #Population size
GA_MAXITER = 3000
GA_TIMELIMIT = None
GA_ELITRATE = 0.05 #Elitism rate
GA_MUTATIONRATE = 0.25 #Mutation rate
NO_IMPROVEMENT_LIMIT = 50  #Local optimum threshold

#Problem (TSP, BIN_PACK)
PROBLEM = "TSP"

#Crossover mode (options: PMX, OX, CX, ER)
CROSSOVER_TYPE = "CX"

#Parent selection method (TOP_HALF_UNIFORM ,RWS, SUS, TOURNAMENT_DET, TOURNAMENT_STOCH)
PARENT_SELECTION_METHOD = "TOURNAMENT_DET"

#Tournament Parameters
TOURNAMENT_K = 15
TOURNAMENT_P = 0.86

#Mutation types (displacement, swap, insertion, simple_inversion, inversion, scramble)
MUTATION_TYPE = "insertion"

# Mutation Control Methods
POP_MUTATION_CONTROL_METHOD = "TRIG-HYPER"  # Options: "NON-LINEAR", "TRIG-HYPER", "NONE"
TRIG_HYPER_TRIGGER = "BEST_FIT"       # Options: "AVG_FIT", "BEST_FIT", "STD_FIT"
HIGH_MUTATION_START_VAL = None        # Internal trigger tracking

IND_MUTATION_CONTROL_METHOD = "AGE"  # Options: "FIT", "AGE", "NONE"

#Fitness Computation Mode
FITNESS_COMPUTATION_MODE = "NONE"  # (NONE, AGE, NOVELITY)
WEIGHT_FACTOR = 0.5
NOVELITY_K = 10

ENABLE_FITNESS_SHARING = False  # Set to True / False to enable/disable fitness sharing
ENABLE_THRESHOLD_SPECIATION = False  # Switch True / False this to use Threshold Speciation

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

#A function to extract the data from a binpack file
def read_binpack_file(filepath):
    weights = []
    with open(filepath, 'r') as file:
        lines = file.readlines()

    #extracting the bin size and optimal number of bins
    try: 
        data = lines[2].strip().split()
        bin_max = int(data[0])
        optimal = int(data[2])
    except ValueError:
        print("Error: Invalid Format")
        return None
    
    #extracting the items weights
    for line in lines[3:]:
        line = line.strip()
        if line.startswith('u'):
            break #stop reading weights
        if line: 
            weights.append(int(line))

    return bin_max, optimal, weights

#A function to extract the TSP problem optimal tour length
def read_opt_tour(filepath):

    #detecting the optimal tour file
    opt_path = filepath.replace(".tsp", ".opt.tour")
    if not os.path.exists(opt_path):
        return None

    #extracting the optimal tour length
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

class BaseIndividual:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = None
        self.raw_fitness = None
        self.rank = None
        self.age = 0

    def mutate(self):
        raise NotImplementedError

    def calculate_fitness(self, *args, **kwargs):
        raise NotImplementedError

    def displacement_mutate(self, seq):
        i, j = sorted(random.sample(range(len(seq)), 2))
        segment = seq[i:j + 1]
        remainder = seq[:i] + seq[j + 1:]
        insert_pos = random.randint(0, len(remainder))
        return remainder[:insert_pos] + segment + remainder[insert_pos:]

    def swap_mutate(self, seq):
        seq = seq[:]
        i, j = random.sample(range(len(seq)), 2)
        seq[i], seq[j] = seq[j], seq[i]
        return seq

    def insertion_mutate(self, seq):
        seq = seq[:]
        i, j = random.sample(range(len(seq)), 2)
        gene = seq.pop(i)
        seq.insert(j, gene)
        return seq

    def simple_inversion_mutate(self, seq):
        seq = seq[:]
        i, j = sorted(random.sample(range(len(seq)), 2))
        seq[i:j + 1] = reversed(seq[i:j + 1])
        return seq

    def inversion_mutate(self, seq):
        i, j = sorted(random.sample(range(len(seq)), 2))
        segment = seq[i:j + 1]
        segment.reverse()
        remainder = seq[:i] + seq[j + 1:]
        insert_pos = random.randint(0, len(remainder))
        return remainder[:insert_pos] + segment + remainder[insert_pos:]

    def scramble_mutate(self, seq):
        seq = seq[:]
        i, j = sorted(random.sample(range(len(seq)), 2))
        segment = seq[i:j + 1]
        random.shuffle(segment)
        return seq[:i] + segment + seq[j + 1:]


class TSPIndividual(BaseIndividual):
    def __init__(self, genome):
        super().__init__(genome)

    def calculate_fitness(self, dist_matrix, optimal_given=None, bin_size=None):
        optimal = optimal_given if optimal_given is not None else 0
        tour1, tour2 = self.genome
        d1 = sum(dist_matrix[tour1[i]][tour1[(i + 1) % len(tour1)]] for i in range(len(tour1)))
        d2 = sum(dist_matrix[tour2[i]][tour2[(i + 1) % len(tour2)]] for i in range(len(tour2)))
        self.raw_fitness = max(d1, d2) - optimal
        self.fitness = self.raw_fitness

    def mutate(self):
        tour1, tour2 = self.genome
        self.original_tour1 = tour1[:]
        self.original_tour2 = tour2[:]
        max_attempts = 1000

        def mutate_pair(t1, t2):
            fn = getattr(self, f"{MUTATION_TYPE}_mutate")
            return fn(t1), fn(t2)

        def are_edge_disjoint(t1, t2):
            def edges(tour):
                return set([(tour[i], tour[(i + 1) % len(tour)]) for i in range(len(tour))] +
                           [(tour[(i + 1) % len(tour)], tour[i]) for i in range(len(tour))])
            return edges(t1).isdisjoint(edges(t2))

        new_tour1, new_tour2 = mutate_pair(tour1[:], tour2[:])
        if are_edge_disjoint(new_tour1, new_tour2):
            self.fitness += 1000
        self.genome = (new_tour1, new_tour2)

#----------------------------------------------------------------------------

class BinPackIndividual(BaseIndividual):
    def __init__(self, genome):
        super().__init__(genome)

    def calculate_fitness(self, dist_matrix=None, optimal_given=None, bin_size=None):
        optimal = optimal_given if optimal_given is not None else 0
        total_bins = 0
        bin_sum = 0
        for element in self.genome:
            weight = element[1]
            bin_sum += weight
            if bin_sum > bin_size:
                total_bins += 1
                bin_sum = weight
        self.raw_fitness = total_bins - optimal
        self.fitness = self.raw_fitness

    def mutate(self):
        fn = getattr(self, f"{MUTATION_TYPE}_mutate")
        self.genome = fn(self.genome)

#----------------------------------------------------------------------------

class BasePopulation:
    def __init__(self, size):
        self.size = size
        self.individuals = []
        self.parents = None  # to store the parents selected by the SUS method
        self.average_fitness = []
        self.best_fitness = []
        self.std_devs = []


    def evaluate_fitness(self):
        raise NotImplementedError

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
        self.parents = self.parents[1:]
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
        return (child)

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

    def apply_novelty_fitness(self, k=10, weight_factor=0.5):
        if PROBLEM == "TSP":
            edge_sets = []
            for ind in self.individuals:
                tour1, tour2 = ind.genome
                edges = set()
                for tour in [tour1, tour2]:
                    for i in range(len(tour)):
                        a = tour[i]
                        b = tour[(i + 1) % len(tour)]
                        edges.add((a, b))
                        edges.add((b, a))
                edge_sets.append(edges)

            for i, ind in enumerate(self.individuals):
                intersections = []
                for j, other_edges in enumerate(edge_sets):
                    if i == j:
                        continue
                    overlap = len(edge_sets[i].intersection(other_edges))
                    intersections.append(overlap)
                intersections.sort(reverse=True)
                novelty_score = -sum(intersections[:k]) / k
                ind.fitness = weight_factor * ind.fitness + (1 - weight_factor) * novelty_score

        elif PROBLEM == "BIN_PACK":
            bin_signatures = []
            for ind in self.individuals:
                bins = split_bins(ind.genome, self.bin_max)
                bin_sums = set(sum(bin) for bin in bins)
                bin_signatures.append(bin_sums)

            for i, ind in enumerate(self.individuals):
                similarities = []
                for j, other_bins in enumerate(bin_signatures):
                    if i == j:
                        continue
                    overlap = len(bin_signatures[i].intersection(other_bins))
                    similarities.append(overlap)
                similarities.sort(reverse=True)
                novelty_score = -sum(similarities[:k]) / k
                ind.fitness = weight_factor * ind.fitness + (1 - weight_factor) * novelty_score



    def apply_fitness_sharing(self, sigma=0.5, alpha=1):
        TOP_K = 40  # you can adjust this number
        if len(self.individuals) <= TOP_K:
            subset = self.individuals
        else:
            subset = sorted(self.individuals, key=lambda ind: ind.fitness)[:TOP_K]

        distance_matrix = np.zeros((len(subset), len(subset)))

        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                if PROBLEM == "TSP":
                    dist = distance_between_genomes_TSP(subset[i], subset[j])
                elif PROBLEM == "BIN_PACK":
                    dist = distance_between_genomes_BinPack(subset[i], subset[j], self.bin_max)
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist

        for i in range(len(subset)):
            sharing_sum = 0
            for j in range(len(subset)):
                dist = distance_matrix[i][j]
                if dist < sigma:
                    sharing_sum += 1 - (dist / sigma) ** alpha
            sharing_sum = max(sharing_sum, 1e-6)
            subset[i].fitness = subset[i].raw_fitness * (1 / sharing_sum)

    def apply_threshold_speciation(self, threshold=0.5):
        species = []
        species_count = 0
        for ind in self.individuals:
            placed = False
            for s in species:
                if distance_between_genomes_TSP(ind, s[0]) < threshold:
                    s.append(ind)
                    placed = True
                    break
            if not placed:
                species.append([ind])
                species_count += 1
        print(f"Number of species (Threshold Speciation): {species_count}")
        # Optional: average fitness per species, max per species etc.


    def generation_stats(self):
        fitnesses = [ind.fitness for ind in self.individuals]
        self.average_fitness.append(sum(fitnesses) / len(fitnesses))
        self.best_fitness.append(min(fitnesses))
        variance = sum((f - self.average_fitness[-1]) ** 2 for f in fitnesses) / len(fitnesses)
        std_dev = math.sqrt(variance)
        self.std_devs.append(std_dev)

    def non_linear_mutation_policy(self):
        global GA_MUTATIONRATE
        min_mutation_rate = 0.05
        p_max = 0.5
        r = 0.01
        generation = len(self.average_fitness)
        exp_term = math.exp(-r * generation)
        f_t = (2 * p_max * exp_term) / (1 + exp_term)
        GA_MUTATIONRATE = max(f_t, min_mutation_rate)
        print(f"Mutation rate (Non-linear): {GA_MUTATIONRATE:.3f}")

    def trigger_hyper_mutation_policy(self):
        global GA_MUTATIONRATE, HIGH_MUTATION_START_VAL
        high = 0.5
        low = 0.25
        k = 10
        if TRIG_HYPER_TRIGGER == "BEST_FIT":
            if HIGH_MUTATION_START_VAL and HIGH_MUTATION_START_VAL - self.best_fitness[-1] > HIGH_MUTATION_START_VAL * 0.01:
                GA_MUTATIONRATE = low
                HIGH_MUTATION_START_VAL = None
            elif not HIGH_MUTATION_START_VAL and len(self.best_fitness) > k and self.best_fitness[-1] == self.best_fitness[-k]:
                GA_MUTATIONRATE = high
                HIGH_MUTATION_START_VAL = self.best_fitness[-1]

    def fit_based_ind_mutation(self):
        avg_fit = sum(ind.fitness for ind in self.individuals) / len(self.individuals)
        for ind in self.individuals:
            rel_fit = ind.fitness / avg_fit if avg_fit > 0 else 1
            mut_rate = max(0.05, GA_MUTATIONRATE * (1 - rel_fit))
            if random.random() < mut_rate:
                ind.mutate()

    def age_based_ind_mutation(self):
        p_min = 0.05
        alpha = 0.05
        for ind in self.individuals:
            mutation_rate = min(GA_MUTATIONRATE, p_min + alpha * ind.age)
            if random.random() < mutation_rate:
                ind.mutate()

class TSPPopulation(BasePopulation):
    def __init__(self, coords, size, optimal_distance=None):
        super().__init__(size)
        self.dist_matrix = compute_distance_matrix(coords)
        self.individuals = self.init_population(coords)
        self.raw_fitness = None  
        self.optimal = optimal_distance

    def init_population(self, items):
      base = list(range(len(items)))
      population = []

      for _ in range(self.size):
          tour1 = random.sample(base, len(base))

          # Build edge set from tour1 (both directions)
          edges1 = set()
          for i in range(len(tour1)):
              a = tour1[i]
              b = tour1[(i + 1) % len(tour1)]
              edges1.add((a, b))
              edges1.add((b, a))

          # Try to generate tour2 disjoint in edges
          while True:
              tour2 = random.sample(base, len(base))
              valid = True
              for i in range(len(tour2)):
                  a = tour2[i]
                  b = tour2[(i + 1) % len(tour2)]
                  if (a, b) in edges1 or (b, a) in edges1:
                      valid = False
                      break
              if valid:
                  break

          population.append(TSPIndividual((tour1, tour2)))

      return population


    def evaluate_fitness(self):
        for ind in self.individuals:
            ind.calculate_fitness(self.dist_matrix, self.optimal)

        if FITNESS_COMPUTATION_MODE == "AGE":
            for ind in self.individuals:
                ind.fitness = WEIGHT_FACTOR * ind.fitness + (1 - WEIGHT_FACTOR) * ind.age




    def build_disjoint_tour(self, tour1, max_attempts=2000):
        n = len(tour1)
        used_edges = set((min(tour1[i], tour1[(i + 1) % n]), max(tour1[i], tour1[(i + 1) % n])) for i in range(n))

        for _ in range(max_attempts):
            remaining = set(tour1)
            current = random.choice(list(remaining))
            tour2 = [current]
            remaining.remove(current)
            edges2 = set()

            valid = True
            while remaining:
                candidates = [city for city in remaining if (min(current, city), max(current, city)) not in used_edges and (min(current, city), max(current, city)) not in edges2]
                if not candidates:
                    valid = False
                    break
                next_city = min(candidates, key=lambda city: self.dist_matrix[current][city])
                edges2.add((min(current, next_city), max(current, next_city)))
                tour2.append(next_city)
                remaining.remove(next_city)
                current = next_city

            # Final edge back to start
            if valid:
                final_edge = (min(tour2[-1], tour2[0]), max(tour2[-1], tour2[0]))
                if final_edge in used_edges or final_edge in edges2:
                    continue
                tour2.append(tour2[0])  # temporarily close cycle to validate
                return tour2[:-1]  # strip redundant end node

        raise RuntimeError("Failed to build fully edge-disjoint tour2.")



    def two_opt(self, tour):
        improved = True
        while improved:
            improved = False
            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour)):
                    if j - i == 1:
                        continue
                    before = self.dist_matrix[tour[i - 1]][tour[i]] + self.dist_matrix[tour[j]][tour[(j + 1) % len(tour)]]
                    after = self.dist_matrix[tour[i - 1]][tour[j]] + self.dist_matrix[tour[i]][tour[(j + 1) % len(tour)]]
                    if before > after:
                        tour[i:j + 1] = reversed(tour[i:j + 1])
                        improved = True
        return tour


    def mate(self):
        self.evaluate_fitness()

        # Apply population-based mutation control
        self.generation_stats()
        if POP_MUTATION_CONTROL_METHOD == "NON-LINEAR":
            self.non_linear_mutation_policy()
        elif POP_MUTATION_CONTROL_METHOD == "TRIG-HYPER":
            self.trigger_hyper_mutation_policy()

        if FITNESS_COMPUTATION_MODE == "NOVELTY":
          self.apply_novelty_fitness(k=NOVELITY_K, weight_factor=WEIGHT_FACTOR)

        # Optional: apply fitness sharing
        if ENABLE_FITNESS_SHARING:
            self.apply_fitness_sharing(sigma=0.5, alpha=1)

        # Optional: apply threshold speciation
        if ENABLE_THRESHOLD_SPECIATION:
            self.apply_threshold_speciation(threshold=0.5)

        # Optional: monitor population diversity
        if self.size >= 2:
            dist = distance_between_genomes_TSP(self.individuals[0], self.individuals[1])
            print(f"KendallTau DTSP Distance: {dist:.4f}")

        # Elitism based on raw or shared fitness
        elite_count = max(1, int(GA_ELITRATE * self.size))
        if ENABLE_FITNESS_SHARING:
            new_pop = sorted(self.individuals, key=lambda x: x.raw_fitness)[:elite_count]
        else:
            new_pop = sorted(self.individuals, key=lambda x: x.fitness)[:elite_count]

        for ind in new_pop:
          ind.age += 1

        while len(new_pop) < self.size:
            _, parent1 = self.select_parents()
            _, parent2 = self.select_parents()

            t1a, t2a = parent1
            t1b, t2b = parent2

            # Perform crossover once for child_tour1
            child_tour1 = self.crossover(t1a, t1b)

            # Always generate a guaranteed disjoint second tour
            child_tour2 = self.build_disjoint_tour(child_tour1)

            child = TSPIndividual((child_tour1, child_tour2))
            child.calculate_fitness(self.dist_matrix, self.optimal)

            if random.random() < GA_MUTATIONRATE:
                child.mutate()
                # Rebuild tour2 again after mutation to preserve edge-disjointness
                t1, _ = child.genome
                t2 = self.build_disjoint_tour(t1)
                child.genome = (t1, t2)
                # Apply 2-opt local optimization to high-quality children (e.g., elite portion)
            if len(new_pop) < elite_count * 2:  # Only apply to top ~10% of next gen
                t1, t2 = child.genome
                t1 = self.two_opt(t1)
                t2 = self.two_opt(t2)
                child.genome = (t1, t2)

            # Final safety: recheck and rebuild if somehow disjointness broke
            t1, t2 = child.genome
            edges1 = set((t1[i], t1[(i + 1) % len(t1)]) for i in range(len(t1)))
            edges2 = set((t2[i], t2[(i + 1) % len(t2)]) for i in range(len(t2)))
            if not edges1.isdisjoint(edges2):
                # Repair it one last time
                t2 = self.build_disjoint_tour(t1)
                child.genome = (t1, t2)

            child.calculate_fitness(self.dist_matrix, self.optimal)
            new_pop.append(child)

        self.individuals = new_pop
        # Apply individual-based mutation control
        if IND_MUTATION_CONTROL_METHOD == "FIT":
            self.fit_based_ind_mutation()
        elif IND_MUTATION_CONTROL_METHOD == "AGE":
            self.age_based_ind_mutation()
        else:
            for ind in self.individuals:
                if random.random() < GA_MUTATIONRATE:
                    ind.mutate()

    def best_individual(self):
        best = min(self.individuals, key=lambda ind: ind.raw_fitness if ENABLE_FITNESS_SHARING else ind.fitness)
        return best.genome, best.raw_fitness if ENABLE_FITNESS_SHARING else best.fitness

#----------------------------------------------------------------------------

class BinPackPopulation(BasePopulation):
    def __init__(self, items, size, optimal, bin_max):
        super().__init__(size)
        self.items = [(index, weight) for index, weight in enumerate(items)]
        self.individuals = self.init_population()
        self.optimal = optimal
        self.bin_max = bin_max

    def init_population(self):
        return [BinPackIndividual(random.sample(self.items, len(self.items))) for _ in range(self.size)]

    def evaluate_fitness(self):
        for ind in self.individuals:
            ind.calculate_fitness(None, self.optimal, self.bin_max)

        if FITNESS_COMPUTATION_MODE == "AGE":
            for ind in self.individuals:
                ind.fitness = WEIGHT_FACTOR * ind.fitness + (1 - WEIGHT_FACTOR) * ind.age


    def mate(self):
        self.evaluate_fitness()

        # Apply population-based mutation control
        self.generation_stats()
        if POP_MUTATION_CONTROL_METHOD == "NON-LINEAR":
            self.non_linear_mutation_policy()
        elif POP_MUTATION_CONTROL_METHOD == "TRIG-HYPER":
            self.trigger_hyper_mutation_policy()

        if FITNESS_COMPUTATION_MODE == "NOVELTY":
          self.apply_novelty_fitness(k=NOVELITY_K, weight_factor=WEIGHT_FACTOR)

        # Optional: apply fitness sharing
        if ENABLE_FITNESS_SHARING:
            self.apply_fitness_sharing(sigma=0.5, alpha=1)

        # Optional: monitor population diversity
        if self.size >= 2:
            dist = distance_between_genomes_BinPack(self.individuals[0], self.individuals[1], self.bin_max)
            print(f"Hungarian BinPack Distance: {dist:.4f}")

        # Elitism based on raw or shared fitness
        elite_count = max(1, int(GA_ELITRATE * self.size))
        if ENABLE_FITNESS_SHARING:
            new_pop = sorted(self.individuals, key=lambda x: x.raw_fitness)[:elite_count]
        else:
            new_pop = sorted(self.individuals, key=lambda x: x.fitness)[:elite_count]

        for ind in new_pop:
          ind.age += 1

        while len(new_pop) < self.size:
            _, parent1 = self.select_parents()
            _, parent2 = self.select_parents()

            child_genome = self.crossover(parent1, parent2)
            child = BinPackIndividual(child_genome)


            if random.random() < GA_MUTATIONRATE:
                child.mutate()

            child.calculate_fitness(None, self.optimal, self.bin_max)
            new_pop.append(child)

        self.individuals = new_pop
        # Apply individual-based mutation control
        if IND_MUTATION_CONTROL_METHOD == "FIT":
            self.fit_based_ind_mutation()
        elif IND_MUTATION_CONTROL_METHOD == "AGE":
            self.age_based_ind_mutation()
        else:
            for ind in self.individuals:
                if random.random() < GA_MUTATIONRATE:
                    ind.mutate()


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
        return bins, best.raw_fitness

#----------------------------------------------------------------------------

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

def distance_between_genomes_TSP(ind1, ind2):
    def get_edges(tour):
        return set((min(tour[i], tour[(i + 1) % len(tour)]),
                    max(tour[i], tour[(i + 1) % len(tour)])) for i in range(len(tour)))

    edges1 = get_edges(ind1.genome[0]).union(get_edges(ind1.genome[1]))
    edges2 = get_edges(ind2.genome[0]).union(get_edges(ind2.genome[1]))

    shared_edges = len(edges1.intersection(edges2))
    total_edges = len(edges1.union(edges2))
    return 1 - shared_edges / total_edges  # normalized distance


def distance_between_genomes_BinPack(ind1, ind2, bin_capacity):
    bins1 = split_bins(ind1.genome, bin_capacity)
    bins2 = split_bins(ind2.genome, bin_capacity)

    size = max(len(bins1), len(bins2))
    while len(bins1) < size:
        bins1.append([])
    while len(bins2) < size:
        bins2.append([])

    cost_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            set1 = set(bins1[i])
            set2 = set(bins2[j])
            cost_matrix[i][j] = len(set1.symmetric_difference(set2))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_diff = cost_matrix[row_ind, col_ind].sum()
    total_items = sum(len(b) for b in bins1)

    return total_diff / total_items if total_items > 0 else 0

def split_bins(flat_genome, bin_capacity):
    bins = []
    current_bin = []
    current_sum = 0
    for _, weight in flat_genome:
        if current_sum + weight > bin_capacity:
            bins.append([w for _, w in current_bin])
            current_bin = []
            current_sum = 0
        current_bin.append((_, weight))
        current_sum += weight
    if current_bin:
        bins.append([w for _, w in current_bin])
    return bins

#A function that linearly scales the fitness values
def linear_scaling(individuals, a,b):
    scaled_fitness = [a * ind.fitness + b for ind in individuals]
    return scaled_fitness

def main(filepath):

    #Initializing the time
    start_time = time.time()

    #Extracting the coordinates from the file
    if PROBLEM == "TSP":
        items = read_tsp_file(filepath)
        optimal = read_opt_tour(filepath)
    elif PROBLEM == "BIN_PACK":
        bin_max, optimal, items = read_binpack_file(filepath)

    #Initializing the population
    population = TSPPopulation(items, GA_POPSIZE, optimal) if PROBLEM == "TSP" else BinPackPopulation(items, GA_POPSIZE, optimal, bin_max)

    #Initiallizing variables to detect local convergence
    best_fit_so_far = float('inf')
    best_gen_so_far = None
    no_improvement_count = 0

    for gen in range(GA_MAXITER):
        population.mate()
        best = population.best_individual()
        
        #Check for convergence
        #global optimum check
        if best and best[1] == 0:
            print("Global optimum convergence.")
            break
        #local optimum check
        if best and best[1] < best_fit_so_far:
            best_gen_so_far = best[0]
            best_fit_so_far = best[1]
            best_adaptive_fit_so_far = best[1]
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

        #Printing the best genome and its fitness achieved in the current generation
        best_genome = best[0] if best else None
        best_fitness = round(best[1], 2) if best else None 
        print(f"Gen {gen:3}. Best = {best_genome} ({best_fitness})")
        print("")

    #printing the best results achieved throughout the generations
    print("\nBest tour:")
    print(best_gen_so_far)
    print(f"Best Fitness Achieved: {best_adaptive_fit_so_far:.2f}")
    print(f"Best Distance Achieved: {best_fit_so_far + optimal:.2f}")

if __name__ == "__main__":

    #Checking validity of the number of arguments
    if len(os.sys.argv) != 4:
        print("Usage: python lab2.py <time_limit> <problem_type> <file_path>")
        exit(1)

    #Checking validity of the time limit (it should be a positive integer)
    try:
        GA_TIMELIMIT = int(os.sys.argv[1])
        if GA_TIMELIMIT <= 0:
            raise ValueError
    except ValueError:
        print("Time limit should be a positive integer.")
        exit(1)

    #Extracting the problem type 
    try:
        PROBLEM = os.sys.argv[2]
        if PROBLEM not in ["TSP", "BIN_PACK"]:
            raise ValueError 
    except ValueError:
        print("Problem should be either TSP or BIN_PACK.")
        exit(1)

    #Extracting the tsp file path and checking its validity
    file_path = os.sys.argv[3]
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        exit(1)

    main(file_path)