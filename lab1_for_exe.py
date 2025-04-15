import random
import time
import math
import matplotlib.pyplot as plt
import sys
import os
import json

#GA Parameters
GA_POPSIZE       = 2048    #Population size
GA_MAXITER       = 120     #Maximum number of iterations
GA_ELITRATE      = 0.05    #Elitism rate
GA_MUTATIONRATE  = 0.25    #Mutation rate
GA_TARGET        = "Hello world!" #Target string
GA_CHARSIZE      = 90      #Range of characters (roughly ' ' to '~')
NO_IMPROVEMENT_LIMIT = 50  #Local optimum threshold

#Problem (TARGET_STRING, MATRIX_TRANSFORM, BIN_PACKING)
PROBLEM = "TARGET_STRING" 

#Crossover mode (options: SINGLE, TWO, UNIFORM, NO_CROSSOVER)
CROSSOVER_TYPE = "UNIFORM"

#Fitness mode (options: DISTANCE, LCS, BINS_DIFF)
FITNESS_MODE = "LCS"

#Parent selection method (TOP_HALF_UNIFORM ,RWS, SUS, TOURNAMENT_DET, TOURNAMENT_STOCH, SHUFFLE)
PARENT_SELECTION_METHOD = "TOP_HALF_UNIFORM"

#Tournament Parameters
TOURNAMENT_K = 49
TOURNAMENT_P = 0.86

#Survivor selection method (STANDARD, AGING)
SURVIVOR_SELECTION_METHOD = "STANDARD"
AGE_LIMIT = 14

#Individual class representing a single solution
class Individual:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = None
        self.age = 0
        self.rank = None

    #A fuction to calculate the fitness of the individual
    def calculate_fitness(self, target):
        if FITNESS_MODE == "DISTANCE": #The distance fitness is the sum of absolute differences between the genome and target
            self.fitness = sum(abs(ord(g) - ord(t)) for g, t in zip(self.genome, target))
        elif FITNESS_MODE == "LCS":
            self.fitness = self.fitness_by_lcs(self.genome, target)
        elif FITNESS_MODE == "BINS_DIFF":
            self.fitness = self.fitness_by_bins(self.genome, target)
        else:
            raise ValueError("Invalid fitness mode")

    #computeing the fitness of the individual using the "LCS" method
    #The function works by finding the length of the longest common subsequence (LCS) between the individual and the target, and adding a bonus for each LCS character that is in the right position
    def fitness_by_lcs(self, a, b):
        m = len(a)
        n = len(b)
        L = [[0] * (n + 1) for _ in range(m + 1)]

        #Filling the LCS table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])
        lcs_length = L[m][n]

        #Backtracking to find the number of LCS characters that are in the right position
        correct_chars_count = 0
        bonus = 4
        while m > 0 and n > 0:
            if a[m - 1] == b[n - 1]:
                m -= 1
                n -= 1
                if m == n: #ckecks if the LCS genome character is in the right position
                    correct_chars_count += 1
            elif L[m - 1][n] > L[m][n - 1]:
                m -= 1
            else:
                n -= 1
        max_possible = (bonus+1) * len(b)
        return max_possible - (lcs_length + bonus * correct_chars_count)
    
    #A functtion to compute the fitness of the individual in Bin Packing problem
    def fitness_by_bins(self, genome, target):
        return len(genome) - target #Difference between the number of bins used and the number of the target bins
    
    #A function to create a genome using the best fit algorithm 
    def best_fit(self, initial_genome, bin_capacity):
        random.shuffle(initial_genome)
        genome = []
        for item in [item for bin in initial_genome for item in bin]: 
            best_bin_index = -1
            min_space_left = bin_capacity+ 1
            for i, bin in enumerate(genome):
                space_left = bin_capacity - sum(bin)
                if space_left >= item and space_left - item < min_space_left:
                    best_bin_index = i
                    min_space_left = space_left - item
            if best_bin_index == -1: #no suitable bin found
                genome.append([item]) #creating a new bin
            else: #adds the item to the best bin found
                genome[best_bin_index].append(item) 

        return genome
    
    #A function to create a genome using the first fit algorithm
    def first_fit(self, initial_genome, bin_capacity):
        random.shuffle(initial_genome)
        genome = []
        for item in [item for bin in initial_genome for item in bin]:
            placed = False
            for bin in genome:
                if sum(bin) + item <= bin_capacity: #A suitable bin found
                    bin.append(item)
                    placed = True
                    break
            if not placed: #no suitable bin found
                genome.append([item]) #creating a new bin
        return genome

    #A function to rearrange the genome by keeping the top 50% bins and repacking the rest
    def genome_rearrange(self, genome, bin_capacity):
        # Flatten all items
        all_bins = [list(bin) for bin in genome] 
        bin_loads = [(sum(b), idx) for idx, b in enumerate(all_bins)]
        bin_loads.sort(reverse=True) #sorting the bins by their loads

        #Determining the bins to keep and items to repack
        keep_count = len(bin_loads) // 2
        keep_indices = set(idx for _, idx in bin_loads[:keep_count])
        kept_bins = [all_bins[i] for i in range(len(all_bins)) if i in keep_indices]
        repack_items = [item for i in range(len(all_bins)) if i not in keep_indices for item in all_bins[i]]

        random.shuffle(repack_items) #shuffling the items to be repacked

        #Re-Packing the items: 
        for item in repack_items:
            best_bin_index = -1
            min_space_left = bin_capacity + 1

            for i, bin in enumerate(kept_bins):
                space_left = bin_capacity - sum(bin)
                if space_left >= item and space_left - item < min_space_left:
                    best_bin_index = i
                    min_space_left = space_left - item

            if best_bin_index == -1: #no suitable bin found
                kept_bins.append([item]) #creating a new bin
            else: #adds the item to the best bin found
                kept_bins[best_bin_index].append(item)

        return kept_bins


#Population class representing a collection of individuals
class Population:
    def __init__(self, size, target, initial_genome, bin_capacity = None):
        self.size = size
        self.target = target if PROBLEM == "TARGET_STRING" or PROBLEM == "BIN_PACKING" else self.matrix_to_string(target)
        self.bin_capacity = bin_capacity
        self.individuals = self.init_population(initial_genome) if PROBLEM == "TARGET_STRING" or PROBLEM == "MATRIX_TRANSFORM" else self.bin_packing_init_population(initial_genome)
        self.best_fitness_list = []
        self.avg_fitness_list = []
        self.worst_fitness_list = []
        self.fitness_history = []
        self.generation_fitness_var = []
        self.top_avg_select_ratio = []
        self.avg_dist_list = [] 
        self.distinct_alleles_list = [] 
        self.shannon_entropy_list = []


    #A function to initialize the population with random individuals
    def init_population(self, initial_genome):
        tsize = len(self.target)
        population = []
        if initial_genome is not None:
            genome = initial_genome if isinstance(initial_genome, str) else self.matrix_to_string(initial_genome)
            for _ in range(self.size):
                individual = Individual(genome)
                population.append(individual)
        else:
            for _ in range(self.size):
                if PROBLEM == "TARGET_STRING":
                    genome = ''.join(chr(random.randint(32, 32 + GA_CHARSIZE - 1)) for _ in range(tsize))
                elif PROBLEM == "MATRIX_TRANSFORM":
                    genome = ''.join(str(random.randint(0, 9)) for _ in range(tsize))
                individual = Individual(genome)
                population.append(individual)
        return population

    #A function to initialize the population for the Bin Packing problem u
    def bin_packing_init_population(self, initial_genome):
        population = []
        for _ in range(self.size):
            individual = Individual(initial_genome)
            genome = individual.best_fit(initial_genome, self.bin_capacity) #creating a genome using the best fit algorithm
            individual.genome = genome
            population.append(individual)
        return population

    #A function to update the fitness of all individuals in the population
    def update_fitness(self):
        for ind in self.individuals:
            ind.calculate_fitness(self.target)

    #A function to sort the population by their fitness
    def sort_by_fitness(self):
        self.individuals.sort(key=lambda ind: ind.fitness)
    
    #A function to select elitism individuals based on their fitness
    def elitism(self, buffer, esize):
        for i in range(esize):
            buffer[i] = self.individuals[i]
        return buffer
    
    #A function that computes and prints the generation best and worst individuals, fitness range, average fitness, and standard deviation (task 1)
    def generation_stats_update(self, generation):
        best_fit  = self.individuals[0].fitness
        worst_fit = self.individuals[-1].fitness
        fitness_range = worst_fit - best_fit
        sum_fit   = sum(ind.fitness for ind in self.individuals)
        avg_fit   = sum_fit / self.size
        variance  = sum((ind.fitness - avg_fit)**2 for ind in self.individuals) / self.size
        std_dev   = math.sqrt(variance) if variance>0 else 0     

        if PROBLEM == "BIN_PACKING":
            best_fit += GA_TARGET
            worst_fit += GA_TARGET
            avg_fit += GA_TARGET

        best_ind = self.individuals[0].genome if PROBLEM == "TARGET_STRING" or PROBLEM == "BIN_PACKING" else self.string_to_matrix(self.individuals[0].genome)
        worst_ind = self.individuals[-1].genome if PROBLEM == "TARGET_STRING" or PROBLEM == "BIN_PACKING" else self.string_to_matrix(self.individuals[-1].genome)

        print(f"Gen{generation}." 
                f" Best: {best_ind} ({best_fit})", 
                f" Worst: {worst_ind} ({worst_fit}) ",
                f" Fitness Range: {fitness_range} ",
                f" Avg: {avg_fit:.2f} ",
                f" Std: {std_dev:.2f} ")

        #Storing the best, average, and worst fitness for line plots use (task 3a)
        self.best_fitness_list.append(best_fit)
        self.avg_fitness_list.append(avg_fit)
        self.worst_fitness_list.append(worst_fit)

        #Storing the distribution of fitness for boxplots use (task 3b)
        gen_fitness_list = [ind.fitness for ind in self.individuals]
        self.fitness_history.append(gen_fitness_list)

    #A function that plots the best, average, and worst fitness over generations (task 3a)
    def fitness_plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_list,  label="Best Fitness")
        plt.plot(self.avg_fitness_list,   label="Average Fitness")
        plt.plot(self.worst_fitness_list, label="Worst Fitness")
        plt.title(f"GA Fitness Over Generations\nCrossover={CROSSOVER_TYPE}, Fitness={FITNESS_MODE}")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.show()

    #A function that plots the boxplots for each generation's distribution (task 3b)
    def fitness_boxplot(self):
        for g,data in enumerate(self.fitness_history):
            plt.figure(figsize=(4,5))
            plt.boxplot(data, showfliers=True)
            plt.title(f"Box Plot of Fitness - Gen {g} ({CROSSOVER_TYPE}, {FITNESS_MODE})")
            plt.ylabel("Fitness")
            plt.show()

    #A function to plot the selection pressure statistics over generations (task 8)
    def plot_selection_pressure(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.generation_fitness_var, label="Fitness Variance") #Factor 1: Fitness Variance
        plt.plot(self.top_avg_select_ratio, label="Top-Average Ratio") #Factor 2: Top Average Selection Probability Ratio
        plt.title("Selection Pressure Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Exploitation Factor")
        plt.legend()
        plt.grid(True)
        plt.show()

    #A function to plot the genetic diversity metrics over generations (task 9)
    def plot_diversity(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.avg_dist_list, label="Avg Pairwise Distance") #Factor 1: Average Distance
        plt.plot(self.distinct_alleles_list, label="Distinct Alleles") #Factor 2: Distinct Alleles Count
        plt.plot(self.shannon_entropy_list, label="Shannon Entropy") #Factor 3: Shannon Entropy
        plt.title("Genetic Diversity Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Diversity Factor")
        plt.legend()
        plt.grid(True)
        plt.show()

    #A function that transforms a 2D matrix into a string (Section 12)
    def matrix_to_string(self, matrix):
        string = []
        for row in matrix:
            string.extend(row)
        return "".join(str(num) for num in string)

    #A function that transforms a string into a 2D matrix (Section 12)
    def string_to_matrix(self, string):
        size = int(math.sqrt(len(string)))
        matrix = []
        for i in range(size):
            row = [int(c) for c in string[i * size:(i + 1) * size]]
            matrix.append(row)
        return matrix


#A function to mutate an individual by changing a random character in its genome
def mutate(individual):
    tsize = len(individual.genome)
    ipos = random.randint(0, tsize - 1) #choosing a random character 
    if PROBLEM == "TARGET_STRING":
        old_char_val = ord(individual.genome[ipos]) #extracting the ASCII value of the character
        delta = random.randint(32, 32 + GA_CHARSIZE - 1) #choosing a random value to add to the ASCII value
        new_char_val = (old_char_val + delta) % 126 #modulo 126 to keep it within the ASCII range
        if new_char_val < 32: 
            new_char_val += 32
    elif PROBLEM == "MATRIX_TRANSFORM":
        new_char_val = random.randint(48, 57) #choosing a random value between 0 and 9

    #Updating the genome with the new character instead of the old one
    genome_list = list(individual.genome)
    genome_list[ipos] = chr(new_char_val)
    individual.genome = "".join(genome_list)

#Single-point crossover function to combine two parents into a child
#The function works by selecting a random crossover point, in which the first part of the child is taken from the first parent and the second part from the second parent
def single_point_crossover(p1, p2):
    tsize = len(p1)
    spos = random.randint(0, tsize - 1)
    return p1[:spos] + p2[spos:]

#Two-point crossover function to combine two parents into a child
#The function works by selecting two random crossover points, in which the first and third part of the child is taken from the first parent and the second part from the second parent
def two_point_crossover(p1, p2):
    tsize = len(p1)
    point1 = random.randint(0, tsize - 1)
    point2 = random.randint(point1, tsize - 1)
    return p1[:point1] + p2[point1:point2] + p1[point2:]

#Uniform crossover function to combine two parents into a child
#The function works by randomly selecting each gene from either parent
def uniform_crossover(p1, p2):
    child = []
    for ch1, ch2 in zip(p1, p2):
        if random.random() < 0.5:
            child.append(ch1)
        else:
            child.append(ch2)
    return "".join(child)

#A function to mate the population
#The function works by selecting two random parents from the top half of the population, and creating a child using the crossover function. It do this for 1-elitism_rate of the population size,
def mate(population, buffer, target):
    pop_size = population.size
    esize = int(pop_size * GA_ELITRATE)
    
    #Age update and removal of old individuals
    if SURVIVOR_SELECTION_METHOD == "AGING":
        aging(population)

    esize = min(esize, len(population.individuals)) #Ensuring esize does not exceed the population size
    buffer = population.elitism(buffer, esize) #updates the first esize individuals in the buffer with the best individuals from the population

    select_count = [0] * pop_size #tracking the number of times each individual is chosen as a parent

    parents = [] #used to store the selected parents if SUS method is used
    for i in range(esize, pop_size):
        
        #Selecting two random parents from the top half of the population
        if PARENT_SELECTION_METHOD == "TOP_HALF_UNIFORM":
            i1, p1 = top_half_uniform_selection(population)
            i2, p2 = top_half_uniform_selection(population)
        elif PARENT_SELECTION_METHOD == "RWS":
            i1, p1 = rws_selection(population.individuals)
            i2, p2 = rws_selection(population.individuals)
        elif PARENT_SELECTION_METHOD == "SUS":
            if not parents:
                parents = sus_selection(population.individuals, 2)
            i1, p1 = parents[0]
            i2, p2 = parents[1]
            parents = parents[2:] #removing the two chosen parents from the list
        elif PARENT_SELECTION_METHOD == "TOURNAMENT_DET":
            individuals_ranked = fitness_ranking(population.individuals)
            i1, p1 = tournament_selection_deter(individuals_ranked)
            i2, p2 = tournament_selection_deter(individuals_ranked)
        elif PARENT_SELECTION_METHOD == "TOURNAMENT_STOCH":
            individuals_ranked = fitness_ranking(population.individuals)
            i1, p1 = tournament_selection_stoch(individuals_ranked)
            i2, p2 = tournament_selection_stoch(individuals_ranked)
        elif PARENT_SELECTION_METHOD == "SHUFFLE":
            child_genome = population.individuals[i].genome_rearrange(population.individuals[i].genome, population.bin_capacity)
            population.individuals[i].genome = child_genome
        else:
            raise ValueError("Invalid parent selection method")

        #Creating a child using the crossover function and inserting it into the buffer
        if CROSSOVER_TYPE == "SINGLE":
            child_genome = single_point_crossover(p1, p2)
        elif CROSSOVER_TYPE == "TWO":
            child_genome = two_point_crossover(p1, p2)
        elif CROSSOVER_TYPE == "UNIFORM":
            child_genome = uniform_crossover(p1, p2)
        elif not CROSSOVER_TYPE == "NO_CROSSOVER":
            raise ValueError("Invalid crossover type")

        buffer[i] = Individual(child_genome)
        buffer[i].calculate_fitness(target)

        #Mutating the child with a GA_MUTATIONRATE probability
        if random.random() < GA_MUTATIONRATE and PROBLEM != "BIN_PACKING":
            mutate(buffer[i])

        #Updating the number of times each individual is chosen as a parent
        if PROBLEM != "BIN_PACKING":
            select_count[i1] += 1
            select_count[i2] += 1
        else:
            select_count[i] += 1

    #computing the variance of selecting probability
    fitness_var = fitness_variance(select_count, pop_size) 
    population.generation_fitness_var.append(fitness_var) 
    #computing the top-average selection probability ratio
    top_avg = top_avg_select_ratio(select_count)
    population.top_avg_select_ratio.append(top_avg)

    #computing the genetic diversity of the population 
    if PROBLEM != "BIN_PACKING":
        distance_avg = distance_average(population.individuals)
        distinct_alleles_count = distinct_alleles(population.individuals)
        shannon_entropy_value = shannon_entropy(population.individuals)
        population.avg_dist_list.append(distance_avg)
        population.distinct_alleles_list.append(distinct_alleles_count)
        population.shannon_entropy_list.append(shannon_entropy_value)

#A method that computes and prints the CPU time and elapsed time (task 2)
def time_compute(start_cpu_time, start_wall_time):
    ticks_cpu = time.process_time() - start_cpu_time
    elapsed   = time.time() - start_wall_time
    print(f"    Ticks CPU: {ticks_cpu:.4f}, Elapsed: {elapsed:.4f}s")

#A function that computes the variance of selecting probability (task 8)
def fitness_variance(select_count, population_size):
    choose_prob = [count / population_size for count in select_count]
    avg_choose_prob = sum(choose_prob) / population_size
    variance = sum((p - avg_choose_prob) ** 2 for p in choose_prob) / population_size
    return variance

#A function that computes the top average selection probability ratio (task 8)
def top_avg_select_ratio(select_count):
    top_half = select_count[:len(select_count)//2]
    top_avg = sum(top_half) / len(top_half)
    all_avg = sum(select_count) / len(select_count)
    return top_avg / all_avg if all_avg != 0 else 0

#A function to compute the average distance between individuals in the population (section 9)
def distance_average(individuals):
    if len(individuals) < 2:
        return 0.0

    genome_length = len(individuals[0].genome)
    total_distance = 0
    n = len(individuals)

    for pos in range(genome_length):
        #Counting the frequency of each character at the current position
        freq = {}
        for ind in individuals:
            ch = ind.genome[pos]
            if ch in freq:
                freq[ch] += 1
            else:
                freq[ch] = 1

        #Computing the total distance for the current position
        chars = list(freq.items())
        for i in range(len(chars)):
            char_i, count_i = chars[i]
            for j in range(i + 1, len(chars)):
                char_j, count_j = chars[j]
                diff = abs(ord(char_i) - ord(char_j))
                total_distance += count_i * count_j * diff

    total_pairs = n * (n - 1) // 2 
    return total_distance / total_pairs

#A function that computes the number of different alleles in the population (section 9)
def distinct_alleles(individuals):
    all_chars = set(ch for ind in individuals for ch in ind.genome)
    return len(all_chars)

#A function that computes the Shannon entropy of the population (section 9)
def shannon_entropy(individuals):
    char_counts = {}
    total_chars = 0
    for ind in individuals:
        for ch in ind.genome:
            char_counts[ch] = char_counts.get(ch, 0) + 1
            total_chars += 1
    entropy = -sum((count / total_chars) * math.log2(count / total_chars) for count in char_counts.values() if count > 0)
    return entropy

#A function that selects a parent using Top Half Uniform  
def top_half_uniform_selection(population):
    rand = random.randint(0, population.size // 2 - 1)
    return rand, population.individuals[rand].genome

#A function that selects a parent using RWS (section 10)
def rws_selection(individuals):
    
    #Scaling the fitness values using linear scaling    
    max_fitness = max(ind.fitness for ind in individuals)
    scaled_fitnesses = linear_scaling(individuals, -1, max_fitness)
    
    #Converting fitness values to probabilities
    total_scaled = sum(scaled_fitnesses)
    selection_probs = [fit / total_scaled for fit in scaled_fitnesses]
    
    if total_scaled == 0:
        random_choice = random.randint(0, len(individuals) - 1)
        return random_choice, individuals[random_choice].genome
    
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
            return i, individuals[i].genome
    return len(individuals) - 1, individuals[-1].genome 

#A function that selects a list of parents using SUS (Section 10)
def sus_selection(individuals, num_parents):
    
    #Scaling the fitness values using linear scaling
    max_fitness = max(ind.fitness for ind in individuals)
    scaled_fitness = linear_scaling(individuals, -1, max_fitness)

    #Converting fitness values to probabilities
    total_fitness = sum(scaled_fitness)
    if total_fitness == 0:
        return random.choices([(i, ind.genome) for i, ind in enumerate(individuals)],k=num_parents)
    selection_probs = [fit / total_fitness for fit in scaled_fitness]
    
    #Building the "roulete"
    cumulative_probs = []
    cumsum = 0
    for prob in selection_probs:
        cumsum += prob
        cumulative_probs.append(cumsum)

    #Selecting a random starting point, then parents
    rand = random.random() / num_parents
    parents = []
    for i in range(num_parents):
        target = rand + i / num_parents
        for j, prob in enumerate(cumulative_probs):
            if target < prob:
                parents.append((j, individuals[j].genome))
                break
    return parents

#A function that selects a parent using Deterministic Tournament Selection (Section 10)
def tournament_selection_deter(individuals):
    tournament = random.sample(range(len(individuals)), TOURNAMENT_K) #choosing K random indices 
    tournament.sort(key=lambda ind: individuals[ind].rank) #sorting the corresponding individuals
    return tournament[0], individuals[tournament[0]].genome

#A function that selects a parent using Stochastic Tournament Selection (Section 10)
def tournament_selection_stoch(individuals):
    tournament = random.sample(range(len(individuals)), TOURNAMENT_K) #choosing K random indices
    tournament.sort(key=lambda ind: individuals[ind].rank) #sorting the corresponding individuals
    for i in range(TOURNAMENT_K):
        if random.random() < TOURNAMENT_P: 
            return tournament[i], individuals[tournament[i]].genome
    return tournament[-1], individuals[tournament[-1]].genome

#A function that linearly scales the fitness values (Section 10)
def linear_scaling(individuals, a,b):
    scaled_fitness = [a * ind.fitness + b for ind in individuals]
    return scaled_fitness

#A function that rank the fitnesses by converting them to ranks (Section 10)
def fitness_ranking(individuals, reverse=False):
    sorted_inds = sorted(individuals, key=lambda ind: ind.fitness, reverse=reverse)
    for rank, ind in enumerate(sorted_inds):
        ind.rank = rank + 1
    return sorted_inds

#A function to increment the age of each individual in the population and remove the old between them
def aging(population):
    for ind in population.individuals:
        ind.age += 1
    population.individuals = [ind for ind in population.individuals if ind.age < AGE_LIMIT]


def main(max_time, initial):
    random.seed(time.time())

    #Initializing the population and buffer
    if PROBLEM == "TARGET_STRING" or PROBLEM == "MATRIX_TRANSFORM":
        population = Population(GA_POPSIZE, GA_TARGET, initial)
    elif PROBLEM == "BIN_PACKING":
        bin_capacity = int(initial[0])
        target = int(initial[1])
        initial_genome = [[item] for item in initial[2]]
        population = Population(GA_POPSIZE, target, initial_genome, bin_capacity)
    buffer = [ind for ind in population.individuals] 

    #Initializing the CPU and elapsed time
    start_wall_time = time.time()
    start_cpu_time  = time.process_time()

    #Variables to detect convergence
    best_fit_so_far = float('inf')
    no_improvement_count = 0
    for generation in range(GA_MAXITER):

        #Updating the fitness of the population and sorting its population
        population.update_fitness()
        population.sort_by_fitness()

        #Computing and printing the generation best and worst individuals, fitness range, average fitness, and standard deviation (task 1)
        population.generation_stats_update(generation)

        #Computing and printing the CPU time and elapsed time (task 2)
        time_compute(start_cpu_time, start_wall_time)

        #Checking for convergence
        if population.individuals[0].fitness == 0:
            print("Global optimum found!")
            break

        if population.individuals[0].fitness < best_fit_so_far:
            best_fit_so_far = population.individuals[0].fitness
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= NO_IMPROVEMENT_LIMIT:
            print("No improvement => Local optimum convergence.")
            break

        if time.time() - start_wall_time > max_time:
            print("Time limit exceeded.")
            break

        #Mating the population
        mate(population, buffer, population.target)

        #Updating the population of the next generation
        population.individuals = buffer

    #Final values of time 
    end_wall_time = time.time()
    end_cpu_time  = time.process_time()
    total_wall    = end_wall_time - start_wall_time
    total_cpu     = end_cpu_time - start_cpu_time

    print(f"Finished after {len(population.fitness_history)} generations.")
    print(f"Total wall-clock time: {total_wall:.2f}s, Total CPU time: {total_cpu:.2f}s")

    #Plotting the best, average, and worst fitness over generations (task 3a)
    population.fitness_plot()

    #Plotting the boxplots for each generation's distribution (task 3b)
    population.fitness_boxplot()

    #Plotting the chosen individuals' fitness variance over generations (task 8)
    population.plot_selection_pressure()

    #Plotting the genetic diversity of the population over generations (task 9)
    if PROBLEM != "BIN_PACKING":
        population.plot_diversity()

def interactive():
    import tkinter as tk
    from tkinter import filedialog

    global CROSSOVER_TYPE, FITNESS_MODE, PARENT_SELECTION_METHOD

    def ask_choice(prompt, options, default_value):
        print(f"{prompt}")
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")
        while True:
            choice = input("Choose (1/2/3 or Enter for default): ").strip()
            if choice == "":
                print(f"Using default: {default_value}\\n")
                return default_value
            elif choice in {"1", "2", "3"} and int(choice) <= len(options):
                print(f"Selected: {options[int(choice)-1]}\\n")
                return options[int(choice)-1]
            else:
                print("Invalid choice. Please enter 1, 2, 3 or press Enter.")

    CROSSOVER_TYPE = ask_choice("Select crossover type:", ["SINGLE", "TWO", "UNIFORM"], CROSSOVER_TYPE)
    FITNESS_MODE = ask_choice("Select fitness mode:", ["DISTANCE", "LCS", "BINS_DIFF"], FITNESS_MODE)
    PARENT_SELECTION_METHOD = ask_choice("Select parent selection method:", ["TOP_HALF_UNIFORM", "RWS", "TOURNAMENT_DET"], PARENT_SELECTION_METHOD)

    print("Would you like to load a file? (txt/json) or press Enter to skip")
    file_path = input("Enter path or press Enter: ").strip()

    if file_path:
        if not (file_path.endswith(".txt") or file_path.endswith(".json")):
            print("Invalid file format. Only .txt or .json supported.")
            sys.exit(1)
        elif not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            sys.exit(1)
        else:
            sys.argv.append("placeholder_time")
            sys.argv.append(file_path)

if __name__ == "__main__":
    
    interactive()

    initial = None
    
    #Check validity of the number of arguments
    if len(sys.argv) < 2 or len(sys.argv) > 5:
        print("Error: Invalid number of arguments.")        
        print("Usage:")
        print("  python script.py <max_time>")
        print("  python script.py <max_time> <target_individual>")
        print("  python script.py <max_time> <initial_individual> <target_individual>")
        print("  python script.py <max_time> <json_file>")
        sys.exit(1)

    #The case in which initial and target given explicitly as arguments
    if len(sys.argv) == 4:  
        initial = sys.argv[2]
        target = sys.argv[3]

        #Convert the initial and target matrices to strings if they are not
        if not isinstance(initial, str):
            initial = "".join(str(num) for num in initial)
        if not isinstance(target, str):
            target = "".join(str(num) for num in target)
            PROBLEM = "MATRIX_TRANSFORM" 
        
        GA_TARGET = target #updating the target variable

    elif len(sys.argv) == 3:
        sec_arg = sys.argv[2]

        #The case in which initial and target given inside a JSON file
        if os.path.exists(sec_arg) and sec_arg.endswith('.json'):
            with open(sec_arg, 'r') as file:
                data = json.load(file)
                initial = data["test"][0].get("input")
                target = data["test"][0].get("output")

            #Convert the initial and target matrices to strings
            initial = "".join(str(num) for row in initial for num in row)
            GA_TARGET = "".join(str(num) for row in target for num in row)
            PROBLEM = "MATRIX_TRANSFORM" 

        #The case in which initial and target given inside a text file
        elif os.path.exists(sec_arg) and sec_arg.endswith('.txt'):
            with open(sec_arg, 'r') as file:
                PROBLEM = "BIN_PACKING"
                PARENT_SELECTION_METHOD = "SHUFFLE"
                CROSSOVER_TYPE = "NO_CROSSOVER"
                FITNESS_MODE = "BINS_DIFF"
                lines = file.readlines()
                input = lines[2].strip().split() #extracting the number of items, bin size, and optimal solution
                initial = [int(input[0]), int(input[2])]
                items = []
                GA_TARGET = int(input[2])
                for line in lines[3: (3 + int(input[1]))]:
                    items.append(int(line.strip()))
                initial.append(items)

        #The case in which only target given as argument
        else:
            if not isinstance(sec_arg, str):
                sec_arg = "".join(str(num) for num in sec_arg)
                PROBLEM = "MATRIX_TRANSFORM"
            GA_TARGET = sec_arg

    #Extracting the maximum time for the algorithm to run
    try:
        max_time = float(sys.argv[1])
        if max_time <= 0:
            raise ValueError
    except ValueError:
        print("Error: max_time must be a positive number.")
        sys.exit(1)

    main(max_time, initial)