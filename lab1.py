import random
import time
import math
import matplotlib.pyplot as plt

#GA Parameters
GA_POPSIZE       = 16384     #Population size
GA_MAXITER       = 100    #Maximum number of iterations
GA_ELITRATE      = 0.05     #Elitism rate
GA_MUTATIONRATE  = 0.25     #Mutation rate
GA_TARGET        = "Hello world!" #Target string
GA_CHARSIZE      = 90       #Range of characters (roughly ' ' to '~')
NO_IMPROVEMENT_LIMIT = 40  #Local optimum threshold

#Crossover mode (options: SINGLE, TWO, UNIFORM)
CROSSOVER_TYPE = "UNIFORM"

#Fitness mode (options: DISTANCE, LCS)
FITNESS_MODE = "LCS"

#Individual class representing a single solution
class Individual:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = None

    #A fuction to calculate the fitness of the individual
    #The fitness is the sum of absolute differences between the genome and target
    def calculate_fitness(self, target):
        if FITNESS_MODE == "DISTANCE":
            self.fitness = sum(abs(ord(g) - ord(t)) for g, t in zip(self.genome, target))
        #LCS fitness is the length of the longest common subsequence and bonus of 4 for each correct character (in the right position) of the characters of the LCS
        elif FITNESS_MODE == "LCS":
            self.fitness = self.fitness_by_lcs(self.genome, target)
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
                if m == n: #ckecks the LCS genome character is in the right position
                    correct_chars_count += 1
            elif L[m - 1][n] > L[m][n - 1]:
                m -= 1
            else:
                n -= 1
        max_possible = (bonus+1) * len(b)
        return max_possible - (lcs_length + bonus * correct_chars_count)


#Population class representing a collection of individuals
class Population:
    def __init__(self, size, target):
        self.size = size
        self.target = target
        self.individuals = self.init_population()

    #A function to initialize the population with random individuals
    def init_population(self):
        tsize = len(self.target)
        population = []
        for _ in range(self.size):
            genome = ''.join(chr(random.randint(32, 32 + GA_CHARSIZE - 1)) for _ in range(tsize))
            individual = Individual(genome)
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

#A function to mutate an individual by changing a random character in its genome
def mutate(individual):
    tsize = len(individual.genome)
    ipos = random.randint(0, tsize - 1) #choosing a random character 
    old_char_val = ord(individual.genome[ipos]) #extracting the ASCII value of the character
    delta = random.randint(32, 32 + GA_CHARSIZE - 1) #choosing a random value to add to the ASCII value
    new_char_val = (old_char_val + delta) % 126 #modulo 126 to keep it within the ASCII range
    if new_char_val < 32: 
        new_char_val += 32

    #Updating the genome with the new character instead of the old one
    genome_list = list(individual.genome)
    genome_list[ipos] = chr(new_char_val)
    individual.genome = "".join(genome_list)

#Single-point crossover function to combine two parents into a child
#The function works by selecting a random crossover point, in which the first part of the child is taken from the first parent and the second part from the second parent
def single_point_crossover(p1, p2):
    """ Single-point crossover """
    tsize = len(p1)
    spos = random.randint(0, tsize - 1)
    return p1[:spos] + p2[spos:]

#Two-point crossover function to combine two parents into a child
#The function works by selecting two random crossover points, in which the first and third part of the child is taken from the first parent and the second part from the second parent
def two_point_crossover(p1, p2):
    """ Two-point crossover """
    tsize = len(p1)
    point1 = random.randint(0, tsize - 1)
    point2 = random.randint(point1, tsize - 1)
    return p1[:point1] + p2[point1:point2] + p1[point2:]

#Uniform crossover function to combine two parents into a child
#The function works by randomly selecting each gene from either parent
def uniform_crossover(p1, p2):
    """ Uniform crossover: each gene randomly from p1 or p2 """
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
    buffer = population.elitism(buffer, esize) #updates the first esize individuals in the buffer with the best individuals from the population

    for i in range(esize, pop_size):
        
        #Selecting two random parents from the top half of the population
        i1 = random.randint(0, pop_size // 2 - 1)
        i2 = random.randint(0, pop_size // 2 - 1)
        p1 = population.individuals[i1].genome
        p2 = population.individuals[i2].genome

        #Creating a child using the crossover function and inserting it into the buffer
        if CROSSOVER_TYPE == "SINGLE":
            child_genome = single_point_crossover(p1, p2)
        elif CROSSOVER_TYPE == "TWO":
            child_genome = two_point_crossover(p1, p2)
        elif CROSSOVER_TYPE == "UNIFORM":
            child_genome = uniform_crossover(p1, p2)
        else:
            raise ValueError("Invalid crossover type")
        
        buffer[i] = Individual(child_genome)
        buffer[i].calculate_fitness(target)

        #Mutating the child with a GA_MUTATIONRATE probability
        if random.random() < GA_MUTATIONRATE:
            mutate(buffer[i])

def main():
    random.seed(time.time())

    #Initializing the population and buffer
    population = Population(GA_POPSIZE, GA_TARGET) 
    buffer = [ind for ind in population.individuals] 

    #Initializing the CPU and elapsed time
    start_wall_time = time.time()
    start_cpu_time  = time.process_time()

    #Variables to detect convergence
    best_fit_so_far = float('inf')
    no_improvement_count = 0

    #Data structures for line plots use
    best_fitness_list = []
    avg_fitness_list  = []
    worst_fitness_list = []

    #Data structure for boxplots use
    fitness_history = []
    final_generation = 0

    for generation in range(GA_MAXITER):
        #Updating the fitness of the population and sorting its population
        population.update_fitness()
        population.sort_by_fitness()

        #Computing and printing the generation best and worst individuals, fitness range, average fitness, and standard deviation (task 1)
        best_fit  = population.individuals[0].fitness
        worst_fit = population.individuals[-1].fitness
        fitness_range = worst_fit - best_fit
        sum_fit   = sum(ind.fitness for ind in population.individuals)
        avg_fit   = sum_fit / population.size
        variance  = sum((ind.fitness - avg_fit)**2 for ind in population.individuals) / population.size
        std_dev   = math.sqrt(variance) if variance>0 else 0

        print(f"Gen{generation}." 
                f" Best: {population.individuals[0].genome} ({best_fit})", 
                f" Worst: {population.individuals[-1].genome} ({worst_fit}) ",
                f" Fitness Range: {fitness_range} ",
                f" Avg: {avg_fit:.2f} ",
                f" Std: {std_dev:.2f} ")

        #Computing and printing the CPU time and elapsed time (task 2)
        ticks_cpu = time.process_time() - start_cpu_time
        elapsed   = time.time() - start_wall_time

        print(f"    Ticks CPU: {ticks_cpu:.4f}, Elapsed: {elapsed:.2f}s")

        #Storing the best, average, and worst fitness for line plots (task 3a)
        best_fitness_list.append(best_fit)
        avg_fitness_list.append(avg_fit)
        worst_fitness_list.append(worst_fit)

        #Storing the distribution of fitness for boxplots (task 3b)
        gen_fitness_list = [ind.fitness for ind in population.individuals]
        fitness_history.append(gen_fitness_list)

        #Checking for convergence
        if best_fit == 0:
            print("Global optimum found!")
            final_generation = generation + 1
            break

        if best_fit < best_fit_so_far:
            best_fit_so_far = best_fit
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= NO_IMPROVEMENT_LIMIT:
            print("No improvement => Local optimum convergence.")
            final_generation = generation + 1
            break

        #Mating the population
        mate(population, buffer, GA_TARGET)

        #Updating the population of the next generation
        population.individuals, buffer = buffer, population.individuals
        final_generation = generation + 1

    #Final values of time 
    end_wall_time = time.time()
    end_cpu_time  = time.process_time()
    total_wall    = end_wall_time - start_wall_time
    total_cpu     = end_cpu_time - start_cpu_time

    print(f"Finished after {final_generation} generations.")
    print(f"Total wall-clock time: {total_wall:.2f}s, Total CPU time: {total_cpu:.2f}s")

    #Plotting the best, average, and worst fitness over generations (task 3a)
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_list,  label="Best Fitness")
    plt.plot(avg_fitness_list,   label="Average Fitness")
    plt.plot(worst_fitness_list, label="Worst Fitness")
    plt.title(f"GA Fitness Over Generations (Crossover: {CROSSOVER_TYPE})")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()

    #Plotting the boxplots for each generation's distribution (task 3b)
    for g in range(final_generation):
        gen_data = fitness_history[g]
        plt.figure(figsize=(4,5))
        plt.boxplot(gen_data, showfliers=True)
        plt.title(f"Box Plot of Fitness - Gen {g} ({CROSSOVER_TYPE})")
        plt.ylabel("Fitness")
        plt.show()

if __name__ == "__main__":
    main()
