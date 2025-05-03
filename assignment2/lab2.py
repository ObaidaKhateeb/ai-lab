import random
import math

#Parent selection method (TOP_HALF_UNIFORM ,RWS, SUS, TOURNAMENT_DET, TOURNAMENT_STOCH, SHUFFLE)
PARENT_SELECTION_METHOD = "TOP_HALF_UNIFORM"

#Tournament Parameters
TOURNAMENT_K = 49
TOURNAMENT_P = 0.86

#Mutation types (displacement, swap, insertion, simple_inversion, inversion, scramble)
MUTATION_TYPE = "scramble" 

#A function to extract the data from a csv file 
def read_tsp_file(filepath):
    coords = []
    with open(filepath, 'r') as f:
        relevant_line = False
        for line in f:
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

class TSPIndividual:
    def __init__(self, genome):
        self.genome = genome  
        self.fitness = None
        self.rank = None

    def calculate_fitness(self, dist_matrix):
        total = 0
        for i in range(len(self.genome)):
            curr_city = self.genome[i]
            next_city = self.genome[(i + 1) % len(self.genome)]
            total += dist_matrix[curr_city][next_city]
        self.fitness = total

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
    def __init__(self, coords, size):
        self.dist_matrix = compute_distance_matrix(coords)
        self.size = size
        self.individuals = self.init_population(coords)

    def init_population(self, coords):
        base = list(range(len(coords))) 
        return [TSPIndividual(random.sample(base, len(base))) for _ in range(self.size)] #randomly shuffle the coordinates of the cities

    def evaluate_fitness(self):
        for ind in self.individuals:
            ind.calculate_fitness(self.dist_matrix)

    def select_parents(self):
        if PARENT_SELECTION_METHOD == "TOP_HALF_UNIFORM":
            return self.top_half_uniform_selection()
        elif PARENT_SELECTION_METHOD == "RWS":
            return self.rws_selection()
        elif PARENT_SELECTION_METHOD == "SUS":
            return self.sus_selection(2)
        elif PARENT_SELECTION_METHOD == "TOURNAMENT_DET":
            self.fitness_ranking(reverse=True)
            return self.tournament_selection_deter()
        elif PARENT_SELECTION_METHOD == "TOURNAMENT_STOCH":
            self.fitness_ranking(reverse=True)
            return self.tournament_selection_stoch()
        else:
            raise ValueError(f"Wrong parent selection method: {PARENT_SELECTION_METHOD}")

    #A function that selects a parent using Top Half Uniform  
    def top_half_uniform_selection(self):
        rand = random.randint(0, len(self.individuals) // 2 - 1)
        return rand, self.individuals[rand].genome

    #A function that selects a parent using RWS (section 10)
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

    #A function that selects a list of parents using SUS (Section 10)
    def sus_selection(self, num_parents):
        
        #Scaling the fitness values using linear scaling
        max_fitness = max(ind.fitness for ind in self.individuals)
        scaled_fitness = linear_scaling(self.individuals, -1, max_fitness)

        #Converting fitness values to probabilities
        total_fitness = sum(scaled_fitness)
        if total_fitness == 0:
            return random.choices([(i, ind.genome) for i, ind in enumerate(self.individuals)],k=num_parents)
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
                    parents.append((j, self.individuals[j].genome))
                    break
        return parents

    #A function that selects a parent using Deterministic Tournament Selection (Section 10)
    def tournament_selection_deter(self):
        tournament = random.sample(range(len(self.individuals)), TOURNAMENT_K) #choosing K random indices 
        tournament.sort(key=lambda ind: self.individuals[ind].rank) #sorting the corresponding individuals
        return tournament[0], self.individuals[tournament[0]].genome

    #A function that selects a parent using Stochastic Tournament Selection (Section 10)
    def tournament_selection_stoch(self):
        tournament = random.sample(range(len(self.individuals)), TOURNAMENT_K) #choosing K random indices
        tournament.sort(key=lambda ind: self.individuals[ind].rank) #sorting the corresponding individuals
        for i in range(TOURNAMENT_K):
            if random.random() < TOURNAMENT_P: 
                return tournament[i], self.individuals[tournament[i]].genome
        return tournament[-1], self.individuals[tournament[-1]].genome

    def fitness_ranking(self, reverse=False):
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=reverse)
        for rank, ind in enumerate(sorted_inds):
            ind.rank = rank + 1

    def crossover(self, p1, p2):
        self.order_crossover(p1.genome, p2.genome)

    def order_crossover(self, p1, p2):
        size = len(p1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = p1[start:end]
        pos = end
        for gene in p2:
            if gene not in child:
                while child[pos % size] is not None:
                    pos += 1
                child[pos % size] = gene
        return TSPIndividual(child)

    def evolve(self, elitism=0.1, mutation_rate=0.2):
        self.evaluate_fitness()
        new_pop = sorted(self.individuals, key=lambda x: x.fitness)[:int(elitism * self.size)]

        while len(new_pop) < self.size:
            _, p1 = self.select_parents()
            _, p2 = self.select_parents()
            child = self.order_crossover(p1, p2)
            if random.random() < mutation_rate:
                child.mutate()
            child.calculate_fitness(self.dist_matrix)
            new_pop.append(child)

        self.individuals = new_pop

    def best_individual(self):
        return min(self.individuals, key=lambda x: x.fitness)


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





#A function that linearly scales the fitness values (Section 10)
def linear_scaling(individuals, a,b):
    scaled_fitness = [a * ind.fitness + b for ind in individuals]
    return scaled_fitness
# -------- Main Runner --------

def main(filepath, pop_size=100, generations=300):
    coords = read_tsp_file(filepath)
    population = TSPPopulation(coords, pop_size)

    for gen in range(generations):
        population.evolve()
        best = population.best_individual()
        print(f"Gen {gen:3}. Best = {best.genome} ({best.fitness:.2f})")

    best = population.best_individual()
    print("\nBest tour:")
    print(best.genome)
    print(f"Total distance: {best.fitness:.2f}")


if __name__ == "__main__":
    main("salesman_inputs/ulysses16.tsp")
