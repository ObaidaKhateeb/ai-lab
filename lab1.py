import random
import time
import math
import matplotlib.pyplot as plt

# -----------------------------
# GA PARAMETERS
# -----------------------------
GA_POPSIZE       = 16384     # Population size
GA_MAXITER       = 100    # Maximum number of iterations
GA_ELITRATE      = 0.05     # Elitism rate
GA_MUTATIONRATE  = 0.25     # Mutation rate
GA_TARGET        = "Hello world!"
GA_CHARSIZE      = 90       # Range of characters (roughly ' ' to '~')
NO_IMPROVEMENT_LIMIT = 40  # local optimum threshold

# -----------------------------
# CROSSOVER MODES
# -----------------------------
# Choose one of: "SINGLE", "TWO", "UNIFORM"
CROSSOVER_TYPE = "TWO"

def init_population(pop_size, target):
    population = []
    tsize = len(target)
    for _ in range(pop_size):
        genome = ''.join(chr(random.randint(32, 32 + GA_CHARSIZE - 1)) for _ in range(tsize))
        population.append({"genome": genome, "fitness": 0})
    return population

def calc_fitness(population, target):
    tsize = len(target)
    for individual in population:
        fitness_score = 0
        for i in range(tsize):
            fitness_score += abs(ord(individual["genome"][i]) - ord(target[i]))
        individual["fitness"] = fitness_score

def sort_by_fitness(population):
    population.sort(key=lambda ind: ind["fitness"])

def elitism(population, buffer, esize):
    for i in range(esize):
        buffer[i]["genome"]  = population[i]["genome"]
        buffer[i]["fitness"] = population[i]["fitness"]

def mutate(individual, target):
    tsize = len(target)
    ipos = random.randint(0, tsize - 1)
    old_char_val = ord(individual["genome"][ipos])
    delta = random.randint(32, 32 + GA_CHARSIZE - 1)
    new_char_val = (old_char_val + delta) % 126
    if new_char_val < 32:
        new_char_val += 32
    genome_list = list(individual["genome"])
    genome_list[ipos] = chr(new_char_val)
    individual["genome"] = "".join(genome_list)

def single_point_crossover(p1, p2):
    """ Single-point crossover """
    tsize = len(p1)
    spos = random.randint(0, tsize - 1)
    return p1[:spos] + p2[spos:]

def two_point_crossover(p1, p2):
    """ Two-point crossover """
    tsize = len(p1)
    point1 = random.randint(0, tsize - 1)
    point2 = random.randint(point1, tsize - 1)
    return p1[:point1] + p2[point1:point2] + p1[point2:]

def uniform_crossover(p1, p2):
    """ Uniform crossover: each gene randomly from p1 or p2 """
    child = []
    for ch1, ch2 in zip(p1, p2):
        if random.random() < 0.5:
            child.append(ch1)
        else:
            child.append(ch2)
    return "".join(child)

def mate(population, buffer, target):
    pop_size = len(population)
    esize = int(pop_size * GA_ELITRATE)
    #print the buffer before and after elitism
    print("Buffer before elitism:", buffer)
    elitism(population, buffer, esize)
    print("Buffer after elitism:", buffer)

    for i in range(esize, pop_size):
        i1 = random.randint(0, pop_size // 2 - 1)
        i2 = random.randint(0, pop_size // 2 - 1)

        p1 = population[i1]["genome"]
        p2 = population[i2]["genome"]

        if CROSSOVER_TYPE == "SINGLE":
            child_genome = single_point_crossover(p1, p2)
        elif CROSSOVER_TYPE == "TWO":
            child_genome = two_point_crossover(p1, p2)
        else:  # "UNIFORM"
            child_genome = uniform_crossover(p1, p2)

        buffer[i]["genome"]  = child_genome
        buffer[i]["fitness"] = 0

        if random.random() < GA_MUTATIONRATE:
            mutate(buffer[i], target)

def main():
    random.seed(time.time())

    pop_alpha = init_population(GA_POPSIZE, GA_TARGET)
    pop_beta  = [{"genome": "", "fitness": 0} for _ in range(GA_POPSIZE)]
    population = pop_alpha
    buffer     = pop_beta

    start_wall_time = time.time()
    start_cpu_time  = time.process_time()

    best_fit_so_far = float('inf')
    no_improvement_count = 0

    # For the line plot
    best_fitness_list = []
    avg_fitness_list  = []
    worst_fitness_list = []

    # For boxplots, store per-generation distributions
    fitness_history = []

    final_generation = 0

    for generation in range(GA_MAXITER):
        calc_fitness(population, GA_TARGET)
        sort_by_fitness(population)

        best_fit  = population[0]["fitness"]
        worst_fit = population[-1]["fitness"]
        fitness_range = worst_fit - best_fit
        sum_fit   = sum(ind["fitness"] for ind in population)
        avg_fit   = sum_fit / len(population)

        variance  = sum((ind["fitness"] - avg_fit)**2 for ind in population) / len(population)
        std_dev   = math.sqrt(variance) if variance>0 else 0

        ticks_cpu = time.process_time() - start_cpu_time
        elapsed   = time.time() - start_wall_time

        best_fitness_list.append(best_fit)
        avg_fitness_list.append(avg_fit)
        worst_fitness_list.append(worst_fit)

        # store distribution for boxplot
        gen_fitness_list = [ind["fitness"] for ind in population]
        fitness_history.append(gen_fitness_list)

        # Print EXACT text
        print(f"Gen={generation:4d}, "
              f"Best={best_fit}, "
              f"Worst={worst_fit}, "
              f"Range={fitness_range}, "
              f"Avg={avg_fit:.2f}, "
              f"Std={std_dev:.2f}, "
              f"TicksCPU={ticks_cpu:.4f}, "
              f"Elapsed={elapsed:.2f}s, "
              f"BestString='{population[0]['genome']}'")

        if best_fit < best_fit_so_far:
            best_fit_so_far = best_fit
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if best_fit == 0:
            print("Global optimum found!")
            final_generation = generation + 1
            break

        if no_improvement_count >= NO_IMPROVEMENT_LIMIT:
            print("No improvement => Local optimum convergence.")
            final_generation = generation + 1
            break

        mate(population, buffer, GA_TARGET)
        population, buffer = buffer, population
        final_generation = generation + 1

    end_wall_time = time.time()
    end_cpu_time  = time.process_time()
    total_wall    = end_wall_time - start_wall_time
    total_cpu     = end_cpu_time - start_cpu_time

    print(f"Finished after {final_generation} generations.")
    print(f"Total wall-clock time: {total_wall:.2f}s, Total CPU time: {total_cpu:.2f}s")

    # --------------------------------------------------
    # Plot (1) Best, Avg, Worst
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Plot (2) Box plots for each generation's distribution (optional)
    # --------------------------------------------------
    for g in range(final_generation):
        gen_data = fitness_history[g]
        plt.figure(figsize=(4,5))
        plt.boxplot(gen_data, showfliers=True)
        plt.title(f"Box Plot of Fitness - Gen {g} ({CROSSOVER_TYPE})")
        plt.ylabel("Fitness")
        plt.show()

if __name__ == "__main__":
    main()
