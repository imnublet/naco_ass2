'''This file contains a skeleton implementation for the practical assignment 
for the NACO 21/22 course. 

Your Genetic Algorithm should be callable like so:
    >>> problem = ioh.get_problem(...)
    >>> ga = GeneticAlgorithm(...)
    >>> ga(problem)

In order to ensure this, please inherit from the provided Algorithm interface,
in a similar fashion as the RandomSearch Example:
    >>> class GeneticAlgorithm(Algorithm):
    >>>     ...

Please be sure to use don't change this name (GeneticAlgorithm) for your implementation.

If you override the constructor in your algoritm code, please be sure to 
call super().__init__, in order to correctly setup the interface. 

Only use keyword arguments for your __init__ method. This is a requirement for the
test script to properly evaluate your algoritm.
'''

import ioh
import random
import numpy as np
from algorithm import Algorithm

class GeneticAlgorithm(Algorithm):



    def __init__(self, pop_size=10, dim=10, seed=10, mutation_type = "swap"):
        super().__init__(max_iterations=1000)
        self.mutation_type = mutation_type
        self.dim = dim
        self.seed = seed
        self.dim = dim
        self.population_size = pop_size
        np.random.seed(self.seed)
        self.population = self.generate_population()
        self.y_best = 0.0
        self.x_best : list[int] = []


    """
    Generate an candidate i for the population self.population
    """
    def generate_candidate(self):
        candidate = np.random.randint(2, size=self.dim, dtype=int)
        return candidate

        """
        Generate a candidate for every i in self.population_size
        """
    def generate_population(self):
        return [self.generate_candidate() for i in range(self.population_size)]

    """
    Print the population size
    """
    def print_pop_size(self):
        print(self.population_size)

        """
        * Calculate fitness score/weight for every candidate
        * Randomly select k candidates given selection probabilities
        * return a list of two candidates
        """
    def selection(self, k=2):
        candidates = np.empty(shape=[0, 2])

        for candidate in self.population:
            candidates = np.append(candidates, [[candidate, self.selection_probability(candidate)]], axis=0)

        weights = np.array(candidates[:,1], dtype=float)
        np_candidates = np.array(candidates[:,0])

        return np.random.choice(np_candidates, p = weights, size = 2)

        """
        Return candidate fitness score / population fitness score
        """
    def selection_probability(self, candidate):
        return (self.fitness(candidate) / self.population_fitness())

        """
        For every child in the offspring apply either a swap mutatation or an insert mutation
        """
    def mutation(self, offspring):
        if self.mutation_type == "swap":
            for child in offspring:
                child = self.swap_mutation(child)
        else:
            for child in offspring:
                child = self.insert_mutation(child)
        return offspring

    """
    Apply a swap mutation, by randomly selecting 2 genes and swapping them.
    """
    def swap_mutation(self, child):
        rand_idx = np.random.randint(self.dim,size=2)
        temp = child[rand_idx[1]]
        child[rand_idx[1]] = child[rand_idx[0]]
        child[rand_idx[0]] = temp
        return child

    """
    * Apply an insert mutation, by randomly selecting an i and a j in the genome.
    * Insert j at position i+1 and delete it's previous index
    """
    def insert_mutation(self, child):
        child = list(child)
        rand_idx = np.random.randint(self.dim,size=2)
        while rand_idx[0] > rand_idx[1]:
            rand_idx = np.random.randint(self.dim,size=2)
        if rand_idx[0] != rand_idx[1]:
            child.insert(rand_idx[1]-1, child[rand_idx[0]])
            child.pop(rand_idx[0])
        return np.array(child, dtype=int)


    """
    For every child in the offspring apply a swap mutation.
    """
    def crossover(self, parents):
        if(len(parents) != 2):
            raise ValueError("There should be 2 parents")
        offspring = []
        for i in range(self.population_size):
            offspring.append(self.crossover_operator(parents))

        return offspring

    """
    * Given two parents, create a child based on a random uniform probability
    * Children are created by a simple treshold. If the probability of a certain index in idx_prob is higher than 0.5, 
    * then a gene from parent A is given, else: parent B
    * This is done for n = self.population_size 
    """
    def crossover_operator(self, parents):
        parentA = np.array(parents[0], dtype=int)
        parentB = np.array(parents[1], dtype=int)
        child = np.empty(shape=[0,1], dtype=int)
        idx_prob = np.random.random_sample(self.dim)
        for i in range(len(idx_prob)):
            if idx_prob[i] > 0.5:
                child = np.append(child, parentA[i])
            else:
                child = np.append(child, parentB[i])
        return child


    """   
    Return the occurrences of 1s
    """
    def fitness(self, candidate):
        return np.count_nonzero(candidate)
    """"
    Calculate the fitness score for the whole population
    Calculation of the fitness score is explained in fitness(self, candidate)
    """
    def population_fitness(self):
        pop_fitness = 0
        for candidate in self.population:
            pop_fitness += self.fitness(candidate)
        return pop_fitness

    def __call__(self, problem: ioh.problem.Integer) -> None:
        self.dim = problem.meta_data.n_variables
        self.population = self.generate_population()
        while (problem.state.evaluations < self.max_iterations) and (self.x_best != problem.objective.x):
            parents = self.selection(self.population)
            offspring = self.crossover(parents)
            mutated_offspring = self.mutation(offspring)
            for candidate in self.population:
                new_y = self.fitness(candidate)
                if new_y > self.y_best:
                    self.y_best = new_y
                    self.x_best = list(candidate)
                    print(self.x_best)
            self.population = mutated_offspring

        problem(self.x_best)
        return problem.state.current_best





def main():

    # Set a random seed in order to get reproducible results
    random.seed(42)

    # Get a problem from the IOHexperimenter environment
    problem: ioh.problem.Integer = ioh.get_problem(1, 1, 5, "Integer")

    # Instantiate the algoritm, you should replace this with your GA implementation
    algorithm = GeneticAlgorithm(mutation_type="insert")

    # Run the algoritm on the problem
    algorithm(problem)

    # Inspect the results
    print("Best solution found:")
    print("".join(map(str, problem.state.current_best.x)))
    print("With an objective value of:", problem.state.current_best.y)
    print()

# main()

if __name__ == '__main__':
    main()

