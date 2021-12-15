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


class RandomSearch(Algorithm):
    '''An example of Random Search.'''

    def __call__(self, problem: ioh.problem.Integer) -> None:
        self.y_best: float = float("inf")
        for iteration in range(self.max_iterations):
            # Generate a random bit string
            x: list[int] = [random.randint(0, 1) for _ in range(problem.meta_data.n_variables)]
            # Call the problem in order to get the y value    
            y: float = problem(x)
            # update the current state
            self.y_best = max(self.y_best, y)


class GeneticAlgorithm(Algorithm):

    def __init__(self, pop_size=22, dim=10, seed=23, base=2, mutation_type="swap", mutation_rate=0.5,
                 crossover="uniform", selection="proportional"):
        super().__init__(max_iterations=10000)
        assert selection == "proportional" or \
               selection == "rank", "selection only accepts \"proportional\" or \"rank\""
        assert crossover == "uniform" or \
               crossover == "single_point", "crossover only accepts \"crossover\" or \"single_point\""
        self.mutation_type = mutation_type
        self.dim = dim
        self.seed = seed
        self.base = base
        self.population_size = pop_size
        np.random.seed(self.seed)
        self.population = self.generate_population()
        self.y_best = 0.0
        self.x_best: list[int] = []
        self.fitness_scores = []
        self.mutation_rate = mutation_rate
        self.selection_method = selection
        self.crossover_method = crossover

    def generate_candidate(self):
        """
        Generate a candidate i for the population self.population
        """
        candidate = np.random.randint(self.base, size=self.dim, dtype=int)
        return candidate

    def generate_population(self):
        """
        Generate a candidate for every i in self.population_size
        """
        return [self.generate_candidate() for i in range(self.population_size)]

    def print_pop_size(self):
        """
        Print the population size
        """
        print(self.population_size)

    def evaluate_population(self, problem):
        [self.fitness(self.population[i], problem) for i in range(self.population_size)]

    def selection(self, problem, k=2):
        """
        * Calculate fitness score/weight for every candidate
        * Randomly select k=2 candidates given selection probabilities
        * Selection is performed with either rank selection or proportional slection
        * return a list of two candidates
        """
        candidates = np.empty(shape=[0, 2])
        self.evaluate_population(problem)
        for candidate in range(len(self.population)):
            candidates = np.append(candidates, [[self.population[candidate], self.selection_probabilities(candidate)]],
                                   axis=0)
        weights = np.array(candidates[:, 1], dtype=float)
        np_candidates = np.array(candidates[:, 0])

        if self.selection_method == "rank":
            sorted_candidates = np.asarray(sorted(candidates, key=lambda l: l[1]))
            np_candidates = np.array(sorted_candidates[:, 0])
            rank_weights = []
            for i in range(1, self.population_size + 1):
                rank_weights.append(i)
            sum_rank_weights = sum(rank_weights)
            for i in range(self.population_size):
                rank_weights[i] = rank_weights[i] / sum_rank_weights
            selected = np.random.choice(np_candidates, p=rank_weights, size=2)
        else:
            selected = np.random.choice(np_candidates, p=weights, size=2)
        return selected

    def selection_probabilities(self, candidate_idx):
        """
        Return candidate fitness score / population fitness score
        """
        prob = (self.fitness_scores[candidate_idx] / self.population_fitness())
        return prob

    def mutation(self, offspring):
        """
        For every child in the offspring apply either a swap, reverse or an insert mutation
        Mutation are only performed for a self.mutation rate
        """
        new_offspring = []
        prob = np.random.random_sample(self.population_size)
        idx = 0
        if self.mutation_type == "swap":
            for child in offspring:
                if prob[idx] > 1 - self.mutation_rate:
                    child = self.swap_mutation(child)
                    new_offspring.append(child)
                else:
                    new_offspring.append(child)
                idx += 1
        elif self.mutation_type == "insert":
            for child in offspring:
                if prob[idx] > 1 - self.mutation_rate:
                    child = self.insert_mutation(child)
                    new_offspring.append(child)
                else:
                    new_offspring.append(child)
                idx += 1
        else:
            for child in offspring:
                if prob[idx] > 1 - self.mutation_rate:
                    child = self.reverse_mutation(child)
                    new_offspring.append(child)
                else:
                    new_offspring.append(child)
                idx += 1

        return new_offspring

    def swap_mutation(self, child):
        """
        Apply a swap mutation, by randomly selecting 2 genes and swapping them.
        """
        rand_idx = np.random.randint(self.dim, size=2)
        temp = child[rand_idx[1]]
        child[rand_idx[1]] = child[rand_idx[0]]
        child[rand_idx[0]] = temp
        return child

    def insert_mutation(self, child):
        """
        * Apply an insert mutation, by randomly selecting an i and a j in the genome.
        * Insert j at position i+1 and delete it's previous index
        """
        child = list(child)
        i_idx = np.random.randint(len(child))
        j_idx = np.random.randint(len(child))

        while i_idx == j_idx:
            i_idx = np.random.randint(len(child))
            j_idx = np.random.randint(len(child))
        child.insert(i_idx + 1, child[j_idx])
        if j_idx > i_idx:
            child.pop(j_idx + 1)
        else:
            child.pop(j_idx)
        return child

    def reverse_mutation(self, child):
        """
        * Apply an reverse mutation, by randomly selecting an i and a j in the genome.
        * Select the subarray between i and j and then reverse this subarray
        """
        child = list(child)
        i_idx = np.random.randint(len(child))
        j_idx = np.random.randint(len(child))

        while i_idx == j_idx or i_idx > j_idx:
            i_idx = np.random.randint(len(child))
            j_idx = np.random.randint(len(child))
        sublist = child[i_idx:j_idx]
        sublist.reverse()
        child[i_idx:j_idx] = sublist
        return child

    def crossover(self, parents):
        """
        For every child in the offspring apply a crossover.
        Either uniformly or using singlepoint crossover
        """
        if (len(parents) != 2):
            raise ValueError("There should be 2 parents")
        offspring = []
        if self.crossover_method == "uniform":
            for i in range(self.population_size):
                offspring.append(self.crossover_uniform(parents))
        else:
            for i in range(self.population_size):
                offspring.append(self.crossover_singlepoint(parents))

        return offspring

    def crossover_singlepoint(self, parents):
        """
        * Given two parents, create a child based on singlepoint crossover
        * Given a random i, create a child with the genes [0,i] of parentA and the genes [1,len(parent)] of parentB
        """
        parentA = np.array(parents[0], dtype=int)
        parentB = np.array(parents[1], dtype=int)
        child = np.empty(shape=[0, 1], dtype=int)
        rand_int = np.random.randint(0, self.dim)

        for i in range(self.dim):
            if i <= rand_int:
                child = np.append(child, parentA[i])
            else:
                child = np.append(child, parentB[i])
        return child

    def crossover_uniform(self, parents):
        """
        * Given two parents, create a child based on a random uniform probability
        * Children are created by a simple treshold. If the probability of a certain index in idx_prob is higher than 0.5,
        * then a gene from parent A is given, else: parent B
        """
        parentA = np.array(parents[0], dtype=int)
        parentB = np.array(parents[1], dtype=int)
        child = np.empty(shape=[0, 1], dtype=int)
        idx_prob = np.random.random_sample(self.dim)
        for i in range(len(idx_prob)):
            if idx_prob[i] > 0.5:
                child = np.append(child, parentA[i])
            else:
                child = np.append(child, parentB[i])
        return child

    def fitness(self, candidate, problem):
        """
        Return the fitness metric
        """
        fitness = problem(candidate)
        self.fitness_scores.append(fitness)
        return fitness

    def population_fitness(self):
        """"
        Calculate the fitness score for the whole population
        Calculation of the fitness score is explained in fitness(self, candidate)
        """
        pop_fitness = sum(self.fitness_scores)
        return pop_fitness

    def __call__(self, problem: ioh.problem.Integer) -> None:
        self.dim = problem.meta_data.n_variables
        self.population = self.generate_population()
        while (problem.state.evaluations < self.max_iterations) \
                and (problem.state.current_best.x != 100.0 or problem.state.current_best.x != problem.objective.x):
            parents = self.selection(problem=problem)
            offspring = self.crossover(parents)
            mutated_offspring = self.mutation(offspring)
            self.population = mutated_offspring
            for cand in range(len(self.population)):
                new_y = self.fitness_scores[cand]
                if new_y > self.y_best:
                    self.y_best = new_y
                    self.x_best = list(self.population[cand])
            self.fitness_scores.clear()
        problem(self.x_best)
        # print('evaluations: ', problem.state.evaluations)
        pass


def main():
    # Set a random seed in order to get reproducible results
    random.seed(42)

    # Instantiate the algoritm, you should replace this with your GA implementation
    algorithm = GeneticAlgorithm(mutation_type="insert")

    # Get a problem from the IOHexperimenter environment
    problem: ioh.problem.Integer = ioh.get_problem(1, 1, 5, "Integer")

    # Run the algoritm on the problem
    algorithm(problem)

    # Inspect the results
    print("Best solution found:")
    print("".join(map(str, problem.state.current_best.x)))
    print("With an objective value of:", problem.state.current_best.y)
    print()


if __name__ == '__main__':
    main()
