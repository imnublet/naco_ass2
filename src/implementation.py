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


    def __init__(self, pop_size=22, dim=10, seed=1, base=2, mutation_type = "swap"):
        super().__init__(max_iterations=500000)
        self.mutation_type = mutation_type
        self.dim = dim
        self.seed = seed
        self.base = base
        self.population_size = pop_size
        np.random.seed(self.seed)
        self.population = self.generate_population()
        self.y_best = 0.0
        self.x_best : list[int] = []
        self.fitness_scores = []


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
        * Randomly select k candidates given selection probabilities
        * return a list of two candidates
        """
        candidates = np.empty(shape=[0, 2])
        self.evaluate_population(problem)
        for candidate in range(len(self.population)):
            candidates = np.append(candidates, [[self.population[candidate], self.selection_probabilities(candidate)]], axis=0)
        weights = np.array(candidates[:,1], dtype=float)
        np_candidates = np.array(candidates[:,0])
        selected = np.random.choice(np_candidates, p = weights, size = 2)

        return selected


    def selection_probabilities(self, candidate_idx):
        """
        Return candidate fitness score / population fitness score
        """
        prob = (self.fitness_scores[candidate_idx] / self.population_fitness())
        return prob


    def mutation(self, offspring):
        """
        For every child in the offspring apply either a swap mutatation or an insert mutation
        """
        new_offspring = []
        if self.mutation_type == "swap":
            for child in offspring:
                # if prob[idx] > 0.5:
                child = self.swap_mutation(child)
                new_offspring.append(child)
        elif self.mutation_type == "insert":
            for child in offspring:
                child = self.insert_mutation(child)
                new_offspring.append(child)
        else:
            for child in offspring:
                child = self.reverse_mutation(child)
                new_offspring.append(child)

        return new_offspring


    def swap_mutation(self, child):
        """
        Apply a swap mutation, by randomly selecting 2 genes and swapping them.
        """
        rand_idx = np.random.randint(self.dim,size=2)
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
        child.insert(i_idx+1, child[j_idx])
        if j_idx > i_idx:
            child.pop(j_idx+1)
        else:
            child.pop(j_idx)
        return child

    def reverse_mutation(self, child):
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
        """
        if(len(parents) != 2):
            raise ValueError("There should be 2 parents")
        offspring = []
        for i in range(self.population_size):
            offspring.append(self.crossover_operator(parents))

        return offspring


    def crossover_operator(self, parents):
        """
        * Given two parents, create a child based on a random uniform probability
        * Children are created by a simple treshold. If the probability of a certain index in idx_prob is higher than 0.5,
        * then a gene from parent A is given, else: parent B
        * This is done for n = self.population_size
        """
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
        # assert problem.state.evaluations > self.max_iterations
        # print('best solution', problem.state.current_best.x)
        print('objective:',problem.objective.x )
        # while (problem.state.evaluations < self.max_iterations) and (problem.state.current_best.x != problem.objective.x):
        while (problem.state.evaluations < self.max_iterations) and (problem.state.current_best.y != problem.objective.y):
            # print('test 3')
            parents = self.selection(problem=problem)
            offspring = self.crossover(parents)
            mutated_offspring = self.mutation(offspring)
            self.population = mutated_offspring
            for cand in range(len(self.population)):
                new_y = self.fitness_scores[cand]
                if new_y > self.y_best:
                    self.y_best = new_y
                    self.x_best = list(self.population[cand])
                    # print('best x: ', self.x_best, 'with y value of:', self.y_best)
                    print('sum of 1s:',sum(self.x_best))
            self.fitness_scores.clear()
        problem(self.x_best)
        print('evaluations: ', problem.state.evaluations)
        # return problem.state.current_best
        pass








# def main():
#     # Set a random seed in order to get reproducible results
#     random.seed(42)
#
#     # Instantiate the algoritm, you should replace this with your GA implementation
#     algorithm = GeneticAlgorithm(mutation_type="insert")
#
#     # Get a problem from the IOHexperimenter environment
#     problem: ioh.problem.Integer = ioh.get_problem(1, 1, 5, "Integer")
#
#     # Run the algoritm on the problem
#     algorithm(problem)
#
#     # Inspect the results
#     print("Best solution found:")
#     print("".join(map(str, problem.state.current_best.x)))
#     print("With an objective value of:", problem.state.current_best.y)
#     print()
#
#
# if __name__ == '__main__':
#     main()
