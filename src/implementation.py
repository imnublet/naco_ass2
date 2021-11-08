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
    '''A skeleton (minimal) implementation of your Genetic Algorithm.'''
    
    def __call__(self, problem: ioh.problem.Integer) -> None:
        pass
    
            
def main():
    # Set a random seed in order to get reproducible results
    random.seed(42)

    # Instantiate the algoritm, you should replace this with your GA implementation 
    algorithm = RandomSearch()

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
