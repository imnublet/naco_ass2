import typing
import ioh

from implementation import RandomSearch


class CellularAutomata:
    '''Skeleton CA, you should implement this.'''
    
    def __init__(self, rule_number: int):
        pass

    def __call__(self, c0: typing.List[int], t: int) -> typing.List[int]:
        '''Evaluate for T timesteps. Return Ct for a given C0.'''
        pass


def objective_function(c0_prime: typing.List[int]) -> float:
    '''Skeleton objective function. You should implement a method
    which computes a similarity measure between c0_prime a suggested by your
    GA, with the true c0 state for the ct state given in the sup. material. '''
    
    ct, rule, t = None, None, None # Given by the sup. material 

    ca = CellularAutomata(rule)
    ct_prime = ca(c0_prime, t)
    similarity = 0.0 # You should implement this

    return similarity

        
def example():
    '''An example of wrapping a objective function in ioh and collecting data
    for inputting in the analyzer.'''
    
    algorithm = RandomSearch()

    # Wrap objective_function as an ioh problem
    problem = ioh.problem.wrap_integer_problem(
            objective_function,
            "objective_function_ca_1",
            60, 
            ioh.OptimizationType.Maximization,
            ioh.IntegerConstraint([0]*60, [1]*60)
    )
    # Attach a logger to the problem
    logger = ioh.logger.Analyzer(store_positions=True)
    problem.attach_logger(logger)

    # run your algoritm on the problem
    algorithm(problem)



if __name__ == '__main__':
    example()
