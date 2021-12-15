#Authors: Sayf El Kaddouri, Luca Goeman & Mustafa Darwesh
#How to run this file:
#Make sure to install the requirements with 'pip3 install -r requirements'
import typing
import ioh
import csv
import math
import numpy as np

from implementation import RandomSearch
from implementation import GeneticAlgorithm

CURR_LINE = 0

class CellularAutomata:
    '''Skeleton CA, you should implement this.'''


    def __init__(self, rule_number: int, neighborhood_radius = 1, length = 10, edge=0, base = 2):
        """
        Constructor function
        """
        self.state = []
        self.rule_number = int(rule_number)
        self.radius = int(neighborhood_radius)
        self.length = length
        self.edge = edge
        self.base = int(base)
        self.possible_rules = int(math.pow(self.base,2*self.radius+1))
        pass

    def rule_to_base_k(self):
        """
        Convert a rule to a certain base-k representation,
        and then save it to a list
        """
        rule_converted = np.base_repr(self.rule_number, base=self.base)
        rule_list = [int(x) for x in str(rule_converted)]
        while len(rule_list) != self.possible_rules:
            rule_list.insert(0,0)
        return rule_list

    def convert_to_binary(self, num, length):
        """
        Convert a base 10 number to a base 2 representation, given a certain length for the bitstring
        """
        bitstring = np.binary_repr(num, width=length)
        print('type of bitstring: ', type(bitstring))
        return bitstring

    def convert_to_other_base(self, num, length):
        """
        Convert a base 10 number to a base k representation, given a certain length for the output
        """
        converted_num = np.base_repr(num, base=self.base)
        while len(converted_num) != length:
            converted_num_list = [int(x) for x in str(converted_num)]
            converted_num_list.insert(0,0)
            converted_str = ''
            converted_num = converted_str.join(map(str, converted_num_list))
        return converted_num

    def possible_neighborhood_states(self):
        """
        This function returns a list with the possible states, that will lead to a non-zero cell in the next time step.
        ie, if the list contains 101 for base 2, it means if for a certain cell C the neighborhood looks like 101 =>
        Cell C will be 1 in timestep t+1
        """
        evaluate_rule = self.rule_to_base_k()
        neighborhood_states = []
        neighborhood_state = self.possible_rules
        next_gen = []
        for i in range(0, self.possible_rules):
            if evaluate_rule[i]:
                neighborhood_state = self.possible_rules-i-1
                neighborhood_state_bits = self.convert_to_other_base(neighborhood_state, length = 2 * self.radius+1)
                neighborhood_states.append(neighborhood_state_bits)
                next_gen.append(evaluate_rule[i])
        return neighborhood_states, next_gen

    def compare_state_rule(self, neighborhood, cell):
        """
        Returns whether a neighborhood is contaminated in possible_neighborhood_states()
        If this is the case return the value of a cell in the next gen
        Else return 0
        """
        rule_states, next_gen = self.possible_neighborhood_states()
        if neighborhood in rule_states:
            idx = rule_states.index(neighborhood)
            return next_gen[idx]
        else:
            return 0

    def get_neighbors(self, state):
        """
        Given a certain state S
        for every cell in S lookup the neighbors
        For example, given S = [1,0,0,0,0,0,0,0,1,1], base = 2 and radius = 1
        This would return the following dict for cell 8:
        8: {'left': 0, 'cell': 1, 'right': 1},
        """
        left = 0
        right = 0
        neighborhood = {}
        neighborhood_of_cell = {}

        for cell in range(len(state)):
            if cell == 0:
                left = 0
            else: left = state[cell-1]
            if cell == len(state)-1:
                right = 0
            else: right = state[cell+1]

            neighborhood_of_cell = {
                "left" : left,
                "cell" : state[cell],
                "right" : right
            }
            neighborhood[cell] = neighborhood_of_cell
        return neighborhood

    def step(self, state):
        """
        Do a timestep in the cellular automaton
        Given a certain state perform a rule-update (transition)
        Return the new state as a list
        """
        next_state = []
        neighborhood_dict = self.get_neighbors(state)
        for cell in neighborhood_dict:
            left = neighborhood_dict[cell]["left"]
            cell_ = neighborhood_dict[cell]["cell"]
            right = neighborhood_dict[cell]["right"]
            neighbors_bits = f"{str(left)}{str(cell_)}{str(right)}"

            next_gen = self.compare_state_rule(neighbors_bits, cell)
            if next_gen:
                next_state.append(next_gen)
            else:
                next_state.append(0)
        return next_state

    def __call__(self, c0: typing.List[int], t: int) -> typing.List[int]:
        '''Evaluate for T timesteps. Return Ct for a given C0.'''
        self.state = c0
        for i in range(t):
            self.state = self.step(self.state)
        return self.state


def hamming_dist(list1, list2, quadratic=None):
    """
    Calculate inverted hemming distance
    Return this as a percentage => [0,100]
    """
    assert len(list1) == len(list2), "The length of list1 is not equal to the length of list2"

    same_count = 0
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            same_count += 1
    if quadratic:
        hamming_dist_perc = math.pow(same_count,2)/math.pow(len(list1),2)
    else:
        hamming_dist_perc = same_count/len(list1)
    return hamming_dist_perc*100

def change_currline(line):
    global CURR_LINE
    CURR_LINE = line

def read_input(line_idx=0):
    config_list = []
    with open('ca_input.csv', newline='',) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(csvfile)
        for line in reader:
            config_list.append(line)
    selected_line = config_list[line_idx]

    # Parsing the ct into a list
    ct = selected_line[3].strip('],[').split(',')
    ct = list(map(int, ct))

    # Return base_k, rule number, Timesteps, Ct respectively
    return selected_line[0], selected_line[1], selected_line[2], ct


def objective_function_1(c0_prime: typing.List[int]) -> float:
    """
    This functions calculates the fitness score based on some inverted hamming distance measure
    If the target is reached fitness = 100.0
    """

    base_k, rule, t, ct = get_problem(PROBLEM_LINE)
    ca = CellularAutomata(rule_number=rule, base=base_k, length=len(ct))
    ct_prime = ca(c0_prime, t)
    similarity = hamming_dist(ct_prime, ct)
    return similarity


def objective_function_2(c0_prime: typing.List[int]) -> float:
    """
    This functions calculates the quadratic fitness score
    If the target is reached fitness = 100.0
    """

    base_k, rule, t, ct = get_problem(PROBLEM_LINE)
    ca = CellularAutomata(rule_number=rule, base=base_k, length=len(ct))
    ct_prime = ca(c0_prime, t)
    similarity = hamming_dist(ct_prime, ct, quadratic=True)
    return similarity


def get_problem(line_idx=0):
    base_k, rule, t, ct = read_input(line_idx=line_idx) # Given by the sup. material
    t = int(t)
    base_k = int(base_k)
    rule = int(rule)

    return base_k, rule, t, ct


if __name__ == '__main__':
    PROBLEM_LINE = 0
    base_k, rule, t, ct = get_problem(line_idx=PROBLEM_LINE)
    prob_len = len(ct)

    for obj in [objective_function_1, objective_function_2]:
        problem: ioh.problem.Integer = ioh.problem.wrap_integer_problem(
            obj,
            f'{obj}',
            60,
            ioh.OptimizationType.Maximization,
            ioh.IntegerConstraint([0]*60, [2]*60)
        )
        for line in range(4):
            for select in ['proportional', 'rank']:
                for cross_method in ['uniform', 'single_point']:
                    for mut in ['reverse', 'swap', 'insert']:
                        for run in range(5):
                            # All parameters below can be adjusted for different results
                            pop = 20
                            seed = np.random.randint(1000)
                            mut_rate = 0.3
                            # Uncomment below and replace all ga instances with randomSearch
                            # randomSearch = RandomSearch()
                            ga = GeneticAlgorithm(mutation_type=mut, pop_size=pop, base=base_k, seed=seed,
                                                  mutation_rate=mut_rate, crossover=cross_method, selection=select)
                            logger = ioh.logger.Analyzer(store_positions=True, algorithm_name="genetic algorithm",
                                                         algorithm_info=f'select: {select},'
                                                                        f' crossover:{cross_method}, mut_type: {mut},'
                                                                        f' using {obj} as obj function')
                            # logger.add_run_attributes(obj, ['seed', 'mutation_rate',
                            #                                 'crossover', 'selection', 'mutation_type', 'selection'])
                            problem.attach_logger(logger)
                            ga(problem)
                            print("Best solution found:")
                            print("".join(map(str, problem.state.current_best.x)))
                            print("With an objective value of:", problem.state.current_best.y)
                            problem.reset()
                            print(f'run {run} completed for problem {PROBLEM_LINE} and the following config:')
                            print(f'select: {select}, crossover:{cross_method}, mut_type: {mut} ')
            PROBLEM_LINE = PROBLEM_LINE+1
            change_currline(PROBLEM_LINE)
    print('complete')
    pass