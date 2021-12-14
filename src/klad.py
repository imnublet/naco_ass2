import ioh
import typing

def objective_function(c0_prime: typing.List[int]) -> float:
    return


problem: ioh.problem.Integer = ioh.problem.wrap_integer_problem(
    objective_function,
    "objective_function_ca_1",
    60,
    ioh.OptimizationType.Maximization,
    ioh.IntegerConstraint([0]*60, [1]*60)
)
# print('variables:', problem.meta_data.n_variables)s
print('objective.y = ', problem.objective.y)

