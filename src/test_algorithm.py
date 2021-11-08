import argparse
import ioh

TYPE_NAME = "GeneticAlgorithm"
MAX_EVALS = 50_000
DIMENSION = 100 

def test_algoritm(filename: str):
    module_name, _ = filename.rsplit(".", 1)
    module = __import__(module_name, fromlist=[TYPE_NAME]) 
    algorithm = getattr(module, TYPE_NAME)()   

    test_problem = ioh.get_problem(2, 1, DIMENSION, 'Integer')
    algorithm(test_problem)
    assert test_problem.state.optimum_found and test_problem.state.evaluations <= MAX_EVALS, (
        f"Your algorithm did not find the optimum after {MAX_EVALS} iterations."
    )      
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    args = parser.parse_args()
    test_algoritm(args.filename)

