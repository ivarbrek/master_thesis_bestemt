from src.models.basic_model import BasicModel
from src.read_problem_data import ProblemData


# problem_data = ProblemData('../../data/input_data/small_testcase_one_vessel.xlsx')
# problem_data = ProblemData('../../data/input_data/small_testcase.xlsx')
# problem_data = ProblemData('../../data/input_data/medium_testcase.xlsx')
# problem_data = ProblemData('../../data/input_data/large_testcase.xlsx')
problem_data = ProblemData('../../data/input_data/larger_testcase.xlsx')
# problem_data = ProblemData('../../data/input_data/larger_testcase_4vessels.xlsx')

extensions = False  # extensions include inventory reward and soft time windows only, precedence is always included
problem_data.soft_tw = extensions
model = BasicModel(problem_data, extended_model=extensions)
model.solve(time_limit=30)
model.print_result()
