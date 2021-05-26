import sys
import os
from random import random, sample

sys.path.append(os.getcwd())

from numpy.lib import math

from src.alns.solution import ProblemDataExtended, ProblemData
from src.models.production_model import ProductionModel
from src.alns.production_problem_heuristic import ProductionProblemHeuristic, ProductionProblem

import json
from typing import Tuple, Dict
import pandas as pd
import argparse
import random as random
from pyomo.opt import TerminationCondition


def load_demands(inst_id: str) -> Dict[str, str]:
    filepath = 'data/input_data/demands/' + inst_id + '.txt'
    with open(filepath) as file:
        d = json.load(file)
    return d


def sample_from_dict(d: Dict, sample: int = 20):
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))


def parse_dict_string(string: str) -> Dict[Tuple[str, str, int], int]:
    d: Dict[Tuple[str, str, int], int] = {}
    string = string.strip("{}(")
    l = string.split(", (")
    for kv in l:
        k, quantity = kv.split("): ")[0], int(kv.split(": ")[1])
        ks = k.split(", ")
        factory, product, time = ks[0].strip('\''), ks[1].strip('\''), int(ks[2])
        d[(factory, product, time)] = quantity
    return d


class ProductionCheckExecutor:
    prbl: ProblemData
    prbl_ext: ProblemDataExtended
    prbl_prod: ProductionProblem
    pp_model: ProductionModel
    production_heuristic: ProductionProblemHeuristic
    out_filepath: str
    inst_id: str
    demands: Dict[Tuple[str, str, int], int]

    def __init__(self,
                 prbl_filepath: str,
                 inst_id: str,
                 init_demands: Dict[Tuple[str, str, int], int],  # demands: Dict[str, Dict[Tuple[str, str, int], int]]
                 ):
        self.prbl_ext = ProblemDataExtended(prbl_filepath + inst_id + '.xlsx')
        self.prbl = ProblemData(file_path=prbl_filepath + inst_id + '.xlsx')
        self.prbl_prod = ProductionProblem(base_problem_data=self.prbl_ext)
        self.inst_id: str
        self.pp_model = ProductionModel(prbl=self.prbl_ext, demands=init_demands, inventory_reward_extension=False)
        self.production_heuristic = ProductionProblemHeuristic(prbl=self.prbl_prod)

    def solve_production_exact(self, demand: Dict[Tuple[str, str, int], int], time_limit: int):
        cost = self.pp_model.get_production_cost(verbose=True, time_limit=time_limit, demand_dict=demand)
        feasible = True if cost < math.inf else False
        try:
            if self.pp_model.results.solver.termination_condition in [TerminationCondition.infeasible]:
                lb = -1
                ub = -1
                mip_gap = -1
                time = round(self.pp_model.results.solver.time, 2)
            else:
                print(f"Termination condition: {self.pp_model.results.solver.termination_condition}")
                lb = self.pp_model.results.Problem._list[0].lower_bound
                ub = self.pp_model.results.Problem._list[0].upper_bound
                mip_gap = float(((ub - lb) / ub) * 100)
                time = round(self.pp_model.results.solver.time, 2)
        except (ValueError, AttributeError, TypeError) as e:
            print(f"Error: {e}; setting default values...")
            lb = -1
            ub = -1
            mip_gap = -1
            time = time_limit

        return feasible, cost, time, lb, ub, mip_gap

    def solve_production_heuristic(self, demand: Dict[Tuple[str, str, int], int]):
        cost, time = self.production_heuristic.get_cost(demands=demand)
        feasible = True if cost > 0 else False
        return feasible, cost, time

    def write_comparison_sheet(self, excel_writer, id, demand):
        e120_feasible, e120_cost, e120_time, e120_lb, e120_ub, e120_mip = self.solve_production_exact(demand,
                                                                                                      time_limit=120)
        e30_feasible, e30_cost, e30_time, e30_lb, e30_ub, e30_mip = self.solve_production_exact(demand,
                                                                                                time_limit=30)
        h_feasible, h_cost, h_time = self.solve_production_heuristic(demand=demand)

        ind = pd.Index(['exact_120', 'exact_30', 'heuristic'])
        df: pd.DataFrame = pd.DataFrame(index=ind)

        df['feasible'] = df.index.to_series().map({'exact_120': e120_feasible, 'exact_30': e30_feasible,
                                                   'heuristic': h_feasible})
        df['cost'] = df.index.to_series().map({'exact_120': e120_cost, 'exact_30': e30_cost,
                                               'heuristic': h_cost})
        df['time'] = df.index.to_series().map({'exact_120': e120_time, 'exact_30': e30_time,
                                               'heuristic': h_time})
        df['mip_gap'] = df.index.to_series().map({'exact_120': e120_mip, 'exact_30': e30_mip,
                                                  'heuristic': 'na'})
        df['optimal'] = df.index.to_series().map({'exact_120': 0 <= e120_mip < 0.01, 'exact_30': 0 <= e30_mip < 0.01,
                                                  'heuristic': h_cost == e120_cost and 0 <= e120_mip < 0.01})
        df['lower_bound'] = df.index.to_series().map({'exact_120': e120_lb, 'exact_30': e30_lb,
                                                      'heuristic': 'na'})
        df['upper_bound'] = df.index.to_series().map({'exact_120': e120_ub, 'exact_30': e30_ub,
                                                      'heuristic': 'na'})
        df = df.transpose()
        df.to_excel(excel_writer, sheet_name=str(id))


if __name__ == '__main__':

    # inst_ids = ['performance-3-1-10-l-0', 'performance-3-1-10-l-1', 'performance-3-1-10-l-2',
    #             'performance-3-1-10-h-0', 'performance-3-1-10-h-1', 'performance-3-1-10-h-2',
    #             'performance-3-1-15-l-0', 'performance-3-1-15-l-1', 'performance-3-1-15-l-2',
    #             'performance-3-1-15-h-0', 'performance-3-1-15-h-1', 'performance-3-1-15-h-2',
    #             'performance-3-1-20-l-0', 'performance-3-1-20-l-1', 'performance-3-1-20-l-2',
    #             'performance-3-1-20-h-0', 'performance-3-1-20-h-1', 'performance-3-1-20-h-2',
    #             'performance-3-1-40-l-0', 'performance-3-1-40-l-1', 'performance-3-1-40-l-2',
    #             'performance-3-1-40-h-0', 'performance-3-1-40-h-1', 'performance-3-1-40-h-2',
    #             'performance-3-1-60-l-0', 'performance-3-1-60-l-1', 'performance-3-1-60-l-2',
    #             'performance-3-1-60-h-0', 'performance-3-1-60-h-1', 'performance-3-1-60-h-2',
    #             'performance-5-2-10-l-0', 'performance-5-2-10-l-1', 'performance-5-2-10-l-2',
    #             'performance-5-2-10-h-0', 'performance-5-2-10-h-1', 'performance-5-2-10-h-2',
    #             'performance-5-2-15-l-0', 'performance-5-2-15-l-1', 'performance-5-2-15-l-2',
    #             'performance-5-2-15-h-0', 'performance-5-2-15-h-1', 'performance-5-2-15-h-2',
    #             'performance-5-2-20-l-0', 'performance-5-2-20-l-1', 'performance-5-2-20-l-2',
    #             'performance-5-2-20-h-0', 'performance-5-2-20-h-1', 'performance-5-2-20-h-2',
    #             'performance-5-2-40-l-0', 'performance-5-2-40-l-1', 'performance-5-2-40-l-2',
    #             'performance-5-2-40-h-0', 'performance-5-2-40-h-1', 'performance-5-2-40-h-2',
    #             'performance-5-2-60-l-0', 'performance-5-2-60-l-1', 'performance-5-2-60-l-2',
    #             'performance-5-2-60-h-0', 'performance-5-2-60-h-1', 'performance-5-2-60-h-2']
    prbl_filepath = 'data/input_data/'
    out_filepath_base = 'data/output_data/'

    parser = argparse.ArgumentParser(description='run production methods')
    parser.add_argument('inst_id', type=str, help='instance id')
    args = parser.parse_args()

    inst_id = args.inst_id
    print(f"Starting on instance {inst_id}...")
    out_filepath = out_filepath_base + inst_id + ".xlsx"

    # Get demands
    demands = load_demands(inst_id=inst_id)
    init_demands = demands['0']
    sample = min(len(demands.keys()), 20)
    demands = sample_from_dict(demands, sample=sample)

    num_production_calls = len(demands.keys())
    if num_production_calls == 0:
        print(f"Production sub-problem is never solved.")
    excel_writer = pd.ExcelWriter(out_filepath, engine='openpyxl', mode='w', options={'strings_to_formulas': False})
    comp = ProductionCheckExecutor(prbl_filepath=prbl_filepath, inst_id=inst_id,
                                   init_demands=parse_dict_string(init_demands))

    it = 0
    for j in demands.keys():
        it += 1
        print(f"Handling production problem call number {it} for instance {inst_id} "
              f"out of a total of {num_production_calls} calls...")
        demand = parse_dict_string(demands[j])
        comp.write_comparison_sheet(excel_writer=excel_writer, id=it-1, demand=demand)

    excel_writer.close()
