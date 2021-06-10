module load Python/3.8.6-GCCcore-10.2.0
module load gurobi/9.1

cd ../../storage/users/ivarbrek/master_thesis/


# 3-1 instances
python src/alns/alns.py data/input_data/performance_testing/performance-3-1-8-h-1p-0.xlsx 1 None 10000
python src/alns/alns.py data/input_data/performance_testing/performance-3-1-8-h-5p-0.xlsx 1 None 10000
python src/alns/alns.py data/input_data/performance_testing/performance-3-1-8-l-1p-0.xlsx 1 None 10000
python src/alns/alns.py data/input_data/performance_testing/performance-3-1-8-l-5p-0.xlsx 1 None 10000

python src/alns/alns.py data/input_data/performance_testing/performance-3-1-10-h-1p-0.xlsx 1 None 10000
python src/alns/alns.py data/input_data/performance_testing/performance-3-1-10-h-5p-0.xlsx 1 None 10000
python src/alns/alns.py data/input_data/performance_testing/performance-3-1-10-l-1p-0.xlsx 1 None 10000
python src/alns/alns.py data/input_data/performance_testing/performance-3-1-10-l-5p-0.xlsx 1 None 10000