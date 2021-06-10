module load Python/3.8.6-GCCcore-10.2.0
module load gurobi/9.1

cd ../../storage/users/ivarbrek/master_thesis/

python src/alns/alns.py data/input_data/performance_testing/performance-5-2-15-h-1p-0.xlsx 1 None 10000
python src/alns/alns.py data/input_data/performance_testing/performance-5-2-15-h-5p-0.xlsx 1 None 10000
python src/alns/alns.py data/input_data/performance_testing/performance-5-2-15-l-1p-0.xlsx 1 None 10000
python src/alns/alns.py data/input_data/performance_testing/performance-5-2-15-l-5p-0.xlsx 1 None 10000

python src/alns/alns.py data/input_data/performance_testing/performance-5-2-18-h-1p-0.xlsx 1 None 10000
python src/alns/alns.py data/input_data/performance_testing/performance-5-2-18-h-5p-0.xlsx 1 None 10000
python src/alns/alns.py data/input_data/performance_testing/performance-5-2-18-l-1p-0.xlsx 1 None 10000
python src/alns/alns.py data/input_data/performance_testing/performance-5-2-18-l-5p-0.xlsx 1 None 10000