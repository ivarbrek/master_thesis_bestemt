module load Python/3.8.6-GCCcore-10.2.0
module load gurobi/9.1

cd ../../storage/users/ivarbrek/master_thesis/

python src/models/ffprp_model.py data/input_data/performance_testing/performance-5-2-20-l-1p-0.xlsx 3600
python src/models/ffprp_model.py data/input_data/performance_testing/performance-5-2-20-l-5p-0.xlsx 3600

python src/models/ffprp_model.py data/input_data/performance_testing/performance-5-2-25-h-1p-0.xlsx 3600
python src/models/ffprp_model.py data/input_data/performance_testing/performance-5-2-25-h-5p-0.xlsx 3600
python src/models/ffprp_model.py data/input_data/performance_testing/performance-5-2-25-l-1p-0.xlsx 3600
python src/models/ffprp_model.py data/input_data/performance_testing/performance-5-2-25-l-5p-0.xlsx 3600

