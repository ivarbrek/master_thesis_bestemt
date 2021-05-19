echo "Master gurobi script starting"
screen -X screen ssh compute-4-37 sh gurobi_run1.sh
screen -X screen ssh compute-4-38 sh gurobi_run2.sh
screen -X screen ssh compute-4-39 sh gurobi_run3.sh
screen -X screen ssh compute-4-40 sh gurobi_run4.sh