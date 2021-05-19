echo "Master ALNS script starting"
screen -X screen ssh compute-4-37 sh alns_run1.sh
screen -X screen ssh compute-4-38 sh alns_run2.sh
screen -X screen ssh compute-4-39 sh alns_run3.sh
screen -X screen ssh compute-4-40 sh alns_run4.sh