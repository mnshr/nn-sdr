#!/bin/bash

# Catch termination signal `SIGINT` and invoke `user_interupt`
trap user_interupt SIGINT

# Reports an user interupt and exits the simulation script (do not continue next
# statement, allows to exit bash loop with `^C`)
user_interupt() {
    echo -e '\nUser Interrupt -> stopping simulation\n'
    exit
}

command="Rscript simulations_binary.R --reps=100 --dataset=B1 --hidden_units=1024 --epochs=5,10"
echo -e "\n$command"
time eval "$command"
command="Rscript simulations_binary.R --reps=100 --dataset=B2"
echo -e "\n$command"
time eval "$command"
command="Rscript simulations_binary.R --reps=100 --dataset=B3 --hidden_units=1024,128,32 --epochs=10,200"
echo -e "\n$command"
time eval "$command"
command="Rscript simulations_binary.R --reps=100 --dataset=B4 --hidden_units=1024,128,32 --epochs=10,200"
echo -e "\n$command"
time eval "$command"
