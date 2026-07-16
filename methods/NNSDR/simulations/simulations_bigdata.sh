#!/bin/bash

# Catch termination signal `SIGINT` and invoke `user_interupt`
trap user_interupt SIGINT

# Reports an user interupt and exits the simulation script (do not continue next
# statement, allows to exit bash loop with `^C`)
user_interupt() {
    echo -e '\nUser Interrupt -> stopping simulation\n'
    exit
}

# # Simulation for big data with `p` proportional to `sqrt(n)`
# for ds in 6 8; do
#     command="Rscript simulations_bigdata.R --reps=10 --run_mave=FALSE --run_cve=FALSE --dataset=$ds --n=1000 --p=32 --epochs=200,400"
#     echo -e "\n$command"
#     time eval "$command"
#     command="Rscript simulations_bigdata.R --reps=10 --run_mave=FALSE --run_cve=FALSE --dataset=$ds --n=4000 --p=63 --epochs=100,200"
#     echo -e "\n$command"
#     time eval "$command"
#     command="Rscript simulations_bigdata.R --reps=10 --run_mave=FALSE --run_cve=FALSE --dataset=$ds --n=16000 --p=126 --epochs=50,100"
#     echo -e "\n$command"
#     time eval "$command"
#     command="Rscript simulations_bigdata.R --reps=10 --run_mave=FALSE --run_cve=FALSE --dataset=$ds --n=64000 --p=253 --epochs=25,50"
#     echo -e "\n$command"
#     time eval "$command"
#     command="Rscript simulations_bigdata.R --reps=10 --run_mave=FALSE --run_cve=FALSE --dataset=$ds --n=256000 --p=506 --epochs=12,25"
#     echo -e "\n$command"
#     time eval "$command"
# done

# Simulation for big data with `p` proportional to `n` (note: for the base case
# of `n = 1000`, `p = 32` see above)
# For i = 1, ..., 4 the sample size `n = 1000 * 4^i`, number of predictors
# `p = 32 * 4^i` and the training epochs `epochs ~ (200, 400) * 1.5^(-i)`
for ds in 6 8; do
    command="Rscript simulations_bigdata.R --reps=10 --run_mave=TRUE --run_cve=TRUE --dataset=$ds --n=4000 --p=128 --epochs=133,266"
    echo -e "\n$command"
    time eval "$command"
    command="Rscript simulations_bigdata.R --reps=10 --run_mave=FALSE --run_cve=FALSE --dataset=$ds --n=16000 --p=512 --epochs=88,176"
    echo -e "\n$command"
    time eval "$command"
    command="Rscript simulations_bigdata.R --reps=10 --run_mave=FALSE --run_cve=FALSE --dataset=$ds --n=64000 --p=2048 --epochs=59,118"
    echo -e "\n$command"
    time eval "$command"
    command="Rscript simulations_bigdata.R --reps=10 --run_mave=FALSE --run_cve=FALSE --dataset=$ds --n=256000 --p=8192 --epochs=39,78"
    echo -e "\n$command"
    time eval "$command"
done
