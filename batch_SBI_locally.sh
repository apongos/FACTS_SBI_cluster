#!/bin/bash

num_j=600
for ((jj=581; jj<=$num_j; jj++)); do
    num_iterations=6
    for ((i=1; i<=$num_iterations; i++)); do
    	echo ${jj}_${i}
        python FACTS_with_SBI_2_locally.py --num_sim ${jj}_${i} &
        # python FACTS_with_SBI_2_locally_multiround.py --num_sim ${jj}_${i} &
    done
    wait
done