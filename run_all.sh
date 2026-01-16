#!/bin/bash


# Get the current date/time info as a prefix to be used for a running log file
time_stamp=$(date +"%Y-%m-%d_%H-%M-%S") 

for dft_task in "SGLPT"; do
    for nanopore_name in "GRAPHENE_NANOPORE"; do
        bash run_dft.sh --dft_task $dft_task --nanopore_name $nanopore_name --wall_time "00:30:00" --time_stamp ${time_stamp}
    done
done

for dft_task in "SGLPT"; do
    for molecule_name in "ALA"; do
        bash run_dft.sh --dft_task $dft_task --molecule_name $molecule_name --wall_time "00:30:00" -time_stamp ${time_stamp}
    done
done


for dft_task in "SGLPT"; do
    for nanopore_name in "GRAPHENE_NANOPORE"; do
        for molecule_name in "ALA"; do
            for theta_angle in $(seq 0 30 0); do
                for phi_angle in $(seq 0 30 0); do
                    for psi_angle in $(seq 0 30 0); do
                        bash run_dft.sh --dft_task $dft_task --nanopore_name $nanopore_name --molecule_name $molecule_name --theta_angle $theta_angle --phi_angle $phi_angle --psi_angle $psi_angle --wall_time "00:30:00" -time_stamp ${time_stamp}
                    done
                done
            done
        done
    done
done
