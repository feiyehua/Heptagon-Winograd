#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Please choose an operation:"
    echo "1. Profile"
    echo "2. Timer"
    echo "3. Verify"
    echo "4. Small Verify"
    echo "5. Memory"
    read choice
else
    choice=$1
fi

case $choice in
1) sbatch ./slurm_scripts/profile.sh ;;
2) sbatch ./slurm_scripts/timer.sh ;;
3) sbatch ./slurm_scripts/verify.sh ;;
4) sbatch ./slurm_scripts/small_verify.sh ;;
5) sbatch ./slurm_scripts/memory.sh ;;
*) echo "Invalid choice, please try again." ;;
esac
