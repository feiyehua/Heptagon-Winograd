#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Please choose an operation:"
    echo "1. Profile"
    echo "2. Timer"
    echo "3. Verify"
    read choice
else
    choice=$1
fi

case $choice in
1) sbatch ./slurm_scripts/profile.sh ;;
2) sbatch ./slurm_scripts/timer.sh ;;
3) sbatch ./slurm_scripts/verify.sh ;;
*) echo "Invalid choice, please try again." ;;
esac
