#!/bin/bash


ROOT_DIR=$PWD

WALL_TIME="01:00:00"

NK_OPTIC=3

cd ${ROOT_DIR}

for jdir in OPTIC_*; do

    cd $jdir
    
    # find "." -type f ! -name "*.fdf" ! -name "*.psf" ! -name "*.XV" ! -name "*.DM" ! -name "*.sh" -delete

    rm -rf *.err *.out

    mv input.fdf optic.fdf

    # sed -i '/SaveRho/d' optic.fdf
    sed -i "s|$(grep -w  SCF.MustConverge  optic.fdf)|SCF.MustConverge    F|g" optic.fdf
    sed -i "s|$(grep -w  MaxSCFIterations  optic.fdf)|MaxSCFIterations    1|g" optic.fdf
    sed -i "s|$(grep -w Diag.ParallelOverK optic.fdf)|Diag.ParallelOverK  F|g" optic.fdf
    sed -i "s|$(grep -w ElectronicTemperature optic.fdf)|ElectronicTemperature 10 K|g" optic.fdf
    echo " " >> optic.fdf

    echo "###OPTICAL PARAMETERS###" >> optic.fdf
    echo "OpticalCalculation T" >> optic.fdf
    echo "Optical.Energy.Minimum 0 eV" >> optic.fdf
    echo "Optical.Energy.Maximum 10 eV" >> optic.fdf
    echo "Optical.Broaden 0.05 eV" >> optic.fdf
    echo "Optical.PolarizationType polarized" >> optic.fdf
    #echo "Optical.NumberOfBands 2000" >> optic.fdf

    echo " " >> optic.fdf

    echo "%block  Optical.Mesh" >> optic.fdf
    echo "${NK_OPTIC} ${NK_OPTIC} 1" >> optic.fdf
    echo "%endblock  Optical.Mesh" >> optic.fdf

    echo " " >> optic.fdf

    echo "%block  Optical.Vector" >> optic.fdf
    echo "1.0 0.0 0.0" >> optic.fdf
    echo "%endblock  Optical.Vector" >> optic.fdf

    sed -i 's/^#SBATCH --job-name=.*/#SBATCH --job-name=OPTIC/' "run.sh"
    sed -i 's/^#SBATCH --nodes=.*/#SBATCH --nodes=1/' "run.sh"
    sed -i 's/^#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=32/' "run.sh"
    sed -i 's/^#SBATCH --cpus-per-task=.*/#SBATCH --cpus-per-task=4/' "run.sh"
    sed -i "s/^#SBATCH --time=.*/#SBATCH --time=${WALL_TIME}/" "run.sh"
    sed -i '/^srun siesta */c\srun siesta < optic.fdf > output' "run.sh"

    sbatch run.sh

    cd ..

done
