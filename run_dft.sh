#!/bin/bash

###############################################################

# Function to calculate column width
calculate_column_width() {
    column_width=0
    for value in "$@"; do
        length=${#value}
        if ((length > column_width)); then
            column_width=$length
        fi
    done
    # # Add some padding for better readability
    column_width=$((column_width + 2))
    echo "$column_width"
}

###############################################################

# Function to print horizontal line
print_horizontal_line() {
    column_width_new=$((column_width + 2))
    line="+"
    for ((i = 1; i <= 7; i++)); do # 10, 13
        line+="$(printf "%-${column_width_new}s" "" | tr ' ' -)+"
    done
    printf "%s\n" "$line"
}

###############################################################

# Function to print table row
print_table_row() {
    values=("$@")
    line="|"
    for value in "${values[@]}"; do
        line+=" $(printf "%-${column_width}s" "$value") |"
    done
    printf "%s\n" "$line"
}

###############################################################

# Default values for command-line parsing variables
dft_task=""
nanopore_name=""
# nanopore_shape=""
# nanopore_size=""
# adsorb_type=""
molecule_name=""
theta_angle=""
phi_angle=""
psi_angle=""
vdw_type=""

num_nodes=""
ntasks_per_node=""
cpus_per_task=""
wall_time=""

###############################################################

# Function to display parsing variables
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "A simple Bash script with options."
    echo "Options:"
    echo "  --help               Show the help message"
    echo "  --dft_task           Set the DFT task (SGLPT | SCF | PDOS | WAVE | CHAR | LEAD | DEVICE)"
    echo "  --nanopore_name      Specify the name of the nanopore"
    # echo "  --nanopore_shape     Specify the shape of the nanopore"
    # echo "  --nanopore_size      Specify the size of the nanopore"
    # echo "  --adsorb_type        Spefify the type of the adsorbant"
    echo "  --molecule_name      Specify the name of the molecule"
    echo "  --theta_angle        Specify the theta angle of the rotation"
    echo "  --phi_angle          Specify the phi angle of the rotation"
    echo "  --psi_angle          Specify the psi angle of the rotation"
    echo "  --vdw_type           Specify the type of the vdw correction"
    echo "  --num_nodes          Specify the the number of nodes"
    echo "  --ntasks_per_node    Specify the the number of MPI tasks per node"
    echo "  --cpus_per_node      Specify the the number of CPUs per task"
    echo "  --wall_time          Specify the CPU time"
    echo "  --time_stamp         Specify the current date/time info for a running log file"
}

###############################################################

# Main script logic
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            show_help
            exit 0
            ;;
        --dft_task)
            dft_task="$2"
            shift 2
            ;;
        --nanopore_name)
            nanopore_name="$2"
            shift 2
            ;;
        # --nanopore_shape)
        #     nanopore_shape="$2"
        #     shift 2
        #     ;;
        # --nanopore_size)
        #     nanopore_size="$2"
        #     shift 2
        #     ;;
        # --adsorb_type)
        #     adsorb_type="$2"
        #     shift 2
        #     ;;
        --molecule_name)
            molecule_name="$2"
            shift 2
            ;;
        --theta_angle)
            theta_angle="$2"
            shift 2
            ;;
        --phi_angle)
            phi_angle="$2"
            shift 2
            ;;
        --psi_angle)
            psi_angle="$2"
            shift 2
            ;;
        --vdw_type)
            vdw_type="$2"
            shift 2
            ;;
        --num_nodes)
            num_nodes="$2"
            shift 2
            ;;
        --ntasks_per_node)
            ntasks_per_node="$2"
            shift 2
            ;;
        --cpus_per_task)
            cpus_per_task="$2"
            shift 2
            ;;
        --wall_time)
            wall_time="$2"
            shift 2
            ;;
        --time_stamp)
            time_stamp="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

###############################################################

# force and displacment criteria for relaxation convergence
maxforce="0.01"    # MD.MaxForceTol (eV/Ang): Maximum residual force tolerance
maxdispl="0.1"     # MD.MaxCGDispl (Ang): Maximum atomic displacement
maxrelax="2000"    # MD.NumCGsteps: Maximum number of relaxation steps

# mixing parameters for scf convergence
scfmixer="0.01"    # DM.MixingWeight: mixing weight used to mix the electron density.
numpulay="20"      # DM.NumberPulay: number of previous SCF steps used for mixing
maxscfit="2000"   # MaxSCFIterations: Maximum number of SCF iterations

###############################################################

ROOT_DIR=$PWD

echo "ROOT_DIR = ${ROOT_DIR}"

DFT_DIR="${ROOT_DIR}/DFT"

mkdir -p ${DFT_DIR}

INPUT_DIR="${DFT_DIR}/INPUTS"

mkdir -p ${INPUT_DIR}

PSEUDO_DIR="${DFT_DIR}/INPUTS"

mkdir -p ${PSEUDO_DIR}

###############################################################

if [[ -n $nanopore_size ]]; then
    nanopore_size=$(printf "%02d" $nanopore_size)
fi

if [[ -n $theta_angle ]]; then
    theta_angle=$(printf "%03d" $theta_angle)
fi

if [[ -n $phi_angle ]]; then
    phi_angle=$(printf "%03d" $phi_angle)
fi

if [[ -n $psi_angle ]]; then
    psi_angle=$(printf "%03d" $psi_angle)
fi

###############################################################

DFT_TASK=${dft_task^^}
NANOPORE_NAME=${nanopore_name^^}
# NANOPORE_SHAPE=${nanopore_shape^^}
# NANOPORE_SIZE=${nanopore_size^^}
# ADSORB_TYPE=${adsorb_type^^}
MOLECULE_NAME=${molecule_name^^}
THETA_ANGLE=${theta_angle^^}
PHI_ANGLE=${phi_angle^^}
PSI_ANGLE=${psi_angle^^}
VDW_TYPE=${vdw_type^^}

###############################################################

WORK_DIR=""

if [[ -n ${NANOPORE_NAME} && -z ${MOLECULE_NAME} ]]; then
    WORK_DIR+="${NANOPORE_NAME}"
fi

if [[ -z ${NANOPORE_NAME} && -n ${MOLECULE_NAME} ]]; then
    WORK_DIR+="BIOMOL_${MOLECULE_NAME}"
fi

if [[ -n ${NANOPORE_NAME} && -n ${MOLECULE_NAME} ]]; then
    WORK_DIR+="${NANOPORE_NAME}_BIOMOL_${MOLECULE_NAME}"
fi

# WORK_DIR="${DFT_TASK}_${NANOPORE_NAME}"

# if [[ -n $NANOPORE_SHAPE ]]; then
#     WORK_DIR+="_SHAPE_${NANOPORE_SHAPE}"
# fi

# if [[ -n $NANOPORE_SIZE ]]; then
#     WORK_DIR+="_SIZE_${NANOPORE_SIZE}"
# fi

# if [[ -n $ADSORB_TYPE ]]; then
#     WORK_DIR+="_ADSORB_${ADSORB_TYPE}"
# fi

# if [[ -n $MOLECULE_NAME ]]; then
#     WORK_DIR+="_BIOMOL_${MOLECULE_NAME}"
# fi
#
# if [[ -n $VDW_TYPE ]]; then
#     WORK_DIR+="_VDW_${VDW_TYPE}"
# fi

WORK_DIR="${DFT_DIR}/${WORK_DIR}"

mkdir -p ${WORK_DIR}

echo "WORK_DIR = ${WORK_DIR}"

###############################################################

JOB_DIR="${DFT_TASK}"

if [[ -n $THETA_ANGLE ]]; then
    JOB_DIR+="_THETA_${THETA_ANGLE}"
fi

if [[ -n $PHI_ANGLE ]]; then
    JOB_DIR+="_PHI_${PHI_ANGLE}"
fi

if [[ -n $PSI_ANGLE ]]; then
    JOB_DIR+="_PSI_${PSI_ANGLE}"
fi

JOB_DIR="${WORK_DIR}/${JOB_DIR}"

mkdir -p ${JOB_DIR}

echo "JOB_DIR = ${JOB_DIR}"

###############################################################

FDF_NAME="${DFT_TASK}"

# if [[ -n $NANOPORE_SHAPE ]]; then
#     FDF_NAME+="_SHAPE_${NANOPORE_SHAPE}"
# fi

# if [[ -n $NANOPORE_SIZE ]]; then
#     FDF_NAME+="_SIZE_${NANOPORE_SIZE}"
# fi

# if [[ -n $ADSORB_TYPE ]]; then
#     FDF_NAME+="_ADSORB_${ADSORB_TYPE}"
# fi

if [[ -n $NANOPORE_NAME ]]; then
    FDF_NAME+="_${NANOPORE_NAME}"
fi

if [[ -n $MOLECULE_NAME ]]; then
    FDF_NAME+="_BIOMOL_${MOLECULE_NAME}"
fi

if [[ -n $THETA_ANGLE ]]; then
    FDF_NAME+="_THETA_${THETA_ANGLE}"
fi

if [[ -n $PHI_ANGLE ]]; then
    FDF_NAME+="_PHI_${PHI_ANGLE}"
fi

if [[ -n $PSI_ANGLE ]]; then
    FDF_NAME+="_PSI_${PSI_ANGLE}"
fi

if [[ -n $VDW_TYPE ]]; then
    FDF_NAME+="_VDW_${VDW_TYPE}"
fi

echo "FDF_NAME = ${FDF_NAME}"

###############################################################

# Define the table columns
# columns=("NANOPORE_NAME" "NANOPORE_SHAPE" "NANOPORE_SIZE" "ADSORB_TYPE" "MOLECULE_NAME" "VDW_TYPE" "THETA_ANGLE" "PHI_ANGLE" "COMPLETE" "CONVERGE")
# column_width=$(calculate_column_width $columns[@])

columns=("NANOPORE_NAME" "MOLECULE_NAME" "VDW_TYPE" "THETA_ANGLE" "PHI_ANGLE" "PSI_ANGLE" "COMPLETE" "CONVERGE")
column_width=$(calculate_column_width $columns[@])

# Print the table header into the screen
# print_horizontal_line
# print_table_row "NANOPORE_NAME" "NANOPORE_SHAPE" "NANOPORE_SIZE" "ADSORB_TYPE" "MOLECULE_NAME" "VDW_TYPE" "THETA_ANGLE" "PHI_ANGLE" "COMPLETE" "CONVERGE"
# print_horizontal_line

print_horizontal_line
print_table_row "NANOPORE_NAME" "MOLECULE_NAME" "VDW_TYPE" "THETA_ANGLE" "PHI_ANGLE" "PSI_ANGLE" "COMPLETE" "CONVERGE"
print_horizontal_line

###############################################################

LOG_DIR="${DFT_DIR}/OUTLOG"

mkdir -p ${LOG_DIR}

LOG_FILE="${DFT_TASK}"

if [[ -n $NANOPORE_NAME ]]; then
    LOG_FILE+="_${NANOPORE_NAME}"
fi
# if [[ -n $NANOPORE_SHAPE ]]; then
#     LOG_FILE+="_SHAPE_${NANOPORE_SHAPE}"
# fi
# if [[ -n $NANOPORE_SIZE ]]; then
#     LOG_FILE+="_SIZE_${NANOPORE_SIZE}"
# fi
# if [[ -n $ADSORB_TYPE ]]; then
#     LOG_FILE+="_ADSORB_${ADSORB_TYPE}"
# fi
if [[ -n $MOLECULE_NAME ]]; then
    LOG_FILE+="_BIOMOL_${MOLECULE_NAME}"
fi
# if [[ -n $VDW_TYPE ]]; then
#     LOG_FILE+="_VDW_${VDW_TYPE}"
# fi

if [[ -n $time_stamp ]]; then
    LOG_FILE="${LOG_DIR}/${LOG_FILE}_${time_stamp}.log"
else
    LOG_FILE="${LOG_DIR}/${LOG_FILE}.log"
fi

if [ ! -f $LOG_FILE ]; then
    # Print the table header to the file
    print_horizontal_line >> $LOG_FILE
    print_table_row "${columns[@]}" >> $LOG_FILE
    print_horizontal_line >> $LOG_FILE
fi

###############################################################

cd ${JOB_DIR}

COMPLETE="NO"
if [ -e "output" ]; then
    if grep -q "Job completed" "output"; then
        COMPLETE="YES"
        cd ${ROOT_DIR}
    else
        cd ${ROOT_DIR}
    fi
else
    cd ${ROOT_DIR}
fi

###############################################################

cd ${JOB_DIR}

CONVERGE="NO"
if [ -e "output" ]; then
    if grep -q "SCF cycle converged" "output"; then
        CONVERGE="YES"
        cd ${ROOT_DIR}
    else
        cd ${ROOT_DIR}
    fi
else
    cd ${ROOT_DIR}
fi

###############################################################

# Print the table data into the screen
# print_table_row "$NANOPORE_NAME" "$NANOPORE_SHAPE" "$NANOPORE_SIZE" "$ADSORB_TYPE" "$MOLECULE_NAME" "$VDW_TYPE" "$THETA_ANGLE" "$PHI_ANGLE" "$COMPLETE" "$CONVERGE"
# print_horizontal_line

print_table_row "$NANOPORE_NAME" "$MOLECULE_NAME" "$VDW_TYPE" "$THETA_ANGLE" "$PHI_ANGLE" "$PSI_ANGLE" "$COMPLETE" "$CONVERGE"
print_horizontal_line

###############################################################

if [[ "$COMPLETE" == "YES" && "$CONVERGE" == "YES" ]]; then
    exit 0 # 0: exit with success; 1: exit with general error
else
    # Print the table data into the file
    # print_table_row "$NANOPORE_NAME" "$NANOPORE_SHAPE" "$NANOPORE_SIZE" "$ADSORB_TYPE" "$MOLECULE_NAME" "$VDW_TYPE" "$THETA_ANGLE" "$PHI_ANGLE" "$COMPLETE" "$CONVERGE" >> $LOG_FILE
    # print_horizontal_line >> $LOG_FILE

    print_table_row "$NANOPORE_NAME" "$MOLECULE_NAME" "$VDW_TYPE" "$THETA_ANGLE" "$PHI_ANGLE" "$PSI_ANGLE" "$COMPLETE" "$CONVERGE" >> $LOG_FILE
    print_horizontal_line >> $LOG_FILE
fi

###############################################################

rm -rf ${JOB_DIR}

mkdir -p ${JOB_DIR}

cd ${JOB_DIR}

###############################################################

file_name="run.sh"

rm -rf $file_name

cat >${file_name} <<EOF
#!/bin/bash

#SBATCH --job-name=${DFT_TASK}
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=2
#SBATCH --time=${wall_time}
#SBATCH -o ./%J.out
#SBATCH -e ./%J.err

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

module reset
module load phys/Siesta/5.2.2-foss-2023b-mpi

# rm -rf WFS.nc
# ln -s /dev/shm/WFS.nc WFS.nc
EOF

echo "     " >> ${file_name}

# echo 'srun -n $SLURM_NTASKS_PER_NODE siesta < input.fdf > output' >> ${file_name}
echo 'srun siesta < input.fdf > output' >> ${file_name}

###############################################################

# cp $ROOTDIR/INPUT/C.psf C.psf
# cp $ROOTDIR/INPUT/H.psf H.psf
# cp $ROOTDIR/INPUT/O.psf O.psf
# cp $ROOTDIR/INPUT/N.psf N.psf
# cp $ROOTDIR/INPUT/B.psf B.psf
# cp $ROOTDIR/INPUT/Mo.psf Mo.psf
# cp $ROOTDIR/INPUT/S.psf S.psf

# List of elements for pseudopotentials
ELEMENTS=("C" "H" "O" "N" "B" "Mo" "S")

# Loop through the list to copy psf files
for ELEMENT in "${ELEMENTS[@]}"; do
    PSF_FILE="${PSEUDO_DIR}/${ELEMENT}.psf"
    if [[ -f "$PSF_FILE" ]]; then
        cp "$PSF_FILE" "${ELEMENT}.psf"
        echo "Successfully copied ${ELEMENT}.psf"
    else
        echo "Warning: ${ELEMENT}.psf not found in ${PSF_DIR}. Skipping..."
        exit 1
    fi
done

###############################################################

# cp ${INPUT_DIR}/${FDF_NAME}.fdf input.fdf

FDF_FILE="${INPUT_DIR}/${FDF_NAME}.fdf"

if [[ -f "$FDF_FILE" ]]; then
    cp ${FDF_FILE} input.fdf
    echo "Successfully copied ${FDF_NAME}.fdf"
else
    echo "Warning: ${FDF_NAME}.fdf not found in ${INPUT_DIR}. Skipping..."
    exit 1
fi

###############################################################

# # Adjust the relax parameters
# if grep -q "MD.MaxForceTol" "input.fdf"; then
#     sed -i "s|$(grep -w  MD.MaxForceTol input.fdf)|MD.MaxForceTol       ${maxforce}|g" input.fdf
# fi
#
# if grep -q "MD.MaxCGDispl" "input.fdf"; then
#     sed -i "s|$(grep -w  MD.MaxCGDispl input.fdf)|MD.MaxCGDispl        ${maxdispl}|g" input.fdf
# fi
#
# if grep -q "MD.NumCGsteps" "input.fdf"; then
#     sed -i "s|$(grep -w  MD.NumCGsteps input.fdf)|MD.NumCGsteps        ${maxrelax}|g" input.fdf
# fi

# Adjust the scf parameters

if grep -q "DM.MixingWeight" "input.fdf"; then
    sed -i "s|$(grep -w  DM.MixingWeight input.fdf)|DM.MixingWeight    ${scfmixer}|g" input.fdf
fi

if grep -q "DM.NumberPulay" "input.fdf"; then
    sed -i "s|$(grep -w  DM.NumberPulay input.fdf)|DM.NumberPulay     ${numpulay}|g" input.fdf
fi

if grep -q "MaxSCFIterations" "input.fdf"; then
    sed -i "s|$(grep -w  MaxSCFIterations input.fdf)|MaxSCFIterations   ${maxscfit}|g" input.fdf
fi


###############################################################

rm -rf *.out *.err

###############################################################

sbatch run.sh

###############################################################
