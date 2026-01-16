#!/bin/bash

ROOT_DIR=$PWD

ANGLE_MIN=0   # in degree
ANGLE_MAX=330 # in degree
ANGLE_STP=30  # in degree

for NANOPORE_NAME in "GRAPHENE"; do # "GRAPHENE" "MOS2_2H"; do

    for MOLECULE_NAME in "ALA"; do

        SOURCE_DIR=/scratch/hpc-prf-meop/Projects/2DMaterNanopore/DFT+NEGF+ML/DFT/OPTIC/${NANOPORE_NAME}_NANOPORE_BIOMOL_${MOLECULE_NAME}

        OBJECT_DIR=${ROOT_DIR}/DFT/OPTIC/${NANOPORE_NAME}_NANOPORE_BIOMOL_${MOLECULE_NAME}

        rm -rf $OBJECT_DIR

        mkdir -p $OBJECT_DIR

        for THETA_ANGLE in $(seq ${ANGLE_MIN} ${ANGLE_STP} ${ANGLE_MAX}); do

            THETA_ANGLE=$(printf "%03d" ${THETA_ANGLE})

            for PHI_ANGLE in $(seq ${ANGLE_MIN} ${ANGLE_STP} ${ANGLE_MAX}); do

                PHI_ANGLE=$(printf "%03d" ${PHI_ANGLE})

                for PSI_ANGLE in $(seq ${ANGLE_MIN} ${ANGLE_STP} ${ANGLE_MAX}); do

                    PSI_ANGLE=$(printf "%03d" ${PSI_ANGLE})

                    WORK_DIR="OPTIC_THETA_${THETA_ANGLE}_PHI_${PHI_ANGLE}_PSI_${PSI_ANGLE}"

                    mkdir -p ${OBJECT_DIR}/${WORK_DIR}

                    TransferPDB2LOC ${SOURCE_DIR}/${WORK_DIR}/optic.fdf ${OBJECT_DIR}/${WORK_DIR}
                    TransferPDB2LOC ${SOURCE_DIR}/${WORK_DIR}/output ${OBJECT_DIR}/${WORK_DIR}
                    TransferPDB2LOC ${SOURCE_DIR}/${WORK_DIR}/run.sh ${OBJECT_DIR}/${WORK_DIR}
                    TransferPDB2LOC ${SOURCE_DIR}/${WORK_DIR}/siesta.EIG ${OBJECT_DIR}/${WORK_DIR}
                    TransferPDB2LOC ${SOURCE_DIR}/${WORK_DIR}/siesta.KP ${OBJECT_DIR}/${WORK_DIR}
                    TransferPDB2LOC ${SOURCE_DIR}/${WORK_DIR}/siesta.EPSIMG ${OBJECT_DIR}/${WORK_DIR}

                done

            done

        done

    done

done
