#!/bin/bash
# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================

# Global Paths
export BASE_DIR="/home/bholdren/systems"
export IMPACT_PATH="${BASE_DIR}/impact"
export PYTHON_SCRIPT="find_nmers.py"

# Filenames
export XTC_NAME="traj_ri_protein.xtc"
export TPR_NAME="topol_protein.tpr"

# Trajectory parameters
export START_TIME=200000        
export END_TIME=500000                     
export FRAME_CHUNK_SIZE=3751      
export DT=2.0
export OFFSET=200000.0

# N-mer extraction parameters
export N_MIN=2
export N_MAX=6
export N_CHUNKS=40

# Systems to process1
export MOLS=(
    "mol-13"
    "mol-14"
    "mol-20"
    "mol-22"
    "mol-26"
    "mol-36"
    "mol-37"
    "mol-53"
    "hexamer"
)
