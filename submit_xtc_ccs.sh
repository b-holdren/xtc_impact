#!/bin/bash
#SBATCH --job-name=xtc_chunk
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00                
#SBATCH --array=0-39                   
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user=bradyholdren@ucsb.edu
#SBATCH --mail-type=all

#==================================================
# GLOBAL CONFIG
SKIP=1 
START_TIME=200000 
END_TIME=500000 
CHUNKS=40 
IMPACT_PATH="/home/bholdren/systems/impact"
#==================================================

if [ ! -f "systems.txt" ]; then
    echo "ERROR: Could not find systems.txt in this folder!"
    exit 1
fi

mapfile -t RUN_FOLDER < <(grep -v '^\s*#' systems.txt | grep -v '^\s*$')

i=$SLURM_ARRAY_TASK_ID
set -euo pipefail
module load gromacs/2025.1

DURATION=$(awk "BEGIN {print $END_TIME - $START_TIME}")
CHUNK_SIZE=$(awk "BEGIN {print $DURATION / $CHUNKS}")
START_T=$(awk "BEGIN {print $START_TIME + ($i * $CHUNK_SIZE)}")

if [ $i -eq $((CHUNKS - 1)) ]; then
    END_T=$END_TIME
else
    END_T=$(awk "BEGIN {print $START_T + $CHUNK_SIZE}")
fi



for CURRENT_RUN in "${RUN_FOLDER[@]}"; do
    #====================================================
    #FILE WRITING PATHS FOR XTC AND TPR
    XTC_FILE="${CURRENT_RUN}/traj_comp.xtc"
    TPR_FILE="${CURRENT_RUN}/topol.tpr"
    OUT_DIR="${CURRENT_RUN}/scratch"                   
    DATA="${OUT_DIR}/chunk${i}_data.tmp"
    OUT_PATTERN="$OUT_DIR/chunk${i}_frame.pdb"
    #====================================================
    mkdir -p "$OUT_DIR"
    rm -f "$DATA"

    echo "========================================================="
    # GROMACS
    echo "Run $CURRENT_RUN | Array Job $i: Extracting ${START_T}ps to ${END_T}ps..."
    echo -e "1\n1\n" | gmx trjconv -s "$TPR_FILE" -f "$XTC_FILE" -o "$OUT_PATTERN" \
        -pbc mol -center -sep -skip "$SKIP" -b "$START_T" -e "$END_T" > /dev/null

    # IMPACT
    echo "Run $CURRENT_RUN | Array Job $i: Calculating CCS..."
    for f in "$OUT_DIR"/chunk${i}_frame*.pdb; do
        
        [ -e "$f" ] || continue
        
        TIME_PS=$(grep -m 1 "t=" "$f" | awk -F "t=" '{print $2}' | awk '{print $1}' || true)
        
        if [ -z "$TIME_PS" ]; then
            echo "ERROR: Could not find time in $f!"
            continue
        fi
        
        CCS=$("$IMPACT_PATH" "$f" | grep "CCS 1" | awk '{print $(NF-1)}' || true)
        
        if [ ! -z "$CCS" ]; then
            echo -e "${TIME_PS}\t${CCS}" >> "$DATA"
        fi
        rm -f "$f"
    done

done

echo "Array Job $i successfully completed all runs!"
